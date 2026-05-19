# -*- coding: utf-8 -*-
import os
import time
import json
import subprocess
import signal
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import requests
from tqdm import tqdm

# ===== 1) 蓝绿切换参数（可按需改） =====
MODEL = "qwen2.5:7b"        # 改成你的 ollama 模型名
PRIMARY_PORT = 11435
SECONDARY_PORT = 11434
RESTART_EVERY = 200          # 每处理多少条 text 切一次；建议 50~300
WAIT_READY_TIMEOUT = 300
REQ_TIMEOUT = 600
START_ENV = {
    "OLLAMA_FLASH_ATTENTION": "1",
    "OLLAMA_KV_CACHE_TYPE": "q8_0",
}
# ====================================

os.environ["NO_PROXY"] = "127.0.0.1,localhost"

# 你自己的任务列表
TASKS = [


    {
        "query": "The statements and stances of the Syrian government and authorities",
        "input_json": r"./QFESum/syria_crisis/unstructured_evidence/government_authorities.json",

    }
]






TEMPERATURE = 0.0
NUM_PREDICT = 600
MAX_TEXT_CHARS = 6000   # 单条 text 最长截断长度，防止过长

SYSTEM_PROMPT = "You are a long-context query-focused summarization assistant."

USER_PROMPT_TEMPLATE = """Your task is to read the source text and answer the following query by first extracting evidence.

Query: {question_text}

You should identify every passage in the source text that is directly relevant to answering the query.
Please copy the exact text of each passage (do NOT paraphrase).

After extracting the evidence, write a brief response to the query based only on the extracted evidence.

Please limit to at most 10 pieces of evidence.

Here is the source text:
{context}

**OUTPUT FORMAT**
Output your response as:
EVIDENCE:
[1] Extracted passage 1
[2] Extracted passage 2
...
[N] Extracted passage N
RESPONSE:
response
"""


def base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def is_ready(port: int) -> bool:
    try:
        r = requests.get(base_url(port) + "/api/version", timeout=3)
        return r.ok
    except Exception:
        return False


def start_serve_on(port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(START_ENV)
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    p = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )
    return p


def stop_serve(proc: subprocess.Popen):
    if proc and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            for _ in range(10):
                if proc.poll() is not None:
                    break
                time.sleep(0.5)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def wait_ready(port: int, timeout: int = WAIT_READY_TIMEOUT) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if is_ready(port):
            return True
        time.sleep(0.5)
    return False


def blue_green_switch(active_port: int, standby_port: int, active_proc: subprocess.Popen):
    new_proc = start_serve_on(standby_port)
    ok = wait_ready(standby_port)
    if not ok:
        stop_serve(new_proc)
        raise RuntimeError(f"新 serve 在 {standby_port} 启动失败")
    stop_serve(active_proc)
    return standby_port, new_proc


def ensure_no_brew_service():
    try:
        subprocess.run(
            ["brew", "services", "stop", "ollama"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception:
        pass


def chat_once(port: int, system_prompt: str, user_prompt: str) -> str:
    url = base_url(port) + "/api/chat"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT
        }
    }
    r = requests.post(url, json=payload, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"]
    return data.get("response", "")


def parse_evidence_and_response(text: str) -> Tuple[List[str], str]:
    upper = text.upper()
    ev_pos = upper.find("EVIDENCE:")
    resp_pos = upper.find("RESPONSE:")

    if ev_pos == -1 and resp_pos == -1:
        return [], text.strip()

    if ev_pos != -1 and resp_pos != -1 and ev_pos < resp_pos:
        evidence_block = text[ev_pos + len("EVIDENCE:"):resp_pos].strip()
        response_block = text[resp_pos + len("RESPONSE:"):].strip()
    elif resp_pos != -1:
        evidence_block = ""
        response_block = text[resp_pos + len("RESPONSE:"):].strip()
    else:
        evidence_block = text[ev_pos + len("EVIDENCE:"):].strip()
        response_block = ""

    evidence_items = []
    pattern = re.compile(r"\[\d+\]\s*(.*?)(?=(?:\n\[\d+\])|\Z)", re.S)
    for m in pattern.finditer(evidence_block):
        item = m.group(1).strip()
        if item:
            evidence_items.append(item)

    return evidence_items, response_block


def build_prompt(query: str, text: str) -> str:
    text = (text or "").strip()
    text = text[:MAX_TEXT_CHARS]
    return USER_PROMPT_TEMPLATE.format(
        question_text=query,
        context=text
    )


def process_one_file(task: Dict[str, str], active_port: int, processed_cnt: int) -> int:
    query = task["query"]
    input_json = Path(task["input_json"])

    print(f"\n🚀 开始处理文件: {input_json}")
    print(f"Query: {query}")

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{input_json} must be a JSON list.")

    results = []
    for item in tqdm(data, desc=f"Extracting evidence from {input_json.name}"):
        text = str(item.get("text", "")).strip()

        if not text:
            item["query"] = query
            item["evidence"] = []
            item["response"] = ""
            item["raw_output"] = ""
            item["evidence_error"] = "Empty text"
            results.append(item)
            processed_cnt += 1
            continue

        prompt = build_prompt(query, text)

        try:
            raw_output = chat_once(
                port=active_port,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt
            ).strip()

            evidences, response = parse_evidence_and_response(raw_output)

            item["query"] = query
            item["evidence"] = evidences
            item["response"] = response
            item["raw_output"] = raw_output

        except requests.exceptions.RequestException as e:
            item["query"] = query
            item["evidence"] = []
            item["response"] = ""
            item["raw_output"] = ""
            item["evidence_error"] = f"Request error: {str(e)}"

        except Exception as e:
            item["query"] = query
            item["evidence"] = []
            item["response"] = ""
            item["raw_output"] = ""
            item["evidence_error"] = f"Processing error: {str(e)}"

        results.append(item)
        processed_cnt += 1

    # 按你的要求：输出路径和输入路径相同，直接覆盖写回
    with open(input_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 文件处理完成: {input_json}")
    return processed_cnt


def main():
    ensure_no_brew_service()

    active_port = PRIMARY_PORT
    standby_port = SECONDARY_PORT
    active_proc = start_serve_on(active_port)
    if not wait_ready(active_port):
        stop_serve(active_proc)
        raise RuntimeError("初始 serve 启动失败")

    processed_cnt = 0

    try:
        for task in TASKS:
            # 文件开始前看是否需要切换
            if processed_cnt > 0 and processed_cnt % RESTART_EVERY == 0:
                print(f"\n==> 达到 {RESTART_EVERY} 条 text，进行蓝绿切换刷新性能……")
                active_port, active_proc = blue_green_switch(active_port, standby_port, active_proc)
                standby_port = PRIMARY_PORT if active_port == SECONDARY_PORT else SECONDARY_PORT

            processed_cnt = process_one_file(task, active_port, processed_cnt)

    finally:
        stop_serve(active_proc)
        print("🎉 全部任务完成")


if __name__ == "__main__":
    main()