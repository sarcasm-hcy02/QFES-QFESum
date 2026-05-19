# -*- coding: utf-8 -*-
import os
import re
import json
import time
import signal
import subprocess
from pathlib import Path
from typing import List, Dict, Any

import requests
from tqdm import tqdm

# =========================================================
# 1) CONFIG
# =========================================================

tasks = [
    {
        "json_path": "./QFESum/yemen/unstructured_evidence/Protests.json",
        "query": "Protests by the masses and students",
        "reference": """Hundreds of students and other protesters gather at Sana'a University, calling for an end to the 32-year rule of President Ali Abdullah Saleh.The demonstrators were apparently inspired by the protests that led to the ouster of Tunisia's President.Yemeni protesters chant “the people want the regime to fall.”A day of antigovernment protests brings more than 20,000 people onto the streets in Sanaa.Tens of thousands of opposition activists demand the ouster of Saleh. Nearly 100,000 protest in Yemen's capital, deeming it the “Friday of No Return.”President Ali Abdullah Saleh's rule is further weakened when five key generals defect to join anti-government protesters.Thousands rally against him in “Day of Departure” protests.Thousands of anti-Saleh protesters march in the southern port of Aden. Huge crowds across Yemen demand Saleh leave.Opposition parades through Sanaa the bodies of 50 people it says were killed in clashes with Saleh's forces. Protesters have been struggling for months to overthrow President Ali Abdullah Saleh.Protests escalate as security forces fire on demonstrators, killing 21 and wounding dozens.
              """
    }
]


OUTPUT_JSON = r"./QFESum/yemen/unstructured_evidence/generated_summary_results.json"

MODEL = "qwen2.5:7b"   # change to your Ollama model name

PRIMARY_PORT = 11435
SECONDARY_PORT = 11434
RESTART_EVERY = 3               # switch Ollama serve every N tasks
WAIT_READY_TIMEOUT = 300
REQ_TIMEOUT = 1200

START_ENV = {
    "OLLAMA_FLASH_ATTENTION": "1",
    "OLLAMA_KV_CACHE_TYPE": "q8_0",
}

os.environ["NO_PROXY"] = "127.0.0.1,localhost"

MAX_CHUNK_RECORDS = 20
MAX_BOOK_CHARS = 12000
MAX_PASSAGES_CHARS = 8000
MAX_DRAFT_SUMMARY_CHARS = 6000

TEMPERATURE_REFINE = 0.3
TEMPERATURE_VALIDATE = 0.0
TEMPERATURE_CITE = 0.2

NUM_PREDICT_REFINE = 1000
NUM_PREDICT_VALIDATE = 50
NUM_PREDICT_CITE = 1200

# =========================================================
# 2) PROMPTS (ENGLISH ONLY)
# =========================================================

PROMPT_REFINE = """
Imagine that you are writing a research summary based on source texts.

These are the source texts:

{book}

Here is a query about the source texts:

{question}

This is the draft summary that you are refining:

{summary}

Please rewrite this response so that it is totally accurate and fully addresses the query.

Please make the response as specific and detail oriented as possible. The following passages from the source texts should help in crafting the response:

{passages}

**OUTPUT FORMAT**

Please wrap the content of the summary you write in a markdown codeblock, in other words, like:
"""

PROMPT_VALIDATE = """
Imagine that you are judging the quality of a summary of source texts. These are the source texts:

{book}

Here is a query about the source texts:

{question}

And here is the summary which addresses the query:

{summary}

Please judge if you think that the summary meets ALL of the following criteria:

1) The summary is absolutely faithful to the source texts
2) The summary FULLY addresses the query

Please think carefully about your answer. If you think that ALL of the criteria are met, please simply respond with "YES".

Otherwise, please simply respond with "NO".
"""

PROMPT_CITE = """
Imagine that you have written a research summary about source texts. You have also extracted passages from the source texts which you used to write the summary.
Your job is to add citations to the summary which properly reference the passages that you have extracted.

Here is the summary:

{essay}

And here are the evidence passages from the source texts, each of which is given a number:

{evidence}

Please add citations to all citation-worthy statements in the summary using the numbered evidence list, by indicating the citation numbers of the corresponding evidence.
More specifically, add the citation number at the end of each relevant sentence in the summary before the punctuation mark, e.g., 'This work shows the effectiveness of problem X [1].' when the passage [1] in the evidence list provides full support for the statement.
Only add a citation if it is fully relevant and unambiguously supportive of that sentence. Not all evidences may be relevant, so only cite those that directly support the statement.
Please do not add any explanations or justifications for the evidence, simply indicate the evidence numbers if they are relevant.
If a sentence does not use any of the provided evidence, please simply copy the sentence as is and do not add anything to the end of it.
If multiple evidences support a statement, please cite them together (e.g., [1][2]).
For each citation-worthy statement, you only need to add at least one citation, so if multiple evidences support the statement, just add the most relevant citation to the sentence.
"""

# =========================================================
# 3) OLLAMA BLUE-GREEN UTILITIES
# =========================================================

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
        raise RuntimeError(f"Failed to start new Ollama serve on port {standby_port}")
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

def chat_once(port: int, prompt: str, num_predict: int, temperature: float) -> str:
    url = base_url(port) + "/api/chat"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict
        }
    }
    r = requests.post(url, json=payload, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"]
    return data.get("response", "")

# =========================================================
# 4) TEXT UTILITIES
# =========================================================

CODE_BLOCK_REGEX = r"```(?:\w+)?\s*\n(.*?)(?=^```)\s*```"

def extract_codeblock(text: str) -> str:
    matches = re.findall(CODE_BLOCK_REGEX, text, re.DOTALL | re.MULTILINE)
    if matches:
        return matches[0].strip()
    return text.strip()

def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())

def dedup_texts(texts: List[str]) -> List[str]:
    seen = set()
    out = []
    for t in texts:
        nt = normalize_text(t)
        if nt and nt not in seen:
            seen.add(nt)
            out.append(t.strip())
    return out

def join_with_limit(texts: List[str], max_chars: int) -> str:
    out = []
    total = 0
    for t in texts:
        t = str(t).strip()
        if not t:
            continue
        add_len = len(t) + 2
        if total + add_len > max_chars:
            break
        out.append(t)
        total += add_len
    return "\n\n".join(out)

def chunk_records(records: List[Dict[str, Any]], chunk_size: int) -> List[List[Dict[str, Any]]]:
    chunks = []
    for i in range(0, len(records), chunk_size):
        chunks.append(records[i:i + chunk_size])
    return chunks

# =========================================================
# 5) PREPARE CHUNK MATERIAL
# =========================================================

def prepare_chunk_material(chunk: List[Dict[str, Any]]):
    texts = []
    responses = []
    evidence_all = []

    for rec in chunk:
        text = str(rec.get("text", "")).strip()
        if text:
            texts.append(text)

        response = str(rec.get("response", "")).strip()
        if response:
            responses.append(response)

        evs = rec.get("evidence", []) or []
        for ev in evs:
            ev = str(ev).strip()
            if ev:
                evidence_all.append(ev)

    evidence_all = dedup_texts(evidence_all)

    book_text = join_with_limit(texts, MAX_BOOK_CHARS)
    draft_summary = join_with_limit(responses, MAX_DRAFT_SUMMARY_CHARS)
    passages_text = join_with_limit(evidence_all, MAX_PASSAGES_CHARS)

    return book_text, draft_summary, evidence_all, passages_text

# =========================================================
# 6) CHUNK-LEVEL GENERATION
# =========================================================

def generate_chunk_summary(port: int, query: str, chunk: List[Dict[str, Any]]) -> Dict[str, Any]:
    book_text, draft_summary, evidence_list, passages_text = prepare_chunk_material(chunk)

    if not draft_summary:
        draft_summary = "No draft summary available."

    # Step 1: refine
    p_refine = PROMPT_REFINE.format(
        book=book_text,
        question=query,
        summary=draft_summary,
        passages=passages_text
    )
    refined_raw = chat_once(port, p_refine, NUM_PREDICT_REFINE, TEMPERATURE_REFINE)
    refined_summary = extract_codeblock(refined_raw)

    # Step 2: validate
    p_validate = PROMPT_VALIDATE.format(
        book=book_text,
        question=query,
        summary=refined_summary
    )
    validate_raw = chat_once(port, p_validate, NUM_PREDICT_VALIDATE, TEMPERATURE_VALIDATE).strip()

    # Step 3: cite
    numbered_evidence = "\n".join([f"[{i+1}] {ev}" for i, ev in enumerate(evidence_list)])
    p_cite = PROMPT_CITE.format(
        essay=refined_summary,
        evidence=numbered_evidence
    )
    cited_raw = chat_once(port, p_cite, NUM_PREDICT_CITE, TEMPERATURE_CITE).strip()

    return {
        "refined_summary": refined_summary,
        "validated": validate_raw,
        "cited_summary": cited_raw,
        "evidence": evidence_list,
        "chunk_size": len(chunk)
    }

# =========================================================
# 7) FINAL SUMMARY GENERATION
# =========================================================

def generate_final_summary(port: int, query: str, chunk_outputs: List[Dict[str, Any]], all_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    # collect chunk summaries
    chunk_summaries = [
        x["cited_summary"] for x in chunk_outputs
        if str(x.get("cited_summary", "")).strip()
    ]
    draft_summary = join_with_limit(chunk_summaries, MAX_DRAFT_SUMMARY_CHARS)

    # collect full source text
    all_texts = [
        str(x.get("text", "")).strip()
        for x in all_records
        if str(x.get("text", "")).strip()
    ]
    book_text = join_with_limit(all_texts, MAX_BOOK_CHARS)

    # collect global evidence
    all_evidence = []
    for x in chunk_outputs:
        all_evidence.extend(x.get("evidence", []))
    all_evidence = dedup_texts(all_evidence)
    passages_text = join_with_limit(all_evidence, MAX_PASSAGES_CHARS)

    # Step 1: refine final summary
    p_refine = PROMPT_REFINE.format(
        book=book_text,
        question=query,
        summary=draft_summary,
        passages=passages_text
    )
    refined_raw = chat_once(port, p_refine, NUM_PREDICT_REFINE, TEMPERATURE_REFINE)
    refined_summary = extract_codeblock(refined_raw)

    # Step 2: validate
    p_validate = PROMPT_VALIDATE.format(
        book=book_text,
        question=query,
        summary=refined_summary
    )
    validate_raw = chat_once(port, p_validate, NUM_PREDICT_VALIDATE, TEMPERATURE_VALIDATE).strip()

    # Step 3: cite final summary
    numbered_evidence = "\n".join([f"[{i+1}] {ev}" for i, ev in enumerate(all_evidence)])
    p_cite = PROMPT_CITE.format(
        essay=refined_summary,
        evidence=numbered_evidence
    )
    cited_raw = chat_once(port, p_cite, NUM_PREDICT_CITE + 300, TEMPERATURE_CITE).strip()

    return {
        "refined_summary": refined_summary,
        "validated": validate_raw,
        "generated_summary": cited_raw,
        "global_evidence": all_evidence
    }

# =========================================================
# 8) PROCESS ONE TASK
# =========================================================

def process_one_task(task: Dict[str, str], active_port: int) -> Dict[str, Any]:
    query = task["query"]
    json_path = task["json_path"]
    reference_summary = task.get("reference", "")

    print(f"\n=== Processing task ===")
    print(f"Query: {query}")
    print(f"Input JSON: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"{json_path} must be a JSON list.")

    valid_records = []
    for rec in records:
        has_response = bool(str(rec.get("response", "")).strip())
        has_evidence = bool(rec.get("evidence", []))
        if has_response or has_evidence:
            valid_records.append(rec)

    if not valid_records:
        return {
            "query": query,
            "json_path": json_path,
            "reference_summary": reference_summary,
            "generated_summary": "",
            "global_evidence": [],
            "num_records": len(records),
            "num_valid_records": 0,
            "num_chunks": 0,
            "chunk_outputs": [],
            "note": "No valid records with response/evidence."
        }

    chunks = chunk_records(valid_records, MAX_CHUNK_RECORDS)

    chunk_outputs = []
    for chunk in tqdm(chunks, desc=f"Chunk summarization"):
        out = generate_chunk_summary(active_port, query, chunk)
        chunk_outputs.append(out)

    if len(chunk_outputs) == 1:
        final_generated_summary = chunk_outputs[0]["cited_summary"]
        global_evidence = chunk_outputs[0]["evidence"]
        final_validation = chunk_outputs[0]["validated"]
    else:
        final_out = generate_final_summary(active_port, query, chunk_outputs, valid_records)
        final_generated_summary = final_out["generated_summary"]
        global_evidence = final_out["global_evidence"]
        final_validation = final_out["validated"]

    return {
        "query": query,
        "json_path": json_path,
        "reference_summary": reference_summary,
        "generated_summary": final_generated_summary,
        "global_evidence": global_evidence,
        "final_validation": final_validation,
        "num_records": len(records),
        "num_valid_records": len(valid_records),
        "num_chunks": len(chunks),
        "chunk_outputs": chunk_outputs
    }

# =========================================================
# 9) MAIN
# =========================================================

def main():
    ensure_no_brew_service()

    active_port = PRIMARY_PORT
    standby_port = SECONDARY_PORT
    active_proc = start_serve_on(active_port)

    if not wait_ready(active_port):
        stop_serve(active_proc)
        raise RuntimeError("Initial Ollama serve failed to start.")

    results = []

    try:
        for idx, task in enumerate(tasks):
            if idx > 0 and idx % RESTART_EVERY == 0:
                print(f"\n==> Reached {RESTART_EVERY} tasks, switching Ollama serve...")
                active_port, active_proc = blue_green_switch(active_port, standby_port, active_proc)
                standby_port = PRIMARY_PORT if active_port == SECONDARY_PORT else SECONDARY_PORT

            try:
                result = process_one_task(task, active_port)
            except requests.exceptions.RequestException as e:
                result = {
                    "query": task["query"],
                    "json_path": task["json_path"],
                    "reference_summary": task.get("reference", ""),
                    "generated_summary": "",
                    "error": f"Request error: {str(e)}"
                }
            except Exception as e:
                result = {
                    "query": task["query"],
                    "json_path": task["json_path"],
                    "reference_summary": task.get("reference", ""),
                    "generated_summary": "",
                    "error": f"Processing error: {str(e)}"
                }

            results.append(result)

            # save after each task
            output_path = Path(OUTPUT_JSON)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"Saved intermediate results to: {OUTPUT_JSON}")

    finally:
        stop_serve(active_proc)
        print("All tasks finished.")

if __name__ == "__main__":
    main()