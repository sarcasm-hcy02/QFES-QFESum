# -*- coding: utf-8 -*-
import os
import time
import json
import subprocess
import signal
import requests
from tqdm import tqdm
from prompt_event_extract import template


# ===== 1) Ollama / LLM configuration =====
MODEL = "qwen2.5:7b"

PRIMARY_PORT = 11434
SECONDARY_PORT = 11435

# Restart Ollama after processing this many documents.
# You can reduce this value if long-running inference becomes slow.
RESTART_EVERY = 1307

WAIT_READY_TIMEOUT = 300
REQ_TIMEOUT = 600

START_ENV = {
    "OLLAMA_FLASH_ATTENTION": "1",
    "OLLAMA_KV_CACHE_TYPE": "q8_0",
}

os.environ["NO_PROXY"] = "127.0.0.1,localhost"


# ===== 2) Task configuration =====
# The script reads each input_path, extracts query-focused events,
# then writes the results back to the same file by default.
#
# If you do not want to overwrite the input file, add "output_path" to each task.
TASKS = [
    {
        "query": "The emotions and protests of the masses",
        "input_path": r"./QFESum/egypt/RAT/mass_protest.json"
    },
    {
        "query": "The authorities' attitude towards protests and trials",
        "input_path": r"./QFESum/egypt/RAT/authorities_attitude.json"
    }
]


def base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def is_ready(port: int) -> bool:
    try:
        r = requests.get(base_url(port) + "/api/version", timeout=3)
        return r.ok
    except Exception:
        return False


def start_serve_on(port: int) -> subprocess.Popen:
    """
    Start a new Ollama serve instance on the specified port.
    """
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
    """
    Stop an Ollama serve process gracefully.
    """
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


def chat_once(port: int, prompt: str) -> str:
    """
    Call Ollama /api/chat once and return the model response text.
    """
    url = base_url(port) + "/api/chat"

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    r = requests.post(url, json=payload, timeout=REQ_TIMEOUT)
    r.raise_for_status()

    data = r.json()

    if "message" in data and "content" in data["message"]:
        return data["message"]["content"]

    return data.get("response", "")


def blue_green_switch(active_port: int, standby_port: int, active_proc: subprocess.Popen):
    """
    Start a new Ollama instance on the standby port, wait until it is ready,
    switch to the new instance, and stop the old one.
    """
    new_proc = start_serve_on(standby_port)
    ok = wait_ready(standby_port)

    if not ok:
        stop_serve(new_proc)
        raise RuntimeError(f"New Ollama serve failed to start on port {standby_port}.")

    stop_serve(active_proc)

    return standby_port, new_proc


def ensure_no_brew_service():
    """
    Stop the Homebrew-managed Ollama service if it occupies port 11434.
    This is mainly useful on macOS.
    """
    try:
        subprocess.run(
            ["brew", "services", "stop", "ollama"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception:
        pass


def parse_llm_events(resp_text: str):
    """
    Parse the LLM response and return a list of raw event strings.

    Expected LLM output:
    {
      "events": [
        "event sentence 1",
        "event sentence 2"
      ]
    }

    If the response is invalid or empty, return an empty list.
    """
    if not isinstance(resp_text, str) or not resp_text.strip():
        return [], "Empty response", resp_text

    try:
        parsed = json.loads(resp_text)
    except json.JSONDecodeError:
        return [], "Invalid JSON response", resp_text

    raw_events = parsed.get("events", [])

    if not isinstance(raw_events, list):
        return [], "Response missing valid 'events' list", resp_text

    event_texts = []

    for event in raw_events:
        if isinstance(event, str):
            text = event.strip()
            if text:
                event_texts.append(text)

        elif isinstance(event, dict):
            text = event.get("text", "")
            if isinstance(text, str) and text.strip():
                event_texts.append(text.strip())

    return event_texts, None, resp_text


def add_global_event_ids_after_extraction(data, start_eid: int = 1):
    """
    Add global integer event IDs after all documents in one task have been processed.

    Before this function:
        item["events"] = [
            "event sentence 1",
            "event sentence 2"
        ]

    After this function:
        item["events"] = [
            {"eid": 1, "text": "event sentence 1"},
            {"eid": 2, "text": "event sentence 2"}
        ]

    The eid is globally unique within the output file of one task.
    """
    next_eid = start_eid

    for item in data:
        raw_events = item.get("events", [])

        normalized_events = []

        if isinstance(raw_events, list):
            for event in raw_events:
                if isinstance(event, str):
                    text = event.strip()
                    if text:
                        normalized_events.append({
                            "eid": next_eid,
                            "text": text
                        })
                        next_eid += 1

                elif isinstance(event, dict):
                    text = event.get("text", "")
                    if isinstance(text, str) and text.strip():
                        normalized_events.append({
                            "eid": next_eid,
                            "text": text.strip()
                        })
                        next_eid += 1

        item["events"] = normalized_events

    return data


def main():
    ensure_no_brew_service()

    active_port = PRIMARY_PORT
    standby_port = SECONDARY_PORT

    active_proc = start_serve_on(active_port)

    if not wait_ready(active_port):
        stop_serve(active_proc)
        raise RuntimeError("Initial Ollama serve failed to start.")

    processed_cnt = 0

    try:
        for task in TASKS:
            query = task["query"]
            input_path = task["input_path"]
            output_path = task.get("output_path", input_path)

            print(f"\n🚀 Processing file: {input_path}")
            print(f"🔎 Query: {query}")

            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            results = []

            for i, item in enumerate(
                tqdm(data, desc=f"Extracting events from {os.path.basename(input_path)}")
            ):
                text = item.get("text", "")

                if not isinstance(text, str) or not text.strip():
                    item["events"] = []
                    item["event_extraction_error"] = "Empty input text"
                    results.append(item)
                    processed_cnt += 1
                    continue

                prompt = template.format(query=query, text=text)

                try:
                    if processed_cnt > 0 and processed_cnt % RESTART_EVERY == 0:
                        print(f"\n==> Processed {RESTART_EVERY} documents. Switching Ollama instance...")
                        active_port, active_proc = blue_green_switch(
                            active_port,
                            standby_port,
                            active_proc
                        )
                        standby_port = PRIMARY_PORT if active_port == SECONDARY_PORT else SECONDARY_PORT

                    resp_text = chat_once(active_port, prompt).strip()

                    event_texts, error_msg, raw_response = parse_llm_events(resp_text)

                    # At this stage, events are kept as raw strings.
                    # Global eid will be added only after all documents are processed.
                    item["events"] = event_texts

                    if error_msg is not None:
                        item["event_extraction_error"] = error_msg
                        item["raw_llm_response"] = raw_response

                except requests.exceptions.RequestException as e:
                    item["events"] = []
                    item["event_extraction_error"] = f"Request error: {str(e)}"

                except Exception as e:
                    item["events"] = []
                    item["event_extraction_error"] = f"Processing error: {str(e)}"

                results.append(item)
                processed_cnt += 1

            # Add eid only after all events in this task have been extracted.
            results = add_global_event_ids_after_extraction(results, start_eid=1)

            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            total_events = sum(len(item.get("events", [])) for item in results)

            print(f"✅ Finished: {output_path}")
            print(f"📌 Total extracted events with eid: {total_events}")

    finally:
        stop_serve(active_proc)
        print("🎉 All tasks completed.")


if __name__ == "__main__":
    main()