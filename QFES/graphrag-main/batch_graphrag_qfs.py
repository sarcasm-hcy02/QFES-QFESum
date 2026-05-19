#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量 GraphRAG QFS：PAIRS 中的 text_path 指向 JSON/JSONL/JSON(.gz) 或 .txt
- 若为 JSON/JSONL：提取其中的 `text` 字段作为输入文本（可用 text_key 覆盖）
- 同一路径仅索引一次，后续 query 复用该索引
- 输出仅包含：
  {"generated_summary": "...", "reference_summary": "..."}
"""

import os, json, gzip, subprocess, shutil, time, sys, uuid, pathlib, re
from uuid import uuid4

# ========= 可按需修改的全局配置 =========
TEMPLATE_ROOT = "./ragtest"     # 模板 root：内含 .env 与 settings.yaml
OUTPUT_JSON   = "./ragtest/output/graphrag_results.json"
METHOD        = "global"        # global / local / drift
COMMUNITY_LVL = 1               # 1 更稳，2 更聚焦，0 更鸟瞰
KEEP_JOBS = True          # True: 保留每个临时工程目录（排错用）
DEFAULT_TEXT_KEY = "text"       # JSON 中取文本的字段名（可在每条中用 "text_key" 覆盖）

# 你的输入：一一对应 (JSON文件/文本路径, query, reference_summary)
PAIRS = [
    # 把示例路径改成你的实际路径（绝对或正确相对路径）
    {
      "text_path": "./QFESum/syria_t17/RAT/Arab_League.json",          # ← JSON 对象/数组/JSONL/JSON.gz 均可
      "query": "Arab League Involvement and intervention",
      "reference_summary": """Arab League chief Nabil Elaraby says he has agreed a series of measures with Assad to help end violence.Syria agrees to an Arab League plan to withdraw its army from cities, free political prisoners and hold talks with the opposition.The Arab League voted to suspend Syria, accusing it of failing to implement an Arab peace plan, and imposed sanctions.Elaraby meets representatives of Arab civil society and agrees to send a 500-strong fact-finding committee to Syria.Arab foreign ministers give Damascus 3 days to implement a road map to end the bloodshed and allow in observers.An Arab League deadline for Syria to end its repression passes with no sign of violence abating.The Arab League rejects a request by Damascus to amend plans to send a 500-strong monitoring mission to Syria.At an extraordinary meeting in Cairo, Arab foreign ministers tell Syria to work to end months of bloodshed "before it's too late."It will send Arab League Secretary-General Nabil Elaraby to Damascus to push for political and economic reforms.
      """

    }
]

# ======================================

def run(cmd, cwd=None, capture=True):
    p = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True
    )
    out, err = p.communicate()
    return p.returncode, (out or ""), (err or "")

def backoff_sleep(attempt, base=2.0, cap=30.0):
    time.sleep(min(cap, base * (attempt + 1)))

def make_job_root(template_root, base_dir):
    job_root = os.path.join(base_dir, f"job_{uuid.uuid4().hex[:8]}")
    print(f"  job_root: {job_root}")
    os.makedirs(job_root, exist_ok=True)
    # 拷贝 .env / settings.yaml
    for name in (".env", "settings.yaml"):
        src = os.path.join(template_root, name)
        dst = os.path.join(job_root, name)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"模板缺少 {name}: {src}")
        shutil.copyfile(src, dst)
    # 目录结构
    os.makedirs(os.path.join(job_root, "input"), exist_ok=True)
    os.makedirs(os.path.join(job_root, "logs"), exist_ok=True)
    return job_root

def write_text(job_root, text, filename):
    ip = os.path.join(job_root, "input", filename)
    with open(ip, "w", encoding="utf-8") as f:
        f.write((text or "").strip() + "\n")
    return ip

def slug(s, default="doc"):
    s = str(s) if s is not None else default
    s = re.sub(r"[^\w\-]+", "_", s)
    s = s.strip("_")
    return s or default

def open_textfile(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_json(path):
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8") as f:
        return json.load(f)

def iter_jsonl(path):
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def extract_text_from_source(path, text_key):
    p = path.lower()
    if p.endswith(".json") or p.endswith(".json.gz"):
        obj = read_json(path)
        if isinstance(obj, dict):
            # 单对象：直接取 text_key
            val = obj.get(text_key, "")
            return val if isinstance(val, str) else str(val)
        elif isinstance(obj, list):
            # 数组：拼接其中每个对象的 text_key
            parts = []
            for it in obj:
                if isinstance(it, dict) and text_key in it and isinstance(it[text_key], str):
                    parts.append(it[text_key])
            return "\n\n".join(parts)
        else:
            raise ValueError("JSON 结构既不是对象也不是数组")
    if p.endswith(".jsonl") or p.endswith(".jsonl.gz"):
        parts = []
        for it in iter_jsonl(path):
            if isinstance(it, dict) and text_key in it and isinstance(it[text_key], str):
                parts.append(it[text_key])
        return "\n\n".join(parts)
    # 非 JSON：当作普通文本文件
    return open_textfile(path)

def main():
    # 基本检查
    if not os.path.isdir(TEMPLATE_ROOT):
        print(f"✖ TEMPLATE_ROOT 不存在：{TEMPLATE_ROOT}", file=sys.stderr); sys.exit(1)
    if not os.path.isfile(os.path.join(TEMPLATE_ROOT, ".env")):
        print(f"✖ 模板缺少 .env：{TEMPLATE_ROOT}/.env", file=sys.stderr); sys.exit(1)
    if not os.path.isfile(os.path.join(TEMPLATE_ROOT, "settings.yaml")):
        print(f"✖ 模板缺少 settings.yaml：{TEMPLATE_ROOT}/settings.yaml", file=sys.stderr); sys.exit(1)
    if not PAIRS:
        print("✖ PAIRS 为空：请在脚本顶部填写 {text_path, query, reference_summary}。", file=sys.stderr); sys.exit(1)

    base_dir = os.path.abspath("./graphrag_jobs")
    os.makedirs(base_dir, exist_ok=True)

    results = []
    total = len(PAIRS)

    # 缓存：同一路径只索引一次
    indexed_roots = {}   # abs_path -> job_root
    created_roots = []

    for i, item in enumerate(PAIRS, 1):
        source_path = item.get("text_path")  # 兼容你现有的键名
        query = (item.get("query") or "").strip()
        ref_sum = item.get("reference_summary", "")
        text_key = item.get("text_key", DEFAULT_TEXT_KEY)

        if not source_path:
            print(f"\n[{i}/{total}] ✖ 缺少 text_path", file=sys.stderr)
            results.append({"generated_summary": "", "reference_summary": ref_sum})
            continue

        abs_path = os.path.abspath(source_path)
        ident = slug(pathlib.Path(abs_path).stem) or f"doc_{i}_{uuid4().hex[:6]}"
        print(f"\n[{i}/{total}] doc={ident}  method={METHOD}  community_level={COMMUNITY_LVL}")

        # 如果该路径还未索引，先读取文本并索引一次
        job_root = indexed_roots.get(abs_path)
        if job_root is None:
            if not os.path.isfile(abs_path):
                print(f"  ✖ 输入文件不存在：{abs_path}", file=sys.stderr)
                results.append({"generated_summary": "", "reference_summary": ref_sum})
                continue

            try:
                text = extract_text_from_source(abs_path, text_key=text_key)
            except Exception as e:
                print(f"  ✖ 解析文本失败（{abs_path}）：{e}", file=sys.stderr)
                results.append({"generated_summary": "", "reference_summary": ref_sum})
                continue

            if not isinstance(text, str) or not text.strip():
                print(f"  ✖ 解析结果为空（{abs_path}；text_key='{text_key}'）", file=sys.stderr)
                results.append({"generated_summary": "", "reference_summary": ref_sum})
                continue

            job_root = make_job_root(TEMPLATE_ROOT, base_dir)
            created_roots.append(job_root)
            write_text(job_root, text, filename=f"{ident}.txt")

            # index（带 429 退避）
            idx_cmd = ["graphrag", "index", "--root", job_root]
            for attempt in range(10):
                rc, out, err = run(idx_cmd)
                if rc == 0:
                    break
                msg = (out + err)
                if "429" in msg or "rate limit" in msg.lower() or "Rate limit" in msg:
                    backoff_sleep(attempt); continue
                else:
                    break
            if rc != 0:
                print(f"  ✖ index 失败：\n{err or out}", file=sys.stderr)
                results.append({"generated_summary": "", "reference_summary": ref_sum})
                if not KEEP_JOBS:
                    try: shutil.rmtree(job_root, ignore_errors=True)
                    except Exception: pass
                continue

            indexed_roots[abs_path] = job_root

        # query（带 429 退避）
        q_cmd = [
            "graphrag", "query",
            "--root", job_root,
            "--method", METHOD,
            "--community_level", str(COMMUNITY_LVL),
            "--query", query
        ]
        answer_text = None
        for attempt in range(12):
            rc, out, err = run(q_cmd)
            if rc == 0:
                answer_text = out.strip()
                break
            msg = (out + err)
            if "429" in msg or "rate limit" in msg.lower() or "Rate limit" in msg:
                backoff_sleep(attempt); continue
            else:
                break

        if answer_text is None:
            print(f"  ✖ query 失败：\n{err or out}", file=sys.stderr)
            results.append({"generated_summary": "", "reference_summary": ref_sum})
        else:
            print("  ✓ 完成")
            results.append({"generated_summary": answer_text, "reference_summary": ref_sum})

    # 统一清理临时工程
    if not KEEP_JOBS:
        for r in created_roots:
            try: shutil.rmtree(r, ignore_errors=True)
            except Exception: pass

    # 写出结果（仅两字段）
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_JSON)), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 已写出：{OUTPUT_JSON}（共 {len(results)} 条）")

if __name__ == "__main__":
    main()
