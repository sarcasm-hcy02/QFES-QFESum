#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Taking the timestamps into account,evaluate whether two prior news events are referring to the same event related
to the keyword. If the two events occur on the same date or within a short time span, and they are about the same topic
related to the keyword, then they should be considered as referring to the same event. If so, please respond directly
with ’yes’. If not, respond with ’no’.
—-
# Keyword
Bill Clinton
# Event 1
January 19, 2001 - The day before leaving office, Clinton agrees to give up his Arkansas law license for five
years, and to pay a $25,000 fine to the state bar association, ending efforts by the Arkansas Supreme Court Committee on
Professional Conduct to disbar him..
# Event 2
January 20, 2001 - Hours before leaving office, Clinton pardons 141 people, including Whitewater figure Susan
McDougal and publishing heiress Patty Hearst. The most controversial pardon is that of financier Marc Rich, who had been
a fugitive in Switzerland. The president also pardons his brother, Roger Clinton, who had been convicted on a cocaine charge
in the 1980s.
# Answer
No.
—-
# Keyword
Tiger Woods
Event 1
June 3, 2012 - With his win at the Memorial Tournament, ties Jack Nicklaus with 73 PGA Tour victories.
# Event 2
July 2, 2012 - Beats Nicklaus’ PGA Tour record with the AT&T National win. Woods’74th PGA Tour win ranks him in second
place on the all-time list.
# Answer
No.
—-
# Keyword
Mitt Romney
# Event 1
November 6, 2012 - Defeated in the general election by President Barack Obama. Romney wins 206 Electoral College
votes to Obama’s 332.
# Event 2
November 6, 2012 - President Barack Obama managed to secure his second term in office, triumphing over his Republican
rival, Mitt Romney.
# Answer
Yes.
—-
# Keyword
{keyword}
# Event 1
{event1}
# Event 2
{event2}
# Answer
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Event clustering + key event selection with QUERY conditioning
LLM "same-event" judging via local Ollama (qwen2.5-7b) or OpenAI-compatible endpoint.

Pipeline:
1) 读取 JSON → 展开内层 events（带格式检查）
2) 用 query 做相关性预筛（topK 或 阈值）
3) 近邻（每条看 top_n=20）+ LLM 判同 + 簇合并
4) 按簇大小排序
5) 以 medoid 选每簇关键事件
6) 保存 JSON

Example (Ollama):
ollama pull qwen2.5:7b-instruct
python cluster_events_qwen_ollama.py \
  --input /path/to/your.json \
  --output /path/to/out.json \
  --query "EU response and interference" \
  --use_llm \
  --llm_provider ollama \
  --ollama_base http://localhost:11434 \
  --llm_model qwen2.5:7b-instruct \
  --top_n 20 --query_topk 1000 --cos_threshold 0.83
"""

import json
import argparse
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np
import requests
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# 若用 OpenAI 兼容端点才需要（保持可选）
try:
    import openai
except Exception:
    openai = None


# ---------------------------
# 1) 读取 JSON 并展开内层 events（带格式检查）
# ---------------------------
def load_event_texts_from_json(json_path: str) -> List[Dict]:
    """
    外层 item 示例：
    {
      "title": null,
      "id": "1622",
      "text": "...",
      "time": "2010-04-06T00:00:00+00:00",
      "keyword": [],
      "score": 69.812,
      "events": [
        {"eid": 1, "text": "Calls were made for BP not to pay dividends ..."}
      ]
    }
    返回“原子事件”列表：
    { 'uid': '1622#1', 'parent_id': '1622', 'eid': 1, 'text': '...', 'score': 69.81, 'time': '...' }
    """
    atoms = []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        parent_id = item.get('id')
        time_val = item.get('time')
        score = item.get('score')
        evs = item.get('events', [])
        if not isinstance(evs, list):
            continue  # 不是列表就跳
        for ev in evs:
            if not isinstance(ev, dict):
                continue
            eid = ev.get('eid')
            raw_text = ev.get('text')  # 兼容不同格式：可能是 str，可能是 dict
            if isinstance(raw_text, str):
                text = raw_text.strip()
            elif isinstance(raw_text, dict):  # 假设字典里有 "content" 或 "value" 这种字段
                text = str(raw_text.get("content") or raw_text.get("value") or "").strip()
            else:
                text = ""
            if eid is None or not text:
                continue
            atoms.append({
                'uid': f"{parent_id}#{eid}",
                'parent_id': parent_id,
                'eid': eid,
                'text': text,
                'score': score,
                'time': time_val,
            })
    return atoms



# ---------------------------
# 2) 文档与向量库
# ---------------------------
def atoms_to_docs(atoms: List[Dict]) -> List[Document]:
    docs, seen = [], set()
    for i, a in enumerate(atoms):
        txt = a['text'].strip()
        if not txt:
            continue
        # 文本去重（可按需去掉）
        if txt in seen:
            continue
        seen.add(txt)
        docs.append(Document(
            page_content=txt,
            metadata={'uid': a['uid'], 'parent_id': a['parent_id'], 'eid': a['eid']}
        ))
    # 重要：为预筛后的“新顺序”写入 idx，避免近邻元数据错位
    for idx, d in enumerate(docs):
        d.metadata['idx'] = idx
    return docs


def init_vector_db(docs: List[Document], embedding_func: SentenceTransformerEmbeddings) -> Chroma:
    return Chroma.from_documents(docs, embedding_func, collection_metadata={"hnsw:space": "cosine"})


# ---------------------------
# 3) 基础嵌入工具
# ---------------------------
def embed_texts(embedding_func: SentenceTransformerEmbeddings, texts: List[str]) -> np.ndarray:
    X = np.array(embedding_func.embed_documents(texts), dtype=np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X


# ---------------------------
# 4) 用 query 做相关性预筛（topK 或 阈值）
# ---------------------------
def prefilter_by_query(
    docs: List[Document],
    embedding_func: SentenceTransformerEmbeddings,
    query: str,
    query_topk: Optional[int] = 1000,
    query_threshold: Optional[float] = None
) -> List[Document]:
    if not query:
        # 如无 query，直接返回
        return docs

    doc_texts = [d.page_content for d in docs]
    X = embed_texts(embedding_func, doc_texts)
    q = embed_texts(embedding_func, [query])[0]
    sims = (X @ q)

    if query_threshold is not None:
        keep_mask = sims >= float(query_threshold)
        kept = [d for d, m in zip(docs, keep_mask) if m]
        # 重写 idx
        for idx, d in enumerate(kept):
            d.metadata['idx'] = idx
        return kept

    K = min(len(docs), int(query_topk) if query_topk is not None else len(docs))
    order = np.argsort(-sims)[:K]
    kept = [docs[i] for i in order]
    for idx, d in enumerate(kept):
        d.metadata['idx'] = idx
    return kept


# ---------------------------
# 5) LLM 判同（Ollama / OpenAI），prompt 含 {keyword}
# ---------------------------
DEFAULT_FEWSHOT = """
Evaluate whether two prior news events are referring to the same event related to the keyword. 
If the two events are about the same topic related to the keyword, then they should be considered as referring to the same event. 
If so, please respond directly with ’yes’. If not, respond with ’no’.
—-
# Keyword
Bill Clinton
# Event 1
The day before leaving office, Clinton agrees to give up his Arkansas law license for five years, 
and to pay a $25,000 fine to the state bar association, ending efforts by the Arkansas Supreme Court Committee on
Professional Conduct to disbar him..
# Event 2
Hours before leaving office, Clinton pardons 141 people, including Whitewater figure Susan
McDougal and publishing heiress Patty Hearst. The most controversial pardon is that of financier Marc Rich, who had been
a fugitive in Switzerland. The president also pardons his brother, Roger Clinton, who had been convicted on a cocaine charge
in the 1980s.
# Answer
No.
—-
# Keyword
Tiger Woods
Event 1
With his win at the Memorial Tournament, ties Jack Nicklaus with 73 PGA Tour victories.
# Event 2
Beats Nicklaus’ PGA Tour record with the AT&T National win. Woods’74th PGA Tour win ranks him in second place on the all-time list.
# Answer
No.
—-
# Keyword
Mitt Romney
# Event 1
Defeated in the general election by President Barack Obama. Romney wins 206 Electoral College votes to Obama’s 332.
# Event 2
President Barack Obama managed to secure his second term in office, triumphing over his Republican rival, Mitt Romney.
# Answer
Yes.
—-
# Keyword
{keyword}
# Event 1
{event1}
# Event 2
{event2}
# Answer
"""

def call_ollama_generate(base_url: str, model: str, prompt: str, temperature: float = 0.0, stop=None, timeout: int = 1200) -> str:
    """
    使用 Ollama 原生 /api/generate 接口，非流式。
    文档: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
    """
    url = base_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if stop:
        payload["stop"] = stop
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()

def same_event_by_llm(
    e1: str,
    e2: str,
    keyword: str,
    provider: str,
    llm_model: str,
    fewshot_tmpl: str,
    ollama_base: Optional[str] = None,
    openai_api_base: Optional[str] = None
) -> bool:
    fewshot = fewshot_tmpl.format(keyword=keyword, event1=e1, event2=e2)
    prompt = f"{fewshot}\nKeyword: {keyword}\nEvent 1: {e1}\nEvent 2: {e2}\nAnswer: "

    # Ollama 分支（推荐）
    if provider == "ollama":
        if not ollama_base:
            raise ValueError("Using provider=ollama requires --ollama_base (e.g., http://localhost:11434)")
        out = call_ollama_generate(
            base_url=ollama_base,
            model=llm_model,
            prompt=prompt,
            temperature=0.0,
            stop=["\n", "----"]
        )
        out = out.strip().lower()
        return out.startswith("yes")

    # OpenAI 兼容分支（可选）
    elif provider == "openai":
        if openai is None:
            raise RuntimeError("openai package missing. Install openai==0.28.* or choose --llm_provider ollama.")
        if openai_api_base:
            openai.api_base = openai_api_base
        openai.api_key = "none"  # 通常本地兼容端点不验 key
        resp = openai.Completion.create(
            model=llm_model,
            prompt=prompt,
            max_tokens=2,
            temperature=0.0,
            stop=['\n', '----']
        )
        out = resp['choices'][0]['text'].strip().lower()
        return out.startswith("yes")

    else:
        raise ValueError("Unknown llm_provider. Use 'ollama' or 'openai'.")


# ---------------------------
# 6) 无时间戳聚类（近邻 + 判同 + 簇合并）
# ---------------------------
def cluster_events_no_time(
    docs: List[Document],
    embedding_func: SentenceTransformerEmbeddings,
    query: str,
    top_n: int,
    llm_provider: str,
    llm_model: str,
    fewshot_tmpl: str,
    ollama_base: Optional[str] = None,
    openai_api_base: Optional[str] = None,
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    返回:
      event_pool: {cluster_id: [doc_idx, ...]}
      event2cluster: {doc_idx: cluster_id}
    """
    db = init_vector_db(docs, embedding_func)
    event_pool: Dict[int, List[int]] = {}
    event2cluster: Dict[int, int] = {}

    # 预嵌入
    X = embed_texts(embedding_func, [d.page_content for d in docs])

    for i, item in enumerate(tqdm(docs, desc="Clustering")):
        query_text = item.page_content

        TOPK = min(len(db.get()['ids']), top_n + 1)
        neighbors = db.similarity_search(query=query_text, k=TOPK)
        neighbors = [d for d in neighbors if d.page_content != query_text]

        matched_any = False
        cls_id = event2cluster.get(i, -1)

        for nb in neighbors:
            j = nb.metadata.get('idx', None)
            if j is None:
                # 兜底：通过内容匹配定位（O(N)，但安全）
                j = next(k for k, d in enumerate(docs) if d.page_content == nb.page_content)

            if i == j:
                continue
            if cls_id != -1 and event2cluster.get(j, -1) == cls_id:
                continue

            # 用 LLM 判同（基于 query 的 few-shot）
            same = same_event_by_llm(
                docs[i].page_content, docs[j].page_content, query,
                provider=llm_provider, llm_model=llm_model, fewshot_tmpl=fewshot_tmpl,
                ollama_base=ollama_base, openai_api_base=openai_api_base
            )
            if not same:
                continue

            neighbor_cls = event2cluster.get(j, -1)

            if cls_id == -1 and neighbor_cls == -1:
                new_cid = max(list(event_pool.keys()) or [-1]) + 1
                event_pool[new_cid] = [i, j]
                event2cluster[i] = new_cid
                event2cluster[j] = new_cid
                cls_id = new_cid
            elif cls_id == -1 and neighbor_cls != -1:
                event_pool[neighbor_cls].append(i)
                event2cluster[i] = neighbor_cls
                cls_id = neighbor_cls
            elif cls_id != -1 and neighbor_cls == -1:
                event_pool[cls_id].append(j)
                event2cluster[j] = cls_id
            elif cls_id != -1 and neighbor_cls != -1 and cls_id != neighbor_cls:
                # 簇合并（并到 cls_id）
                event_pool[cls_id] = list(set(event_pool[cls_id] + event_pool[neighbor_cls]))
                for e in event_pool[cls_id]:
                    event2cluster[e] = cls_id
                del event_pool[neighbor_cls]

            matched_any = True

        if not matched_any and cls_id == -1:
            new_cid = max(list(event_pool.keys()) or [-1]) + 1
            event_pool[new_cid] = [i]
            event2cluster[i] = new_cid

    try:
        db.delete_collection()
    except Exception:
        pass

    return event_pool, event2cluster

def textrank_cluster_sentences(
    sentences,
    embedding_func,
    k=2,
    d=0.85,
    max_iter=50,
    tol=1e-4
):
    """
    对簇内句子运行 TextRank（基于 GTE 余弦相似度）并返回排名前 k 的句子列表。
    """
    if not sentences:
        return []
    if len(sentences) <= k:
        return sentences

    # 1) 向量
    X = np.array(embedding_func.embed_documents(sentences), dtype=np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    # 2) 相似度矩阵（对角置零，避免自环放大）
    S = X @ X.T
    np.fill_diagonal(S, 0.0)

    # 3) 行归一化 -> 转移矩阵
    P = S / (S.sum(axis=1, keepdims=True) + 1e-12)

    # 4) TextRank (PageRank)
    N = len(sentences)
    scores = np.ones(N, dtype=np.float32) / N
    base = (1.0 - d) / N
    for _ in range(max_iter):
        prev = scores.copy()
        scores = base + d * (P.T @ scores)
        if np.linalg.norm(scores - prev, ord=1) < tol:
            break

    # 5) 取 top-k
    order = np.argsort(-scores)[:k]
    return [sentences[i] for i in order]
# ---------------------------
# 7) 选关键事件：每簇选 medoid（或前 topM）
# ---------------------------
def pick_cluster_representatives(
    docs: List[Document],
    embedding_func: SentenceTransformerEmbeddings,
    event_pool: Dict[int, List[int]],
    topM_per_cluster: int = 1
) -> List[Tuple[int, List[int]]]:
    texts = [d.page_content for d in docs]
    X = embed_texts(embedding_func, texts)

    results = []
    for cid, idxs in event_pool.items():
        if not idxs:
            continue
        E = X[idxs]              # [n, dim]
        S = E @ E.T              # 余弦相似度
        D = 1.0 - S              # 距离
        avgD = D.mean(axis=1)
        order = np.argsort(avgD)  # 升序，越小越中心
        reps = [idxs[k] for k in order[:topM_per_cluster]]
        results.append((cid, reps))
    return results


# ---------------------------
# 8) 串起流程并保存结果
# ---------------------------
def run_pipeline(
    input_path: str,
    output_path: str,
    query: str,
    top_l: int,                         # 非默认参数放前面
    k_per_cluster: int,                 # 非默认参数放前面
    summary_mode: str,                  # 非默认参数放前面

    # 下面开始给默认值
    embed_model: str = "./model/GTE_large",
    top_n: int = 20,
    topM_per_cluster: int = 1,          # 保留以兼容 medoid 分支
    # LLM 相关
    use_llm: bool = True,
    llm_provider: str = "ollama",       # 'ollama' or 'openai'
    llm_model: str = "qwen2.5:7b",
    ollama_base: str = "http://127.0.0.1:11434",
    openai_api_base: str = "",
    fewshot_tmpl: str = DEFAULT_FEWSHOT,
    # Query 预筛
    query_topk: Optional[int] = 1000,
    query_threshold: Optional[float] = None,
):
    atoms = load_event_texts_from_json(input_path)
    base_docs = atoms_to_docs(atoms)
    if not base_docs:
        raise ValueError("No events found after loading/cleaning input JSON.")

    embedding_func = HuggingFaceEmbeddings(
        model_name=embed_model,  # 本地 GTE 目录或 HF 名称
        model_kwargs={"local_files_only": True},  # 只用本地
        encode_kwargs={"normalize_embeddings": True}  # ← 别放 show_progress_bar
    )

    # 1) query 预筛
    docs = prefilter_by_query(
        base_docs, embedding_func, query,
        query_topk=query_topk,
        query_threshold=query_threshold
    )
    if not docs:
        raise ValueError("No events left after query prefiltering. Try relaxing --query_topk or lowering --query_threshold.")

    # 2) LLM 判同聚类（你本来就要求必须用 LLM）
    if not use_llm:
        raise ValueError("This pipeline is configured for LLM judging. Set --use_llm.")

    event_pool, event2cluster = cluster_events_no_time(
        docs=docs,
        embedding_func=embedding_func,
        query=query,
        top_n=top_n,
        llm_provider=llm_provider,
        llm_model=llm_model,
        fewshot_tmpl=fewshot_tmpl,
        ollama_base=ollama_base if llm_provider == "ollama" else None,
        openai_api_base=openai_api_base if llm_provider == "openai" else None
    )

    # 3) 簇排序
    clusters_sorted = sorted(event_pool.items(), key=lambda kv: len(kv[1]), reverse=True)

    # 4) 取前 l 个簇
    top_clusters = clusters_sorted[:top_l]

    # 5) 簇内摘要（TextRank + GTE 或 Medoid）
    out_clusters = []
    if summary_mode == "textrank":
        for cid, members in top_clusters:
            sentences = [docs[i].page_content for i in members]
            summary_sents = textrank_cluster_sentences(
                sentences, embedding_func, k=k_per_cluster
            )
            out_clusters.append({
                "cluster_id": cid,
                "size": len(members),
                "summary_sentences": summary_sents,
                "member_uids": [docs[i].metadata['uid'] for i in members],
            })
    elif summary_mode == "medoid":
        event_pool_top = {cid: members for cid, members in top_clusters}
        reps = pick_cluster_representatives(
            docs=docs,
            embedding_func=embedding_func,
            event_pool=event_pool_top,
            topM_per_cluster=k_per_cluster
        )
        cid2reps = dict(reps)
        for cid, members in top_clusters:
            rep_idxs = cid2reps.get(cid, [])
            out_clusters.append({
                "cluster_id": cid,
                "size": len(members),
                "representatives": [docs[i].page_content for i in rep_idxs],
                "member_uids": [docs[i].metadata['uid'] for i in members],
            })
    else:
        raise ValueError("summary_mode must be 'textrank' or 'medoid'.")

    # 6) 保存结果
    # 6) 保存结果
    result = {
        "meta": {
            "input": input_path,
            "query": query,
            "embed_model": embed_model,
            "top_n": top_n,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "summary_mode": summary_mode,
            "top_l": top_l,
            "k_per_cluster": k_per_cluster,
            "query_topk": query_topk,
            "query_threshold": query_threshold,
        },
        "prefiltered_docs": len(docs),
        "num_clusters_total": len(clusters_sorted),
        "num_clusters_kept": len(out_clusters),
        "clusters": out_clusters
    }

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[Done] Saved clustering+summaries to: {output_path}")

    return result

def _collect_summary_sentences(result: Dict) -> List[str]:
    """
    从 run_pipeline 的 result 中提取所有摘要句子（自动兼容 textrank/medoid）。
    """
    sents = []
    mode = result["meta"]["summary_mode"]
    for cl in result.get("clusters", []):
        if mode == "textrank":
            sents.extend(cl.get("summary_sentences", []))
        elif mode == "medoid":
            sents.extend(cl.get("representatives", []))
    # 去重并保序
    seen, dedup = set(), []
    for s in sents:
        ss = s.strip()
        if ss and ss not in seen:
            seen.add(ss)
            dedup.append(ss)
    return dedup

def _sentences_to_paragraph(sentences: List[str]) -> str:
    # 简单空格拼接；需要更自然可加句号补全/规则化
    return " ".join(sentences)

def run_batch(
    manifest_path: str,
    output_path: str,
    # 共享参数：
    embed_model: str,
    top_n: int,
    top_l: int,
    topM_per_cluster: int,
    k_per_cluster: int,
    summary_mode: str,
    use_llm: bool,
    llm_provider: str,
    llm_model: str,
    ollama_base: str,
    openai_api_base: str,
    query_topk: Optional[int],
    query_threshold: Optional[float],
):
    """
    manifest 是一个 JSON 列表：
    [
      {"input": "/path/to/file1.json", "query": "Query 1"},
      {"input": "/path/to/file2.json", "query": "Query 2"}
    ]
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        jobs = json.load(f)
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("batch manifest 必须是非空 JSON 列表，元素为 {input, query}。")

    all_items = []
    for idx, job in enumerate(jobs, start=1):
        in_path = job.get("input")
        query = job.get("query")
        if not in_path or not query:
            raise ValueError(f"第 {idx} 项缺少 'input' 或 'query'。")

        print(f"\n=== Batch {idx}/{len(jobs)} ===")
        res = run_pipeline(
            input_path=in_path,
            output_path=None,  # 批处理时先不各自落盘
            query=query,
            embed_model=embed_model,
            top_n=top_n,
            top_l=top_l,
            topM_per_cluster=topM_per_cluster,
            k_per_cluster=k_per_cluster,
            summary_mode=summary_mode,
            use_llm=use_llm,
            llm_provider=llm_provider,
            llm_model=llm_model,
            ollama_base=ollama_base,
            openai_api_base=openai_api_base,
            query_topk=query_topk,
            query_threshold=query_threshold,
        )
        sents = _collect_summary_sentences(res)
        paragraph = _sentences_to_paragraph(sents)

        all_items.append({
            "input": in_path,
            "query": query,
            "prefiltered_docs": res.get("prefiltered_docs"),
            "num_clusters_total": res.get("num_clusters_total"),
            "num_clusters_kept": res.get("num_clusters_kept"),
            "paragraph": paragraph,      # ← 汇总成段
            "sentences": sents,          # 如不需要可删
            "clusters": res["clusters"], # 如不需要可删
            "meta": res["meta"],         # 保留可复现
        })

    combined = {
        "batch_meta": {
            "manifest": manifest_path,
            "n_jobs": len(all_items),
            "shared_params": {
                "embed_model": embed_model,
                "top_n": top_n,
                "top_l": top_l,
                "topM_per_cluster": topM_per_cluster,
                "k_per_cluster": k_per_cluster,
                "summary_mode": summary_mode,
                "use_llm": use_llm,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "ollama_base": ollama_base,
                "openai_api_base": openai_api_base,
                "query_topk": query_topk,
                "query_threshold": query_threshold,
            }
        },
        "items": all_items
    }

    # （可选）全局合并所有句子为一个大段落：
    # all_sents = []
    # for it in all_items: all_sents.extend(it["sentences"])
    # combined["global_paragraph"] = _sentences_to_paragraph(list(dict.fromkeys(all_sents)))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"[Done] 批量结果已写入: {output_path}")


# ---------------------------
# 9) Main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="面向查询的事件聚类与关键事件选择（支持 LLM 判同；单文件/批处理）。")
    # 单文件参数
    p.add_argument("--input", help="单文件：输入 JSON 路径")
    p.add_argument("--query", help="单文件：用于预筛和 LLM 提示词的查询")
    p.add_argument("--output", required=True, help="输出 JSON 路径（单文件或批量合并）")

    # 批处理清单
    p.add_argument("--batch_manifest", default=None,
                   help="批处理清单 JSON（列表），元素为 {\"input\": 路径, \"query\": 查询}")

    p.add_argument("--embed_model", default="thenlper/gte-large", help="Sentence-Transformer 模型名或本地路径")
    p.add_argument("--top_n", type=int, default=20, help="每个事件取相似邻居的数量")
    p.add_argument("--topM_per_cluster", type=int, default=1, help="medoid模式：每簇代表数")
    p.add_argument("--top_l", type=int, default=5, help="按簇大小选前 l 个簇")
    p.add_argument("--k_per_cluster", type=int, default=2, help="textrank模式：每簇 top-k 句")
    p.add_argument("--summary_mode", choices=["textrank", "medoid"], default="textrank",
                   help="簇内摘要方式：TextRank+GTE 或 medoid")

    # LLM 选项
    p.add_argument("--use_llm", action="store_true", help="启用 LLM 判同")
    p.add_argument("--llm_provider", default="ollama", choices=["ollama", "openai"], help="LLM 提供方")
    p.add_argument("--llm_model", default="qwen2.5:7b-instruct", help="模型名称")
    p.add_argument("--ollama_base", default="http://localhost:11434", help="Ollama 基地址（provider=ollama 时）")
    p.add_argument("--openai_api_base", default="", help="OpenAI 兼容 API base（provider=openai 时）")

    # Query 预筛
    p.add_argument("--query_topk", type=int, default=1000, help="保留与 query 最相似的前 K（与阈值互斥）")
    p.add_argument("--query_threshold", type=float, default=None, help="仅保留余弦(query,event)≥阈值的项")
    return p.parse_args()



if __name__ == "__main__":
    args = parse_args()

    if args.batch_manifest:
        # 批处理模式：多个 {input, query}，写一个合并输出
        run_batch(
            manifest_path=args.batch_manifest,
            output_path=args.output,
            embed_model=args.embed_model,
            top_n=args.top_n,
            top_l=args.top_l,
            topM_per_cluster=args.topM_per_cluster,
            k_per_cluster=args.k_per_cluster,
            summary_mode=args.summary_mode,
            use_llm=args.use_llm,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            ollama_base=args.ollama_base,
            openai_api_base=args.openai_api_base,
            query_topk=args.query_topk,
            query_threshold=args.query_threshold,
        )
    else:
        # 单文件模式：保持原行为，但现在 run_pipeline 会返回结果
        if not args.input or not args.query:
            raise ValueError("单文件模式需要同时提供 --input 与 --query（或使用 --batch_manifest）。")
        _ = run_pipeline(
            input_path=args.input,
            output_path=args.output,
            query=args.query,
            embed_model=args.embed_model,
            top_n=args.top_n,
            top_l=args.top_l,
            topM_per_cluster=args.topM_per_cluster,
            k_per_cluster=args.k_per_cluster,
            summary_mode=args.summary_mode,
            use_llm=args.use_llm,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            ollama_base=args.ollama_base,
            openai_api_base=args.openai_api_base,
            query_topk=args.query_topk,
            query_threshold=args.query_threshold,
        )
