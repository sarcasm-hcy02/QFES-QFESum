from BGE_calculate_id_list_score import calculate_similarity_scores
from LLM_id_list_function import check_relevance
from LLM_retrieve_function import llm_sentencewise_relevance
import json
from tqdm import tqdm  # ✅ NEW: 进度条
import random
from sample_doc_ids import sample_document_ids

# === 配置：选择一种模式 ===
LLM_PROGRESS_MODE = "first_n"   # 选 "first_n" | "random_p" | "reservoir_k"
LLM_PROGRESS_N = 5              # 模式 first_n：只对前 N 个灰区文档开进度
LLM_PROGRESS_P = 0.10           # 模式 random_p：按概率 p 开进度（0~1）
LLM_PROGRESS_K = 5              # 模式 reservoir_k：期望只展示 K 个（流式水库抽样）
MARGIN = 3.0
random.seed(42)

def _should_show_progress(gray_seen: int) -> bool:
    """根据已遇到的灰区计数 gray_seen，决定是否给当前灰区文档开启进度。"""
    mode = LLM_PROGRESS_MODE
    if mode == "first_n":
        return gray_seen <= LLM_PROGRESS_N
    elif mode == "random_p":
        return (random.random() < LLM_PROGRESS_P)
    elif mode == "reservoir_k":
        k = max(1, int(LLM_PROGRESS_K))
        # 流式水库抽样：第 gray_seen 个样本被选中的概率为 k/gray_seen
        if gray_seen <= k:
            return True
        return (random.random() < (k / float(gray_seen)))
    else:
        return False
def _to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x == 1
    if isinstance(x, str):
        return x.strip().lower() in {"true", "1", "yes"}
    return bool(x)

# === NEW: 建一次 id->text 映射，避免反复查找 ===
def _build_id2text(json_file_path: str):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    id2text = {}
    for it in corpus:
        rid = it.get("id") or it.get("uid") or it.get("doc_id")
        if rid is None:
            continue
        try:
            rid_int = int(str(rid).split("#")[0])
        except Exception:
            continue
        txt = it.get("text") or it.get("content") or it.get("body") or ""
        id2text[rid_int] = txt if isinstance(txt, str) else ""
    return corpus, id2text
# 定义输入参数
json_file_path = "./QFESum/syria_crisis/syria_crisis.json"

SAMPLE_RATIO = 0.10
RANDOM_SEED = 42

id_list = sample_document_ids(
    json_file_path=json_file_path,
    sample_ratio=SAMPLE_RATIO,
    seed=RANDOM_SEED
)

TASKS = [
    {
        "query": "The Syrian army's crackdown on protesters",
        "output_json": r"./QFESum/syria_crisis/RAT/crackdown_on_protesters.json"
    },
    {
        "query": "Protests against the Syrian government and Assad",
        "output_json": r"./QFESum/syria_crisis/RAT/Protests.json"
    }
]

def threshold_select(scores: list, labels: list):
    # 1. 构造 id → label 映射
    id_to_label = {int(i): v for i, v in labels}

    # 2. 提取分数与对应标签
    score_label_pairs = [(score, id_to_label[int(doc_id)]) for doc_id, score in scores]

    # 3. 遍历所有分数作为候选阈值
    from sklearn.metrics import accuracy_score

    best_acc = 0
    best_threshold = None

    # 排序后用唯一分数尝试作为阈值
    unique_scores = sorted(set(score for score, _ in score_label_pairs))

    for thresh in unique_scores:
        preds = [s >= thresh for s, _ in score_label_pairs]
        true_labels = [label for _, label in score_label_pairs]
        acc = accuracy_score(true_labels, preds)

        if acc > best_acc:
            best_acc = acc
            best_threshold = thresh

    # ✅ 输出最佳阈值
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Accuracy at threshold: {best_acc:.4f}")
    return best_threshold


full_data, id2text = _build_id2text(json_file_path)

# ✅ 汇总每个 query 的灰区统计
gray_stats_per_query = []

# === 主循环，带任务级进度条 ===
for task_idx, task in enumerate(tqdm(TASKS, desc="Tasks", unit="task")):
    query_text = task["query"]

    # === (不变) 用带标签的 id_list 学习阈值 ===
    labels = check_relevance(
        input_file_path=json_file_path,
        id_list=id_list,
        query=query_text
    )
    scores_labeled = calculate_similarity_scores(json_file_path, id_list, query_text)  # [(doc_id, score), ...]
    print(query_text)
    score_threshold = threshold_select(scores_labeled, labels)

    # === ✅ 关键修改：对“全语料”跑分进行筛选（不是只筛 id_list） ===
    all_ids = list(id2text.keys())
    scores_all = calculate_similarity_scores(json_file_path, all_ids, query_text)      # ← 全部文档

    # === 基于“阈值±MARGIN”的三段式 + 你的 LLM 函数 ===
    low, high = score_threshold - MARGIN, score_threshold + MARGIN
    selected = []  # (doc_id, score, selected_by)

    # ✅ 灰区计数器
    gray_total = 0
    gray_kept_by_llm = 0
    gray_rejected_by_llm = 0

    # === 文档级进度条（针对全语料得分） ===
    for doc_id, s in tqdm(scores_all, desc=f"[{task_idx + 1}/{len(TASKS)}] Screening ALL: {query_text}",
                          unit="doc", leave=False):
        doc_id = int(doc_id)
        if s >= high:
            selected.append((doc_id, s, "score"))
        elif s <= low:
            continue
        else:
            # ✅ 灰区：只对前 N/随机样本开句子级进度条
            gray_total += 1
            text = id2text.get(doc_id, "")
            show_progress = _should_show_progress(gray_total)

            keep_flag = _to_bool(llm_sentencewise_relevance(
                text, query_text,
                timeout=180,
                progress=show_progress,  # ← 这里按需开关
                tqdm_desc=f"LLM judge · doc {doc_id}",  # 可自定义
                tqdm_position=1,  # 外层 position=0，这里用 1
                tqdm_leave=False
            ))

            if keep_flag:
                gray_kept_by_llm += 1
                selected.append((doc_id, s, "llm"))
            else:
                gray_rejected_by_llm += 1

    # === 组装当前任务的输出并写盘（与你之前保存风格一致）===
    selected_ids = {d for d, _, _ in selected}
    score_map = {d: float(sc) for d, sc, _ in selected}
    source_map = {d: via for d, _, via in selected}

    results = []
    for item in full_data:
        rid = item.get("id") or item.get("uid") or item.get("doc_id")
        try:
            rid_int = int(str(rid).split("#")[0])
        except Exception:
            continue
        if rid_int in selected_ids:
            obj = dict(item)
            obj["_similarity"] = round(score_map.get(rid_int, 0.0), 3)
            obj["_selected_by"] = source_map.get(rid_int, "score")
            results.append(obj)

    results.sort(key=lambda x: x.get("_similarity", 0.0), reverse=True)

    output_path = str(task["output_json"]).strip()
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[OK] '{query_text}' -> {len(results)} saved to {output_path} "
          f"(thr={score_threshold:.2f}, margin=±{MARGIN})")

    # ✅ 打印并记录本 query 的灰区统计
    print(f"[GRAY] '{query_text}': total={gray_total}, kept_by_llm={gray_kept_by_llm}, rejected_by_llm={gray_rejected_by_llm}")
    gray_stats_per_query.append({
        "query": query_text,
        "gray_total": gray_total,
        "gray_kept_by_llm": gray_kept_by_llm,
        "gray_rejected_by_llm": gray_rejected_by_llm,
    })

# ✅ 全部任务结束后的灰区汇总
print("\n==== Gray-zone Summary ====")
for stat in gray_stats_per_query:
    print(f"- {stat['query']}: total={stat['gray_total']}, "
          f"kept_by_llm={stat['gray_kept_by_llm']}, rejected_by_llm={stat['gray_rejected_by_llm']}")
