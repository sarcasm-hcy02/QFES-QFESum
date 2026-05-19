# -*- coding: utf-8 -*-
import json
from pathlib import Path
import sys
import numpy as np
from rouge_score import rouge_scorer
import nltk
import re
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
# 首次运行需要下载的 NLTK 资源（仅 WordNet/OMW；我们会手动分词，避免 punkt 依赖）
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Optional metrics packages (loaded lazily)
try:
    from sacrebleu import corpus_bleu
    _HAS_SACREBLEU = True
except Exception:
    _HAS_SACREBLEU = False

try:
    from nltk.translate.meteor_score import single_meteor_score
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False

try:
    from bert_score import score as bertscore
    _HAS_BERTSCORE = True
except Exception:
    _HAS_BERTSCORE = False

# === 直接在这里写你的 JSON 文件路径 ===
JSON_PATH = ("./EQFSum/libya/EG-QFS/summary.json")   # <- 修改成你的实际路径
USE_STEMMER = True                      # 是否启用词干化

# 需要的指标（假定你的环境支持 rouge4）
METRICS = ['rouge1', 'rouge2', 'rougeL', 'rouge4']

# ---- 新增：仅检查 wordnet/omw 是否可用（用于 METEOR 同义词），不再依赖 punkt ----
def ensure_wordnet_ok():
    for pkg, res in [("wordnet", "corpora/wordnet"), ("omw-1.4", "corpora/omw-1.4")]:
        try:
            nltk.data.find(res)
        except LookupError:
            nltk.download(pkg, quiet=True)

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
def simple_en_tokenize(s: str):
    """简单英文分词：小写 + 英数及连字符/撇号内部词，避免依赖 punkt。"""
    return _WORD_RE.findall(s.lower())


def main():
    path = Path(JSON_PATH)
    if not path.exists():
        print(f"找不到文件：{path}", file=sys.stderr)
        sys.exit(1)

    # 读取 JSON（期望为 list，每项包含 generated_summary / reference_summary）
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("输入 JSON 应为列表（list）。", file=sys.stderr)
        sys.exit(2)

    # 构建 ROUGE 计算器（ref 在前，gen 在后）
    try:
        scorer = rouge_scorer.RougeScorer(METRICS, use_stemmer=USE_STEMMER)
    except Exception as e:
        print(f"构建 RougeScorer 失败：{e}", file=sys.stderr)
        return

    # —— 容器 —— #
    buckets = {m: [] for m in METRICS}
    total = 0
    skipped = 0
    refs_all, gens_all = [], []   # 用于 BLEU / BERTScore / METEOR

    # —— 遍历样本 —— #
    for item in data:
        ref = item.get("reference_summary")
        gen = item.get("generated_summary")
        if not isinstance(ref, str) or not isinstance(gen, str):
            skipped += 1
            continue
        ref = ref.strip()
        gen = gen.strip()
        if not ref or not gen:
            skipped += 1
            continue

        total += 1

        # ROUGE
        try:
            scores = scorer.score(ref, gen)
            for m in METRICS:
                if m in scores:
                    buckets[m].append(scores[m].fmeasure)
        except Exception:
            pass

        refs_all.append(ref)
        gens_all.append(gen)

    if total == 0:
        print("没有有效样本（请检查每项是否包含字符串类型的 reference_summary 与 generated_summary）。", file=sys.stderr)
        if skipped:
            print(f"已跳过 {skipped} 条无效记录。", file=sys.stderr)
        sys.exit(3)

    # —— 汇总 ROUGE —— #
    avgs = {m: (float(np.mean(buckets[m])) if buckets[m] else float('nan')) for m in METRICS}

    print("Average ROUGE (F1):")
    for m in METRICS:
        v = avgs[m]
        print(f"{m}: {'N/A' if np.isnan(v) else f'{v:.4f}'}")

    # —— BLEU（sacrebleu）—— #
    if _HAS_SACREBLEU:
        try:
            bleu = corpus_bleu(gens_all, [refs_all])
            print("\nBLEU:")
            print(f"BLEU (sacreBLEU): {bleu.score:.2f}")
            print(f"BLEU (0-1 scale): {bleu.score/100.0:.4f}")
        except Exception as e:
            print("\nBLEU 计算失败：", e, file=sys.stderr)
    else:
        print("\n未安装 sacrebleu，跳过 BLEU。安装：pip install sacrebleu")

    # —— METEOR（英文；手动分词，避免 punkt 依赖） —— #
    if _HAS_NLTK:
        ensure_wordnet_ok()
        meteor_vals = []
        for ref, gen in zip(refs_all, gens_all):
            try:
                ref_tok = simple_en_tokenize(ref)
                gen_tok = simple_en_tokenize(gen)
                if not ref_tok or not gen_tok:
                    meteor_vals.append(None)
                    continue
                meteor_vals.append(float(single_meteor_score(ref_tok, gen_tok)))
            except Exception:
                meteor_vals.append(None)

        valid = [s for s in meteor_vals if isinstance(s, (int, float))]
        if valid:
            print("\nMETEOR (English, tokenized):")
            print(f"METEOR (avg): {float(np.mean(valid)):.4f}")
        else:
            print("\nMETEOR 计算失败或无有效样本（已使用手动分词；若仍失败，多为 wordnet/权限问题）。")
    else:
        print("\n未安装 nltk，跳过 METEOR。安装：pip install nltk")

    # —— BERTScore（英文，本地 roberta-large，无 baseline 重标定） —— #
    if _HAS_BERTSCORE:
        try:
            import os

            # ① 选择设备（若无 CUDA 则回退到 CPU）
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

            # ② 本地模型路径（确保指向包含 config.json 的目录）
            local_model_path = "./model/roberta-large"
            if not os.path.exists(os.path.join(local_model_path, "config.json")):
                # 若是 HF 缓存根目录，自动进入 snapshots/<hash>
                snapshots_dir = os.path.join(local_model_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    subs = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                    if subs:
                        local_model_path = os.path.join(snapshots_dir, subs[0])

            print(f"\n[INFO] Using local BERTScore model from: {local_model_path}")
            print(f"[INFO] BERTScore device: {device}")

            # ③ 计算 BERTScore
            P, R, F1 = bertscore(
                gens_all,
                refs_all,
                model_type=local_model_path,  # 从本地加载 roberta-large
                num_layers=17,  # roberta-large 的最终层
                rescale_with_baseline=False,  # 关闭 baseline（避免负值）
                device=device
            )

            print("\nBERTScore (English, local model, no baseline):")
            print(f"P: {float(P.mean()):.4f}")
            print(f"R: {float(R.mean()):.4f}")
            print(f"F1: {float(F1.mean()):.4f}")

        except Exception as e:
            print("\nBERTScore 计算失败：", e, file=sys.stderr)
            print("提示：可检查本地模型目录是否包含 config.json/pytorch_model.bin，或改用：")
            print("      bertscore(..., model_type='microsoft/deberta-xlarge-mnli', device='cpu')")
    else:
        print("\n未安装 bert-score，跳过 BERTScore。安装：pip install bert-score")

if __name__ == "__main__":
    main()
