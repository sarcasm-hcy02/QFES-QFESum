import json
import requests
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
import os
import re
from openai import OpenAI

client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.environ.get("OPENAI_API_KEY"),    # 阿里云api-key
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )



# Prompt 模板
prompt_event_match = """
**Role: Data Annotator**
**Instructions:**
You are provided with the following materials:
- **Passage**: {passage}
- **Sentence**: {sentence}
**Task**: Assess whether the passage fully supports the sentence.
**Choices**:
1. **Fully Supports**: Select this option if the passage completely and clearly supports every aspect of the sentence.
2. **Does Not Fully Support**: Select this option if any discrepancies, omissions, or inaccuracies in the passage prevent it
from fully supporting the sentence.
**Output**:
- If the passage fully supports the sentence, output "true"
- If it does not, output "false"
**Note**: Please refrain from adding any content not requested in the instructions

Respond with only:

true

or

false

Do not explain.
"""

# 输入摘要（每句为一个事件）

# ===== 配置：一次跑多个路径 =====
INPUT_FILE_PATHS = [

"./EQFSum/libya/EG-QFS/summary.json",


]

def simple_sentence_split(text, min_words=3):
    # 按句号切分（保留你原有的习惯）；太短的句子剔除
    sentences = re.findall(r'[^.]+(?:\.)', text)
    return [s.strip() for s in sentences if len(s.strip().split()) >= min_words]

# === 控制打印细节的开关 ===
DEBUG = True                 # 打开调试
MAX_DEBUG_PAIRS = 10         # 每个文件最多打印前多少个配对细节（避免日志爆炸）

results = []

for input_file_path in INPUT_FILE_PATHS:
    with open(input_file_path, 'r', encoding='utf-8') as f:
        summaries = json.load(f)

    f1_list, precision_list, recall_list = [], [], []

    # === 文件级统计器 ===
    file_api_calls = 0
    file_api_errors = 0
    file_true_outputs = 0
    file_false_outputs = 0
    file_other_outputs = 0

    debug_printed_pairs = 0  # 控制详细打印的数量

    if DEBUG:
        print(f"\n===== 开始处理文件：{input_file_path} =====")
        print(f"摘要对数量：{len(summaries)}")

    for doc_idx, item in enumerate(tqdm(summaries, desc=f"[File] {input_file_path}", leave=False)):
        gen_sents = simple_sentence_split(item.get("generated_summary", "").strip())
        ref_sents = simple_sentence_split(item.get("reference_summary", "").strip())

        if DEBUG:
            print(f"\n--- Doc {doc_idx+1}/{len(summaries)} ---")
            print(f"生成句子数(gen_sents)：{len(gen_sents)}")
            print(f"参考句子数(ref_sents)：{len(ref_sents)}")

        matched_pairs = []
        used_ref_indices = set()

        for i, gen_sent in enumerate(tqdm(gen_sents,
                                          desc=f"  Doc {doc_idx+1}/{len(summaries)}",
                                          leave=False)):
            matched = False
            for j, ref_sent in enumerate(ref_sents):
                if j in used_ref_indices:
                    continue

                prompt = prompt_event_match.format(passage=ref_sent, sentence=gen_sent)
                try:
                    file_api_calls += 1
                    resp = client.chat.completions.create(
                        model="deepseek-v3",
                        messages=[
                            {'role': 'system', 'content': 'You are a helpful assistant.'},
                            {'role': 'user', 'content': prompt}
                        ],
                        temperature=0,
                        stream=False,
                        timeout=180,
                        extra_body={"enable_thinking": False}  # 用 extra_body 传
                    )
                    content = resp.choices[0].message.content.strip().lower()

                    # 统计不同输出
                    if content == "true":
                        file_true_outputs += 1
                    elif content == "false":
                        file_false_outputs += 1
                    else:
                        file_other_outputs += 1

                    # 打印前若干个配对的详细信息
                    if DEBUG and debug_printed_pairs < MAX_DEBUG_PAIRS:
                        print("\n[DEBUG] 一次API返回：")
                        print(f"  gen_idx={i}, ref_idx={j}")
                        print(f"  生成句子(gen): {gen_sent}")
                        print(f"  参考句子(ref): {ref_sent}")
                        print(f"  模型原始输出(content): {repr(resp.choices[0].message.content)}")
                        print(f"  归一化后(content.lower()): {repr(content)}")
                        debug_printed_pairs += 1

                    # 使用结果判断是否匹配
                    if content == "true":
                        matched_pairs.append((gen_sent, ref_sent))
                        used_ref_indices.add(j)
                        matched = True
                        break  # 每个生成句只匹配一个参考句

                except Exception as e:
                    file_api_errors += 1
                    if DEBUG:
                        print("\n[ERROR] API 调用异常：")
                        print(f"  文件: {input_file_path}")
                        print(f"  Doc 索引: {doc_idx}, gen_idx={i}, ref_idx={j}")
                        print(f"  异常类型: {type(e).__name__}")
                        print(f"  异常信息: {e}")
                    # 静默跳过该配对（不抛出，让程序继续）

            if not matched and DEBUG and i < 3:
                # 前几个没有匹配到的生成句，给个提示
                print(f"[DEBUG] 生成句 index {i} 未找到匹配的参考句。句子：{gen_sent}")

        # —— 计算该摘要对的指标 ——
        tp = len(matched_pairs)
        precision = tp / len(gen_sents) if gen_sents else 0.0
        recall = tp / len(ref_sents) if ref_sents else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        if DEBUG:
            print(f"[Doc 结果] TP={tp}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    # —— 文件级平均 ——
    num_docs = len(f1_list)
    avg_precision = sum(precision_list) / num_docs if num_docs else 0.0
    avg_recall = sum(recall_list) / num_docs if num_docs else 0.0
    avg_f1 = sum(f1_list) / num_docs if num_docs else 0.0

    if DEBUG:
        print(f"\n===== 文件级统计：{input_file_path} =====")
        print(f"  摘要对数量: {num_docs}")
        print(f"  总 API 调用次数: {file_api_calls}")
        print(f"  API 异常次数: {file_api_errors}")
        print(f"  输出为 'true' 的次数: {file_true_outputs}")
        print(f"  输出为 'false' 的次数: {file_false_outputs}")
        print(f"  其他输出（既不是 true 也不是 false）次数: {file_other_outputs}")
        print(f"  平均 Precision: {avg_precision:.4f}")
        print(f"  平均 Recall:    {avg_recall:.4f}")
        print(f"  平均 F1:        {avg_f1:.4f}")

    results.append({
        "path": input_file_path,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1
    })

# === 总结输出 ===
print("\n===== Final Results =====")
for r in results:
    print(f"{r['path']}\n  Precision: {r['precision']:.4f}  Recall: {r['recall']:.4f}  F1: {r['f1']:.4f}")