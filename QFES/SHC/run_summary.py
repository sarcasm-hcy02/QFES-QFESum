from bertopic_HDBSCAN_class import EventClusterSummarizer
import sys
from rouge_score import rouge_scorer
from sentence_simi import split_and_deduplicate_flexible
import numpy as np
import itertools
import json
import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)

"""
    记得修改output_path
"""



tasks = [
    {
        "json_path": "./QFESum/libya/RAT/European_Union.json",
        "query": "The response and interference of the European Union",
        "reference": """EU governments approve sanctions against Gaddafi and his closest advisers .The European Union bans the sale of arms and ammunition to Libya and freezes the assets of Gadhafi and five members of his family, while imposing a visa ban on Gadhafi and 15 other people tied to the regime's crackdown.The EU calls for Moammar Kadafi to step down.', 'France recognizes the rebel council in Benghazi as Libya's legitimate authority.The European Union opens an office in the rebel-held Libyan city of Benghazi.EU widens sanctions against the regime.
             """
    },
    {
        "json_path": "./QFESum/libya/RAT/NATO.json",
        "query": "NATO's military strikes and statements",
        "reference": """NATO agrees to take command of the mission enforcing a no-fly zone over Libya.At an international conference in London, the U.S. prepares to hand over control of military action in Libya, but NATO still remains starkly divided over the best possible outcome for the country.Without air support, rebel forces are routed in the western city of Surt, retreating 100 miles to the east, giving up Bin Jawwad and Ras Lanuf in the process.Port Brega is eventually lost as well, reversing the gains of previous days.NATO announces that it has begun Operation Unified Protector in Libya, including an arms embargo, a no-fly zone, and ``actions to protect civilians and civilian centers.''After recriminations for being slow to act and inadvertently killing rebels in a wayward airstrike, NATO increases attacks on Tripoli.In Port Brega, NATO planes mistakenly bomb tanks under rebel control.A Nato missile attack on a house in Tripoli kills Gaddafi 's youngest son and three grandchildren , his government says .NATO announces that it is extending its mission in Libya for 90 days.Waves of NATO fighter planes hit the Libyan capital with one of the largest bombardments of the city since the Western-led alliance began airstrikes almost three months ago.NATO officials admit that the alliance was probably responsible for an airstrike in a densely populated Tripoli neighborhood that Libyan authorities said killed nine people and injured 18.Libyan government officials say 15 people, including three children, were killed in the strike on Khweldi Hamedi's home west of Tripoli.NATO defends the attack as a'a precision strike on a legitimate military target.NATO says it supported rebels by hitting a military facility, armored vehicles, tanks and light military vehicles around Brega.Libyan authorities accuse NATO of a ``massacre'' of 85 villagers in air strikes south of Zliten in western Libya.NATO insists it has no evidence of the civilian deaths.Col. Roland Lavoie, a spokesman for NATO's military operation, tells reporters that anti-Gadhafi forces are now assuming control of the key approaches to Tripoli.'' Meanwhile, a brother of Moussa Ibrahim, the spokesman for the government in Tripoli, was killed Thursday night by NATO aircraft, a Libyan government official said.Nato says Libya's interim rulers have taken full control of the country's stockpile of chemical weapons and nuclear material.
             """
    }
]





def evaluate_parameters(tasks, params):
    rouge1_list = []
    rouge2_list = []
    rougel_list = []
    rouge4_list = []

    for task in tasks:
        summarizer = EventClusterSummarizer(
            json_file_path=task["json_path"],
            query=task["query"],
            model_path="./model/BGE_large_v1.5",  # 替换成你的路径
            threshold_high_ratio=params["threshold_high_ratio"],
            threshold_mid_ratio=params["threshold_mid_ratio"],
            min_dist=params["min_dist"],
            min_cluster_size=params["min_cluster_size"],
            min_samples=params["min_samples"]
        )

        try:
            sentences = summarizer.summarize()
            candidate = split_and_deduplicate_flexible(sentences)
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rouge4'], use_stemmer=True)
            scores = scorer.score(task["reference"], candidate)

            rouge1_list.append(scores["rouge1"].fmeasure)
            rouge2_list.append(scores["rouge2"].fmeasure)
            rougel_list.append(scores["rougeL"].fmeasure)
            rouge4_list.append(scores["rouge4"].fmeasure)

        except Exception as e:
            print(f"⚠️ Error on {task['json_path']}: {e}")
            continue

    if not rouge1_list:
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougel": 0.0,
            "rouge4": 0.0,
            "f_measure": 0.0
        }

    avg_r1 = np.mean(rouge1_list)
    avg_r2 = np.mean(rouge2_list)
    avg_rl = np.mean(rougel_list)
    avg_r4 = np.mean(rouge4_list)
    avg_f = (avg_r1 + avg_r2 + avg_rl) / 3

    return {
        "rouge1": avg_r1,
        "rouge2": avg_r2,
        "rougel": avg_rl,
        "rouge4": avg_r4,
        "f_measure": avg_f
    }




# 参数搜索空间（可根据需要扩展）
param_grid = {
    "threshold_high_ratio": [0.3],
    "threshold_mid_ratio": [0.5],
    "min_dist": [0.05],
    "min_cluster_size": [20],
    "min_samples": [20]
}

# 构造所有组合
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# 存储最佳结果
best_params = None
best_score = -1

for i, combo in enumerate(param_combinations):
    print(f"\n🔍 Testing combination {i+1}/{len(param_combinations)}: {combo}")
    result = evaluate_parameters(tasks, combo)

    print(f"🔹 ROUGE-1 F: {result['rouge1']:.4f}")
    print(f"🔹 ROUGE-2 F: {result['rouge2']:.4f}")
    print(f"🔹 ROUGE-L F: {result['rougel']:.4f}")
    print(f"🔹 ROUGE-4 F: {result['rouge4']:.4f}")
    print(f"✅ Avg F-measure: {result['f_measure']:.4f}")

    if result["f_measure"] > best_score:
        best_score = result["f_measure"]
        best_rouge_1 = result["rouge1"]
        best_rouge_2 = result["rouge2"]
        best_rouge_l = result["rougel"]
        best_rouge_4 = result["rouge4"]
        best_params = combo

# === 在找到最优参数后，再用该参数逐个输出每个任务的摘要 ===
print("\n📝 Generating summaries for best parameters...\n")
summaries = []
for task in tasks:
    summarizer = EventClusterSummarizer(
        json_file_path=task["json_path"],
        query=task["query"],
        model_path="./model/BGE_large_v1.5",
        threshold_high_ratio=best_params["threshold_high_ratio"],
        threshold_mid_ratio=best_params["threshold_mid_ratio"],
        min_dist=best_params["min_dist"],
        min_cluster_size=best_params["min_cluster_size"],
        min_samples=best_params["min_samples"]
    )

    try:
        sentences = summarizer.summarize()
        candidate = split_and_deduplicate_flexible(sentences)
        summaries.append({
            "generated_summary": candidate.strip(),
            "reference_summary": task["reference"].strip()
        })
        print("🔹 Query:", task["query"])
        print("🔸 Reference Summary:")
        print(task["reference"])
        print("🔸 Candidate Summary:")
        print(candidate)
        print("="*80)

    except Exception as e:
        print(f"⚠️ Error while generating summary for {task['json_path']}: {e}")


output_path = "./QFESum/libya/RAT/summary.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(summaries, f, ensure_ascii=False, indent=4)


print("\n🎯 Best Parameters:")
print(best_params)
print(f"🏆 Best Avg F-measure: {best_score:.4f}")
print(f"🏆 Best Avg ROUGE_1: {best_rouge_1:.4f}:")
print(f"🏆 Best Avg ROUGE_2: {best_rouge_2:.4f}:")
print(f"🏆 Best Avg ROUGE_l: {best_rouge_l:.4f}:")
print(f"🏆 Best Avg ROUGE_4: {best_rouge_4:.4f}:")