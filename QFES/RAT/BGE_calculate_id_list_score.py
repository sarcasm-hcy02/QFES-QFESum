# similarity_calculator.py
import json
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def calculate_similarity_scores(json_file_path: str, id_list: list, query_text: str) -> list:
    """
    计算查询文本与JSON文件中指定ID文本的相似度分数

    参数:
        json_file_path (str): JSON文件路径
        id_list (list): 要处理的ID列表
        query_text (str): 查询文本

    返回:
        list: 包含[raw_id, score]的子列表
    """
    # 加载数据
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 构建 id → text 的映射表
    id_to_text = {str(item["id"]): item["text"] for item in data if
                  "id" in item and "text" in item and isinstance(item["text"], str)}

    # 加载 BGE 模型
    tokenizer = AutoTokenizer.from_pretrained(
        "./model/BGE_large_v1.5")
    model = AutoModel.from_pretrained("./model/BGE_large_v1.5",
                                      use_safetensors=False)
    model.eval()

    scores = []

    # 获取 query embedding
    with torch.no_grad():
        query_dict = tokenizer(query_text, max_length=512, padding=True, truncation=True, return_tensors='pt')
        query_outputs = model(**query_dict)
        query_embedding = average_pool(query_outputs.last_hidden_state, query_dict['attention_mask'])
        query_embedding = F.normalize(query_embedding, p=2, dim=1)

    # 遍历 ID 列表，计算相似度
    print("✅ Similarity Scores for Selected IDs:")
    for raw_id in id_list:
        id_str = str(raw_id)
        if id_str not in id_to_text:
            print(f"ID: {id_str} | ⚠️ Text not found in JSON.")
            continue

        text = id_to_text[id_str]

        with torch.no_grad():
            inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
            outputs = model(**inputs)
            embedding = average_pool(outputs.last_hidden_state, inputs["attention_mask"])
            embedding = F.normalize(embedding, p=2, dim=1)

            score = (query_embedding @ embedding.T).item() * 100
            scores.append([raw_id, score])
            print(f"[{raw_id}, {score:.2f}],")

    return scores