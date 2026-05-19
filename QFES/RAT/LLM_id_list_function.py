import json
import requests
from nltk.tokenize import sent_tokenize
import os

os.environ["NO_PROXY"] = "127.0.0.1,localhost"


def check_relevance(
        input_file_path: str,
        id_list: list,
        query: str,
        model_name: str = "qwen2.5:7b",
        api_url: str = "http://127.0.0.1:11434/api/generate",
        verbose: bool = True
) -> list:
    """
    检查JSON文档中指定ID的文本内容是否与查询相关

    参数:
        input_file_path (str): JSON文件路径
        id_list (list): 要检查的ID列表
        query (str): 查询文本
        model_name (str): 使用的LLM模型名称，默认为"qwen2.5:7b"
        api_url (str): API端点URL，默认为"http://127.0.0.1:11434/api/generate"
        verbose (bool): 是否打印处理过程，默认为True

    返回:
        list: 包含[raw_id, is_relevant]的子列表
    """
    # 加载JSON数据
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 创建id→text映射
    id_to_text = {str(item["id"]): item.get("text", "") for item in data if "id" in item and "text" in item}

    labels = []

    # 遍历ID列表并逐句判断相关性
    for raw_id in id_list:
        id_str = str(raw_id)
        content = id_to_text.get(id_str, "")

        if not content.strip():
            if verbose:
                print(f"ID {raw_id}: ❌ Text not found or empty.")
            labels.append([raw_id, False])
            continue

        sentences = sent_tokenize(content)
        is_relevant = False

        for sentence in sentences:
            prompt = f"""
Please determine whether the following sentence is strictly relevant to the query: "{query}". 
Only respond "True" if this sentence is clearly and directly about the topic in the query. Otherwise, respond "False".

Sentence:
\"{sentence.strip()}\"

Answer with a single word: "True" or "False".
"""
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options":{"temperature": 0}
            }

            try:
                response = requests.post(api_url, json=payload)
                response.raise_for_status()
                result_text = response.json().get("response", "").strip().lower()
                if result_text.startswith("true"):
                    is_relevant = True
                    break  # 一句相关即可
            except Exception as e:
                if verbose:
                    print(f"❌ Error processing sentence in ID {raw_id}: {e}")
                break  # 出错就跳出当前项处理

        labels.append([raw_id, is_relevant])
        if verbose:
            print(f"[{raw_id}, {is_relevant}],")

    return labels


