from typing import Union

def split_and_deduplicate_flexible(
    input_text: Union[str, list],
    model_path="./model/GTE_large",
    similarity_threshold=0.9
):
    """
    既支持传入字符串（会按 . 分句），也支持传入句子列表。
    """
    import re
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    if isinstance(input_text, list):
        text = " ".join(input_text)
    else:
        text = input_text

    raw_sentences = re.split(r'\.\s*', text.strip())
    clean_sentences = [s.strip() + '.' for s in raw_sentences if s.strip()]

    model = SentenceTransformer(model_path)
    embeddings = model.encode(clean_sentences, convert_to_tensor=True).detach().cpu().numpy()
    sim_matrix = cosine_similarity(embeddings)

    n = len(clean_sentences)
    selected = []
    used = set()

    for i in range(n):
        if i in used:
            continue
        for j in range(i + 1, n):
            if j not in used and sim_matrix[i][j] >= similarity_threshold:
                used.add(j)
        used.add(i)
        selected.append(clean_sentences[i])

    return " ".join(selected)
