

import json
import numpy as np
import math
import os
import warnings
import hdbscan
from umap import UMAP
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)


import random, torch
os.environ["PYTHONHASHSEED"] = "42"          # 新增：确保哈希顺序稳定（影响 dict/set 迭代等）
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)

def _set_global_seed(seed: int = 42):
    random.seed(seed)                         # 新增
    np.random.seed(seed)                      # 新增
    torch.manual_seed(seed)                   # 新增
    if torch.cuda.is_available():             # 新增：如日后切到GPU
        torch.cuda.manual_seed_all(seed)
        # 为完全确定性（可能略降速），可按需打开下一行
        # torch.use_deterministic_algorithms(True)
class EventClusterSummarizer:
    def __init__(self,
                 json_file_path,
                 query,
                 model_path,
                 threshold_high_ratio=0.3,
                 threshold_mid_ratio=0.6,
                 min_dist=0.03,
                 min_cluster_size=14,
                 min_samples=14,
                 seed: int = 42):

        self.json_file_path = json_file_path
        self.query = query
        self.model_path = model_path
        self.threshold_high_ratio = threshold_high_ratio
        self.threshold_mid_ratio = threshold_mid_ratio
        self.min_dist = min_dist
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.seed = seed

        self.embedding_model = SentenceTransformer(model_path, device="cpu")

    def read_json_events_to_docs(self, min_words=5):
        with open(self.json_file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # 自动兼容单个 dict 或 list[dict]
        if isinstance(raw_data, dict):
            data_list = [raw_data]
        elif isinstance(raw_data, list):
            data_list = raw_data
        else:
            raise ValueError("Unsupported JSON format")

        docs, ids = [], []
        for item in data_list:
            for event in item.get("events", []):
                if not event:
                    continue
                if isinstance(event, dict):
                    raw_text = event.get("text", None)
                    eid = event.get("eid", None)
                    if not isinstance(raw_text, str):
                        continue
                    text = raw_text.strip()
                    if eid is not None and len(text.split()) >= min_words:
                        docs.append(text)
                        ids.append(eid)

        print(f"✅ Loaded {len(docs)} event(s) from {self.json_file_path}")
        return docs, ids

    def rank_events_by_centrality(self, json_file_path, target_eids, model_path):
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        eid_to_text = {}

        for item in data:
            events = item.get("events", [])
            if not isinstance(events, list):
                continue  # 跳过非列表的 events 字段

            for event in events:
                if not isinstance(event, dict):
                    continue  # 跳过非字典结构

                eid = event.get("eid")
                text = event.get("text")

                if isinstance(eid, int) and isinstance(text, str) and eid in target_eids:
                    eid_to_text[eid] = text.strip()

        if not eid_to_text:
            return []

        model = SentenceTransformer(model_path, device='cpu')
        eids = list(eid_to_text.keys())
        texts = [eid_to_text[eid] for eid in eids]
        embeddings = model.encode(texts, normalize_embeddings=True)
        sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix, 0)
        centrality_scores = sim_matrix.mean(axis=1)
        ranked = sorted(zip(eids, centrality_scores, texts), key=lambda x: -x[1])

        top_eid, _, top_text = ranked[0]
        return [(top_eid, top_text)]

    def cluster_events_with_center(self, json_file_path, eid_list, model_path, min_words=5, min_cluster_size=4):
        with open(json_file_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)

        eid_set = set(map(int, eid_list))
        texts, eids = [], []

        for item in data_list:
            for event in item.get("events", []):
                if not event or not isinstance(event, dict):
                    continue
                eid = event.get("eid")
                raw_text = event.get("text", "")
                if eid in eid_set and isinstance(raw_text, str):
                    text = raw_text.strip()
                    if len(text.split()) >= min_words:
                        texts.append(text)
                        eids.append(eid)

        if len(texts) < min_cluster_size:
            return []

        model = SentenceTransformer(model_path, device='cpu')
        embeddings = model.encode(texts, normalize_embeddings=True)
        distance_matrix = cosine_distances(embeddings.astype(np.float64))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed')
        labels = clusterer.fit_predict(distance_matrix)

        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:
                continue
            clusters.setdefault(label, []).append((eids[i], texts[i], embeddings[i]))

        results = []
        for cluster_id, items in clusters.items():
            eid_list, text_list, emb_list = zip(*items)
            emb_array = np.vstack(emb_list)
            sim_matrix = cosine_similarity(emb_array)
            np.fill_diagonal(sim_matrix, 0)
            centrality_scores = sim_matrix.mean(axis=1)
            max_idx = int(np.argmax(centrality_scores))
            center_eid, center_text = eid_list[max_idx], text_list[max_idx]

            results.append({
                "cluster_id": cluster_id,
                "events": [{"eid": eid, "text": text} for eid, text in zip(eid_list, text_list)],
                "center_event": {"eid": center_eid, "text": center_text}
            })
        return results

    def summarize(self):
        docs, ids = self.read_json_events_to_docs()
        print(f"📄 Loaded {len(docs)} docs from {self.json_file_path}")
        if not docs:
            print(f"⚠️ No documents found in {self.json_file_path}, skipping...")
            return  # 或 raise / continue

        umap_model = UMAP(
            n_neighbors=8,
            min_dist=self.min_dist,
            n_components=5,
            metric="cosine",
            random_state=self.seed
        )
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        vectorizer_model = CountVectorizer(stop_words="english")

        topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            calculate_probabilities=True,
            verbose=False
        )

        topics, probs = topic_model.fit_transform(docs)

        # 原始主题分配
        info = topic_model.get_document_info(docs)
        info["eid"] = ids

        # ✅ 加入最大簇限制逻辑
        max_cluster_size = 200
        new_topic_to_docids = {}
        new_topic_counter = 0

        for topic_num, group_df in info.groupby("Topic"):
            if topic_num < 0:
                continue
            eid_list = group_df["eid"].tolist()
            doc_list = group_df["Document"].tolist()

            if len(eid_list) <= max_cluster_size:
                new_topic_to_docids[new_topic_counter] = eid_list
                new_topic_counter += 1
            else:
                # 对过大的主题进一步切割（这里用KMeans做粗分）
                num_sub_clusters = math.ceil(len(eid_list) / max_cluster_size)
                sub_kmeans = KMeans(n_clusters=num_sub_clusters, random_state=self.seed)
                emb_subset = self.embedding_model.encode(doc_list, normalize_embeddings=True)
                sub_labels = sub_kmeans.fit_predict(emb_subset)

                for i in range(num_sub_clusters):
                    sub_eids = [eid_list[j] for j in range(len(sub_labels)) if sub_labels[j] == i]
                    new_topic_to_docids[new_topic_counter] = sub_eids
                    new_topic_counter += 1
        topic_to_docids = new_topic_to_docids
        topic_to_texts = {
            topic_id: [docs[ids.index(eid)] for eid in eid_list if eid in ids]
            for topic_id, eid_list in topic_to_docids.items()
        }
        # === 编码 query 向量 ===
        query_emb = self.embedding_model.encode([self.query], normalize_embeddings=True)[0]

        topic_sim_scores = {}
        topic_texts_embeddings = {}

        # === 计算每个 topic 与 query 的平均相似度 ===
        for topic_id, texts in topic_to_texts.items():
            if topic_id not in topic_to_docids or not texts:
                continue
            embs = self.embedding_model.encode(texts, normalize_embeddings=True)
            sim_scores = np.dot(embs, query_emb)
            avg_sim = float(np.mean(sim_scores))
            topic_sim_scores[topic_id] = avg_sim
            topic_texts_embeddings[topic_id] = (texts, embs)

        # === 相似度排序 ===
        sorted_topic_scores = sorted(topic_sim_scores.items(), key=lambda x: -x[1])

        # 原始排序后的主题分数
        N = len(sorted_topic_scores)

        # 先计算浮点值用于判断
        raw_cut_high = N * self.threshold_high_ratio
        raw_cut_mid = N * self.threshold_mid_ratio

        # 然后用于实际分割的整数索引
        cut_high = round(raw_cut_high)
        cut_mid = round(raw_cut_mid)

        # 判断逻辑根据浮点值决定
        if raw_cut_high < 2:
            # 情况 1：高分区数量不足 1.5，全划为高分区
            high_topics = sorted_topic_scores
            mid_topics = []
            low_topics = []
        elif 2 <= raw_cut_high <= 3:
            # 情况 2：高分区数量在 1.5 到 2.5 之间，高 + 中合并为高
            high_topics = sorted_topic_scores[:cut_mid]
            mid_topics = []
            low_topics = sorted_topic_scores[cut_mid:]
        else:
            # 正常分区
            high_topics = sorted_topic_scores[:cut_high]
            mid_topics = sorted_topic_scores[cut_high:cut_mid]
            low_topics = sorted_topic_scores[cut_mid:]

        # print("\n====== 分区策略提取的中心句 ======")
        sentences = []
        # === 高相关区（min_cluster_size = 2；失败则选中心性句）===
        for topic_id, score in high_topics:
            # topic_words = topic_model.get_topic(topic_id)
            # top_words = ", ".join([w for w, _ in topic_words[:5]])
            id_list = topic_to_docids[topic_id]

            # print(f"\n🟢 Topic {topic_id} | 相似度: {score:.4f} | Top words: {top_words}")

            cluster_results = self.cluster_events_with_center(self.json_file_path, id_list, self.model_path, min_cluster_size=2)

            if cluster_results:
                for cluster in cluster_results:
                    # print(f'"""{cluster["center_event"]["text"].strip()}"""')
                    sentences.append(cluster["center_event"]["text"].strip())
            else:
                selected = self.rank_events_by_centrality(self.json_file_path, id_list, self.model_path)
                for _, text in selected:
                    # print(f'  """{text.strip()}"""')
                    sentences.append(text.strip())

        # === 中相关区（min_cluster_size = 6；失败不给）===
        for topic_id, score in mid_topics:
            # topic_words = topic_model.get_topic(topic_id)
            # top_words = ", ".join([w for w, _ in topic_words[:5]])
            id_list = topic_to_docids[topic_id]

            # print(f"\n🟡 Topic {topic_id} | 相似度: {score:.4f} | Top words: {top_words}")

            cluster_results = self.cluster_events_with_center(self.json_file_path, id_list, self.model_path, min_cluster_size=4)

            if cluster_results:
                for cluster in cluster_results:
                    # print(f'  """{cluster["center_event"]["text"].strip()}"""')
                    sentences.append(cluster["center_event"]["text"].strip())
            else:
                selected = self.rank_events_by_centrality(self.json_file_path, id_list, self.model_path)
                for _, text in selected:
                    # print(f'  """{text.strip()}"""')
                    sentences.append(text.strip())
                # pass

        # === 收集所有 low_topics 中的句子 ===
        all_low_eids = []
        for topic_id, _ in low_topics:
            all_low_eids.extend(topic_to_docids[topic_id])

        # === 从 JSON 中提取这些 eid 对应的文本 ===
        eid_to_text = {}
        with open(self.json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            for event in item.get("events", []):
                if not event or not isinstance(event, dict):
                    continue
                eid = event.get("eid")
                text = event.get("text")
                if eid in all_low_eids and isinstance(text, str):
                    eid_to_text[eid] = text.strip()

        texts = list(eid_to_text.values())
        if texts:
            embs = self.embedding_model.encode(texts, normalize_embeddings=True)
            sim_matrix = cosine_similarity(embs)
            np.fill_diagonal(sim_matrix, 0)
            centrality = sim_matrix.mean(axis=1)
            center_idx = int(np.argmax(centrality))
            #
            # print("\n🔴 低相关区所有主题的整体中心句：")
            # print(f' """{texts[center_idx]}"""')
            sentences.append(texts[center_idx].strip())
        return sentences