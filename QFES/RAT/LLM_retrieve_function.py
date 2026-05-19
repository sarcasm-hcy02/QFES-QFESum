import re
import requests
from typing import Optional
from tqdm import tqdm

# 分句：优先 nltk，兜底正则
try:
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize
    def _sent_tokenize(text: str):
        return _nltk_sent_tokenize(text)
except Exception:
    def _sent_tokenize(text: str):
        return [s.strip() for s in re.split(r'(?<=[\.!\?。！？])\s+', text) if s.strip()]

CHAT_URL = "http://127.0.0.1:11434/api/chat"  # 你之前用的是 /api/chat
MODEL_NAME = "qwen2.5:7b"

def _parse_bool(s: str) -> Optional[bool]:
    """宽容解析 True/False/Yes/No 等返回。"""
    if not isinstance(s, str):
        return None
    low = s.strip().lower()
    m = re.search(r'\b(true|false|yes|no)\b', low)
    if m:
        return m.group(1) in ("true", "yes")
    letters = re.sub(r'[^a-z]', '', low)
    if letters.startswith("true") or letters.startswith("yes"):
        return True
    if letters.startswith("false") or letters.startswith("no"):
        return False
    return None

def llm_sentencewise_relevance(
    text: str,
    query: str,
    *,
    timeout: int = 180,
    return_int: bool = False,
    progress: bool = False,             # ✅ 开关：是否显示进度条
    tqdm_desc: Optional[str] = None,    # ✅ 自定义进度条标题
    tqdm_position: Optional[int] = None,# ✅ 嵌套进度条位置（外层=0，内层可用1）
    tqdm_leave: bool = False            # ✅ 结束后是否保留进度条
) -> bool | int:
    """
    逐句调用 LLM 判断与 query 是否相关；任一句 True 即整体 True。
    - 使用 /api/chat 端点与 messages 负载
    - 宽容解析 True/False
    - 可显示句子级进度条
    """
    if not isinstance(text, str) or not text.strip() or not isinstance(query, str) or not query.strip():
        return 1 if (return_int and False) else (False if not return_int else 0)

    sentences = _sent_tokenize(text)
    keep = False

    # 设置进度条
    bar_iter = tqdm(
        sentences,
        desc=tqdm_desc or "LLM judging",
        unit="sent",
        leave=tqdm_leave,
        position=tqdm_position,
        disable=not progress,
    )

    for sent in bar_iter:
        prompt = f"""
Please determine whether the following sentence is strictly relevant to the query: "{query}". 
Only answer "True" if the sentence explicitly discusses issues related to the query. Otherwise, respond "False".

Sentence:
\"{sent}\"

Answer with a single word: "True" or "False".
""".strip("\n")

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "Reply with only one word: True or False."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0}
        }

        try:
            resp = requests.post(CHAT_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            j = resp.json()
            # /api/chat 常见返回
            result_text = (
                j.get("message", {}).get("content") or
                (j.get("choices") or [{}])[0].get("message", {}).get("content", "")
            )
            verdict = _parse_bool(result_text)
            if progress:
                # 在进度条尾部显示最近一次模型输出的前若干字符
                bar_iter.set_postfix_str((result_text or "")[:20])

            if verdict is True:
                keep = True
                break
            # verdict 为 False 或无法解析(None) 都继续下一句
        except Exception as e:
            if progress:
                bar_iter.set_postfix_str(f"err:{e.__class__.__name__}")
            # 出错不中断，继续下一句
            continue

    if progress:
        # 若中途 break，tqdm 会自动结束；显式 close 以免偶发残影
        bar_iter.close()

    if return_int:
        return 1 if keep else 0
    return keep
