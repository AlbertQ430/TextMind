"""BM25 关键词检索基线（Okapi BM25，自实现）。

评分公式（对查询中每个不重复词项累加）：

    score(D, Q) = Σ idf(qᵢ) · tf(qᵢ, D)(k1 + 1)
                  / (tf(qᵢ, D) + k1 · (1 - b + b · |D| / avgdl))

其中 ``idf(q) = ln((N - df(q) + 0.5) / (df(q) + 0.5) + 1)``，该形式保证
idf 非负，避免高频词产生负分。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

from document.dataset import Document
from retrieval.base import Retriever

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def simple_tokenize(text: str) -> List[str]:
    """默认英文分词：小写 + 字母数字序列。

    中文场景需替换为 jieba 等分词器，通过构造参数注入。
    """
    return _TOKEN_RE.findall(text.lower())


def jieba_tokenize(text: str) -> List[str]:
    """中文分词（jieba），过滤空白 token。

    jieba 在首次调用时加载词典，因此延迟导入本函数内。
    """
    import jieba

    return [token for token in jieba.lcut(text.lower()) if token.strip()]


class BM25Retriever(Retriever):
    """基于倒排索引与文档长度归一化的 BM25 检索器。"""

    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        tokenizer: Callable[[str], Sequence[str]] = simple_tokenize,
    ) -> None:
        if k1 < 0:
            raise ValueError(f"k1 必须 >= 0，实际 {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b 必须在 [0, 1] 内，实际 {b}")
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer
        self._doc_len: Dict[str, int] = {}
        self._postings: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._avgdl = 0.0
        self._n_docs = 0

    def fit(self, documents: Iterable[Document]) -> "BM25Retriever":
        self._doc_len.clear()
        self._postings.clear()
        total_len = 0
        for doc in documents:
            tokens = self.tokenizer(doc.searchable_text)
            self._doc_len[doc.doc_id] = len(tokens)
            total_len += len(tokens)
            for term, tf in Counter(tokens).items():
                self._postings[term][doc.doc_id] = tf
        self._n_docs = len(self._doc_len)
        self._avgdl = total_len / self._n_docs if self._n_docs else 0.0
        return self

    def _idf(self, term: str) -> float:
        df = len(self._postings.get(term, ()))
        if df == 0:
            return 0.0
        return math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if top_k <= 0:
            raise ValueError(f"top_k 必须为正数，实际 {top_k}")
        if self._n_docs == 0:
            raise RuntimeError("索引尚未构建，请先调用 fit()")

        avgdl = self._avgdl or 1.0
        scores: Dict[str, float] = {}
        # BM25 对同一查询词只计一次，因此先去重
        for term in set(self.tokenizer(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for doc_id, tf in postings.items():
                dl = self._doc_len[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (self.k1 + 1.0) / denom

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]
