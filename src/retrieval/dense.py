"""Dense Retrieval：Embedding 编码 + 余弦相似度检索。

设计：
- 编码器通过构造参数注入（只需实现 ``encode`` 方法），便于单元测试
  在不加载真实模型的情况下验证检索逻辑；
- 文档与查询向量均做 L2 归一化，内积等价于余弦相似度；
- 检索为暴力矩阵乘法，SciFact 规模（数千文档）下足够快，
  不引入额外索引库。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from document.dataset import Document
from retrieval.base import Retriever


class Encoder:
    """编码器接口：把文本列表编码为二维向量矩阵。

    ``sentence_transformers.SentenceTransformer`` 已满足该接口。
    """

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:  # pragma: no cover - 接口定义
        raise NotImplementedError


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class DenseRetriever(Retriever):
    """基于预训练 Embedding 模型的向量检索器。"""

    def __init__(
        self,
        encoder: Encoder,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须为正整数，实际 {batch_size}")
        self.encoder = encoder
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.doc_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.encoder.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            normalize_embeddings=True,
        )
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"编码器应返回二维矩阵，实际形状 {matrix.shape}")
        return _l2_normalize(matrix)

    def fit(self, documents: Iterable[Document]) -> "DenseRetriever":
        documents = list(documents)
        if not documents:
            raise ValueError("文档集合为空，无法构建索引")
        self.doc_ids = [doc.doc_id for doc in documents]
        self.embeddings = self._encode([doc.searchable_text for doc in documents])
        return self

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if top_k <= 0:
            raise ValueError(f"top_k 必须为正数，实际 {top_k}")
        if self.embeddings is None:
            raise RuntimeError("索引尚未构建，请先调用 fit()")

        query_vector = self._encode([query])[0]
        scores = self.embeddings @ query_vector
        order = np.argsort(-scores)[:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in order]
