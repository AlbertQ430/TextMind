"""Dense Retrieval：Embedding 编码 + 余弦相似度检索。

设计：
- 编码器通过构造参数注入（只需实现 ``encode`` 方法），便于单元测试
  在不加载真实模型的情况下验证检索逻辑；
- 文档与查询向量均做 L2 归一化，内积等价于余弦相似度；
- 检索为暴力矩阵乘法，SciFact 规模（数千文档）下足够快，
  不引入额外索引库；
- 可选 embedding 缓存：按 ``cache_key``（模型标识）+ 语料内容指纹
  保存文档向量，重复实验无需重新编码。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

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
        cache_dir: Optional[Path] = None,
        cache_key: str = "encoder",
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须为正整数，实际 {batch_size}")
        self.encoder = encoder
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_key = cache_key
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

    def _fingerprint(self, texts: Sequence[str]) -> str:
        digest = hashlib.sha256()
        digest.update(self.cache_key.encode("utf-8"))
        for text in texts:
            digest.update(text.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()[:16]

    def _cache_paths(self, texts: Sequence[str]) -> Tuple[Path, Path]:
        assert self.cache_dir is not None
        slug = re.sub(r"[^A-Za-z0-9._-]+", "__", self.cache_key)[:80]
        base = self.cache_dir / f"{slug}_{self._fingerprint(texts)}"
        return base.with_name(base.name + ".npy"), base.with_name(base.name + ".json")

    def _load_cached(self, texts: Sequence[str]) -> Optional[np.ndarray]:
        if self.cache_dir is None:
            return None
        npy_path, meta_path = self._cache_paths(texts)
        if not (npy_path.exists() and meta_path.exists()):
            return None
        try:
            embeddings = np.load(npy_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if (
            meta.get("doc_ids") != self.doc_ids
            or embeddings.ndim != 2
            or embeddings.shape[0] != len(self.doc_ids)
        ):
            return None
        return embeddings.astype(np.float32)

    def _save_cached(self, embeddings: np.ndarray, texts: Sequence[str]) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        npy_path, meta_path = self._cache_paths(texts)
        np.save(npy_path, embeddings)
        meta_path.write_text(
            json.dumps(
                {
                    "cache_key": self.cache_key,
                    "doc_ids": self.doc_ids,
                    "dim": int(embeddings.shape[1]),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def fit(self, documents: Iterable[Document]) -> "DenseRetriever":
        documents = list(documents)
        if not documents:
            raise ValueError("文档集合为空，无法构建索引")
        self.doc_ids = [doc.doc_id for doc in documents]
        texts = [doc.searchable_text for doc in documents]
        self.embeddings = self._load_cached(texts)
        if self.embeddings is None:
            self.embeddings = self._encode(texts)
            self._save_cached(self.embeddings, texts)
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
