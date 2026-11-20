"""Dense Retrieval：Embedding 编码 + 余弦相似度检索。

设计：
- 编码器通过构造参数注入（只需实现 ``encode`` 方法），便于单元测试
  在不加载真实模型的情况下验证检索逻辑；
- 文档与查询向量均做 L2 归一化，内积等价于余弦相似度；
- 检索为暴力矩阵乘法，SciFact 规模（数千文档）下足够快，
  不引入额外索引库；
- 文档与查询向量分别缓存，缓存元数据记录模型名与版本；
- 耗时拆分为：文档缓存加载、语料编码、查询缓存加载、查询编码、向量检索。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from document.dataset import Document
from retrieval.base import Retriever

CACHE_VERSION = 2


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
        model_name: Optional[str] = None,
        model_revision: Optional[str] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须为正整数，实际 {batch_size}")
        self.encoder = encoder
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_key = cache_key
        self.model_name = model_name
        self.model_revision = model_revision
        self.doc_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.timing: Dict[str, float] = {
            "doc_cache_load": 0.0,
            "corpus_encode": 0.0,
            "query_cache_load": 0.0,
            "query_encode": 0.0,
        }

    # ------------------------------------------------------------------ 编码
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

    def _meta_matches(self, meta: dict) -> bool:
        """校验缓存版本与模型标识，不匹配时缓存失效。"""
        if meta.get("version") != CACHE_VERSION:
            return False
        if self.model_name is not None and meta.get("model_name") != self.model_name:
            return False
        if self.model_revision is not None and meta.get("revision") != self.model_revision:
            return False
        return True

    # ------------------------------------------------------------------ 缓存
    def _fingerprint(self, texts: Sequence[str]) -> str:
        digest = hashlib.sha256()
        digest.update(self.cache_key.encode("utf-8"))
        for text in texts:
            digest.update(text.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()[:16]

    def _slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "__", self.cache_key)[:80]

    def _doc_cache_paths(self, texts: Sequence[str]) -> Tuple[Path, Path]:
        assert self.cache_dir is not None
        base = self.cache_dir / f"{self._slug()}_{self._fingerprint(texts)}"
        return base.with_name(base.name + ".npy"), base.with_name(base.name + ".json")

    def _query_cache_paths(self, queries: Sequence[str]) -> Tuple[Path, Path]:
        assert self.cache_dir is not None
        base = self.cache_dir / f"{self._slug()}_queries_{self._fingerprint(queries)}"
        return base.with_name(base.name + ".npy"), base.with_name(base.name + ".json")

    def _load_cached(self, texts: Sequence[str]) -> Optional[np.ndarray]:
        if self.cache_dir is None:
            return None
        npy_path, meta_path = self._doc_cache_paths(texts)
        if not (npy_path.exists() and meta_path.exists()):
            return None
        try:
            embeddings = np.load(npy_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not self._meta_matches(meta):
            return None
        if (
            meta.get("doc_ids") != self.doc_ids
            or embeddings.ndim != 2
            or embeddings.shape[0] != len(self.doc_ids)
            or meta.get("dim") != embeddings.shape[1]
        ):
            return None
        return embeddings.astype(np.float32)

    def _save_cached(self, embeddings: np.ndarray, texts: Sequence[str]) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        npy_path, meta_path = self._doc_cache_paths(texts)
        np.save(npy_path, embeddings)
        meta_path.write_text(
            json.dumps(
                {
                    "version": CACHE_VERSION,
                    "cache_key": self.cache_key,
                    "model_name": self.model_name,
                    "revision": self.model_revision,
                    "doc_ids": self.doc_ids,
                    "dim": int(embeddings.shape[1]),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _load_query_cached(self, queries: Sequence[str]) -> Optional[np.ndarray]:
        if self.cache_dir is None:
            return None
        npy_path, meta_path = self._query_cache_paths(queries)
        if not (npy_path.exists() and meta_path.exists()):
            return None
        try:
            vectors = np.load(npy_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not self._meta_matches(meta):
            return None
        if (
            meta.get("queries") != list(queries)
            or vectors.shape[0] != len(queries)
            or meta.get("dim") != vectors.shape[1]
        ):
            return None
        return vectors.astype(np.float32)

    def _save_query_cached(self, vectors: np.ndarray, queries: Sequence[str]) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        npy_path, meta_path = self._query_cache_paths(queries)
        np.save(npy_path, vectors)
        meta_path.write_text(
            json.dumps(
                {
                    "version": CACHE_VERSION,
                    "cache_key": self.cache_key,
                    "model_name": self.model_name,
                    "revision": self.model_revision,
                    "queries": list(queries),
                    "dim": int(vectors.shape[1]),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ 索引
    def fit(self, documents: Iterable[Document]) -> "DenseRetriever":
        documents = list(documents)
        if not documents:
            raise ValueError("文档集合为空，无法构建索引")
        self.doc_ids = [doc.doc_id for doc in documents]
        texts = [doc.searchable_text for doc in documents]

        start = time.perf_counter()
        cached = self._load_cached(texts)
        self.timing["doc_cache_load"] = time.perf_counter() - start
        if cached is not None:
            self.embeddings = cached
        else:
            start = time.perf_counter()
            self.embeddings = self._encode(texts)
            self.timing["corpus_encode"] = time.perf_counter() - start
            self._save_cached(self.embeddings, texts)
        return self

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        """批量编码查询（带磁盘缓存），返回 (Q, D) 矩阵。"""
        queries = list(queries)
        if not queries:
            raise ValueError("查询集合为空")
        start = time.perf_counter()
        cached = self._load_query_cached(queries)
        self.timing["query_cache_load"] = time.perf_counter() - start
        if cached is not None:
            return cached
        start = time.perf_counter()
        vectors = self._encode(queries)
        self.timing["query_encode"] = time.perf_counter() - start
        self._save_query_cached(vectors, queries)
        return vectors

    # ------------------------------------------------------------------ 检索
    def search_vector(self, vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """按文档向量与查询向量的内积返回 Top-K。"""
        if top_k <= 0:
            raise ValueError(f"top_k 必须为正数，实际 {top_k}")
        if self.embeddings is None:
            raise RuntimeError("索引尚未构建，请先调用 fit()")
        scores = self.embeddings @ vector
        order = np.argsort(-scores)[:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in order]

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if top_k <= 0:
            raise ValueError(f"top_k 必须为正数，实际 {top_k}")
        if self.embeddings is None:
            raise RuntimeError("索引尚未构建，请先调用 fit()")
        vector = self._encode([query])[0]
        return self.search_vector(vector, top_k)
