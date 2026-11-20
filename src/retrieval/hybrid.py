"""Hybrid Retrieval：多路检索结果用 RRF 融合。

- 每路子检索器可配置独立的 query 变换：例如 Dense 需要模型指令前缀，
  而 BM25 必须接收原始 query；
- ``search_batch`` 批量执行，并分别统计查询编码、BM25 检索、Dense 向量
  检索与 RRF 融合耗时；
- 端到端检索耗时口径与 Dense/BM25 一致：不含查询编码。
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from retrieval.base import Retriever

QueryTransform = Callable[[str], str]


class HybridRetriever(Retriever):
    """组合多个检索器的 RRF 融合检索器。"""

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        rrf_k: int = 60,
        candidate_depth: Optional[int] = None,
        query_transforms: Optional[Sequence[Optional[QueryTransform]]] = None,
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        if not retrievers:
            raise ValueError("至少需要一个子检索器")
        if rrf_k <= 0:
            raise ValueError(f"rrf_k 必须为正整数，实际 {rrf_k}")
        if candidate_depth is not None and candidate_depth <= 0:
            raise ValueError(f"candidate_depth 必须为正整数，实际 {candidate_depth}")
        if query_transforms is not None and len(query_transforms) != len(retrievers):
            raise ValueError("query_transforms 长度必须与 retrievers 一致")
        if weights is not None:
            if len(weights) != len(retrievers):
                raise ValueError("weights 长度必须与 retrievers 一致")
            if any(weight < 0 for weight in weights):
                raise ValueError(f"weights 必须非负，实际 {weights}")
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k
        self.candidate_depth = candidate_depth
        self.query_transforms = (
            list(query_transforms)
            if query_transforms is not None
            else [None] * len(self.retrievers)
        )
        self.weights = (
            list(weights) if weights is not None else [1.0] * len(self.retrievers)
        )
        self.last_batch_timing: Dict[str, object] = {}

    def fit(self, documents: Iterable) -> "HybridRetriever":
        documents = list(documents)
        for retriever in self.retrievers:
            retriever.fit(documents)
        return self

    def _transform(self, index: int, query: str) -> str:
        transform = self.query_transforms[index]
        return transform(query) if transform else query

    def _depth(self, top_k: int) -> int:
        return self.candidate_depth or max(top_k, 100)

    @staticmethod
    def rrf_fuse(
        rankings: Sequence[Sequence[Tuple[str, float]]],
        rrf_k: int,
        top_k: int,
        weights: Optional[Sequence[float]] = None,
    ) -> List[Tuple[str, float]]:
        """加权 Reciprocal Rank Fusion：``score(d) = Σ wᵢ / (rrf_k + rank_i)``。"""
        fused: Dict[str, float] = {}
        for index, ranking in enumerate(rankings):
            weight = weights[index] if weights is not None else 1.0
            if weight == 0:
                continue
            for rank, (doc_id, _) in enumerate(ranking, start=1):
                fused[doc_id] = fused.get(doc_id, 0.0) + weight / (rrf_k + rank)
        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if top_k <= 0:
            raise ValueError(f"top_k 必须为正数，实际 {top_k}")
        depth = self._depth(top_k)
        rankings = [
            retriever.search(self._transform(index, query), top_k=depth)
            for index, retriever in enumerate(self.retrievers)
        ]
        return self.rrf_fuse(rankings, self.rrf_k, top_k, self.weights)

    def search_batch(
        self, queries: Sequence[str], top_k: int = 10
    ) -> List[List[Tuple[str, float]]]:
        """批量检索，返回每条 query 的融合结果。

        耗时写入 ``self.last_batch_timing``：
        ``query_encode`` / ``bm25_search`` / ``dense_search`` / ``rrf_fusion``
        以及逐 query 端到端耗时 ``per_query_end_to_end``（不含查询编码）。
        """
        queries = list(queries)
        if not queries:
            raise ValueError("查询集合为空，无法批量检索")
        if top_k <= 0:
            raise ValueError(f"top_k 必须为正数，实际 {top_k}")
        depth = self._depth(top_k)
        per_query = [0.0] * len(queries)
        timing: Dict[str, object] = {
            "query_encode": 0.0,
            "bm25_search": 0.0,
            "dense_search": 0.0,
            "rrf_fusion": 0.0,
            "per_query_end_to_end": per_query,
        }

        rankings_per: List[List[List[Tuple[str, float]]]] = []
        for index, retriever in enumerate(self.retrievers):
            texts = [self._transform(index, query) for query in queries]
            rankings: List[List[Tuple[str, float]]] = []
            if hasattr(retriever, "encode_queries"):
                start = time.perf_counter()
                vectors = retriever.encode_queries(texts)
                timing["query_encode"] += time.perf_counter() - start
                for position, vector in enumerate(vectors):
                    start = time.perf_counter()
                    rankings.append(retriever.search_vector(vector, top_k=depth))
                    elapsed = time.perf_counter() - start
                    timing["dense_search"] += elapsed
                    per_query[position] += elapsed
            else:
                for position, text in enumerate(texts):
                    start = time.perf_counter()
                    rankings.append(retriever.search(text, top_k=depth))
                    elapsed = time.perf_counter() - start
                    timing["bm25_search"] += elapsed
                    per_query[position] += elapsed
            rankings_per.append(rankings)

        fused = []
        for position in range(len(queries)):
            start = time.perf_counter()
            fused.append(
                self.rrf_fuse(
                    [rankings[position] for rankings in rankings_per],
                    self.rrf_k,
                    top_k,
                    self.weights,
                )
            )
            elapsed = time.perf_counter() - start
            timing["rrf_fusion"] += elapsed
            per_query[position] += elapsed

        self.last_batch_timing = timing
        return fused
