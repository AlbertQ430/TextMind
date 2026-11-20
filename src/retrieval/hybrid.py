"""Hybrid Retrieval：多路检索结果用 RRF 融合。

Reciprocal Rank Fusion（RRF）：

    score(d) = Σ 1 / (rrf_k + rank_i(d))

其中 rank 从 1 开始，文档未出现在某一路结果中则不贡献分数。
融合只依赖排名，不需要归一化不同检索器的原始分数。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from retrieval.base import Retriever


class HybridRetriever(Retriever):
    """组合多个检索器的 RRF 融合检索器。"""

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        rrf_k: int = 60,
        candidate_depth: Optional[int] = None,
    ) -> None:
        if not retrievers:
            raise ValueError("至少需要一个子检索器")
        if rrf_k <= 0:
            raise ValueError(f"rrf_k 必须为正整数，实际 {rrf_k}")
        if candidate_depth is not None and candidate_depth <= 0:
            raise ValueError(f"candidate_depth 必须为正整数，实际 {candidate_depth}")
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k
        self.candidate_depth = candidate_depth

    def fit(self, documents: Iterable) -> "HybridRetriever":
        documents = list(documents)
        for retriever in self.retrievers:
            retriever.fit(documents)
        return self

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if top_k <= 0:
            raise ValueError(f"top_k 必须为正数，实际 {top_k}")

        depth = self.candidate_depth or max(top_k, 100)
        fused: Dict[str, float] = {}
        for retriever in self.retrievers:
            for rank, (doc_id, _) in enumerate(
                retriever.search(query, top_k=depth), start=1
            ):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]
