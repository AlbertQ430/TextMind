"""检索评价指标：Precision@K / Recall@K / MRR / NDCG@K。

约定：
- ``retrieved`` 是按相关度降序排列的 doc_id 列表；
- ``relevant`` 是相关 doc_id 集合，NDCG 额外接受 ``{doc_id: relevance}``；
- 无相关标注的 query 由 ``evaluate`` 统一跳过并计数（不在本模块平均）。
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """前 K 个结果中相关文档的比例，分母固定为 K（TREC 口径）。"""
    if k <= 0:
        raise ValueError(f"k 必须为正数，实际 {k}")
    relevant_set = set(relevant)
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant_set)
    return hits / k


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """前 K 个结果召回了全部相关文档的比例。"""
    if k <= 0:
        raise ValueError(f"k 必须为正数，实际 {k}")
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant_set)
    return hits / len(relevant_set)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """第一个相关结果排名的倒数，没有命中时为 0。"""
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[str],
    relevance: Dict[str, int],
    k: int,
) -> float:
    """归一化折损累计增益，gain 直接使用 qrels 标注值。"""
    if k <= 0:
        raise ValueError(f"k 必须为正数，实际 {k}")

    def dcg(gains: Iterable[int]) -> float:
        return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))

    gains = [relevance.get(doc_id, 0) for doc_id in retrieved[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg(gains) / ideal_dcg


def evaluate(
    retrieved: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    ks: Sequence[int] = (1, 5, 10, 100),
) -> dict:
    """按 query 宏平均计算全部指标。

    参数
    ----
    retrieved: ``{query_id: [doc_id, ...]}``，已按相关度降序。
    qrels: ``{query_id: {doc_id: relevance}}``。

    返回
    ----
    包含 ``metrics``（均值）、``num_queries``（参与平均的 query 数）、
    ``num_skipped``（无相关标注被跳过的 query 数）的字典。
    """
    if not retrieved:
        raise ValueError("retrieved 为空，无法评估")

    ks = tuple(dict.fromkeys(ks))
    if not ks:
        raise ValueError("ks 不能为空")
    if any(k <= 0 for k in ks):
        raise ValueError(f"ks 必须全为正整数，实际 {ks}")

    sums: Dict[str, float] = {}
    for k in ks:
        sums[f"P@{k}"] = 0.0
        sums[f"R@{k}"] = 0.0
        sums[f"NDCG@{k}"] = 0.0
    sums["MRR"] = 0.0

    num_queries = 0
    num_skipped = 0
    for query_id, rels in qrels.items():
        relevant = {doc_id: score for doc_id, score in rels.items() if score > 0}
        if not relevant:
            num_skipped += 1
            continue
        top = retrieved.get(query_id, [])
        for k in ks:
            sums[f"P@{k}"] += precision_at_k(top, relevant, k)
            sums[f"R@{k}"] += recall_at_k(top, relevant, k)
            sums[f"NDCG@{k}"] += ndcg_at_k(top, relevant, k)
        sums["MRR"] += reciprocal_rank(top, relevant)
        num_queries += 1

    if num_queries == 0:
        raise ValueError("没有任何 query 带有相关标注，无法计算指标")

    metrics = {name: value / num_queries for name, value in sums.items()}
    return {
        "metrics": metrics,
        "num_queries": num_queries,
        "num_skipped": num_skipped,
    }
