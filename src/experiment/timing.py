"""实验耗时记录器。

固定 14 项计时字段，语义与 SDS §8.5 一致：
- 数字：该阶段适用并实际执行；
- `0.0`：适用，但缓存命中或耗时可忽略；
- `None`：该阶段对当前变体不适用。
"""

from __future__ import annotations

import math

TIMING_FIELDS: tuple[str, ...] = (
    "dataset_load_seconds",
    "model_load_seconds",
    "index_build_seconds",
    "doc_cache_load_seconds",
    "corpus_encode_seconds",
    "query_cache_load_seconds",
    "query_encode_seconds",
    "bm25_search_total_seconds",
    "dense_search_total_seconds",
    "rrf_fusion_seconds",
    "retrieval_total_seconds",
    "retrieval_mean_seconds",
    "retrieval_median_seconds",
    "evaluation_seconds",
)

HYBRID_SUBTOTAL_FIELDS: tuple[str, ...] = (
    "bm25_search_total_seconds",
    "dense_search_total_seconds",
    "rrf_fusion_seconds",
)


class DurationRecorder:
    """校验并汇总单个变体的阶段耗时。"""

    def __init__(self) -> None:
        self._values: dict[str, float | None] = {
            field: None for field in TIMING_FIELDS
        }

    @staticmethod
    def _check_field(field: str) -> None:
        if field not in TIMING_FIELDS:
            raise KeyError(f"未知计时字段: {field}")

    @staticmethod
    def _check_seconds(field: str, seconds: float) -> float:
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise TypeError(f"{field} 必须是数字，实际 {type(seconds).__name__}")
        value = float(seconds)
        if not math.isfinite(value):
            raise ValueError(f"{field} 必须是有限数值，实际 {value}")
        if value < 0:
            raise ValueError(f"{field} 必须非负，实际 {value}")
        return value

    def record(self, field: str, seconds: float) -> None:
        self._check_field(field)
        self._values[field] = self._check_seconds(field, seconds)

    def add(self, field: str, seconds: float) -> None:
        self._check_field(field)
        value = self._check_seconds(field, seconds)
        current = self._values[field] or 0.0
        self._values[field] = current + value

    def mark_not_applicable(self, field: str) -> None:
        self._check_field(field)
        self._values[field] = None

    def value(self, field: str) -> float | None:
        self._check_field(field)
        return self._values[field]

    def snapshot(self) -> dict[str, float | None]:
        """返回全部 14 个字段的副本。"""
        return dict(self._values)


def hybrid_subtotal(timing: dict[str, float | None]) -> float:
    """Hybrid 三个子项之和；缺失子项按错误处理。"""
    total = 0.0
    for field in HYBRID_SUBTOTAL_FIELDS:
        value = timing.get(field)
        if value is None:
            raise ValueError(f"Hybrid 计时缺少 {field}")
        total += value
    return total


def hybrid_total_matches(
    timing: dict[str, float | None], tolerance: float = 1e-6
) -> bool:
    """retrieval_total_seconds 是否与三个子项之和在容差内一致。"""
    total = timing.get("retrieval_total_seconds")
    if total is None:
        return False
    try:
        subtotal = hybrid_subtotal(timing)
    except ValueError:
        return False
    return abs(total - subtotal) <= tolerance
