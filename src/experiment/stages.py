"""实验执行阶段模型。

阶段顺序固定，执行器按顺序报告 `started` / `completed` / `skipped` 事件。
适用性由检索方法决定：BM25 不需要加载模型和编码查询。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

STAGE_PREPARE = "prepare"
STAGE_LOAD_DATASET = "load_dataset"
STAGE_LOAD_MODEL = "load_model"
STAGE_BUILD_INDEX = "build_index"
STAGE_ENCODE_QUERIES = "encode_queries"
STAGE_RETRIEVE = "retrieve"
STAGE_EVALUATE = "evaluate"

# 生命周期阶段固定顺序
STAGES: tuple[str, ...] = (
    STAGE_PREPARE,
    STAGE_LOAD_DATASET,
    STAGE_LOAD_MODEL,
    STAGE_BUILD_INDEX,
    STAGE_ENCODE_QUERIES,
    STAGE_RETRIEVE,
    STAGE_EVALUATE,
)

STAGE_LABELS: dict[str, str] = {
    STAGE_PREPARE: "准备",
    STAGE_LOAD_DATASET: "加载数据集",
    STAGE_LOAD_MODEL: "加载模型",
    STAGE_BUILD_INDEX: "构建索引",
    STAGE_ENCODE_QUERIES: "编码查询",
    STAGE_RETRIEVE: "检索",
    STAGE_EVALUATE: "评价",
}

StageEventType = Literal["started", "completed", "skipped"]
STAGE_EVENTS: tuple[str, ...] = ("started", "completed", "skipped")

# BM25 不适用的阶段
_BM25_NOT_APPLICABLE: frozenset[str] = frozenset(
    {STAGE_LOAD_MODEL, STAGE_ENCODE_QUERIES}
)


@dataclass(frozen=True)
class StageEvent:
    """单个阶段状态事件；时间戳必须带时区。"""

    stage: str
    event: StageEventType
    timestamp: datetime

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"未知阶段: {self.stage}")
        if self.event not in STAGE_EVENTS:
            raise ValueError(f"未知阶段事件: {self.event}")
        if self.timestamp.tzinfo is None:
            raise ValueError("阶段事件时间必须包含时区")


def not_applicable_stages(method: str) -> tuple[str, ...]:
    """返回该方法不适用的阶段（按固定顺序）。"""
    if method == "bm25":
        return tuple(stage for stage in STAGES if stage in _BM25_NOT_APPLICABLE)
    return ()


def applicable_stages(method: str) -> tuple[str, ...]:
    skipped = set(not_applicable_stages(method))
    return tuple(stage for stage in STAGES if stage not in skipped)


def stage_is_applicable(method: str, stage: str) -> bool:
    if stage not in STAGES:
        raise ValueError(f"未知阶段: {stage}")
    return stage not in not_applicable_stages(method)
