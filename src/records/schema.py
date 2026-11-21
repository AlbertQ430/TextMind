"""运行记录模型与 JSON Schema 双向校验。

记录格式版本为 1.0，只描述一次单变体运行；结果来源由所在目录决定，
因此模型禁止 `source`、`result_source`、`import_time` 等来源或导入时间字段。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator, FormatChecker
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from paths import SCHEMAS_DIR

RUN_RECORD_SCHEMA_PATH: Path = SCHEMAS_DIR / "run_record.schema.json"
FORMAT_VERSION = "1.0"
RUN_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
TERMINAL_STATUSES = ("Completed", "Failed", "Cancelled", "Interrupted")

_TIMING_FIELDS = (
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


class RunRecordError(Exception):
    """运行记录校验失败；消息中包含字段路径。"""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        self.errors = list(errors or [])
        super().__init__(message)


class _RecordModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class StageProgress(_RecordModel):
    completed: tuple[str, ...]
    current: str | None
    not_applicable: tuple[str, ...]


class DatasetInfo(_RecordModel):
    name: str = Field(min_length=1)
    split: str = Field(min_length=1)
    corpus_size: int | None = Field(ge=0)
    query_count: int | None = Field(ge=0)
    qrels_pairs: int | None = Field(ge=0)


class TimingRecord(_RecordModel):
    dataset_load_seconds: float | None = Field(ge=0)
    model_load_seconds: float | None = Field(ge=0)
    index_build_seconds: float | None = Field(ge=0)
    doc_cache_load_seconds: float | None = Field(ge=0)
    corpus_encode_seconds: float | None = Field(ge=0)
    query_cache_load_seconds: float | None = Field(ge=0)
    query_encode_seconds: float | None = Field(ge=0)
    bm25_search_total_seconds: float | None = Field(ge=0)
    dense_search_total_seconds: float | None = Field(ge=0)
    rrf_fusion_seconds: float | None = Field(ge=0)
    retrieval_total_seconds: float | None = Field(ge=0)
    retrieval_mean_seconds: float | None = Field(ge=0)
    retrieval_median_seconds: float | None = Field(ge=0)
    evaluation_seconds: float | None = Field(ge=0)


class ChartRef(_RecordModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    path: str = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def _validate_chart_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("图表路径不能为空")
        if value.startswith("/"):
            raise ValueError("图表路径必须是记录目录内的相对路径")
        if value[1:2] == ":":
            raise ValueError("图表路径不能包含盘符")
        if "\\" in value:
            raise ValueError("图表路径必须使用 POSIX 分隔符")
        parts = value.split("/")
        if any(part in ("", ".", "..") for part in parts):
            raise ValueError("图表路径不能包含空段、. 或 ..")
        return value


class ErrorInfo(_RecordModel):
    message: str = Field(min_length=1)
    detail: str | None = None
    stage: str | None = None


class RunRecord(_RecordModel):
    format_version: Literal["1.0"]
    run_id: str = Field(pattern=RUN_ID_PATTERN)
    plan_id: str = Field(min_length=1)
    plan_version: str = Field(min_length=1)
    variant_id: str = Field(min_length=1)
    plan_name: str = Field(min_length=1)
    variant_name: str = Field(min_length=1)
    status: Literal["Completed", "Failed", "Cancelled", "Interrupted"]
    stage_progress: StageProgress
    started_at: AwareDatetime
    finished_at: AwareDatetime
    cancelled_at: AwareDatetime | None = None
    dataset: DatasetInfo
    variant_config: dict[str, Any]
    metrics: dict[str, float] | None = None
    num_queries: int | None = Field(default=None, ge=0)
    num_skipped: int | None = Field(default=None, ge=0)
    timing: TimingRecord
    environment: dict[str, Any] = Field(default_factory=dict)
    charts: tuple[ChartRef, ...] = ()
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def _validate_terminal_state(self) -> "RunRecord":
        if self.status == "Completed":
            if not self.metrics:
                raise ValueError("Completed 记录必须包含非空 metrics")
            if self.num_queries is None:
                raise ValueError("Completed 记录必须包含 num_queries")
            if self.num_skipped is None:
                raise ValueError("Completed 记录必须包含 num_skipped")
            if self.error is not None:
                raise ValueError("Completed 记录的 error 必须为 null")
        elif self.error is None:
            raise ValueError(f"{self.status} 记录必须包含 error")

        if self.status == "Cancelled" and self.cancelled_at is None:
            raise ValueError("Cancelled 记录必须包含非空 cancelled_at")
        return self


_schema_cache: dict[str, Any] | None = None


def run_record_schema() -> dict[str, Any]:
    """读取并缓存 run_record.schema.json。"""
    global _schema_cache
    if _schema_cache is None:
        _schema_cache = json.loads(
            RUN_RECORD_SCHEMA_PATH.read_text(encoding="utf-8")
        )
    return _schema_cache


def _json_path(error) -> str:
    path = "$"
    for part in error.absolute_path:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return path


def _format_pydantic_error(error: ValidationError) -> str:
    lines: list[str] = []
    for item in error.errors():
        loc = "$"
        for part in item["loc"]:
            loc += f"[{part}]" if isinstance(part, int) else f".{part}"
        lines.append(f"{loc}: {item['msg']}")
    return "\n".join(lines)


def validate_run_record_schema(data: dict[str, Any]) -> None:
    """按 JSON Schema 校验原始记录；失败时抛出 RunRecordError。"""
    if not isinstance(data, dict):
        raise RunRecordError("运行记录顶层必须是 JSON 对象")
    validator = Draft202012Validator(
        run_record_schema(), format_checker=FormatChecker()
    )
    errors = sorted(validator.iter_errors(data), key=lambda item: list(item.absolute_path))
    if errors:
        messages = [f"{_json_path(error)}: {error.message}" for error in errors]
        raise RunRecordError(
            "运行记录不符合 run_record.schema.json：\n" + "\n".join(messages),
            messages,
        )


def parse_run_record(data: dict[str, Any]) -> RunRecord:
    """先按 JSON Schema，再按 Pydantic 校验并返回运行记录。"""
    validate_run_record_schema(data)
    try:
        return RunRecord.model_validate(data)
    except ValidationError as exc:
        raise RunRecordError(
            "运行记录校验失败：\n" + _format_pydantic_error(exc)
        ) from exc
