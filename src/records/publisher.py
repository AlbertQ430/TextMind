"""终态结果发布器与重启对齐。

`RunRecordPublisher` 同时实现 T-004 的 `TerminalPublisher` 与
`ResultReconciler`：先写 staging、校验、原子发布目录，失败时保持非终态；
重启时优先使用合法最终 JSON，其次发布唯一合法 staging 记录。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from experiment.executor import ExecutionResult, RunCancelled
from experiment.stages import not_applicable_stages
from experiment.timing import TIMING_FIELDS
from records.charts import generate_charts
from records.repository import ResultConflictError, ResultRecordError, ResultRepository
from records.schema import RunRecord
from tasks.models import TaskRecord, TaskStatus


@dataclass(frozen=True)
class RecoveryResult:
    """重启对齐结果：记录中的终态和结束时间。"""

    status: TaskStatus
    ended_at: datetime
    cancelled_at: datetime | None = None


class ResultReconciler(Protocol):
    def reconcile(self, task: TaskRecord) -> RecoveryResult | None: ...


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


def _snapshot(task: TaskRecord) -> dict[str, Any]:
    snapshot = task.config_snapshot
    return snapshot if isinstance(snapshot, dict) else {}


def _variant_snapshot(task: TaskRecord) -> dict[str, Any]:
    variant = _snapshot(task).get("variant")
    return variant if isinstance(variant, dict) else {}


def _dataset_from_snapshot(task: TaskRecord) -> dict[str, Any]:
    dataset = _snapshot(task).get("dataset")
    dataset = dataset if isinstance(dataset, dict) else {}
    return {
        "name": str(dataset.get("name") or task.plan_id),
        "split": str(dataset.get("split") or "unknown"),
        "corpus_size": None,
        "query_count": None,
        "qrels_pairs": None,
    }


def _base_record(
    task: TaskRecord, status: str, finished_at: datetime
) -> dict[str, Any]:
    snapshot = _snapshot(task)
    variant = _variant_snapshot(task)
    method = variant.get("method")
    return {
        "format_version": "1.0",
        "run_id": task.run_id,
        "plan_id": task.plan_id,
        "plan_version": task.plan_version,
        "variant_id": task.variant_id,
        "plan_name": str(snapshot.get("plan_name") or task.plan_id),
        "variant_name": str(snapshot.get("variant_name") or task.variant_id),
        "status": status,
        "stage_progress": {
            "completed": list(task.completed_stages),
            "current": task.current_stage,
            "not_applicable": list(
                not_applicable_stages(method if isinstance(method, str) else "")
            ),
        },
        "started_at": _iso(task.started_at or task.submitted_at),
        "finished_at": _iso(finished_at),
        "dataset": _dataset_from_snapshot(task),
        "variant_config": dict(variant),
        "metrics": None,
        "num_queries": None,
        "num_skipped": None,
        "timing": {field: None for field in TIMING_FIELDS},
        "environment": {},
        "charts": [],
        "error": None,
    }


def assemble_completed(
    task: TaskRecord,
    execution: ExecutionResult,
    charts: list[dict[str, str]],
    finished_at: datetime,
) -> dict[str, Any]:
    record = _base_record(task, "Completed", finished_at)
    record["stage_progress"] = {
        "completed": list(execution.stage_progress.completed),
        "current": execution.stage_progress.current,
        "not_applicable": list(execution.stage_progress.not_applicable),
    }
    record["dataset"] = {
        "name": str(execution.dataset["name"]),
        "split": str(execution.dataset["split"]),
        "corpus_size": execution.dataset["corpus_size"],
        "query_count": execution.dataset["query_count"],
        "qrels_pairs": execution.dataset["qrels_pairs"],
    }
    record["variant_config"] = dict(execution.variant_config)
    record["metrics"] = dict(execution.metrics)
    record["num_queries"] = execution.num_queries
    record["num_skipped"] = execution.num_skipped
    record["timing"] = dict(execution.timing)
    record["environment"] = dict(execution.environment)
    record["charts"] = list(charts)
    record["error"] = None
    return record


def assemble_failed(
    task: TaskRecord, error: BaseException, finished_at: datetime
) -> dict[str, Any]:
    record = _base_record(task, "Failed", finished_at)
    message = str(error) or type(error).__name__
    record["stage_progress"]["current"] = task.current_stage
    record["error"] = {
        "message": message,
        "detail": f"{type(error).__name__}: {error}",
        "stage": task.current_stage,
    }
    return record


def assemble_cancelled(
    task: TaskRecord, cancelled: RunCancelled, finished_at: datetime
) -> dict[str, Any]:
    record = _base_record(task, "Cancelled", finished_at)
    record["cancelled_at"] = _iso(finished_at)
    record["stage_progress"]["current"] = cancelled.stage
    record["error"] = {
        "message": "运行已取消",
        "detail": None,
        "stage": cancelled.stage,
    }
    return record


def assemble_interrupted(
    task: TaskRecord, reason: str, finished_at: datetime
) -> dict[str, Any]:
    record = _base_record(task, "Interrupted", finished_at)
    record["error"] = {
        "message": reason,
        "detail": "SERVICE_INTERRUPTED",
        "stage": task.current_stage,
    }
    return record


class RunRecordPublisher:
    """把任务终态发布为 `results/runs/<run_id>/run.json`。"""

    def __init__(
        self,
        results_root: Path | str | None = None,
        *,
        now: Callable[[], datetime] | None = None,
        chart_generator: Callable = generate_charts,
    ) -> None:
        self._repository = ResultRepository(results_root)
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._chart_generator = chart_generator

    @property
    def repository(self) -> ResultRepository:
        return self._repository

    # ------------------------------------------------------------------ 发布
    def publish_completed(
        self, task: TaskRecord, execution: ExecutionResult
    ) -> None:
        staging = self._repository.create_staging(task.run_id)
        charts = self._chart_generator(task, execution, staging / "charts")
        data = assemble_completed(task, execution, charts, self._now())
        self._repository.write_record(staging, data)
        self._repository.validate_chart_paths(data, staging)
        self._repository.publish(staging, task.run_id)

    def publish_failed(self, task: TaskRecord, error: BaseException) -> None:
        data = assemble_failed(task, error, self._now())
        self._publish_simple(task, data)

    def publish_cancelled(self, task: TaskRecord, cancelled: RunCancelled) -> None:
        data = assemble_cancelled(task, cancelled, self._now())
        self._publish_simple(task, data)

    def publish_interrupted(self, task: TaskRecord, reason: str) -> None:
        data = assemble_interrupted(task, reason, self._now())
        self._publish_simple(task, data)

    def _publish_simple(self, task: TaskRecord, data: dict[str, Any]) -> None:
        staging = self._repository.create_staging(task.run_id)
        self._repository.write_record(staging, data)
        self._repository.publish(staging, task.run_id)

    # ------------------------------------------------------------------ 恢复对齐
    @staticmethod
    def _identity_matches(record: RunRecord, task: TaskRecord) -> bool:
        return (
            record.run_id == task.run_id
            and record.plan_id == task.plan_id
            and record.plan_version == task.plan_version
            and record.variant_id == task.variant_id
        )

    def reconcile(self, task: TaskRecord) -> RecoveryResult | None:
        record = self._repository.read(task.run_id)
        if record is not None:
            if not self._identity_matches(record, task):
                raise ResultRecordError(
                    "最终记录身份与任务不一致: "
                    f"{record.run_id}/{record.plan_id}/{record.plan_version}/"
                    f"{record.variant_id}"
                )
            return self._recovery(record)

        candidates = self._repository.find_staging_candidates(task.run_id)
        valid: list[tuple[Path, RunRecord]] = []
        for path in candidates:
            try:
                staging_record = self._repository.read_at(path)
            except ResultRecordError:
                continue
            if staging_record is None or not self._identity_matches(
                staging_record, task
            ):
                continue
            try:
                self._repository.validate_chart_paths(
                    staging_record.model_dump(mode="json"), path
                )
            except ResultRecordError:
                continue
            valid.append((path, staging_record))

        if not valid:
            return None
        if len(valid) > 1:
            raise ResultConflictError(
                f"存在多个合法 staging 记录: {[str(path) for path, _ in valid]}"
            )
        path, staging_record = valid[0]
        self._repository.publish(path, task.run_id)
        return self._recovery(staging_record)

    @staticmethod
    def _recovery(record: RunRecord) -> RecoveryResult:
        return RecoveryResult(
            status=TaskStatus(record.status),
            ended_at=record.finished_at,
            cancelled_at=record.cancelled_at,
        )
