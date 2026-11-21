"""Web 路由与 API 序列化辅助。"""

from __future__ import annotations

from datetime import datetime, timezone

from experiment.plans import (
    PlanCatalog,
    PlanDefinition,
    ReferenceVariant,
    RunVariant,
)
from experiment.stages import not_applicable_stages
from tasks.models import TaskRecord, TaskStatus

ACTIVE_STATUSES = frozenset(
    {TaskStatus.PREPARING, TaskStatus.RUNNING, TaskStatus.CANCELLING}
)

RESOURCE_PENDING = "待资源检查"


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


def find_plan(catalog: PlanCatalog, plan_id: str) -> PlanDefinition | None:
    for plan in catalog.plans:
        if plan.plan_id == plan_id:
            return plan
    return None


def plan_summary(plan: PlanDefinition) -> dict:
    run_variants = [
        variant for variant in plan.variants if isinstance(variant, RunVariant)
    ]
    references = [
        variant for variant in plan.variants if isinstance(variant, ReferenceVariant)
    ]
    return {
        "plan_id": plan.plan_id,
        "plan_version": plan.plan_version,
        "name": plan.name,
        "description": plan.description,
        "dataset": {
            "name": plan.dataset.name,
            "language": plan.dataset.language,
            "split": plan.dataset.split,
        },
        "variant_count": len(plan.variants),
        "run_variant_count": len(run_variants),
        "reference_count": len(references),
    }


def variant_summary(variant) -> dict:
    if isinstance(variant, ReferenceVariant):
        return {
            "variant_id": variant.variant_id,
            "name": variant.name,
            "kind": "reference",
            "method": None,
            "runnable": False,
            "resource_status": "参照结果，不可提交",
            "preset_run_id": None,
            "source_plan_id": variant.source_plan_id,
            "source_plan_version": variant.source_plan_version,
            "source_variant_id": variant.source_variant_id,
        }

    data: dict = {
        "variant_id": variant.variant_id,
        "name": variant.name,
        "kind": "run",
        "method": variant.method,
        "runnable": True,
        "resource_status": RESOURCE_PENDING,
        "preset_run_id": variant.preset_run_id,
        "preset_source": variant.preset_source,
    }
    if variant.method in ("bm25", "hybrid"):
        data.update(
            {
                "k1": variant.k1,
                "b": variant.b,
                "tokenizer": variant.tokenizer,
            }
        )
    if variant.method in ("dense", "hybrid"):
        data.update(
            {
                "model": {
                    "name": variant.model.name,
                    "directory_name": variant.model.directory_name,
                    "revision": variant.model.revision,
                },
                "query_prefix": variant.query_prefix,
                "batch_size": variant.batch_size,
                "normalize_embeddings": variant.normalize_embeddings,
            }
        )
    if variant.method == "hybrid":
        data.update(
            {
                "rrf_k": variant.rrf_k,
                "rrf_weights": list(variant.rrf_weights),
            }
        )
    return data


def plan_detail(plan: PlanDefinition) -> dict:
    detail = plan_summary(plan)
    detail["evaluation"] = {
        "top_k": plan.evaluation.top_k,
        "k_values": list(plan.evaluation.k_values),
    }
    detail["variants"] = [variant_summary(variant) for variant in plan.variants]
    return detail


def task_summary(task: TaskRecord) -> dict:
    snapshot = task.config_snapshot if isinstance(task.config_snapshot, dict) else {}
    variant = snapshot.get("variant") if isinstance(snapshot.get("variant"), dict) else {}
    method = variant.get("method")
    elapsed: float | None = None
    if task.started_at is not None:
        end = task.ended_at or datetime.now(timezone.utc)
        elapsed = max(0.0, (end - task.started_at).total_seconds())
    return {
        "run_id": task.run_id,
        "plan_id": task.plan_id,
        "plan_version": task.plan_version,
        "variant_id": task.variant_id,
        "plan_name": str(snapshot.get("plan_name") or task.plan_id),
        "variant_name": str(snapshot.get("variant_name") or task.variant_id),
        "method": method if isinstance(method, str) else None,
        "not_applicable_stages": list(
            not_applicable_stages(method if isinstance(method, str) else "")
        ),
        "status": task.status.value,
        "current_stage": task.current_stage,
        "completed_stages": list(task.completed_stages),
        "submitted_at": _iso(task.submitted_at),
        "started_at": _iso(task.started_at) if task.started_at else None,
        "ended_at": _iso(task.ended_at) if task.ended_at else None,
        "cancel_requested": task.cancel_requested,
        "elapsed_seconds": round(elapsed, 3) if elapsed is not None else None,
        "error_summary": task.error_summary,
    }


def task_snapshot(tasks: list[TaskRecord]) -> dict:
    summaries = [task_summary(task) for task in tasks]
    current = next(
        (
            summary
            for summary in summaries
            if summary["status"] in {status.value for status in ACTIVE_STATUSES}
        ),
        None,
    )
    queue = [
        summary for summary in summaries if summary["status"] == TaskStatus.PENDING.value
    ]
    return {
        "current": current,
        "queue": queue,
        "tasks": summaries,
    }
