"""任务提交、查询与取消 API。"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from resources.checker import ResourceUnavailableError
from resources.settings import SettingsInvalidError, SettingsReadError
from tasks.models import TaskNotFoundError
from web.routes import task_snapshot, task_summary
from web.security import ApiError

router = APIRouter(prefix="/api/v1/tasks")


class SubmitTaskRequest(BaseModel):
    plan_id: str = Field(min_length=1)
    plan_version: str = Field(min_length=1)
    variant_id: str = Field(min_length=1)


@router.post("", status_code=201)
def submit_task(payload: SubmitTaskRequest, request: Request) -> dict:
    service = request.app.state.context.task_service
    try:
        run_id = service.submit(
            payload.plan_id, payload.plan_version, payload.variant_id
        )
    except ResourceUnavailableError as exc:
        raise ApiError(
            409, "RESOURCE_UNAVAILABLE", exc.message, exc.missing
        ) from None
    except (SettingsReadError, SettingsInvalidError) as exc:
        raise ApiError(422, "SETTINGS_INVALID", str(exc)) from None
    return {"run_id": run_id, "task": task_summary(service.get(run_id))}


@router.get("")
def list_tasks(request: Request) -> dict:
    tasks = request.app.state.context.task_service.list_tasks()
    snapshot = task_snapshot(tasks)
    return {
        **snapshot,
        "counts": {
            "total": len(tasks),
            "pending": len(snapshot["queue"]),
            "active": 1 if snapshot["current"] is not None else 0,
        },
    }


@router.get("/{run_id}")
def get_task(run_id: str, request: Request) -> dict:
    task = request.app.state.context.task_service.get(run_id)
    return {"task": task_summary(task)}


@router.post("/{run_id}/cancel")
def cancel_task(run_id: str, request: Request) -> dict:
    service = request.app.state.context.task_service
    service.cancel(run_id)
    try:
        task = service.get(run_id)
    except TaskNotFoundError:
        return {"run_id": run_id, "status": "Deleted"}
    return {"run_id": run_id, "status": task.status.value}
