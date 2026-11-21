"""任务状态、记录模型与 Run ID 生成。

任务状态机按 SDS §7.2 固定；非法转换抛出 `TaskStateError`，不得静默覆盖。
"""

from __future__ import annotations

import secrets
import string
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping

RUN_ID_SUFFIX_LENGTH = 6


class TaskStatus(StrEnum):
    PENDING = "Pending"
    PREPARING = "Preparing"
    RUNNING = "Running"
    CANCELLING = "Cancelling"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"
    INTERRUPTED = "Interrupted"


TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset(
    {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.INTERRUPTED,
    }
)

# Pending 取消是删除任务行，不产生 Cancelled 行，因此不在状态转换表中。
ALLOWED_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PENDING: frozenset({TaskStatus.PREPARING}),
    TaskStatus.PREPARING: frozenset(
        {
            TaskStatus.RUNNING,
            TaskStatus.FAILED,
            TaskStatus.CANCELLING,
            TaskStatus.INTERRUPTED,
        }
    ),
    TaskStatus.RUNNING: frozenset(
        {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLING,
            TaskStatus.INTERRUPTED,
        }
    ),
    TaskStatus.CANCELLING: frozenset(
        {
            TaskStatus.CANCELLED,
            # 取消期间底层执行抛出普通异常时按失败处理，保留取消请求。
            TaskStatus.FAILED,
            TaskStatus.INTERRUPTED,
        }
    ),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
    TaskStatus.INTERRUPTED: frozenset(),
}

NOTICE_QUEUE_CANCELLED_ON_SHUTDOWN = "queue_cancelled_on_shutdown"
NOTICE_QUEUE_CANCELLED_ON_RESTART = "queue_cancelled_on_restart"
NOTICE_KINDS: frozenset[str] = frozenset(
    {NOTICE_QUEUE_CANCELLED_ON_SHUTDOWN, NOTICE_QUEUE_CANCELLED_ON_RESTART}
)


class TaskStateError(ValueError):
    """当前状态不允许目标状态。"""


class TaskConflictError(TaskStateError):
    """任务已处于终态，操作冲突。"""


class TaskNotFoundError(KeyError):
    """任务不存在。"""


class TaskSubmitError(ValueError):
    """提交参数无法解析为唯一可运行变体。"""


class ServiceNotRunningError(RuntimeError):
    """任务服务未启动或已停止接单。"""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def is_terminal(status: TaskStatus) -> bool:
    return status in TERMINAL_STATUSES


def ensure_transition(current: TaskStatus, target: TaskStatus) -> None:
    if target not in ALLOWED_TRANSITIONS[current]:
        raise TaskStateError(
            f"非法任务状态转换: {current.value} -> {target.value}"
        )


def generate_run_id(
    now: datetime | None = None, suffix: str | None = None
) -> str:
    """生成 ``YYYYMMDDTHHMMSSffffffZ-<6 位小写字母数字>`` 形式的 Run ID。"""
    moment = (now or utc_now()).astimezone(timezone.utc)
    if suffix is None:
        alphabet = string.ascii_lowercase + string.digits
        suffix = "".join(
            secrets.choice(alphabet) for _ in range(RUN_ID_SUFFIX_LENGTH)
        )
    return f"{moment.strftime('%Y%m%dT%H%M%S%fZ')}-{suffix}"


@dataclass(frozen=True)
class TaskRecord:
    run_id: str
    plan_id: str
    plan_version: str
    variant_id: str
    status: TaskStatus
    current_stage: str | None
    completed_stages: tuple[str, ...]
    submitted_at: datetime
    started_at: datetime | None
    ended_at: datetime | None
    cancel_requested: bool
    config_snapshot: Mapping[str, Any]
    error_summary: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


@dataclass(frozen=True)
class Notice:
    id: int
    kind: str
    count: int
    created_at: datetime


def build_task_record(
    run_id: str,
    plan_id: str,
    plan_version: str,
    variant_id: str,
    config_snapshot: Mapping[str, Any],
    *,
    submitted_at: datetime | None = None,
    status: TaskStatus = TaskStatus.PENDING,
) -> TaskRecord:
    """构造 Pending 任务记录（提交或测试使用）。"""
    return TaskRecord(
        run_id=run_id,
        plan_id=plan_id,
        plan_version=plan_version,
        variant_id=variant_id,
        status=status,
        current_stage=None,
        completed_stages=(),
        submitted_at=submitted_at or utc_now(),
        started_at=None,
        ended_at=None,
        cancel_requested=False,
        config_snapshot=dict(config_snapshot),
        error_summary=None,
    )
