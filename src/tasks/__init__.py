"""SQLite 任务队列、状态机与单工作线程服务。"""

from tasks.locking import InstanceLock, InstanceLockError
from tasks.models import (
    ALLOWED_TRANSITIONS,
    NOTICE_KINDS,
    NOTICE_QUEUE_CANCELLED_ON_RESTART,
    NOTICE_QUEUE_CANCELLED_ON_SHUTDOWN,
    TERMINAL_STATUSES,
    Notice,
    ServiceNotRunningError,
    TaskConflictError,
    TaskNotFoundError,
    TaskRecord,
    TaskStateError,
    TaskStatus,
    TaskSubmitError,
    build_task_record,
    ensure_transition,
    generate_run_id,
    is_terminal,
    utc_now,
)
from tasks.repository import TaskRepository
from tasks.service import TaskService, TerminalPublisher

__all__ = [
    "ALLOWED_TRANSITIONS",
    "NOTICE_KINDS",
    "NOTICE_QUEUE_CANCELLED_ON_RESTART",
    "NOTICE_QUEUE_CANCELLED_ON_SHUTDOWN",
    "TERMINAL_STATUSES",
    "InstanceLock",
    "InstanceLockError",
    "Notice",
    "ServiceNotRunningError",
    "TaskConflictError",
    "TaskNotFoundError",
    "TaskRecord",
    "TaskRepository",
    "TaskService",
    "TaskStateError",
    "TaskStatus",
    "TaskSubmitError",
    "TerminalPublisher",
    "build_task_record",
    "ensure_transition",
    "generate_run_id",
    "is_terminal",
    "utc_now",
]
