"""应用依赖容器与默认服务装配。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from experiment.executor import ExperimentExecutor
from experiment.plans import PlanCatalog, load_plan_catalog
from paths import ensure_app_dirs
from records.catalog import ResultCatalog
from records.importer import ImportService
from records.publisher import RunRecordPublisher
from records.repository import ResultRepository
from tasks.repository import TaskRepository
from tasks.service import TaskService
from web.session import SessionStore


@dataclass
class AppContext:
    """一个 Web 实例共享的目录、任务服务与会话存储。"""

    catalog: PlanCatalog
    task_service: Any
    session_store: SessionStore
    result_repository: ResultRepository | None = None
    result_catalog: Any | None = None
    import_service: Any | None = None
    sse_poll_interval: float = 0.5
    sse_heartbeat_interval: float = 15.0
    started: bool = False

    def start(self) -> None:
        if self.started:
            return
        ensure_app_dirs()
        self.task_service.start()
        self.started = True

    def stop(self) -> None:
        if not self.started:
            return
        self.task_service.stop_normal()
        self.started = False


def build_default_context() -> AppContext:
    """默认装配：真实方案目录、任务服务、执行器与结果发布器。"""
    catalog = load_plan_catalog()
    repository = TaskRepository()
    executor = ExperimentExecutor()
    publisher = RunRecordPublisher()
    service = TaskService(
        catalog,
        repository,
        executor,
        publisher,
        reconciler=publisher,
    )
    record_catalog = ResultCatalog(catalog=catalog)
    return AppContext(
        catalog=catalog,
        task_service=service,
        session_store=SessionStore(),
        result_repository=publisher.repository,
        result_catalog=record_catalog,
        import_service=ImportService(catalog=record_catalog),
    )
