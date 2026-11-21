"""单工作线程任务服务。

职责：FIFO 队列、任务生命周期、阶段进度、阶段边界取消、正常关闭、
启动恢复以及与终态发布接口（`TerminalPublisher`）的顺序约定。
本任务不写运行结果 JSON，发布实现由 T-005 提供。
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime
from typing import Any, Callable, Protocol

from experiment.executor import ExecutionResult, ExperimentExecutor, RunCancelled
from experiment.plans import PlanCatalog, PlanDefinition, RunVariant
from experiment.stages import StageEvent
from paths import INSTANCE_LOCK_PATH
from records.repository import ResultConflictError
from resources.checker import ResourceUnavailableError
from tasks.locking import InstanceLock
from tasks.models import (
    NOTICE_QUEUE_CANCELLED_ON_RESTART,
    NOTICE_QUEUE_CANCELLED_ON_SHUTDOWN,
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
    utc_now,
)
from tasks.repository import TaskRepository


class TerminalPublisher(Protocol):
    """终态发布接口；发布成功后才允许提交 SQLite 终态。"""

    def publish_completed(
        self, task: TaskRecord, execution: ExecutionResult
    ) -> None: ...

    def publish_failed(self, task: TaskRecord, error: BaseException) -> None: ...

    def publish_cancelled(
        self, task: TaskRecord, cancelled: RunCancelled
    ) -> None: ...

    def publish_interrupted(self, task: TaskRecord, reason: str) -> None: ...


class _ExecutionToken:
    """单次执行的有效期令牌。

    正常关闭接管任务时令牌失效；旧工作线程恢复后只能退出，
    不得继续更新阶段或提交终态。
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._valid = True
        self._lock = threading.Lock()

    @property
    def valid(self) -> bool:
        return self._valid

    def invalidate(self) -> None:
        with self._lock:
            self._valid = False


class _CancellationToken:
    """把服务取消事件和执行令牌适配为执行器取消令牌。"""

    def __init__(self, event: threading.Event, token: _ExecutionToken) -> None:
        self._event = event
        self._token = token

    def raise_if_cancelled(self, stage: str) -> None:
        if not self._token.valid or self._event.is_set():
            raise RunCancelled(stage)


class TaskService:
    """单实例、单工作线程的后端任务服务。"""

    def __init__(
        self,
        catalog: PlanCatalog,
        repository: TaskRepository,
        executor: ExperimentExecutor,
        publisher: TerminalPublisher,
        *,
        reconciler: Any | None = None,
        resource_checker: Any | None = None,
        lock: InstanceLock | None = None,
        poll_interval: float = 0.05,
        join_timeout: float = 5.0,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._catalog = catalog
        self._repository = repository
        self._executor = executor
        self._publisher = publisher
        self._reconciler = reconciler
        self._resource_checker = resource_checker
        self._lock = lock
        self._poll_interval = poll_interval
        self._join_timeout = join_timeout
        self._now = now or utc_now

        self._started = False
        self._accepting = False
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._shutdown_event = threading.Event()
        # 终态发布与取消共用的任务级互斥边界。
        self._publish_lock = threading.Lock()
        # submit 的“接单判断 + 写 Pending”与 stop_normal 的“停止接单 + 清队列”共用。
        self._lifecycle_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._current_run_id: str | None = None
        self._current_cancel_event: threading.Event | None = None
        self._current_token: _ExecutionToken | None = None
        self._last_worker_error: BaseException | None = None

    @property
    def last_worker_error(self) -> BaseException | None:
        """最近一次工作线程内部未预期异常，供诊断和测试使用。"""
        return self._last_worker_error

    # ------------------------------------------------------------------ 生命周期
    def start(self) -> None:
        if self._started:
            raise ServiceNotRunningError("任务服务已经启动")
        lock = self._lock or InstanceLock(INSTANCE_LOCK_PATH)
        acquired = False
        try:
            lock.acquire()
            acquired = True
            self._repository.initialize()
            self._recover()
        except BaseException:
            if acquired:
                lock.release()
            raise
        self._lock = lock
        self._stop_event.clear()
        self._shutdown_event.clear()
        self._accepting = True
        self._started = True
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="textmind-task-worker",
            daemon=True,
        )
        self._worker.start()

    def stop_normal(self, reason: str = "服务关闭") -> None:
        # 与 submit 的接单判断和 Pending 写入互斥，避免关闭后残留 Pending。
        with self._lifecycle_lock:
            if not self._started:
                return
            self._accepting = False
            self._shutdown_event.set()

            self._repository.delete_all_pending_with_notice(
                NOTICE_QUEUE_CANCELLED_ON_SHUTDOWN
            )

            if self._current_cancel_event is not None:
                self._current_cancel_event.set()
            with self._publish_lock:
                token = self._current_token
                if token is not None:
                    # 撤销旧工作线程的阶段更新权和终态发布权。
                    token.invalidate()
                run_id = self._current_run_id
                if run_id is not None:
                    task = self._repository.get(run_id)
                    if task is not None:
                        self._interrupt_one(task, reason, reason)

            self._stop_event.set()
            self._wake_event.set()

            worker = self._worker
            if worker is not None and worker.is_alive():
                worker.join(timeout=self._join_timeout)
            self._worker = None

            # 兜底：关闭期间被工作线程取走但未能中断的任务。
            for task in self._repository.active_tasks():
                with self._publish_lock:
                    current = self._repository.get(task.run_id)
                    if current is not None:
                        self._interrupt_one(current, reason, reason)

            if self._lock is not None:
                self._lock.release()
                self._lock = None
            self._started = False

    # ------------------------------------------------------------------ 公共接口
    def submit(self, plan_id: str, plan_version: str, variant_id: str) -> str:
        # 接单判断与 Pending 写入必须原子，避免 stop_normal 清队列后残留。
        with self._lifecycle_lock:
            if not self._started or not self._accepting:
                raise ServiceNotRunningError("任务服务未运行或已停止接单")
            plan, variant = self._resolve(plan_id, plan_version, variant_id)
            if self._resource_checker is not None:
                self._resource_checker.require_available(plan, variant)
            now = self._now()
            run_id = generate_run_id(now=now)
            record = build_task_record(
                run_id,
                plan.plan_id,
                plan.plan_version,
                variant.variant_id,
                self._config_snapshot(plan, variant),
                submitted_at=now,
            )
            self._repository.create_pending(record)
        self._wake_event.set()
        return run_id

    def get(self, run_id: str) -> TaskRecord:
        return self._repository.require(run_id)

    def list_tasks(self) -> list[TaskRecord]:
        return self._repository.list_tasks()

    def cancel(self, run_id: str) -> None:
        """取消任务；与终态发布共用同一互斥边界。"""
        with self._publish_lock:
            task = self._repository.get(run_id)
            if task is None:
                raise TaskNotFoundError(run_id)
            if task.status == TaskStatus.PENDING:
                try:
                    self._repository.delete_pending(run_id)
                except TaskConflictError:
                    # 工作线程已取走任务，按运行中取消处理。
                    self._request_cancel(self._repository.require(run_id))
                return
            self._request_cancel(task)

    def _request_cancel(self, task: TaskRecord) -> None:
        if task.status == TaskStatus.CANCELLING:
            return
        if task.is_terminal:
            raise TaskConflictError(
                f"任务已处于终态 {task.status.value}，无法取消"
            )
        try:
            self._repository.transition(
                task.run_id, TaskStatus.CANCELLING, cancel_requested=True
            )
        except TaskStateError:
            latest = self._repository.require(task.run_id)
            if latest.status == TaskStatus.CANCELLING:
                return
            if latest.is_terminal:
                raise TaskConflictError(
                    f"任务已处于终态 {latest.status.value}，无法取消"
                ) from None
            raise TaskConflictError(
                f"任务状态已变化为 {latest.status.value}，无法取消"
            ) from None
        if (
            self._current_run_id == task.run_id
            and self._current_cancel_event is not None
        ):
            self._current_cancel_event.set()

    # ------------------------------------------------------------------ 启动恢复
    def _recover(self) -> None:
        self._repository.delete_all_pending_with_notice(
            NOTICE_QUEUE_CANCELLED_ON_RESTART
        )
        for task in self._repository.active_tasks():
            with self._publish_lock:
                current = self._repository.get(task.run_id)
                if current is None or current.is_terminal:
                    continue
                if self._reconciler is None:
                    self._interrupt_or_force(
                        current, "服务重启，任务中断", "SERVICE_INTERRUPTED"
                    )
                    continue
                try:
                    recovered = self._reconciler.reconcile(current)
                except ResultConflictError as exc:
                    # 冲突：发布基于任务快照的最小 Interrupted 记录，保留原候选。
                    self._interrupt_or_force(
                        current,
                        "服务重启，任务中断",
                        f"恢复冲突: {exc}",
                    )
                    continue
                except BaseException as exc:
                    # 损坏或身份不符：不覆盖现有文件，但 SQLite 必须脱离活跃。
                    self._force_terminal(
                        current.run_id,
                        TaskStatus.INTERRUPTED,
                        self._now(),
                        f"恢复对齐失败: {exc}",
                    )
                    continue
                if recovered is not None:
                    self._force_terminal(
                        current.run_id,
                        recovered.status,
                        recovered.ended_at,
                        None,
                    )
                    continue
                self._interrupt_or_force(
                    current, "服务重启，任务中断", "SERVICE_INTERRUPTED"
                )

    def _interrupt_or_force(
        self, task: TaskRecord, reason: str, error_summary: str
    ) -> None:
        """发布最小 Interrupted 记录；发布失败时只强制 SQLite 脱离活跃。"""
        try:
            self._interrupt_one(task, reason, error_summary)
        except BaseException as exc:
            self._force_terminal(
                task.run_id,
                TaskStatus.INTERRUPTED,
                self._now(),
                f"{error_summary}: {exc}",
            )

    def _force_terminal(
        self,
        run_id: str,
        status: TaskStatus,
        ended_at: datetime,
        error_summary: str | None,
    ) -> None:
        """恢复专用对齐：活跃任务直接采用记录终态，不受运行时状态机限制。

        仅用于启动恢复；正常运行时仍由 `transition` 校验状态机。
        """
        if status not in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.INTERRUPTED,
        ):
            raise ValueError(f"只能对齐到终态，实际 {status.value}")
        connection = sqlite3.connect(
            self._repository.path, timeout=10.0, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE tasks
                SET status = ?, ended_at = ?, current_stage = NULL,
                    error_summary = ?
                WHERE run_id = ? AND status IN (?, ?, ?)
                """,
                (
                    status.value,
                    ended_at.isoformat(),
                    error_summary,
                    run_id,
                    TaskStatus.PREPARING.value,
                    TaskStatus.RUNNING.value,
                    TaskStatus.CANCELLING.value,
                ),
            )
            if cursor.rowcount != 1:
                connection.execute("ROLLBACK")
                return
            connection.execute("COMMIT")
        except BaseException:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            connection.close()

    def _interrupt_one(
        self, task: TaskRecord, reason: str, error_summary: str
    ) -> None:
        if task.is_terminal:
            return
        try:
            self._publisher.publish_interrupted(task, reason)
        except BaseException as exc:
            self._repository.set_error(task.run_id, f"中断发布失败: {exc}")
            return
        try:
            self._repository.transition(
                task.run_id,
                TaskStatus.INTERRUPTED,
                ended_at=self._now(),
                current_stage=None,
                error_summary=error_summary,
            )
        except TaskStateError:
            return

    # ------------------------------------------------------------------ 工作线程
    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                task = self._repository.next_pending()
            except Exception as exc:  # noqa: BLE001 - 线程必须继续
                self._record_worker_error(exc)
                self._wake_event.wait(self._poll_interval)
                self._wake_event.clear()
                continue
            if task is None:
                self._wake_event.wait(self._poll_interval)
                self._wake_event.clear()
                continue
            try:
                self._execute_task(task)
            except BaseException as exc:  # noqa: BLE001 - 线程必须继续
                self._record_worker_error(exc)

    def _record_worker_error(self, exc: BaseException) -> None:
        self._last_worker_error = exc
        logging.getLogger(__name__).error(
            "任务工作线程未预期异常: %s", exc, exc_info=exc
        )

    def _execute_task(self, task: TaskRecord) -> None:
        cancel_event = threading.Event()
        token = _ExecutionToken(task.run_id)
        self._current_run_id = task.run_id
        self._current_cancel_event = cancel_event
        self._current_token = token
        completed: list[str] = []

        def on_stage(event: StageEvent) -> None:
            # 令牌检查与仓储写入必须在同一互斥边界内：
            # stop_normal() 在同一锁内撤销令牌，因此关闭返回时不存在在途阶段写入。
            with self._publish_lock:
                if not token.valid:
                    return
                if event.event == "started":
                    self._repository.update_stage(
                        task.run_id,
                        current_stage=event.stage,
                        completed_stages=tuple(completed),
                    )
                elif event.event == "completed":
                    if event.stage not in completed:
                        completed.append(event.stage)
                    self._repository.update_stage(
                        task.run_id,
                        current_stage=None,
                        completed_stages=tuple(completed),
                    )

        try:
            try:
                self._repository.transition(
                    task.run_id,
                    TaskStatus.PREPARING,
                    started_at=self._now(),
                    cancel_requested=False,
                )
            except TaskStateError:
                return

            plan, variant = self._resolve(
                task.plan_id, task.plan_version, task.variant_id
            )
            # Preparing -> Running 与取消共用同一互斥边界，避免取消先到后仍提交 Running。
            with self._publish_lock:
                latest = self._repository.require(task.run_id)
                if (
                    latest.status == TaskStatus.CANCELLING
                    or latest.cancel_requested
                    or cancel_event.is_set()
                ):
                    raise RunCancelled("prepare", tuple(completed))
                try:
                    self._repository.transition(
                        task.run_id, TaskStatus.RUNNING
                    )
                except TaskStateError:
                    latest = self._repository.require(task.run_id)
                    if latest.status == TaskStatus.CANCELLING:
                        raise RunCancelled(
                            "prepare", tuple(completed)
                        ) from None
                    raise

            result = self._executor.execute(
                plan, variant, on_stage, _CancellationToken(cancel_event, token)
            )
        except RunCancelled as exc:
            self._finish_cancelled(task.run_id, exc, completed, token)
        except BaseException as exc:
            self._finish_failed(task.run_id, exc, completed, token)
        else:
            self._finish_completed(task.run_id, result, completed, token)
        finally:
            self._current_run_id = None
            self._current_cancel_event = None
            self._current_token = None

    # ------------------------------------------------------------------ 终态提交
    def _finish_completed(
        self,
        run_id: str,
        result: ExecutionResult,
        completed: list[str],
        token: _ExecutionToken,
    ) -> None:
        with self._publish_lock:
            if not token.valid:
                return
            task = self._repository.get(run_id)
            if task is None or task.is_terminal:
                return
            if self._shutdown_event.is_set():
                return
            # 取消优先：取消先到、执行结果后到时不得发布 Completed。
            if task.status == TaskStatus.CANCELLING or task.cancel_requested:
                if task.status != TaskStatus.CANCELLING:
                    task = self._repository.transition(
                        run_id, TaskStatus.CANCELLING, cancel_requested=True
                    )
                cancelled = RunCancelled(
                    task.current_stage or "retrieve", tuple(completed)
                )
                self._publish_and_finalize(
                    task,
                    TaskStatus.CANCELLED,
                    lambda current: self._publisher.publish_cancelled(
                        current, cancelled
                    ),
                    error_summary=None,
                )
                return
            if task.status not in (
                TaskStatus.PREPARING,
                TaskStatus.RUNNING,
            ):
                self._repository.set_error(
                    run_id, f"终态发布被拒绝: 当前状态 {task.status.value}"
                )
                return
            self._publish_and_finalize(
                task,
                TaskStatus.COMPLETED,
                lambda current: self._publisher.publish_completed(
                    current, result
                ),
                error_summary=None,
            )

    def _finish_failed(
        self,
        run_id: str,
        error: BaseException,
        completed: list[str],
        token: _ExecutionToken,
    ) -> None:
        del completed  # 阶段进度已由 update_stage 持久化
        with self._publish_lock:
            if not token.valid:
                return
            task = self._repository.get(run_id)
            if task is None or task.is_terminal:
                return
            if self._shutdown_event.is_set():
                return
            if task.status not in (
                TaskStatus.PREPARING,
                TaskStatus.RUNNING,
                TaskStatus.CANCELLING,
            ):
                self._repository.set_error(
                    run_id, f"终态发布被拒绝: 当前状态 {task.status.value}"
                )
                return
            self._publish_and_finalize(
                task,
                TaskStatus.FAILED,
                lambda current: self._publisher.publish_failed(current, error),
                error_summary=str(error),
            )

    def _finish_cancelled(
        self,
        run_id: str,
        cancelled: RunCancelled,
        completed: list[str],
        token: _ExecutionToken,
    ) -> None:
        del completed  # 阶段进度已由 update_stage 持久化
        with self._publish_lock:
            if not token.valid:
                return
            task = self._repository.get(run_id)
            if task is None or task.is_terminal:
                return
            if self._shutdown_event.is_set():
                return
            if task.status in (TaskStatus.PREPARING, TaskStatus.RUNNING):
                task = self._repository.transition(
                    run_id, TaskStatus.CANCELLING, cancel_requested=True
                )
            elif task.status != TaskStatus.CANCELLING:
                return
            self._publish_and_finalize(
                task,
                TaskStatus.CANCELLED,
                lambda current: self._publisher.publish_cancelled(
                    current, cancelled
                ),
                error_summary=None,
            )

    def _publish_and_finalize(
        self,
        task: TaskRecord,
        target: TaskStatus,
        publish: Callable[[TaskRecord], None],
        *,
        error_summary: str | None,
    ) -> None:
        """调用方必须已持有 `_publish_lock` 并重新读取过任务状态。"""
        try:
            ensure_transition(task.status, target)
        except TaskStateError as exc:
            self._repository.set_error(task.run_id, str(exc))
            return
        try:
            publish(task)
        except BaseException as exc:
            self._repository.set_error(
                task.run_id, f"终态发布失败: {exc}"
            )
            return
        try:
            self._repository.transition(
                task.run_id,
                target,
                ended_at=self._now(),
                current_stage=None,
                error_summary=error_summary,
            )
        except TaskStateError as exc:
            # 发布成功后 SQLite 提交失败：保持非终态并记录，交由恢复流程处理。
            self._repository.set_error(
                task.run_id, f"终态提交失败: {exc}"
            )

    # ------------------------------------------------------------------ 解析与快照
    def _resolve(
        self, plan_id: str, plan_version: str, variant_id: str
    ) -> tuple[PlanDefinition, RunVariant]:
        for plan in self._catalog.plans:
            if plan.plan_id != plan_id or plan.plan_version != plan_version:
                continue
            for variant in plan.variants:
                if variant.variant_id != variant_id:
                    continue
                if not isinstance(variant, RunVariant):
                    raise TaskSubmitError(
                        f"变体 {variant_id} 是参照项，不能提交运行"
                    )
                return plan, variant
            raise TaskSubmitError(
                f"方案 {plan_id}/{plan_version} 中没有变体 {variant_id}"
            )
        raise TaskSubmitError(f"未知方案 {plan_id}/{plan_version}")

    @staticmethod
    def _config_snapshot(
        plan: PlanDefinition, variant: RunVariant
    ) -> dict[str, Any]:
        return {
            "plan_id": plan.plan_id,
            "plan_version": plan.plan_version,
            "plan_name": plan.name,
            "variant_id": variant.variant_id,
            "variant_name": variant.name,
            "dataset": {
                "name": plan.dataset.name,
                "language": plan.dataset.language,
                "split": plan.dataset.split,
            },
            "evaluation": {
                "top_k": plan.evaluation.top_k,
                "k_values": list(plan.evaluation.k_values),
            },
            "variant": variant.model_dump(mode="json"),
        }
