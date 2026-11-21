"""T-004 任务服务测试（FakeExecutor + FakeTerminalPublisher）。"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment.executor import ExecutionResult  # noqa: E402
from experiment.plans import load_plan_catalog  # noqa: E402
from experiment.stages import StageEvent  # noqa: E402
from experiment.timing import TIMING_FIELDS  # noqa: E402
from records.schema import StageProgress  # noqa: E402
from tasks.locking import InstanceLock, InstanceLockError  # noqa: E402
from tasks.models import (  # noqa: E402
    NOTICE_QUEUE_CANCELLED_ON_RESTART,
    NOTICE_QUEUE_CANCELLED_ON_SHUTDOWN,
    ServiceNotRunningError,
    TaskConflictError,
    TaskStatus,
    TaskSubmitError,
    build_task_record,
)
from tasks.repository import TaskRepository  # noqa: E402
from tasks.service import TaskService  # noqa: E402

ZERO_TIMING = {field: None for field in TIMING_FIELDS}


def wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def make_execution() -> ExecutionResult:
    return ExecutionResult(
        dataset={
            "name": "scifact",
            "split": "test",
            "corpus_size": 3,
            "query_count": 2,
            "qrels_pairs": 2,
        },
        variant_config={"method": "bm25"},
        metrics={"MRR": 1.0},
        num_queries=2,
        num_skipped=0,
        timing=dict(ZERO_TIMING),
        environment={},
        stage_progress=StageProgress(
            completed=("prepare", "load_dataset", "build_index"),
            current=None,
            not_applicable=(),
        ),
    )


class FakeExecutor:
    """可控制完成/失败/阻塞的假执行器。"""

    def __init__(
        self,
        *,
        fail: bool = False,
        auto_release: bool = True,
        honor_cancellation: bool = True,
    ) -> None:
        self.fail = fail
        self.honor_cancellation = honor_cancellation
        self.calls: list[str] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        if auto_release:
            self.release.set()

    def execute(self, plan, variant, report_stage, cancellation):
        self.calls.append(variant.variant_id)
        self._stage("prepare", report_stage, cancellation)
        self._stage("load_dataset", report_stage, cancellation)
        self.entered.set()
        self.release.wait(timeout=5)
        if self.fail:
            raise RuntimeError("执行失败")
        if self.honor_cancellation:
            self._stage("build_index", report_stage, cancellation)
        return make_execution()

    @staticmethod
    def _stage(stage, report_stage, cancellation) -> None:
        report_stage(StageEvent(stage, "started", datetime.now(timezone.utc)))
        cancellation.raise_if_cancelled(stage)
        report_stage(StageEvent(stage, "completed", datetime.now(timezone.utc)))


class FakeTerminalPublisher:
    def __init__(
        self, *, fail: bool = False, block_completed: bool = False
    ) -> None:
        self.fail = fail
        self.block_completed = block_completed
        self.calls: list[tuple[str, str, TaskStatus]] = []
        self.completed_entered = threading.Event()
        self.completed_release = threading.Event()
        if not block_completed:
            self.completed_release.set()
        self._lock = threading.Lock()

    def _record(self, kind: str, task) -> None:
        with self._lock:
            self.calls.append((kind, task.run_id, task.status))
        if self.fail:
            raise RuntimeError("发布失败")

    def publish_completed(self, task, execution) -> None:
        self._record("completed", task)
        if self.block_completed:
            self.completed_entered.set()
            self.completed_release.wait(timeout=5)

    def publish_failed(self, task, error) -> None:
        self._record("failed", task)

    def publish_cancelled(self, task, cancelled) -> None:
        self._record("cancelled", task)

    def publish_interrupted(self, task, reason) -> None:
        self._record("interrupted", task)

    def kinds(self) -> list[str]:
        return [kind for kind, _, _ in self.calls]


class ServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "tasks.db"
        self.lock_path = Path(self._tmp.name) / "instance.lock"
        self.catalog = load_plan_catalog()
        self.repository = TaskRepository(self.db_path)
        self.publisher = FakeTerminalPublisher()
        self.executor = FakeExecutor()
        self.service = self._make_service()
        self.addCleanup(self._stop)

    def _make_service(
        self,
        *,
        executor=None,
        publisher=None,
        repository=None,
        lock_path=None,
    ) -> TaskService:
        return TaskService(
            self.catalog,
            repository or self.repository,
            executor or self.executor,
            publisher or self.publisher,
            lock=InstanceLock(lock_path or self.lock_path),
            poll_interval=0.01,
            join_timeout=0.2,
        )

    def _start(self, service: TaskService | None = None) -> TaskService:
        service = service or self.service
        service.start()
        return service

    def _stop(self) -> None:
        try:
            self.service.stop_normal()
        except Exception:
            pass

    def _submit_bm25(self, service: TaskService | None = None) -> str:
        return (service or self.service).submit(
            "scifact-comparison", "v1", "scifact-bm25"
        )

    def _wait_status(self, run_id: str, status: TaskStatus, timeout: float = 3.0):
        service = self.service
        self.assertTrue(
            wait_until(
                lambda: service.get(run_id).status == status, timeout
            ),
            f"任务未在超时内达到 {status.value}",
        )


class ServiceQueueTests(ServiceTestCase):
    def test_submit_runs_to_completion(self) -> None:
        self._start()
        run_id = self._submit_bm25()
        self._wait_status(run_id, TaskStatus.COMPLETED)

        task = self.service.get(run_id)
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertIsNotNone(task.started_at)
        self.assertIsNotNone(task.ended_at)
        self.assertEqual(
            task.completed_stages,
            ("prepare", "load_dataset", "build_index"),
        )
        self.assertEqual(self.publisher.calls[0][0], "completed")
        self.assertEqual(self.publisher.calls[0][2], TaskStatus.RUNNING)
        self.assertIn("variant", task.config_snapshot)
        self.assertEqual(
            task.config_snapshot["variant"]["variant_id"], "scifact-bm25"
        )

    def test_submit_rejects_invalid_variants(self) -> None:
        self._start()
        with self.assertRaises(TaskSubmitError):
            self.service.submit("scifact-comparison", "v1", "missing")
        with self.assertRaises(TaskSubmitError):
            self.service.submit("missing-plan", "v1", "scifact-bm25")
        with self.assertRaises(TaskSubmitError):
            self.service.submit(
                "duretrieval-rrf-weight-sweep",
                "v1",
                "duretrieval-hybrid-bge-small-prefix-equal",
            )

    def test_submit_before_start_rejected(self) -> None:
        with self.assertRaises(ServiceNotRunningError):
            self._submit_bm25()

    def test_fifo_single_worker(self) -> None:
        executor = FakeExecutor(auto_release=False)
        service = self._make_service(executor=executor)
        self._start(service)
        self.addCleanup(service.stop_normal)

        first = service.submit("scifact-comparison", "v1", "scifact-bm25")
        self.assertTrue(executor.entered.wait(2))
        second = service.submit(
            "scifact-comparison", "v1", "scifact-dense-qwen3-base"
        )
        self.assertEqual(service.get(second).status, TaskStatus.PENDING)
        self.assertEqual(service.get(first).status, TaskStatus.RUNNING)

        executor.release.set()
        self.assertTrue(
            wait_until(
                lambda: service.get(second).status == TaskStatus.COMPLETED
            )
        )
        self.assertEqual(
            executor.calls, ["scifact-bm25", "scifact-dense-qwen3-base"]
        )

    def test_stage_progress_updated(self) -> None:
        executor = FakeExecutor(auto_release=False)
        service = self._make_service(executor=executor)
        self._start(service)
        self.addCleanup(service.stop_normal)

        run_id = self._submit_bm25(service)
        self.assertTrue(executor.entered.wait(2))
        task = service.get(run_id)
        self.assertEqual(task.status, TaskStatus.RUNNING)
        self.assertEqual(
            task.completed_stages, ("prepare", "load_dataset")
        )
        self.assertIsNone(task.current_stage)
        executor.release.set()
        self.assertTrue(
            wait_until(
                lambda: service.get(run_id).status == TaskStatus.COMPLETED
            )
        )

    def test_list_tasks_fifo(self) -> None:
        executor = FakeExecutor(auto_release=False)
        service = self._make_service(executor=executor)
        self._start(service)
        self.addCleanup(service.stop_normal)

        first = self._submit_bm25(service)
        self.assertTrue(executor.entered.wait(2))
        second = service.submit(
            "scifact-comparison", "v1", "scifact-dense-qwen3-base"
        )
        self.assertEqual(
            [task.run_id for task in service.list_tasks()], [first, second]
        )
        executor.release.set()
        self.assertTrue(
            wait_until(
                lambda: service.get(second).status == TaskStatus.COMPLETED
            )
        )


class ServiceCancellationTests(ServiceTestCase):
    def test_cancel_pending_deletes_row_without_publisher(self) -> None:
        executor = FakeExecutor(auto_release=False)
        service = self._make_service(executor=executor)
        self._start(service)
        self.addCleanup(service.stop_normal)

        running = self._submit_bm25(service)
        self.assertTrue(executor.entered.wait(2))
        pending = service.submit(
            "scifact-comparison", "v1", "scifact-dense-qwen3-base"
        )
        service.cancel(pending)
        self.assertIsNone(self.repository.get(pending))
        self.assertNotIn("cancelled", self.publisher.kinds())

        executor.release.set()
        self.assertTrue(
            wait_until(
                lambda: service.get(running).status == TaskStatus.COMPLETED
            )
        )

    def test_cancel_running_at_stage_boundary(self) -> None:
        executor = FakeExecutor(auto_release=False)
        service = self._make_service(executor=executor)
        self._start(service)
        self.addCleanup(service.stop_normal)

        run_id = self._submit_bm25(service)
        self.assertTrue(executor.entered.wait(2))
        service.cancel(run_id)
        self.assertEqual(service.get(run_id).status, TaskStatus.CANCELLING)
        service.cancel(run_id)  # 幂等

        executor.release.set()
        self.assertTrue(
            wait_until(
                lambda: service.get(run_id).status == TaskStatus.CANCELLED
            )
        )
        interrupted = [
            call for call in self.publisher.calls if call[0] == "cancelled"
        ]
        self.assertEqual(len(interrupted), 1)
        self.assertEqual(interrupted[0][2], TaskStatus.CANCELLING)
        self.assertTrue(service.get(run_id).cancel_requested)

    def test_cancel_terminal_conflict(self) -> None:
        self._start()
        run_id = self._submit_bm25()
        self._wait_status(run_id, TaskStatus.COMPLETED)
        with self.assertRaises(TaskConflictError):
            self.service.cancel(run_id)


class ServiceFailureTests(ServiceTestCase):
    def test_failure_publishes_then_marks_failed(self) -> None:
        executor = FakeExecutor(fail=True)
        service = self._make_service(executor=executor)
        self._start(service)
        self.addCleanup(service.stop_normal)

        run_id = self._submit_bm25(service)
        self._wait_status(run_id, TaskStatus.FAILED)
        self.assertEqual(self.publisher.kinds(), ["failed"])
        self.assertIn("执行失败", service.get(run_id).error_summary)

    def test_publisher_failure_keeps_task_non_terminal(self) -> None:
        publisher = FakeTerminalPublisher(fail=True)
        service = self._make_service(publisher=publisher)
        self._start(service)
        self.addCleanup(service.stop_normal)

        run_id = self._submit_bm25(service)
        self.assertTrue(
            wait_until(
                lambda: "发布失败"
                in (service.get(run_id).error_summary or "")
            )
        )
        task = service.get(run_id)
        self.assertEqual(task.status, TaskStatus.RUNNING)
        self.assertFalse(task.is_terminal)
        self.assertEqual(publisher.kinds(), ["completed"])


class ServiceRaceTests(ServiceTestCase):
    def test_cancel_during_completed_publication_keeps_consistent(self) -> None:
        publisher = FakeTerminalPublisher(block_completed=True)
        service = self._make_service(publisher=publisher)
        self._start(service)
        self.addCleanup(service.stop_normal)

        run_id = self._submit_bm25(service)
        self.assertTrue(publisher.completed_entered.wait(3))

        outcome: dict[str, str] = {}

        def do_cancel() -> None:
            try:
                service.cancel(run_id)
                outcome["result"] = "cancelled"
            except TaskConflictError:
                outcome["result"] = "conflict"

        thread = threading.Thread(target=do_cancel)
        thread.start()
        time.sleep(0.1)
        publisher.completed_release.set()
        thread.join(timeout=3)

        self.assertTrue(
            wait_until(
                lambda: service.get(run_id).status == TaskStatus.COMPLETED
            )
        )
        self.assertEqual(outcome.get("result"), "conflict")
        self.assertEqual(publisher.kinds(), ["completed"])

    def test_cancel_then_executor_failure_marks_failed(self) -> None:
        executor = FakeExecutor(auto_release=False, fail=True)
        service = self._make_service(executor=executor)
        self._start(service)
        self.addCleanup(service.stop_normal)

        run_id = self._submit_bm25(service)
        self.assertTrue(executor.entered.wait(2))
        service.cancel(run_id)
        self.assertEqual(service.get(run_id).status, TaskStatus.CANCELLING)

        executor.release.set()
        self.assertTrue(
            wait_until(
                lambda: service.get(run_id).status == TaskStatus.FAILED
            )
        )
        task = service.get(run_id)
        self.assertTrue(task.cancel_requested)
        self.assertIn("执行失败", task.error_summary)
        self.assertEqual(self.publisher.kinds(), ["failed"])

    def test_concurrent_cancel_is_idempotent(self) -> None:
        executor = FakeExecutor(auto_release=False)
        service = self._make_service(executor=executor)
        self._start(service)
        self.addCleanup(service.stop_normal)

        run_id = self._submit_bm25(service)
        self.assertTrue(executor.entered.wait(2))
        errors: list[BaseException] = []

        def do_cancel() -> None:
            try:
                service.cancel(run_id)
            except BaseException as exc:  # noqa: BLE001 - 测试收集
                errors.append(exc)

        threads = [threading.Thread(target=do_cancel) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)

        self.assertEqual(errors, [])
        self.assertEqual(service.get(run_id).status, TaskStatus.CANCELLING)
        executor.release.set()
        self.assertTrue(
            wait_until(
                lambda: service.get(run_id).status == TaskStatus.CANCELLED
            )
        )
        cancelled_calls = [
            call for call in self.publisher.calls if call[0] == "cancelled"
        ]
        self.assertEqual(len(cancelled_calls), 1)

    def test_cancel_before_result_never_publishes_completed(self) -> None:
        executor = FakeExecutor(
            auto_release=False, honor_cancellation=False
        )
        service = self._make_service(executor=executor)
        self._start(service)
        self.addCleanup(service.stop_normal)

        run_id = self._submit_bm25(service)
        self.assertTrue(executor.entered.wait(2))
        service.cancel(run_id)
        executor.release.set()
        self.assertTrue(
            wait_until(
                lambda: service.get(run_id).status == TaskStatus.CANCELLED
            )
        )
        self.assertEqual(self.publisher.kinds(), ["cancelled"])


class PausingAfterPreparingRepository(TaskRepository):
    """在 Pending -> Preparing 提交后暂停工作线程，用于确定性时序测试。"""

    def __init__(self, db_path) -> None:
        super().__init__(db_path)
        self.preparing_done = threading.Event()
        self.preparing_release = threading.Event()

    def transition(self, run_id, target, **kwargs):
        result = super().transition(run_id, target, **kwargs)
        if target == TaskStatus.PREPARING:
            self.preparing_done.set()
            self.preparing_release.wait(timeout=5)
        return result


class BlockingRunningRepository(TaskRepository):
    """在 Preparing -> Running 转换时阻塞。"""

    def __init__(self, db_path) -> None:
        super().__init__(db_path)
        self.running_entered = threading.Event()
        self.running_release = threading.Event()

    def transition(self, run_id, target, **kwargs):
        if target == TaskStatus.RUNNING:
            self.running_entered.set()
            self.running_release.wait(timeout=5)
        return super().transition(run_id, target, **kwargs)


class BlockingCreateRepository(TaskRepository):
    """第 N 次 Pending 写入前阻塞。"""

    def __init__(self, db_path, *, block_after: int = 1) -> None:
        super().__init__(db_path)
        self.block_after = block_after
        self.calls = 0
        self.create_entered = threading.Event()
        self.create_release = threading.Event()

    def create_pending(self, record):
        self.calls += 1
        if self.calls > self.block_after:
            self.create_entered.set()
            self.create_release.wait(timeout=5)
        return super().create_pending(record)


class ServiceBoundaryRaceTests(ServiceTestCase):
    def test_cancel_before_running_does_not_start_executor(self) -> None:
        repository = PausingAfterPreparingRepository(self.db_path)
        executor = FakeExecutor(auto_release=False)
        service = self._make_service(repository=repository, executor=executor)
        self._start(service)
        self.addCleanup(service.stop_normal)

        run_id = self._submit_bm25(service)
        self.assertTrue(repository.preparing_done.wait(2))
        service.cancel(run_id)
        self.assertEqual(service.get(run_id).status, TaskStatus.CANCELLING)

        repository.preparing_release.set()
        self.assertTrue(
            wait_until(
                lambda: service.get(run_id).status == TaskStatus.CANCELLED
            )
        )
        self.assertEqual(executor.calls, [])
        self.assertEqual(self.publisher.kinds(), ["cancelled"])

    def test_cancel_blocked_during_running_transition(self) -> None:
        repository = BlockingRunningRepository(self.db_path)
        executor = FakeExecutor(auto_release=False)
        service = self._make_service(repository=repository, executor=executor)
        self._start(service)
        self.addCleanup(service.stop_normal)

        run_id = self._submit_bm25(service)
        self.assertTrue(repository.running_entered.wait(2))
        errors: list[BaseException] = []

        def do_cancel() -> None:
            try:
                service.cancel(run_id)
            except BaseException as exc:  # noqa: BLE001 - 测试收集
                errors.append(exc)

        thread = threading.Thread(target=do_cancel)
        thread.start()
        repository.running_release.set()
        thread.join(timeout=3)

        self.assertEqual(errors, [])
        # 取消可能让执行器在启动前就收到取消，也可能在首个阶段边界停止；
        # 两种情况都必须最终为 Cancelled。
        executor.release.set()
        self.assertTrue(
            wait_until(
                lambda: service.get(run_id).status == TaskStatus.CANCELLED
            )
        )
        self.assertEqual(self.publisher.kinds(), ["cancelled"])

    def test_stop_during_blocked_submit_leaves_no_pending(self) -> None:
        repository = BlockingCreateRepository(self.db_path, block_after=1)
        executor = FakeExecutor(auto_release=False)
        service = self._make_service(repository=repository, executor=executor)
        self._start(service)

        first = self._submit_bm25(service)
        self.assertTrue(executor.entered.wait(2))

        submit_result: dict[str, object] = {}

        def submit_b() -> None:
            try:
                submit_result["run_id"] = service.submit(
                    "scifact-comparison", "v1", "scifact-dense-qwen3-base"
                )
            except ServiceNotRunningError:
                submit_result["rejected"] = True

        submit_thread = threading.Thread(target=submit_b)
        submit_thread.start()
        self.assertTrue(repository.create_entered.wait(2))

        stop_done = threading.Event()

        def do_stop() -> None:
            service.stop_normal()
            stop_done.set()

        stop_thread = threading.Thread(target=do_stop)
        stop_thread.start()
        time.sleep(0.05)
        repository.create_release.set()
        submit_thread.join(timeout=3)
        self.assertTrue(stop_done.wait(3))
        executor.release.set()

        self.assertEqual(repository.count_pending(), 0)
        notices = repository.list_notices()
        self.assertEqual(len(notices), 1)
        self.assertEqual(notices[0].kind, NOTICE_QUEUE_CANCELLED_ON_SHUTDOWN)
        self.assertEqual(notices[0].count, 1)
        self.assertEqual(service.get(first).status, TaskStatus.INTERRUPTED)
        if "run_id" in submit_result:
            self.assertIsNone(repository.get(str(submit_result["run_id"])))
        else:
            self.assertTrue(submit_result.get("rejected"))


class CountingUpdateRepository(TaskRepository):
    """统计阶段更新次数，用于验证关闭后旧线程不再写入。"""

    def __init__(self, db_path) -> None:
        super().__init__(db_path)
        self.stage_updates = 0

    def update_stage(self, run_id, **kwargs):
        self.stage_updates += 1
        return super().update_stage(run_id, **kwargs)


class BlockingUpdateRepository(TaskRepository):
    """在第一次 update_stage 内部阻塞，模拟在途阶段回调。"""

    def __init__(self, db_path) -> None:
        super().__init__(db_path)
        self.calls = 0
        self.update_entered = threading.Event()
        self.update_release = threading.Event()

    def update_stage(self, run_id, **kwargs):
        self.calls += 1
        if self.calls == 1:
            self.update_entered.set()
            self.update_release.wait(timeout=10)
        return super().update_stage(run_id, **kwargs)


class ServiceWorkerIsolationTests(ServiceTestCase):
    def test_shutdown_ignores_old_worker_stage_updates(self) -> None:
        # 场景：关闭时执行器尚未产生新阶段回调，释放后不得再写入仓储。
        repository = CountingUpdateRepository(self.db_path)
        executor = FakeExecutor(auto_release=False)
        service = self._make_service(repository=repository, executor=executor)
        self._start(service)

        run_id = self._submit_bm25(service)
        self.assertTrue(executor.entered.wait(2))
        service.stop_normal()
        after_stop = repository.stage_updates
        self.assertGreater(after_stop, 0)

        executor.release.set()
        time.sleep(0.3)

        self.assertEqual(repository.stage_updates, after_stop)
        self.assertEqual(service.get(run_id).status, TaskStatus.INTERRUPTED)

    def test_stop_waits_for_inflight_stage_update(self) -> None:
        # 场景：关闭时已有一个 update_stage 正在进行，关闭必须等待其结束。
        repository = BlockingUpdateRepository(self.db_path)
        executor = FakeExecutor(auto_release=False)
        service = self._make_service(repository=repository, executor=executor)
        self._start(service)

        run_id = self._submit_bm25(service)
        self.assertTrue(repository.update_entered.wait(5))

        stop_done = threading.Event()

        def do_stop() -> None:
            service.stop_normal()
            stop_done.set()

        stop_thread = threading.Thread(target=do_stop)
        stop_thread.start()
        time.sleep(0.2)
        self.assertFalse(stop_done.is_set())

        repository.update_release.set()
        self.assertTrue(stop_done.wait(5))
        stop_thread.join(timeout=5)

        calls_after_stop = repository.calls
        executor.release.set()
        time.sleep(0.3)
        self.assertEqual(repository.calls, calls_after_stop)
        self.assertEqual(service.get(run_id).status, TaskStatus.INTERRUPTED)

    def test_worker_unexpected_exception_is_recorded(self) -> None:
        service = self.service
        self._start(service)
        self.addCleanup(service.stop_normal)

        def boom(*args, **kwargs):
            raise RuntimeError("内部故障")

        service._finish_completed = boom  # type: ignore[method-assign]

        with self.assertLogs("tasks.service", level="ERROR") as captured:
            run_id = self._submit_bm25(service)
            self.assertTrue(self.executor.entered.wait(10))
            self.assertTrue(
                wait_until(
                    lambda: isinstance(service.last_worker_error, RuntimeError),
                    timeout=10,
                )
            )
        self.assertTrue(
            any("内部故障" in message for message in captured.output)
        )
        self.assertEqual(str(service.last_worker_error), "内部故障")
        self.assertEqual(service.get(run_id).status, TaskStatus.RUNNING)

        service.stop_normal()
        self.assertFalse(service._started)


class ServiceShutdownRecoveryTests(ServiceTestCase):
    def test_normal_shutdown_clears_pending_and_interrupts_active(self) -> None:
        executor = FakeExecutor(auto_release=False)
        service = self._make_service(executor=executor)
        self._start(service)

        running = self._submit_bm25(service)
        self.assertTrue(executor.entered.wait(2))
        pending = service.submit(
            "scifact-comparison", "v1", "scifact-dense-qwen3-base"
        )

        service.stop_normal()
        self.assertEqual(service.get(running).status, TaskStatus.INTERRUPTED)
        self.assertIsNone(repository_get(self.repository, pending))
        notices = self.repository.list_notices()
        self.assertEqual(len(notices), 1)
        self.assertEqual(notices[0].kind, NOTICE_QUEUE_CANCELLED_ON_SHUTDOWN)
        self.assertEqual(notices[0].count, 1)
        interrupted = [
            call for call in self.publisher.calls if call[0] == "interrupted"
        ]
        self.assertEqual(len(interrupted), 1)

        executor.release.set()
        time.sleep(0.05)

    def test_restart_recovers_legacy_pending_and_active(self) -> None:
        repository = TaskRepository(self.db_path)
        repository.initialize()
        repository.create_pending(
            build_task_record(
                "run-pending",
                "scifact-comparison",
                "v1",
                "scifact-bm25",
                {"variant": {}},
            )
        )
        repository.create_pending(
            build_task_record(
                "run-active",
                "scifact-comparison",
                "v1",
                "scifact-bm25",
                {"variant": {}},
            )
        )
        repository.transition("run-active", TaskStatus.PREPARING)
        repository.transition("run-active", TaskStatus.RUNNING)

        service = self._make_service(repository=repository)
        service.start()
        self.addCleanup(service.stop_normal)

        self.assertIsNone(repository.get("run-pending"))
        self.assertEqual(repository.get("run-active").status, TaskStatus.INTERRUPTED)
        notices = repository.list_notices()
        self.assertEqual(len(notices), 1)
        self.assertEqual(notices[0].kind, NOTICE_QUEUE_CANCELLED_ON_RESTART)
        self.assertEqual(notices[0].count, 1)
        self.assertEqual(self.publisher.kinds(), ["interrupted"])

    def test_second_instance_cannot_start(self) -> None:
        self._start()
        second = self._make_service(lock_path=self.lock_path)
        with self.assertRaises(InstanceLockError):
            second.start()


def repository_get(repository: TaskRepository, run_id: str):
    return repository.get(run_id)


if __name__ == "__main__":
    unittest.main()
