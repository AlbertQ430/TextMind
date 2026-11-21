"""T-004 SQLite 任务仓储测试。"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tasks.models import (  # noqa: E402
    NOTICE_QUEUE_CANCELLED_ON_RESTART,
    NOTICE_QUEUE_CANCELLED_ON_SHUTDOWN,
    TaskConflictError,
    TaskNotFoundError,
    TaskStateError,
    TaskStatus,
    build_task_record,
    generate_run_id,
)
from tasks.repository import TaskRepository  # noqa: E402

MOMENT = datetime(2026, 11, 21, 10, 0, 0, tzinfo=timezone.utc)


def make_repository(tmp: str) -> TaskRepository:
    repository = TaskRepository(Path(tmp) / "textmind.db")
    repository.initialize()
    return repository


def make_record(
    run_id: str = "run-1", submitted_at: datetime | None = None
):
    return build_task_record(
        run_id,
        "scifact-comparison",
        "v1",
        "scifact-bm25",
        {"variant": {"method": "bm25"}, "evaluation": {"top_k": 100}},
        submitted_at=submitted_at or MOMENT,
    )


class RepositoryInitializationTests(unittest.TestCase):
    def test_initialize_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repository = make_repository(tmp)
            repository.create_pending(make_record())
            repository.initialize()
            repository.initialize()
            self.assertEqual(len(repository.list_tasks()), 1)
            self.assertTrue(Path(tmp, "textmind.db").exists())

    def test_wal_and_foreign_keys_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repository = make_repository(tmp)
            connection = sqlite3.connect(Path(tmp, "textmind.db"))
            try:
                self.assertEqual(
                    connection.execute("PRAGMA journal_mode").fetchone()[0].lower(),
                    "wal",
                )
            finally:
                connection.close()
            connection = sqlite3.connect(Path(tmp, "textmind.db"))
            try:
                connection.execute("PRAGMA foreign_keys = ON")
                self.assertEqual(
                    connection.execute("PRAGMA foreign_keys").fetchone()[0], 1
                )
            finally:
                connection.close()

    def test_missing_parent_directory_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "app" / "textmind.db"
            repository = TaskRepository(path)
            repository.initialize()
            self.assertTrue(path.exists())


class RepositoryTaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repository = make_repository(self._tmp.name)

    def test_create_and_get_roundtrip(self) -> None:
        record = make_record()
        self.repository.create_pending(record)
        loaded = self.repository.get(record.run_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.run_id, record.run_id)
        self.assertEqual(loaded.status, TaskStatus.PENDING)
        self.assertEqual(loaded.submitted_at, MOMENT)
        self.assertEqual(loaded.submitted_at.tzinfo, timezone.utc)
        self.assertEqual(loaded.completed_stages, ())
        self.assertIsNone(loaded.current_stage)
        self.assertEqual(
            loaded.config_snapshot["variant"]["method"], "bm25"
        )
        self.assertFalse(loaded.cancel_requested)

    def test_get_missing_returns_none_and_require_raises(self) -> None:
        self.assertIsNone(self.repository.get("missing"))
        with self.assertRaises(TaskNotFoundError):
            self.repository.require("missing")

    def test_fifo_order_by_submitted_at_then_rowid(self) -> None:
        same = MOMENT
        self.repository.create_pending(make_record("run-a", same))
        self.repository.create_pending(make_record("run-b", same))
        self.repository.create_pending(
            make_record("run-c", datetime(2026, 11, 21, 9, 0, tzinfo=timezone.utc))
        )
        self.assertEqual(
            [task.run_id for task in self.repository.list_tasks()],
            ["run-c", "run-a", "run-b"],
        )
        self.assertEqual(self.repository.next_pending().run_id, "run-c")
        self.repository.delete_pending("run-c")
        self.assertEqual(self.repository.next_pending().run_id, "run-a")

    def test_duplicate_run_id_rejected(self) -> None:
        self.repository.create_pending(make_record())
        with self.assertRaises(TaskConflictError):
            self.repository.create_pending(make_record())

    def test_legal_transition_chain(self) -> None:
        self.repository.create_pending(make_record())
        preparing = self.repository.transition(
            "run-1", TaskStatus.PREPARING, started_at=MOMENT
        )
        self.assertEqual(preparing.status, TaskStatus.PREPARING)
        self.assertEqual(preparing.started_at, MOMENT)

        running = self.repository.transition("run-1", TaskStatus.RUNNING)
        self.assertEqual(running.status, TaskStatus.RUNNING)

        completed = self.repository.transition(
            "run-1", TaskStatus.COMPLETED, ended_at=MOMENT
        )
        self.assertEqual(completed.status, TaskStatus.COMPLETED)
        self.assertTrue(completed.is_terminal)

    def test_illegal_transition_rejected(self) -> None:
        self.repository.create_pending(make_record())
        with self.assertRaises(TaskStateError):
            self.repository.transition("run-1", TaskStatus.COMPLETED)
        self.repository.transition("run-1", TaskStatus.PREPARING)
        with self.assertRaises(TaskStateError):
            self.repository.transition("run-1", TaskStatus.CANCELLED)
        self.assertEqual(
            self.repository.get("run-1").status, TaskStatus.PREPARING
        )

    def test_update_stage(self) -> None:
        self.repository.create_pending(make_record())
        self.repository.transition("run-1", TaskStatus.PREPARING)
        updated = self.repository.update_stage(
            "run-1", current_stage="prepare", completed_stages=()
        )
        self.assertEqual(updated.current_stage, "prepare")
        updated = self.repository.update_stage(
            "run-1",
            current_stage="load_dataset",
            completed_stages=("prepare",),
        )
        self.assertEqual(updated.current_stage, "load_dataset")
        self.assertEqual(updated.completed_stages, ("prepare",))

    def test_update_stage_ignored_for_terminal(self) -> None:
        self.repository.create_pending(make_record())
        self.repository.transition("run-1", TaskStatus.PREPARING)
        self.repository.transition("run-1", TaskStatus.RUNNING)
        self.repository.transition("run-1", TaskStatus.COMPLETED)
        result = self.repository.update_stage(
            "run-1", current_stage="retrieve", completed_stages=("prepare",)
        )
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertIsNone(result.current_stage)

    def test_set_error(self) -> None:
        self.repository.create_pending(make_record())
        self.repository.set_error("run-1", "磁盘不可写")
        self.assertEqual(self.repository.get("run-1").error_summary, "磁盘不可写")

    def test_cancelling_to_failed_is_allowed(self) -> None:
        self.repository.create_pending(make_record())
        self.repository.transition("run-1", TaskStatus.PREPARING)
        self.repository.transition("run-1", TaskStatus.RUNNING)
        self.repository.transition(
            "run-1", TaskStatus.CANCELLING, cancel_requested=True
        )
        failed = self.repository.transition(
            "run-1", TaskStatus.FAILED, error_summary="boom"
        )
        self.assertEqual(failed.status, TaskStatus.FAILED)
        self.assertTrue(failed.cancel_requested)
        self.assertEqual(failed.error_summary, "boom")

    def test_concurrent_transition_only_one_succeeds(self) -> None:
        self.repository.create_pending(make_record())
        self.repository.transition("run-1", TaskStatus.PREPARING)
        self.repository.transition("run-1", TaskStatus.RUNNING)

        successes: list[TaskStatus] = []
        errors: list[TaskStateError] = []

        def attempt() -> None:
            try:
                successes.append(
                    self.repository.transition(
                        "run-1",
                        TaskStatus.CANCELLING,
                        cancel_requested=True,
                    ).status
                )
            except TaskStateError as exc:
                errors.append(exc)

        threads = [threading.Thread(target=attempt) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)

        self.assertEqual(successes, [TaskStatus.CANCELLING])
        self.assertEqual(len(errors), 1)
        self.assertEqual(
            self.repository.get("run-1").status, TaskStatus.CANCELLING
        )

    def test_active_tasks(self) -> None:
        self.repository.create_pending(make_record("run-pending"))
        self.repository.create_pending(make_record("run-preparing"))
        self.repository.transition("run-preparing", TaskStatus.PREPARING)
        self.repository.create_pending(make_record("run-running"))
        self.repository.transition("run-running", TaskStatus.PREPARING)
        self.repository.transition("run-running", TaskStatus.RUNNING)
        self.repository.create_pending(make_record("run-cancelling"))
        self.repository.transition("run-cancelling", TaskStatus.PREPARING)
        self.repository.transition("run-cancelling", TaskStatus.CANCELLING)
        self.repository.create_pending(make_record("run-done"))
        self.repository.transition("run-done", TaskStatus.PREPARING)
        self.repository.transition("run-done", TaskStatus.RUNNING)
        self.repository.transition("run-done", TaskStatus.COMPLETED)

        active = self.repository.active_tasks()
        self.assertEqual(
            {task.run_id for task in active},
            {"run-preparing", "run-running", "run-cancelling"},
        )


class RepositoryCancellationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repository = make_repository(self._tmp.name)

    def test_delete_pending(self) -> None:
        self.repository.create_pending(make_record())
        self.repository.delete_pending("run-1")
        self.assertIsNone(self.repository.get("run-1"))

    def test_delete_non_pending_rejected(self) -> None:
        self.repository.create_pending(make_record())
        self.repository.transition("run-1", TaskStatus.PREPARING)
        with self.assertRaises(TaskConflictError):
            self.repository.delete_pending("run-1")

    def test_delete_all_pending_with_notice(self) -> None:
        self.repository.create_pending(make_record("run-1"))
        self.repository.create_pending(make_record("run-2"))
        self.repository.create_pending(make_record("run-3"))
        self.repository.transition("run-3", TaskStatus.PREPARING)

        count = self.repository.delete_all_pending_with_notice(
            NOTICE_QUEUE_CANCELLED_ON_RESTART
        )
        self.assertEqual(count, 2)
        self.assertEqual(self.repository.count_pending(), 0)

        notices = self.repository.list_notices()
        self.assertEqual(len(notices), 1)
        self.assertEqual(notices[0].kind, NOTICE_QUEUE_CANCELLED_ON_RESTART)
        self.assertEqual(notices[0].count, 2)

    def test_no_notice_when_queue_empty(self) -> None:
        count = self.repository.delete_all_pending_with_notice(
            NOTICE_QUEUE_CANCELLED_ON_SHUTDOWN
        )
        self.assertEqual(count, 0)
        self.assertEqual(self.repository.list_notices(), [])

    def test_invalid_notice_kind(self) -> None:
        with self.assertRaises(ValueError):
            self.repository.delete_all_pending_with_notice("unknown")
        with self.assertRaises(ValueError):
            self.repository.add_notice("unknown", 1)
        with self.assertRaises(ValueError):
            self.repository.add_notice(NOTICE_QUEUE_CANCELLED_ON_SHUTDOWN, 0)


class RepositoryNoticeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repository = make_repository(self._tmp.name)

    def test_add_list_and_ack_notice(self) -> None:
        first = self.repository.add_notice(
            NOTICE_QUEUE_CANCELLED_ON_SHUTDOWN, 2
        )
        second = self.repository.add_notice(
            NOTICE_QUEUE_CANCELLED_ON_RESTART, 3
        )
        notices = self.repository.list_notices()
        self.assertEqual([notice.id for notice in notices], [first, second])
        self.assertEqual(notices[0].count, 2)

        self.assertTrue(self.repository.ack_notice(first))
        self.assertFalse(self.repository.ack_notice(first))
        remaining = self.repository.list_notices()
        self.assertEqual([notice.id for notice in remaining], [second])


class RunIdTests(unittest.TestCase):
    def test_run_id_format(self) -> None:
        run_id = generate_run_id(now=MOMENT)
        self.assertEqual(run_id, "20261121T100000000000Z-" + run_id.split("-")[1])
        self.assertRegex(
            run_id, r"^\d{8}T\d{12}Z-[a-z0-9]{6}$"
        )

    def test_run_id_unique(self) -> None:
        ids = {generate_run_id(now=MOMENT) for _ in range(50)}
        self.assertEqual(len(ids), 50)


if __name__ == "__main__":
    unittest.main()
