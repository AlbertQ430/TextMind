"""T-005 重启恢复与终态对齐测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment.plans import load_plan_catalog  # noqa: E402
from experiment.timing import TIMING_FIELDS  # noqa: E402
from records.publisher import RunRecordPublisher  # noqa: E402
from records.repository import ResultConflictError, ResultRecordError  # noqa: E402
from records.schema import parse_run_record  # noqa: E402
from tasks.locking import InstanceLock  # noqa: E402
from tasks.models import TaskStatus, build_task_record  # noqa: E402
from tasks.repository import TaskRepository  # noqa: E402
from tasks.service import TaskService  # noqa: E402

RUN_ID = "20261121T100000000000Z-rec001"
NOW = datetime(2026, 11, 21, 10, 0, 5, tzinfo=timezone.utc)

SNAPSHOT = {
    "plan_id": "scifact-comparison",
    "plan_version": "v1",
    "plan_name": "SciFact 英文检索方法与模型对比",
    "variant_id": "scifact-bm25",
    "variant_name": "BM25",
    "dataset": {"name": "scifact", "language": "en", "split": "test"},
    "evaluation": {"top_k": 100, "k_values": [1, 5]},
    "variant": {
        "method": "bm25",
        "k1": 1.2,
        "b": 0.75,
        "tokenizer": "simple",
        "model": None,
    },
}


def make_record(
    run_id: str = RUN_ID,
    status: str = "Completed",
    finished_at: str = "2026-11-21T10:00:05Z",
    error: dict | None = None,
) -> dict:
    record = {
        "format_version": "1.0",
        "run_id": run_id,
        "plan_id": "scifact-comparison",
        "plan_version": "v1",
        "variant_id": "scifact-bm25",
        "plan_name": "SciFact 英文检索方法与模型对比",
        "variant_name": "BM25",
        "status": status,
        "stage_progress": {
            "completed": ["prepare", "load_dataset"],
            "current": None,
            "not_applicable": ["load_model", "encode_queries"],
        },
        "started_at": "2026-11-21T10:00:00Z",
        "finished_at": finished_at,
        "dataset": {
            "name": "scifact",
            "split": "test",
            "corpus_size": 3,
            "query_count": 2,
            "qrels_pairs": 2,
        },
        "variant_config": {"method": "bm25"},
        "metrics": {"MRR": 0.75},
        "num_queries": 2,
        "num_skipped": 0,
        "timing": {field: None for field in TIMING_FIELDS},
        "environment": {},
        "charts": [],
        "error": error,
    }
    if status != "Completed":
        record["metrics"] = None
        record["num_queries"] = None
        record["num_skipped"] = None
        record["error"] = error or {"message": "失败", "detail": None, "stage": "retrieve"}
    if status == "Cancelled":
        record["cancelled_at"] = finished_at
    return record


class NoopExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("恢复流程不应执行任务")


class ReconcilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"
        self.publisher = RunRecordPublisher(self.runs, now=lambda: NOW)

    def _write_final(self, record: dict) -> Path:
        path = self.runs / record["run_id"] / "run.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        return path

    def _write_staging(self, record: dict, attempt: str = "attempt1") -> Path:
        staging = self.runs / ".staging" / f"{record['run_id']}-{attempt}"
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "run.json").write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )
        return staging

    def _task_record(self):
        return build_task_record(
            RUN_ID,
            "scifact-comparison",
            "v1",
            "scifact-bm25",
            dict(SNAPSHOT),
            submitted_at=datetime(2026, 11, 21, 10, 0, 0, tzinfo=timezone.utc),
            status=TaskStatus.RUNNING,
        )

    def test_final_record_is_returned(self) -> None:
        self._write_final(make_record())
        result = self.publisher.reconcile(self._task_record())
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertEqual(result.ended_at, NOW)

    def test_missing_records_returns_none(self) -> None:
        self.assertIsNone(self.publisher.reconcile(self._task_record()))

    def test_unique_staging_is_published(self) -> None:
        staging = self._write_staging(make_record())
        result = self.publisher.reconcile(self._task_record())
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertFalse(staging.exists())
        self.assertTrue((self.runs / RUN_ID / "run.json").is_file())

    def test_two_staging_candidates_conflict(self) -> None:
        self._write_staging(make_record(), "attempt1")
        self._write_staging(make_record(), "attempt2")
        with self.assertRaises(ResultConflictError):
            self.publisher.reconcile(self._task_record())
        self.assertFalse((self.runs / RUN_ID).exists())

    def test_corrupt_final_record_raises(self) -> None:
        path = self.runs / RUN_ID / "run.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{bad json", encoding="utf-8")
        with self.assertRaises(ResultRecordError):
            self.publisher.reconcile(self._task_record())
        self.assertFalse((self.runs / RUN_ID / "charts").exists())

    def test_corrupt_staging_is_ignored(self) -> None:
        staging = self.runs / ".staging" / f"{RUN_ID}-attempt1"
        staging.mkdir(parents=True)
        (staging / "run.json").write_text("{bad", encoding="utf-8")
        self.assertIsNone(self.publisher.reconcile(self._task_record()))


class ServiceRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "tasks.db"
        self.runs = Path(self._tmp.name) / "runs"
        self.lock_path = Path(self._tmp.name) / "instance.lock"
        self.catalog = load_plan_catalog()
        self.repository = TaskRepository(self.db_path)
        self.repository.initialize()
        self.executor = NoopExecutor()
        self.publisher = RunRecordPublisher(self.runs, now=lambda: NOW)
        self.service = self._make_service()
        self.addCleanup(self._stop)

    def _make_service(self) -> TaskService:
        return TaskService(
            self.catalog,
            self.repository,
            self.executor,
            self.publisher,
            reconciler=self.publisher,
            lock=InstanceLock(self.lock_path),
            poll_interval=0.01,
            join_timeout=0.1,
        )

    def _stop(self) -> None:
        try:
            self.service.stop_normal()
        except Exception:
            pass

    def _legacy_running(self, run_id: str = RUN_ID) -> None:
        self.repository.create_pending(
            build_task_record(
                run_id,
                "scifact-comparison",
                "v1",
                "scifact-bm25",
                dict(SNAPSHOT),
                submitted_at=datetime(2026, 11, 21, 10, 0, 0, tzinfo=timezone.utc),
            )
        )
        self.repository.transition(run_id, TaskStatus.PREPARING)
        self.repository.transition(run_id, TaskStatus.RUNNING)

    def _legacy_cancelling(self, run_id: str = RUN_ID) -> None:
        self._legacy_running(run_id)
        self.repository.transition(
            run_id, TaskStatus.CANCELLING, cancel_requested=True
        )

    def _write_final(self, record: dict) -> Path:
        path = self.runs / record["run_id"] / "run.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        return path

    def _write_staging(self, record: dict, attempt: str = "attempt1") -> Path:
        staging = self.runs / ".staging" / f"{record['run_id']}-{attempt}"
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "run.json").write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )
        return staging

    def test_aligns_completed_from_final_record(self) -> None:
        self._legacy_running()
        self._write_final(make_record())
        self.service.start()

        task = self.repository.get(RUN_ID)
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertEqual(self.executor.calls, 0)

    def test_generates_minimal_interrupted_without_records(self) -> None:
        self._legacy_running()
        self.service.start()

        task = self.repository.get(RUN_ID)
        self.assertEqual(task.status, TaskStatus.INTERRUPTED)
        record = json.loads(
            (self.runs / RUN_ID / "run.json").read_text(encoding="utf-8")
        )
        self.assertEqual(record["status"], "Interrupted")
        self.assertEqual(record["charts"], [])
        parse_run_record(record)

    def test_publishes_unique_staging_then_aligns(self) -> None:
        self._legacy_running()
        staging = self._write_staging(make_record())
        self.service.start()

        self.assertEqual(self.repository.get(RUN_ID).status, TaskStatus.COMPLETED)
        self.assertFalse(staging.exists())
        self.assertTrue((self.runs / RUN_ID / "run.json").is_file())

    def test_conflict_degrades_to_interrupted_and_keeps_candidates(self) -> None:
        self._legacy_running()
        first = self._write_staging(make_record(), "attempt1")
        second = self._write_staging(make_record(), "attempt2")
        self.service.start()

        task = self.repository.get(RUN_ID)
        self.assertEqual(task.status, TaskStatus.INTERRUPTED)
        self.assertIn("恢复冲突", task.error_summary or "")
        self.assertTrue(first.exists())
        self.assertTrue(second.exists())
        record = json.loads(
            (self.runs / RUN_ID / "run.json").read_text(encoding="utf-8")
        )
        self.assertEqual(record["status"], "Interrupted")

    def test_identity_mismatch_in_final_records_is_rejected(self) -> None:
        cases = [
            ("20261121T100000000000Z-id001", "plan_id", "other-plan"),
            ("20261121T100000000000Z-id002", "plan_version", "v2"),
            ("20261121T100000000000Z-id003", "variant_id", "other-variant"),
        ]
        for run_id, field, value in cases:
            self._legacy_running(run_id)
            record = make_record(run_id)
            record[field] = value
            self._write_final(record)

        self.service.start()
        for run_id, field, value in cases:
            with self.subTest(field=field):
                task = self.repository.get(run_id)
                self.assertEqual(task.status, TaskStatus.INTERRUPTED)
                record = json.loads(
                    (self.runs / run_id / "run.json").read_text(encoding="utf-8")
                )
                self.assertEqual(record[field], value)

    def test_identity_mismatch_in_staging_is_ignored(self) -> None:
        cases = [
            ("20261121T100000000000Z-id011", "plan_id", "other-plan"),
            ("20261121T100000000000Z-id012", "plan_version", "v2"),
            ("20261121T100000000000Z-id013", "variant_id", "other-variant"),
        ]
        staging_dirs = []
        for run_id, field, value in cases:
            self._legacy_running(run_id)
            record = make_record(run_id)
            record[field] = value
            staging_dirs.append(self._write_staging(record))

        self.service.start()
        for (run_id, field, value), staging in zip(cases, staging_dirs):
            with self.subTest(field=field):
                self.assertEqual(
                    self.repository.get(run_id).status, TaskStatus.INTERRUPTED
                )
                self.assertTrue(staging.exists())
                record = json.loads(
                    (self.runs / run_id / "run.json").read_text(encoding="utf-8")
                )
                self.assertEqual(record["status"], "Interrupted")

    def test_aligns_all_terminal_states(self) -> None:
        cases = [
            (TaskStatus.RUNNING, "Completed"),
            (TaskStatus.RUNNING, "Failed"),
            (TaskStatus.RUNNING, "Cancelled"),
            (TaskStatus.RUNNING, "Interrupted"),
            (TaskStatus.CANCELLING, "Completed"),
            (TaskStatus.CANCELLING, "Failed"),
            (TaskStatus.CANCELLING, "Cancelled"),
            (TaskStatus.CANCELLING, "Interrupted"),
        ]
        run_ids = []
        for index, (current, target) in enumerate(cases):
            run_id = f"20261121T100000000000Z-st{index:04d}"
            run_ids.append(run_id)
            if current == TaskStatus.RUNNING:
                self._legacy_running(run_id)
            else:
                self._legacy_cancelling(run_id)
            self._write_final(make_record(run_id, status=target))

        self.service.start()
        for run_id, (_, target) in zip(run_ids, cases):
            with self.subTest(target=target):
                self.assertEqual(
                    self.repository.get(run_id).status, TaskStatus(target)
                )

    def test_corrupt_final_degrades_to_interrupted_without_overwrite(self) -> None:
        self._legacy_running()
        path = self.runs / RUN_ID / "run.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{bad json", encoding="utf-8")

        self.service.start()

        task = self.repository.get(RUN_ID)
        self.assertEqual(task.status, TaskStatus.INTERRUPTED)
        self.assertIn("恢复对齐失败", task.error_summary or "")
        self.assertEqual(path.read_text(encoding="utf-8"), "{bad json")
        self.assertEqual(
            sorted(item.name for item in (self.runs / RUN_ID).iterdir()),
            ["run.json"],
        )

    def test_recovery_is_idempotent(self) -> None:
        self._legacy_running()
        self._write_final(make_record())
        self.service.start()
        self.service.stop_normal()

        record_path = self.runs / RUN_ID / "run.json"
        before = record_path.read_text(encoding="utf-8")

        self.service = self._make_service()
        self.service.start()
        after = record_path.read_text(encoding="utf-8")
        self.assertEqual(before, after)
        self.assertEqual(self.repository.get(RUN_ID).status, TaskStatus.COMPLETED)


if __name__ == "__main__":
    unittest.main()
