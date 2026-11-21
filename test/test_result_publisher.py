"""T-005 终态记录与图表发布测试。"""

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

from experiment.executor import ExecutionResult, RunCancelled  # noqa: E402
from experiment.timing import TIMING_FIELDS  # noqa: E402
from records.publisher import RunRecordPublisher  # noqa: E402
from records.repository import ResultConflictError  # noqa: E402
from records.schema import StageProgress, parse_run_record  # noqa: E402
from tasks.models import TaskRecord, TaskStatus  # noqa: E402

RUN_ID = "20261121T100000000000Z-abc123"
STARTED = datetime(2026, 11, 21, 10, 0, 0, tzinfo=timezone.utc)
FINISHED = datetime(2026, 11, 21, 10, 0, 5, tzinfo=timezone.utc)

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


def make_task(
    status: TaskStatus = TaskStatus.RUNNING,
    *,
    current_stage: str | None = "retrieve",
    completed: tuple[str, ...] = ("prepare", "load_dataset", "build_index", "retrieve"),
) -> TaskRecord:
    return TaskRecord(
        run_id=RUN_ID,
        plan_id="scifact-comparison",
        plan_version="v1",
        variant_id="scifact-bm25",
        status=status,
        current_stage=current_stage,
        completed_stages=completed,
        submitted_at=STARTED,
        started_at=STARTED,
        ended_at=None,
        cancel_requested=False,
        config_snapshot=dict(SNAPSHOT),
        error_summary=None,
    )


def make_execution() -> ExecutionResult:
    timing = {field: None for field in TIMING_FIELDS}
    timing["dataset_load_seconds"] = 0.01
    timing["retrieval_total_seconds"] = 0.002
    return ExecutionResult(
        dataset={
            "name": "scifact",
            "split": "test",
            "corpus_size": 3,
            "query_count": 2,
            "qrels_pairs": 2,
        },
        variant_config={"method": "bm25", "top_k": 100, "k_values": [1, 5]},
        metrics={"P@1": 0.5, "MRR": 0.7},
        num_queries=2,
        num_skipped=0,
        timing=timing,
        environment={"python": "3.11.18", "device": None},
        stage_progress=StageProgress(
            completed=(
                "prepare",
                "load_dataset",
                "build_index",
                "retrieve",
                "evaluate",
            ),
            current=None,
            not_applicable=("load_model", "encode_queries"),
        ),
    )


class PublisherTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "runs"
        self.publisher = RunRecordPublisher(
            self.root, now=lambda: FINISHED
        )

    def record_path(self, run_id: str = RUN_ID) -> Path:
        return self.root / run_id / "run.json"

    def load(self, run_id: str = RUN_ID) -> dict:
        return json.loads(self.record_path(run_id).read_text(encoding="utf-8"))


class CompletedRecordTests(PublisherTestCase):
    def test_completed_record_contains_charts_and_validates(self) -> None:
        self.publisher.publish_completed(make_task(), make_execution())

        record = self.load()
        self.assertEqual(record["status"], "Completed")
        self.assertIsNone(record["error"])
        self.assertEqual(
            [chart["path"] for chart in record["charts"]],
            ["charts/metrics.png", "charts/timing.png"],
        )
        for chart in record["charts"]:
            self.assertTrue((self.root / RUN_ID / chart["path"]).is_file())
        parsed = parse_run_record(record)
        self.assertEqual(parsed.num_queries, 2)
        self.assertEqual(parsed.metrics["MRR"], 0.7)

    def test_completed_has_no_source_fields(self) -> None:
        self.publisher.publish_completed(make_task(), make_execution())
        record = self.load()
        for field in ("source", "result_source", "import_time"):
            self.assertNotIn(field, record)

    def test_only_successful_charts_are_recorded(self) -> None:
        def only_metrics(task, execution, charts_dir):
            charts_dir.mkdir(parents=True, exist_ok=True)
            (charts_dir / "metrics.png").write_bytes(b"png")
            return [{"id": "metrics", "title": "指标", "path": "charts/metrics.png"}]

        publisher = RunRecordPublisher(
            self.root, now=lambda: FINISHED, chart_generator=only_metrics
        )
        publisher.publish_completed(make_task(), make_execution())

        record = self.load()
        self.assertEqual(len(record["charts"]), 1)
        self.assertEqual(record["charts"][0]["path"], "charts/metrics.png")
        self.assertFalse((self.root / RUN_ID / "charts" / "timing.png").exists())
        self.assertEqual(record["status"], "Completed")

    def test_all_chart_failures_still_complete(self) -> None:
        publisher = RunRecordPublisher(
            self.root,
            now=lambda: FINISHED,
            chart_generator=lambda task, execution, charts_dir: [],
        )
        publisher.publish_completed(make_task(), make_execution())
        record = self.load()
        self.assertEqual(record["charts"], [])
        self.assertEqual(record["status"], "Completed")
        parse_run_record(record)

    def test_staging_is_removed_after_publish(self) -> None:
        self.publisher.publish_completed(make_task(), make_execution())
        staging_root = self.root / ".staging"
        leftovers = [
            path for path in staging_root.iterdir() if path.is_dir()
        ] if staging_root.exists() else []
        self.assertEqual(leftovers, [])

    def test_second_publish_does_not_overwrite(self) -> None:
        self.publisher.publish_completed(make_task(), make_execution())
        original = self.record_path().read_text(encoding="utf-8")
        with self.assertRaises(ResultConflictError):
            self.publisher.publish_failed(make_task(), RuntimeError("boom"))
        self.assertEqual(self.record_path().read_text(encoding="utf-8"), original)


class FailureRecordTests(PublisherTestCase):
    def test_failed_record(self) -> None:
        self.publisher.publish_failed(make_task(), RuntimeError("执行失败"))
        record = self.load()
        self.assertEqual(record["status"], "Failed")
        self.assertIsNone(record["metrics"])
        self.assertIsNone(record["num_queries"])
        self.assertEqual(record["charts"], [])
        self.assertIn("执行失败", record["error"]["message"])
        self.assertEqual(record["error"]["stage"], "retrieve")
        self.assertTrue(all(value is None for value in record["timing"].values()))
        self.assertEqual(record["stage_progress"]["completed"][-1], "retrieve")
        parse_run_record(record)

    def test_cancelled_record(self) -> None:
        task = make_task(TaskStatus.CANCELLING)
        self.publisher.publish_cancelled(
            task, RunCancelled("retrieve", ("prepare",))
        )
        record = self.load()
        self.assertEqual(record["status"], "Cancelled")
        self.assertIsNotNone(record["cancelled_at"])
        self.assertEqual(record["error"]["stage"], "retrieve")
        self.assertIsNone(record["metrics"])
        parse_run_record(record)

    def test_interrupted_record_is_minimal(self) -> None:
        task = make_task(
            TaskStatus.RUNNING, current_stage="retrieve", completed=("prepare",)
        )
        self.publisher.publish_interrupted(task, "服务重启，任务中断")
        record = self.load()
        self.assertEqual(record["status"], "Interrupted")
        self.assertEqual(record["charts"], [])
        self.assertEqual(record["environment"], {})
        self.assertIsNone(record["metrics"])
        self.assertIsNone(record["num_queries"])
        self.assertEqual(record["timing"], {field: None for field in TIMING_FIELDS})
        self.assertEqual(record["stage_progress"]["completed"], ["prepare"])
        self.assertIn("服务重启", record["error"]["message"])
        parse_run_record(record)


if __name__ == "__main__":
    unittest.main()
