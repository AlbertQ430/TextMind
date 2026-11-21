"""T-007 结果比较与比较图测试。"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment.plans import load_plan_catalog  # noqa: E402
from experiment.timing import TIMING_FIELDS  # noqa: E402
from records.catalog import ResultCatalog, RunNotFound  # noqa: E402
from records.compare import (  # noqa: E402
    CompareNoMetricsError,
    CompareSelectionError,
    build_compare_chart,
    build_selection,
    compare,
)

CATALOG = load_plan_catalog()
REFERENCE_RUN_ID = "preset-duretrieval-hybrid-bge-small-prefix-equal"


def make_record(
    run_id: str,
    *,
    variant_id: str = "scifact-bm25",
    status: str = "Completed",
    plan_id: str = "scifact-comparison",
    plan_version: str = "v1",
    metrics: dict | None = None,
    config: dict | None = None,
    environment: dict | None = None,
    started_at: str = "2026-11-21T10:00:00Z",
) -> dict:
    record = {
        "format_version": "1.0",
        "run_id": run_id,
        "plan_id": plan_id,
        "plan_version": plan_version,
        "variant_id": variant_id,
        "plan_name": "SciFact 英文检索方法与模型对比",
        "variant_name": variant_id,
        "status": status,
        "stage_progress": {
            "completed": ["prepare"],
            "current": None,
            "not_applicable": [],
        },
        "started_at": started_at,
        "finished_at": "2026-11-21T10:00:05Z",
        "dataset": {
            "name": "scifact",
            "split": "test",
            "corpus_size": None,
            "query_count": None,
            "qrels_pairs": None,
        },
        "variant_config": config or {"method": "bm25", "k1": 1.2},
        "metrics": {"MRR": 0.5} if metrics is None else metrics,
        "num_queries": 1,
        "num_skipped": 0,
        "timing": {field: None for field in TIMING_FIELDS},
        "environment": environment or {},
        "charts": [],
        "error": None,
    }
    if status != "Completed":
        record["metrics"] = None
        record["num_queries"] = None
        record["num_skipped"] = None
        record["error"] = {"message": "失败", "detail": None, "stage": "retrieve"}
    return record


class CompareTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"
        self.catalog = ResultCatalog(
            runs_root=self.runs, catalog=CATALOG
        )

    def write(self, record: dict) -> None:
        directory = self.runs / record["run_id"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "run.json").write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )

    def write_pair(self) -> None:
        self.write(
            make_record(
                "run-a",
                metrics={"P@1": 0.5, "MRR": 0.7},
                environment={"python": "3.11.18", "device": "cpu"},
            )
        )
        self.write(
            make_record(
                "run-b",
                variant_id="scifact-dense-qwen3-base",
                metrics={"P@5": 0.3, "MRR": 0.6},
                config={"method": "dense", "k1": 1.2},
                environment={"python": "3.11.18", "device": "mps"},
                started_at="2026-11-21T11:00:00Z",
            )
        )


class SelectionTests(CompareTestCase):
    def test_too_few_records(self) -> None:
        self.write_pair()
        with self.assertRaises(CompareSelectionError):
            build_selection(self.catalog, ["run-a"])
        with self.assertRaises(CompareSelectionError):
            build_selection(self.catalog, [])

    def test_duplicates_after_dedupe(self) -> None:
        self.write_pair()
        with self.assertRaises(CompareSelectionError):
            build_selection(self.catalog, ["run-a", "run-a"])

    def test_unknown_record_keeps_original_error(self) -> None:
        self.write_pair()
        with self.assertRaises(RunNotFound):
            build_selection(self.catalog, ["run-a", "missing"])

    def test_cross_plan_rejected(self) -> None:
        self.write_pair()
        self.write(
            make_record(
                "run-duretrieval",
                plan_id="duretrieval-comparison",
                variant_id="duretrieval-bm25",
            )
        )
        with self.assertRaises(CompareSelectionError):
            build_selection(self.catalog, ["run-a", "run-duretrieval"])

    def test_incomplete_record_rejected(self) -> None:
        self.write_pair()
        self.write(make_record("run-failed", status="Failed"))
        with self.assertRaises(CompareSelectionError):
            build_selection(self.catalog, ["run-a", "run-failed"])

    def test_reference_cannot_be_selected(self) -> None:
        self.write_pair()
        self.write(
            make_record(
                REFERENCE_RUN_ID,
                plan_id="duretrieval-comparison",
                variant_id="duretrieval-hybrid-bge-small-prefix-equal",
            )
        )
        with self.assertRaises(CompareSelectionError):
            build_selection(self.catalog, ["run-a", REFERENCE_RUN_ID])

    def test_unknown_variant_rejected(self) -> None:
        self.write_pair()
        self.write(make_record("run-unknown", variant_id="unknown-variant"))
        with self.assertRaises(CompareSelectionError):
            build_selection(self.catalog, ["run-a", "run-unknown"])


class CompareDataTests(CompareTestCase):
    def test_metric_alignment_and_differences(self) -> None:
        self.write_pair()
        selection = build_selection(self.catalog, ["run-a", "run-b"])
        data = compare(selection)

        self.assertEqual(data["plan_id"], "scifact-comparison")
        self.assertEqual(data["metric_columns"], ["MRR", "P@1", "P@5"])
        self.assertEqual([row["run_id"] for row in data["rows"]], ["run-a", "run-b"])
        self.assertIsNone(data["rows"][0]["metrics"]["P@5"])
        self.assertEqual(data["rows"][0]["incomparable"], ["P@5"])
        self.assertIsNone(data["rows"][1]["metrics"]["P@1"])
        self.assertEqual(data["config_differences"], {
            "method": {"run-a": "bm25", "run-b": "dense"}
        })
        self.assertEqual(
            data["environment_differences"],
            {"device": {"run-a": "cpu", "run-b": "mps"}},
        )
        self.assertIsNone(data["baseline"])
        self.assertIn("runs=run-a,run-b", data["chart_url"])

    def test_baseline_included(self) -> None:
        self.write_pair()
        self.write(
            make_record(
                REFERENCE_RUN_ID,
                plan_id="duretrieval-comparison",
                variant_id="duretrieval-hybrid-bge-small-prefix-equal",
                metrics={"MRR": 0.8, "P@1": 0.6},
            )
        )
        self.write(
            make_record(
                "run-w",
                plan_id="duretrieval-rrf-weight-sweep",
                variant_id="duretrieval-hybrid-bge-small-prefix-w0.5",
                metrics={"MRR": 0.7},
                started_at="2026-11-21T12:00:00Z",
            )
        )
        self.write(
            make_record(
                "run-w2",
                plan_id="duretrieval-rrf-weight-sweep",
                variant_id="duretrieval-hybrid-bge-small-prefix-w0.3",
                metrics={"MRR": 0.65},
                started_at="2026-11-21T13:00:00Z",
            )
        )
        selection = build_selection(self.catalog, ["run-w", "run-w2"])
        self.assertIsNotNone(selection.baseline)
        data = compare(selection)
        self.assertEqual(data["baseline"]["run_id"], REFERENCE_RUN_ID)
        self.assertEqual(data["baseline"]["metrics"]["MRR"], 0.8)
        self.assertIsNone(data["baseline"]["metrics"].get("P@5"))

    def test_partial_overlap_uses_common_metrics(self) -> None:
        self.write_pair()
        selection = build_selection(self.catalog, ["run-a", "run-b"])
        data = compare(selection)
        self.assertEqual(data["common_metric_columns"], ["MRR"])
        self.assertTrue(data["chart_available"])
        png = build_compare_chart(selection, data)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")

    def test_no_common_metrics_rejects_chart(self) -> None:
        self.write(make_record("run-c", metrics={"P@1": 0.5}))
        self.write(
            make_record(
                "run-d",
                variant_id="scifact-dense-qwen3-base",
                metrics={"MRR": 0.7},
            )
        )
        selection = build_selection(self.catalog, ["run-c", "run-d"])
        data = compare(selection)
        self.assertEqual(data["common_metric_columns"], [])
        self.assertFalse(data["chart_available"])
        self.assertTrue(data["chart_message"])
        with self.assertRaises(CompareNoMetricsError):
            build_compare_chart(selection, data)

    def test_failed_reference_is_not_baseline(self) -> None:
        self.write(
            make_record(
                "run-w",
                plan_id="duretrieval-rrf-weight-sweep",
                variant_id="duretrieval-hybrid-bge-small-prefix-w0.5",
                metrics={"MRR": 0.7},
            )
        )
        self.write(
            make_record(
                "run-w2",
                plan_id="duretrieval-rrf-weight-sweep",
                variant_id="duretrieval-hybrid-bge-small-prefix-w0.3",
                metrics={"MRR": 0.65},
                started_at="2026-11-21T11:00:00Z",
            )
        )
        self.write(
            make_record(
                REFERENCE_RUN_ID,
                plan_id="duretrieval-comparison",
                variant_id="duretrieval-hybrid-bge-small-prefix-equal",
                status="Failed",
            )
        )
        selection = build_selection(self.catalog, ["run-w", "run-w2"])
        self.assertIsNone(selection.baseline)
        data = compare(selection)
        self.assertIsNone(data["baseline"])

    def test_chart_is_valid_png(self) -> None:
        self.write_pair()
        selection = build_selection(self.catalog, ["run-a", "run-b"])
        data = compare(selection)
        png = build_compare_chart(selection, data)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        image = Image.open(io.BytesIO(png))
        self.assertGreater(image.width, 0)
        self.assertGreater(image.height, 0)

    def test_chart_without_metrics_raises(self) -> None:
        self.write_pair()
        selection = build_selection(self.catalog, ["run-a", "run-b"])
        data = {
            "plan_id": "scifact-comparison",
            "plan_version": "v1",
            "metric_columns": ["MRR"],
            "rows": [
                {"variant_name": "A", "metrics": {"MRR": None}},
                {"variant_name": "B", "metrics": {"MRR": None}},
            ],
            "baseline": None,
        }
        with self.assertRaises(CompareNoMetricsError):
            build_compare_chart(selection, data)

    def test_compare_does_not_write_files(self) -> None:
        self.write_pair()
        before = sorted(
            str(path.relative_to(self.runs))
            for path in self.runs.rglob("*")
            if path.is_file()
        )
        selection = build_selection(self.catalog, ["run-a", "run-b"])
        data = compare(selection)
        build_compare_chart(selection, data)
        after = sorted(
            str(path.relative_to(self.runs))
            for path in self.runs.rglob("*")
            if path.is_file()
        )
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
