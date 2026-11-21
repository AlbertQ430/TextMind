"""T-007 结果目录扫描与安全读取测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment.plans import load_plan_catalog  # noqa: E402
from experiment.timing import TIMING_FIELDS  # noqa: E402
from records.catalog import (  # noqa: E402
    ChartNotFound,
    ChartUnavailable,
    ChartUnsafe,
    ResultCatalog,
    RunIdConflict,
    RunNotFound,
    RunUnavailable,
)

CATALOG = load_plan_catalog()
REFERENCE_RUN_ID = "preset-duretrieval-hybrid-bge-small-prefix-equal"


def make_record(
    run_id: str,
    *,
    status: str = "Completed",
    plan_id: str = "scifact-comparison",
    plan_version: str = "v1",
    variant_id: str = "scifact-bm25",
    started_at: str = "2026-11-21T10:00:00Z",
    metrics: dict | None = None,
    charts: list | None = None,
) -> dict:
    record = {
        "format_version": "1.0",
        "run_id": run_id,
        "plan_id": plan_id,
        "plan_version": plan_version,
        "variant_id": variant_id,
        "plan_name": "SciFact 英文检索方法与模型对比",
        "variant_name": "BM25",
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
        "variant_config": {"method": "bm25"},
        "metrics": {"MRR": 0.5} if metrics is None else metrics,
        "num_queries": 1,
        "num_skipped": 0,
        "timing": {field: None for field in TIMING_FIELDS},
        "environment": {},
        "charts": charts or [],
        "error": None,
    }
    if status != "Completed":
        record["metrics"] = None
        record["num_queries"] = None
        record["num_skipped"] = None
        record["error"] = {"message": "失败", "detail": None, "stage": "retrieve"}
    if status == "Cancelled":
        record["cancelled_at"] = "2026-11-21T10:00:05Z"
    return record


class CatalogTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.presets = self.root / "presets"
        self.runs = self.root / "runs"
        self.imports = self.root / "imports"
        self.catalog = ResultCatalog(
            presets_root=self.presets,
            runs_root=self.runs,
            imports_root=self.imports,
            catalog=CATALOG,
        )

    def write(
        self,
        source: str,
        run_id: str,
        record: dict | None = None,
        *,
        raw: str | None = None,
    ) -> Path:
        directory = {"presets": self.presets, "runs": self.runs, "imports": self.imports}[
            source
        ] / run_id
        directory.mkdir(parents=True, exist_ok=True)
        if raw is not None:
            (directory / "run.json").write_text(raw, encoding="utf-8")
        elif record is not None:
            (directory / "run.json").write_text(
                json.dumps(record, ensure_ascii=False), encoding="utf-8"
            )
        return directory


class ScanTests(CatalogTestCase):
    def test_empty_directories(self) -> None:
        scan = self.catalog.scan()
        self.assertEqual(scan.records, [])
        self.assertEqual(scan.unavailable, [])
        self.assertEqual(scan.references, [])

    def test_sources_and_sorting(self) -> None:
        self.write("presets", "run-a", make_record("run-a", started_at="2026-11-21T09:00:00Z"))
        self.write("runs", "run-b", make_record("run-b", started_at="2026-11-21T11:00:00Z"))
        self.write("imports", "run-c", make_record("run-c", started_at="2026-11-21T10:00:00Z"))

        scan = self.catalog.scan()
        self.assertEqual(
            [(entry.run_id, entry.source) for entry in scan.records],
            [("run-b", "runs"), ("run-c", "imports"), ("run-a", "presets")],
        )

    def test_same_started_at_sorted_by_run_id(self) -> None:
        self.write("runs", "run-b", make_record("run-b"))
        self.write("runs", "run-a", make_record("run-a"))
        scan = self.catalog.scan()
        self.assertEqual([entry.run_id for entry in scan.records], ["run-a", "run-b"])

    def test_corrupt_and_missing_records(self) -> None:
        self.write("runs", "run-bad", raw="{bad json")
        self.write("runs", "run-missing")
        scan = self.catalog.scan()
        self.assertEqual(scan.records, [])
        codes = {(item.run_id, item.code) for item in scan.unavailable}
        self.assertIn(("run-bad", "RUN_RECORD_INVALID"), codes)
        self.assertIn(("run-missing", "RUN_RECORD_MISSING"), codes)

    def test_invalid_utf8_is_isolated(self) -> None:
        directory = self.runs / "run-utf8"
        directory.mkdir(parents=True)
        (directory / "run.json").write_bytes(b"\xff\xfe\x00")
        self.write("runs", "run-ok", make_record("run-ok"))

        scan = self.catalog.scan()
        self.assertEqual([entry.run_id for entry in scan.records], ["run-ok"])
        self.assertEqual(scan.unavailable[0].run_id, "run-utf8")
        self.assertEqual(scan.unavailable[0].code, "RUN_RECORD_INVALID")

    def test_run_directory_symlink_escape(self) -> None:
        outside = self.root / "outside-run"
        outside.mkdir()
        (outside / "run.json").write_text(
            json.dumps(make_record("run-link")), encoding="utf-8"
        )
        self.runs.mkdir(parents=True, exist_ok=True)
        (self.runs / "run-link").symlink_to(outside)

        scan = self.catalog.scan()
        self.assertEqual(scan.records, [])
        self.assertEqual(scan.unavailable[0].code, "RUN_RECORD_UNSAFE")

    def test_run_json_symlink_escape(self) -> None:
        outside = self.root / "outside.json"
        outside.write_text(
            json.dumps(make_record("run-link2")), encoding="utf-8"
        )
        directory = self.runs / "run-link2"
        directory.mkdir(parents=True)
        (directory / "run.json").symlink_to(outside)

        scan = self.catalog.scan()
        self.assertEqual(scan.records, [])
        self.assertEqual(scan.unavailable[0].code, "RUN_RECORD_UNSAFE")

    def test_id_mismatch(self) -> None:
        self.write("runs", "run-dir", make_record("run-other"))
        scan = self.catalog.scan()
        self.assertEqual(scan.records, [])
        self.assertEqual(scan.unavailable[0].code, "RUN_RECORD_ID_MISMATCH")

    def test_conflict_across_sources(self) -> None:
        self.write("runs", "run-x", make_record("run-x"))
        self.write("imports", "run-x", make_record("run-x"))
        scan = self.catalog.scan()
        self.assertEqual(scan.records, [])
        self.assertEqual(len(scan.unavailable), 2)
        self.assertTrue(
            all(item.code == "RUN_ID_CONFLICT" for item in scan.unavailable)
        )
        with self.assertRaises(RunIdConflict):
            self.catalog.get("run-x")

    def test_hidden_directories_are_skipped(self) -> None:
        self.write("runs", ".staging", make_record(".staging"))
        self.write("runs", "run-ok", make_record("run-ok"))
        scan = self.catalog.scan()
        self.assertEqual([entry.run_id for entry in scan.records], ["run-ok"])
        self.assertEqual(scan.unavailable, [])

    def test_reference_run_identified(self) -> None:
        self.write(
            "presets",
            REFERENCE_RUN_ID,
            make_record(
                REFERENCE_RUN_ID,
                plan_id="duretrieval-comparison",
                variant_id="duretrieval-hybrid-bge-small-prefix-equal",
            ),
        )
        self.write("runs", "run-normal", make_record("run-normal"))
        scan = self.catalog.scan()
        self.assertEqual([entry.run_id for entry in scan.records], ["run-normal"])
        self.assertEqual(
            [entry.run_id for entry in scan.references], [REFERENCE_RUN_ID]
        )
        self.assertTrue(self.catalog.get(REFERENCE_RUN_ID).is_reference)

    def test_get_errors(self) -> None:
        with self.assertRaises(RunNotFound):
            self.catalog.get("missing")
        self.write("runs", "run-bad", raw="{bad")
        with self.assertRaises(RunUnavailable):
            self.catalog.get("run-bad")

    def test_known_plan_and_eligibility(self) -> None:
        record = make_record("run-a")
        from records.schema import parse_run_record

        parsed = parse_run_record(record)
        self.assertIsNotNone(self.catalog.known_plan(parsed))
        self.assertTrue(self.catalog.comparison_eligible(parsed))

        failed = parse_run_record(make_record("run-b", status="Failed"))
        self.assertFalse(self.catalog.comparison_eligible(failed))

        unknown = parse_run_record(
            make_record("run-c", variant_id="unknown-variant")
        )
        self.assertIsNone(self.catalog.known_plan(unknown))
        self.assertFalse(self.catalog.comparison_eligible(unknown))


class ChartTests(CatalogTestCase):
    def _entry_with_chart(self, relative: str = "charts/metrics.png"):
        directory = self.write(
            "runs",
            "run-chart",
            make_record(
                "run-chart",
                charts=[{"id": "metrics", "title": "指标", "path": relative}],
            ),
        )
        target = directory / relative
        if relative.startswith("charts/"):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"png")
        return self.catalog.get("run-chart")

    def test_chart_file_and_availability(self) -> None:
        entry = self._entry_with_chart()
        self.assertTrue(self.catalog.chart_available(entry, 0))
        self.assertTrue(self.catalog.chart_file(entry, 0).is_file())

    def test_chart_index_out_of_range(self) -> None:
        entry = self._entry_with_chart()
        with self.assertRaises(ChartNotFound):
            self.catalog.chart_file(entry, 3)
        with self.assertRaises(ChartNotFound):
            self.catalog.chart_file(entry, -1)

    def test_missing_chart_file(self) -> None:
        directory = self.write(
            "runs",
            "run-missing-chart",
            make_record(
                "run-missing-chart",
                charts=[
                    {"id": "metrics", "title": "指标", "path": "charts/metrics.png"}
                ],
            ),
        )
        entry = self.catalog.get("run-missing-chart")
        self.assertFalse(self.catalog.chart_available(entry, 0))
        with self.assertRaises(ChartUnavailable):
            self.catalog.chart_file(entry, 0)

    def test_nul_chart_path_does_not_crash(self) -> None:
        self.write(
            "runs",
            "run-nul",
            make_record(
                "run-nul",
                charts=[
                    {
                        "id": "metrics",
                        "title": "指标",
                        "path": "charts/bad\u0000.png",
                    }
                ],
            ),
        )
        entry = self.catalog.get("run-nul")
        self.assertFalse(self.catalog.chart_available(entry, 0))
        with self.assertRaises(ChartUnsafe):
            self.catalog.chart_file(entry, 0)

    def test_unreadable_chart_marks_unavailable(self) -> None:
        entry = self._entry_with_chart()
        with mock.patch("records.catalog._is_readable", return_value=False):
            self.assertFalse(self.catalog.chart_available(entry, 0))
            with self.assertRaises(ChartUnavailable):
                self.catalog.chart_file(entry, 0)

    def test_symlink_escape_rejected(self) -> None:
        outside = self.root / "outside.png"
        outside.write_bytes(b"secret")
        directory = self.write(
            "runs",
            "run-link",
            make_record(
                "run-link",
                charts=[{"id": "metrics", "title": "指标", "path": "charts/metrics.png"}],
            ),
        )
        (directory / "charts").mkdir(parents=True, exist_ok=True)
        (directory / "charts" / "metrics.png").symlink_to(outside)
        entry = self.catalog.get("run-link")
        self.assertFalse(self.catalog.chart_available(entry, 0))
        with self.assertRaises(ChartUnsafe):
            self.catalog.chart_file(entry, 0)


class FingerprintTests(CatalogTestCase):
    def test_fingerprint_changes_with_records(self) -> None:
        first = self.catalog.fingerprint()
        self.write("runs", "run-a", make_record("run-a"))
        second = self.catalog.fingerprint()
        self.assertNotEqual(first, second)

        self.write("runs", "run-b", make_record("run-b"))
        third = self.catalog.fingerprint()
        self.assertNotEqual(second, third)

        # 内容变化（mtime/size）也会改变指纹。
        self.write(
            "runs",
            "run-a",
            make_record("run-a", metrics={"MRR": 0.9, "P@1": 0.4}),
        )
        fourth = self.catalog.fingerprint()
        self.assertNotEqual(third, fourth)


if __name__ == "__main__":
    unittest.main()
