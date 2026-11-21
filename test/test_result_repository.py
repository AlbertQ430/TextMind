"""T-005 结果目录仓储测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment.timing import TIMING_FIELDS  # noqa: E402
from records.repository import (  # noqa: E402
    ResultConflictError,
    ResultPathError,
    ResultRecordError,
    ResultRepository,
)
from records.schema import parse_run_record  # noqa: E402

RUN_ID = "20261121T100000000000Z-abc123"


def make_record(run_id: str = RUN_ID, charts: list | None = None) -> dict:
    return {
        "format_version": "1.0",
        "run_id": run_id,
        "plan_id": "scifact-comparison",
        "plan_version": "v1",
        "variant_id": "scifact-bm25",
        "plan_name": "SciFact 英文检索方法与模型对比",
        "variant_name": "BM25",
        "status": "Completed",
        "stage_progress": {
            "completed": ["prepare"],
            "current": None,
            "not_applicable": ["load_model", "encode_queries"],
        },
        "started_at": "2026-11-21T10:00:00Z",
        "finished_at": "2026-11-21T10:00:01Z",
        "dataset": {
            "name": "scifact",
            "split": "test",
            "corpus_size": 3,
            "query_count": 1,
            "qrels_pairs": 1,
        },
        "variant_config": {"method": "bm25"},
        "metrics": {"MRR": 0.5},
        "num_queries": 1,
        "num_skipped": 0,
        "timing": {field: None for field in TIMING_FIELDS},
        "environment": {},
        "charts": charts or [],
        "error": None,
    }


class ResultRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "runs"
        self.repository = ResultRepository(self.root)

    def test_create_staging_is_unique_and_inside_staging_root(self) -> None:
        first = self.repository.create_staging(RUN_ID)
        second = self.repository.create_staging(RUN_ID)
        self.assertNotEqual(first, second)
        self.assertEqual(first.parent, self.repository.staging_root)
        self.assertTrue(first.name.startswith(RUN_ID + "-"))
        self.assertTrue(first.is_dir())

    def test_publish_moves_staging_to_final(self) -> None:
        staging = self.repository.create_staging(RUN_ID)
        self.repository.write_record(staging, make_record())
        final = self.repository.publish(staging, RUN_ID)
        self.assertEqual(final, self.root / RUN_ID)
        self.assertTrue((final / "run.json").is_file())
        self.assertFalse(staging.exists())

    def test_publish_rejects_existing_final(self) -> None:
        final = self.root / RUN_ID
        final.mkdir(parents=True)
        (final / "run.json").write_text("{}", encoding="utf-8")
        staging = self.repository.create_staging(RUN_ID)
        with self.assertRaises(ResultConflictError):
            self.repository.publish(staging, RUN_ID)
        self.assertEqual((final / "run.json").read_text(encoding="utf-8"), "{}")
        self.assertTrue(staging.exists())

    def test_publish_rejects_outside_staging(self) -> None:
        outside = Path(self._tmp.name) / "outside"
        outside.mkdir()
        with self.assertRaises(ResultPathError):
            self.repository.publish(outside, RUN_ID)

    def test_run_id_path_safety(self) -> None:
        for run_id in ("../escape", "a/b", ".", "..", "", " run"):
            with self.subTest(run_id=run_id):
                with self.assertRaises(ResultPathError):
                    self.repository.record_directory(run_id)

    def test_write_and_read_record(self) -> None:
        staging = self.repository.create_staging(RUN_ID)
        record = self.repository.write_record(staging, make_record())
        self.assertEqual(record.run_id, RUN_ID)
        self.assertEqual(record.status, "Completed")
        final = self.repository.publish(staging, RUN_ID)
        loaded = self.repository.read(RUN_ID)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.run_id, RUN_ID)
        self.assertEqual(loaded.finished_at, parse_run_record(make_record()).finished_at)
        self.assertEqual(final, self.root / RUN_ID)

    def test_read_missing_returns_none(self) -> None:
        self.assertIsNone(self.repository.read(RUN_ID))
        self.assertIsNone(self.repository.read_at(self.root / "missing"))

    def test_read_corrupt_record_raises(self) -> None:
        directory = self.root / RUN_ID
        directory.mkdir(parents=True)
        (directory / "run.json").write_text("{oops", encoding="utf-8")
        with self.assertRaises(ResultRecordError):
            self.repository.read(RUN_ID)

        (directory / "run.json").write_text(
            json.dumps({"format_version": "1.0"}), encoding="utf-8"
        )
        with self.assertRaises(ResultRecordError):
            self.repository.read(RUN_ID)

    def test_find_staging_candidates_matches_exact_run_id(self) -> None:
        first = self.repository.create_staging(RUN_ID)
        second = self.repository.create_staging(RUN_ID)
        other = self.repository.create_staging("20261121T100000000000Z-other1")
        candidates = self.repository.find_staging_candidates(RUN_ID)
        self.assertEqual(candidates, sorted([first, second]))
        self.assertNotIn(other, candidates)

    def test_validate_chart_paths(self) -> None:
        staging = self.repository.create_staging(RUN_ID)
        charts_dir = staging / "charts"
        charts_dir.mkdir()
        (charts_dir / "metrics.png").write_bytes(b"png")
        record = make_record(
            charts=[{"id": "metrics", "title": "指标", "path": "charts/metrics.png"}]
        )
        self.repository.validate_chart_paths(record, staging)

        for path in ("charts/missing.png", "../escape.png", "/abs.png"):
            with self.subTest(path=path):
                bad = make_record(
                    charts=[{"id": "x", "title": "x", "path": path}]
                )
                with self.assertRaises(ResultRecordError):
                    self.repository.validate_chart_paths(bad, staging)

    def test_write_record_rejects_invalid_schema(self) -> None:
        staging = self.repository.create_staging(RUN_ID)
        bad = make_record()
        bad["status"] = "Running"
        with self.assertRaises(ResultRecordError):
            self.repository.write_record(staging, bad)


if __name__ == "__main__":
    unittest.main()
