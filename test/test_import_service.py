"""T-008 单份 JSON 导入服务测试。"""

from __future__ import annotations

import errno
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment.plans import load_plan_catalog  # noqa: E402
from experiment.timing import TIMING_FIELDS  # noqa: E402
from records.catalog import ResultCatalog  # noqa: E402
from records.importer import (  # noqa: E402
    MAX_IMPORT_BYTES,
    ImportFailure,
    ImportService,
)

CATALOG = load_plan_catalog()
RUN_ID = "20261121T120000000000Z-imp001"


def make_record(
    run_id: str = RUN_ID,
    *,
    status: str = "Completed",
    plan_id: str = "scifact-comparison",
    variant_id: str = "scifact-bm25",
    charts: list | None = None,
    metrics: dict | None = None,
    environment: dict | None = None,
) -> dict:
    record = {
        "format_version": "1.0",
        "run_id": run_id,
        "plan_id": plan_id,
        "plan_version": "v1",
        "variant_id": variant_id,
        "plan_name": "SciFact 英文检索方法与模型对比",
        "variant_name": "BM25",
        "status": status,
        "stage_progress": {
            "completed": ["prepare", "load_dataset"],
            "current": None,
            "not_applicable": ["load_model", "encode_queries"],
        },
        "started_at": "2026-11-21T10:00:00Z",
        "finished_at": "2026-11-21T10:00:05Z",
        "dataset": {
            "name": "scifact",
            "split": "test",
            "corpus_size": 3,
            "query_count": 1,
            "qrels_pairs": 1,
        },
        "variant_config": {"method": "bm25", "k1": 1.2},
        "metrics": {"MRR": 0.5} if metrics is None else metrics,
        "num_queries": 1,
        "num_skipped": 0,
        "timing": {field: None for field in TIMING_FIELDS},
        "environment": (
            environment if environment is not None else {"python": "3.11.18"}
        ),
        "charts": charts or [],
        "error": None,
    }
    if status != "Completed":
        record["metrics"] = None
        record["num_queries"] = None
        record["num_skipped"] = None
        record["error"] = {
            "message": "失败",
            "detail": None,
            "stage": "retrieve",
        }
    return record


def dump(record: dict) -> bytes:
    return json.dumps(record, ensure_ascii=False).encode("utf-8")


class ImportServiceTestCase(unittest.TestCase):
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
        self.service = ImportService(self.imports, catalog=self.catalog)

    def assert_failure(self, content: bytes, code: str, filename="run.json"):
        with self.assertRaises(ImportFailure) as context:
            self.service.import_json(filename, content)
        self.assertEqual(context.exception.code, code)
        return context.exception

    def assert_no_staging(self) -> None:
        if not self.imports.is_dir():
            return
        leftovers = [
            path.name
            for path in self.imports.iterdir()
            if path.name.startswith(".tmp-")
        ]
        self.assertEqual(leftovers, [])


class ImportSuccessTests(ImportServiceTestCase):
    def test_import_writes_original_bytes(self) -> None:
        content = dump(make_record())
        result = self.service.import_json("run.json", content)

        self.assertEqual(result.run_id, RUN_ID)
        self.assertEqual(result.bytes_written, len(content))
        target = self.imports / RUN_ID / "run.json"
        self.assertTrue(target.is_file())
        self.assertEqual(target.read_bytes(), content)
        self.assert_no_staging()

        entry = self.catalog.get(RUN_ID)
        self.assertEqual(entry.source, "imports")
        self.assertEqual(entry.record.environment["python"], "3.11.18")
        self.assertEqual(entry.record.started_at.isoformat(), "2026-11-21T10:00:00+00:00")

    def test_import_reports_missing_charts(self) -> None:
        content = dump(
            make_record(
                charts=[
                    {"id": "metrics", "title": "指标", "path": "charts/metrics.png"},
                    {"id": "timing", "title": "耗时", "path": "charts/timing.png"},
                ]
            )
        )
        result = self.service.import_json("run.json", content)
        self.assertEqual(len(result.missing_charts), 2)
        self.assertEqual(result.missing_charts[0]["id"], "metrics")
        self.assertTrue(any("metrics" in warning for warning in result.warnings))
        self.assertEqual(len(result.warnings), 2)

    def test_unknown_plan_and_variant_import(self) -> None:
        content = dump(
            make_record(plan_id="unknown-plan", variant_id="unknown-variant")
        )
        self.service.import_json("run.json", content)
        entry = self.catalog.get(RUN_ID)
        self.assertEqual(entry.record.plan_id, "unknown-plan")
        self.assertFalse(self.catalog.comparison_eligible(entry.record))

    def test_failed_record_import(self) -> None:
        content = dump(make_record(status="Failed"))
        self.service.import_json("run.json", content)
        entry = self.catalog.get(RUN_ID)
        self.assertEqual(entry.record.status, "Failed")
        self.assertIsNone(entry.record.metrics)


class ImportValidationTests(ImportServiceTestCase):
    def test_empty_and_wrong_extension(self) -> None:
        self.assert_failure(b"", "IMPORT_FILE_INVALID")
        self.assert_failure(dump(make_record()), "IMPORT_FILE_INVALID", "run.txt")
        self.assert_failure(dump(make_record()), "IMPORT_FILE_INVALID", "run")

    def test_too_large(self) -> None:
        content = b" " * (MAX_IMPORT_BYTES + 1)
        self.assert_failure(content, "IMPORT_TOO_LARGE")

    def test_invalid_utf8(self) -> None:
        self.assert_failure(b"\xff\xfe\x00", "IMPORT_INVALID_ENCODING")

    def test_duplicate_key_top_level(self) -> None:
        content = b'{"format_version": "1.0", "run_id": "a", "run_id": "b"}'
        error = self.assert_failure(content, "IMPORT_DUPLICATE_KEY")
        self.assertEqual(error.details, ["run_id"])

    def test_duplicate_key_nested(self) -> None:
        record = make_record()
        text = json.dumps(record, ensure_ascii=False)
        text = text.replace(
            '"variant_config": {', '"variant_config": {"method": "bm25", ', 1
        )
        self.assert_failure(text.encode("utf-8"), "IMPORT_DUPLICATE_KEY")

    def test_invalid_json_and_constants(self) -> None:
        self.assert_failure(b"{bad json", "IMPORT_INVALID_JSON")
        record = make_record()
        text = json.dumps(record, ensure_ascii=False).replace(
            '"MRR": 0.5', '"MRR": NaN'
        )
        self.assert_failure(text.encode("utf-8"), "IMPORT_INVALID_JSON")
        text = json.dumps(record, ensure_ascii=False).replace(
            '"MRR": 0.5', '"MRR": Infinity'
        )
        self.assert_failure(text.encode("utf-8"), "IMPORT_INVALID_JSON")

    def test_top_level_must_be_object(self) -> None:
        self.assert_failure(b"[]", "IMPORT_SCHEMA_INVALID")

    def test_unsupported_version(self) -> None:
        record = make_record()
        record["format_version"] = "2.0"
        self.assert_failure(dump(record), "IMPORT_VERSION_UNSUPPORTED")
        del record["format_version"]
        self.assert_failure(dump(record), "IMPORT_VERSION_UNSUPPORTED")

    def test_invalid_run_id(self) -> None:
        record = make_record()
        record["run_id"] = "../escape"
        self.assert_failure(dump(record), "IMPORT_SCHEMA_INVALID")

    def test_schema_errors(self) -> None:
        record = make_record()
        record["unknown_field"] = 1
        self.assert_failure(dump(record), "IMPORT_SCHEMA_INVALID")

        record = make_record(metrics={})
        self.assert_failure(dump(record), "IMPORT_SCHEMA_INVALID")

        record = make_record()
        del record["variant_id"]
        self.assert_failure(dump(record), "IMPORT_SCHEMA_INVALID")

    def test_unsafe_chart_paths(self) -> None:
        for path in (
            "/abs/chart.png",
            "../chart.png",
            "charts\\x.png",
            "charts/./x.png",
            "charts//x.png",
            "C:/x.png",
        ):
            with self.subTest(path=path):
                record = make_record(
                    charts=[{"id": "metrics", "title": "指标", "path": path}]
                )
                self.assert_failure(dump(record), "IMPORT_PATH_INVALID")
                self.assertFalse((self.imports / RUN_ID).exists())

    def test_failures_leave_no_partial_import(self) -> None:
        self.assert_failure(dump(make_record(metrics={})), "IMPORT_SCHEMA_INVALID")
        self.assert_no_staging()
        self.assertFalse((self.imports / RUN_ID).exists())


class ImportConflictTests(ImportServiceTestCase):
    def test_conflict_with_existing_import(self) -> None:
        content = dump(make_record())
        self.service.import_json("run.json", content)
        original = (self.imports / RUN_ID / "run.json").read_bytes()
        self.assert_failure(content, "IMPORT_RUN_ID_CONFLICT")
        self.assertEqual(
            (self.imports / RUN_ID / "run.json").read_bytes(), original
        )

    def test_conflict_with_presets_or_runs(self) -> None:
        for root in (self.presets, self.runs):
            with self.subTest(root=root.name):
                run_id = f"{RUN_ID}-{root.name}"
                (root / run_id).mkdir(parents=True)
                record = make_record(run_id)
                self.assert_failure(
                    dump(record), "IMPORT_RUN_ID_CONFLICT"
                )
                self.assertFalse((self.imports / run_id).exists())

    def test_conflict_with_corrupt_directory(self) -> None:
        (self.runs / RUN_ID).mkdir(parents=True)
        (self.runs / RUN_ID / "run.json").write_text("{bad", encoding="utf-8")
        self.assert_failure(dump(make_record()), "IMPORT_RUN_ID_CONFLICT")
        self.assertFalse((self.imports / RUN_ID).exists())

    def test_write_failure_cleans_staging(self) -> None:
        content = dump(make_record())
        with mock.patch(
            "records.importer.os.rename", side_effect=OSError("boom")
        ):
            with self.assertRaises(ImportFailure) as context:
                self.service.import_json("run.json", content)
        self.assertEqual(context.exception.code, "IMPORT_WRITE_FAILED")
        self.assertFalse((self.imports / RUN_ID).exists())
        self.assert_no_staging()


class ImportHardeningTests(ImportServiceTestCase):
    def test_control_characters_in_chart_path(self) -> None:
        for path in ("charts/bad\u0000.png", "charts/bad\n.png", "charts/bad\t.png"):
            with self.subTest(path=repr(path)):
                record = make_record(
                    charts=[{"id": "metrics", "title": "指标", "path": path}]
                )
                self.assert_failure(dump(record), "IMPORT_PATH_INVALID")
                self.assertFalse((self.imports / RUN_ID).exists())

    def test_exponent_overflow_rejected(self) -> None:
        for value in ("1e999", "-1e999"):
            with self.subTest(value=value):
                record = make_record()
                text = json.dumps(record, ensure_ascii=False).replace(
                    '"environment": {"python": "3.11.18"}',
                    f'"environment": {{"python": {value}}}',
                )
                self.assertIn(value, text)
                self.assert_failure(
                    text.encode("utf-8"), "IMPORT_INVALID_JSON"
                )

    def test_deeply_nested_json_rejected(self) -> None:
        content = b"[" * 2000 + b"]" * 2000
        self.assert_failure(content, "IMPORT_INVALID_JSON")

    def test_concurrent_same_run_id(self) -> None:
        content = dump(make_record())
        barrier = threading.Barrier(2)
        outcomes: list[str] = []

        def worker() -> None:
            barrier.wait()
            try:
                self.service.import_json("run.json", content)
                outcomes.append("ok")
            except ImportFailure as exc:
                outcomes.append(exc.code)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertCountEqual(outcomes, ["ok", "IMPORT_RUN_ID_CONFLICT"])
        self.assertTrue((self.imports / RUN_ID / "run.json").is_file())
        self.assert_no_staging()

    def test_external_race_maps_to_conflict(self) -> None:
        content = dump(make_record())
        with mock.patch(
            "records.importer.os.rename",
            side_effect=OSError(errno.ENOTEMPTY, "Directory not empty"),
        ):
            with self.assertRaises(ImportFailure) as context:
                self.service.import_json("run.json", content)
        self.assertEqual(context.exception.code, "IMPORT_RUN_ID_CONFLICT")
        self.assertFalse((self.imports / RUN_ID).exists())
        self.assert_no_staging()

    def test_external_race_file_exists_maps_to_conflict(self) -> None:
        content = dump(make_record())
        with mock.patch(
            "records.importer.os.rename",
            side_effect=FileExistsError(errno.EEXIST, "File exists"),
        ):
            with self.assertRaises(ImportFailure) as context:
                self.service.import_json("run.json", content)
        self.assertEqual(context.exception.code, "IMPORT_RUN_ID_CONFLICT")
        self.assert_no_staging()


if __name__ == "__main__":
    unittest.main()
