"""T-002 运行记录契约测试。"""

from __future__ import annotations

import copy
import json
import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from jsonschema import Draft202012Validator  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from records.schema import (  # noqa: E402
    RUN_RECORD_SCHEMA_PATH,
    RunRecord,
    RunRecordError,
    parse_run_record,
    validate_run_record_schema,
)

TIMING_NULLS = {
    name: None
    for name in (
        "dataset_load_seconds",
        "model_load_seconds",
        "index_build_seconds",
        "doc_cache_load_seconds",
        "corpus_encode_seconds",
        "query_cache_load_seconds",
        "query_encode_seconds",
        "bm25_search_total_seconds",
        "dense_search_total_seconds",
        "rrf_fusion_seconds",
        "retrieval_total_seconds",
        "retrieval_mean_seconds",
        "retrieval_median_seconds",
        "evaluation_seconds",
    )
}

BASE_RECORD = {
    "format_version": "1.0",
    "run_id": "20261121T081500123456Z-a1b2c3",
    "plan_id": "scifact-comparison",
    "plan_version": "v1",
    "variant_id": "scifact-bm25",
    "plan_name": "SciFact 英文检索方法与模型对比",
    "variant_name": "BM25",
    "status": "Completed",
    "stage_progress": {
        "completed": ["prepare", "write_record"],
        "current": None,
        "not_applicable": ["load_model", "encode_queries"],
    },
    "started_at": "2026-11-21T08:15:00.123456Z",
    "finished_at": "2026-11-21T08:15:02.000000Z",
    "dataset": {
        "name": "scifact",
        "split": "test",
        "corpus_size": 5183,
        "query_count": 300,
        "qrels_pairs": 339,
    },
    "variant_config": {
        "method": "bm25",
        "k1": 1.2,
        "b": 0.75,
        "tokenizer": "simple",
        "top_k": 100,
        "k_values": [1, 5, 10, 100],
    },
    "metrics": {"P@1": 0.5333, "NDCG@10": 0.6605, "R@100": 0.8859, "MRR": 0.6312},
    "num_queries": 300,
    "num_skipped": 0,
    "timing": dict(TIMING_NULLS),
    "environment": {"python": "3.11.18", "platform": "macOS", "device": "cpu"},
    "charts": [
        {"id": "metrics", "title": "指标", "path": "charts/metrics.png"},
    ],
    "error": None,
}


def make_record(**overrides) -> dict:
    data = copy.deepcopy(BASE_RECORD)
    data.update(overrides)
    return data


def make_failed_record(**overrides) -> dict:
    data = make_record(
        status="Failed",
        metrics=None,
        num_queries=None,
        num_skipped=None,
        charts=[],
        error={"message": "检索失败", "detail": "traceback", "stage": "retrieve"},
    )
    data.update(overrides)
    return data


class RunRecordTests(unittest.TestCase):
    def _assert_schema_rejects(self, data: dict) -> None:
        with self.assertRaises(RunRecordError):
            validate_run_record_schema(data)

    def _assert_invalid(self, data: dict, fragment: str | None = None) -> RunRecordError:
        with self.assertRaises(RunRecordError) as context:
            parse_run_record(data)
        if fragment is not None:
            self.assertIn(fragment, str(context.exception))
        return context.exception

    # 正向与往返 -------------------------------------------------------------------

    def test_completed_record_roundtrip(self) -> None:
        record = parse_run_record(make_record())
        self.assertEqual(record.status, "Completed")
        self.assertEqual(record.metrics["MRR"], 0.6312)
        self.assertEqual(record.charts[0].path, "charts/metrics.png")

        dumped = record.model_dump(mode="json")
        self.assertEqual(dumped["started_at"], "2026-11-21T08:15:00.123456Z")
        reparsed = parse_run_record(dumped)
        self.assertEqual(reparsed, record)

    def test_schema_is_draft_2020_12_and_accepts_record(self) -> None:
        schema = json.loads(RUN_RECORD_SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["$schema"], "https://json-schema.org/draft/2020-12/schema"
        )
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(make_record())

    def test_failed_record_with_error_is_accepted(self) -> None:
        record = parse_run_record(make_failed_record())
        self.assertEqual(record.status, "Failed")
        self.assertIsNone(record.metrics)
        self.assertEqual(record.error.message, "检索失败")

    def test_cancelled_and_interrupted_require_error(self) -> None:
        for status in ("Cancelled", "Interrupted"):
            with self.subTest(status=status):
                record = parse_run_record(
                    make_failed_record(status=status, cancelled_at="2026-11-21T08:15:03Z")
                )
                self.assertEqual(record.status, status)

    # 时区与标识 -------------------------------------------------------------------

    def test_missing_timezone_is_rejected_by_schema_and_parser(self) -> None:
        for field in ("started_at", "finished_at"):
            with self.subTest(field=field):
                data = make_record(**{field: "2026-11-21T08:15:00"})
                self._assert_schema_rejects(data)
                self._assert_invalid(data, field)

    def test_invalid_run_id_is_rejected(self) -> None:
        for run_id in ("/absolute", "..", ".hidden", "a" * 129, "has space"):
            with self.subTest(run_id=run_id):
                self._assert_invalid(make_record(run_id=run_id), "run_id")

    def test_unknown_status_is_rejected(self) -> None:
        self._assert_invalid(make_record(status="Running"), "status")

    def test_wrong_format_version_is_rejected(self) -> None:
        self._assert_invalid(make_record(format_version="2.0"), "format_version")

    def test_empty_plan_version_is_rejected(self) -> None:
        self._assert_invalid(make_record(plan_version=""), "plan_version")

    # 终态约束 ---------------------------------------------------------------------

    def test_completed_requires_metrics_and_coverage(self) -> None:
        for field in ("metrics", "num_queries", "num_skipped"):
            with self.subTest(field=field):
                data = make_record()
                del data[field]
                self._assert_invalid(data, field)

    def test_completed_requires_non_empty_metrics(self) -> None:
        for metrics in ({}, None):
            with self.subTest(metrics=metrics):
                data = make_record(metrics=metrics)
                self._assert_schema_rejects(data)
                self._assert_invalid(data, "metrics")
                with self.assertRaises(ValidationError):
                    RunRecord.model_validate(data)

    def test_completed_requires_integer_coverage_fields(self) -> None:
        for field in ("num_queries", "num_skipped"):
            with self.subTest(field=field):
                data = make_record(**{field: None})
                self._assert_schema_rejects(data)
                self._assert_invalid(data, field)

    def test_cancelled_requires_non_empty_cancelled_at(self) -> None:
        missing = make_failed_record(status="Cancelled")
        null = make_failed_record(status="Cancelled", cancelled_at=None)
        for data in (missing, null):
            with self.subTest(data=data.get("cancelled_at", "<missing>")):
                self._assert_schema_rejects(data)
                self._assert_invalid(data, "cancelled_at")
                with self.assertRaises(ValidationError):
                    RunRecord.model_validate(data)

    def test_completed_with_error_is_rejected(self) -> None:
        self._assert_invalid(
            make_record(error={"message": "boom"}), "error"
        )

    def test_non_completed_requires_error(self) -> None:
        data = make_failed_record()
        del data["error"]
        self._assert_invalid(data, "error")

    def test_negative_counts_and_timing_are_rejected(self) -> None:
        self._assert_invalid(make_record(num_queries=-1), "num_queries")
        bad_timing = dict(TIMING_NULLS)
        bad_timing["retrieval_total_seconds"] = -0.1
        self._assert_invalid(make_record(timing=bad_timing), "timing")

    def test_missing_timing_field_is_rejected(self) -> None:
        timing = dict(TIMING_NULLS)
        del timing["evaluation_seconds"]
        self._assert_invalid(make_record(timing=timing), "timing")

    # 图表路径 ---------------------------------------------------------------------

    def test_unsafe_chart_paths_are_rejected_by_schema_and_parser(self) -> None:
        for path in (
            "/abs/chart.png",
            "../chart.png",
            "charts\\x.png",
            "charts//x.png",
            "charts/./x.png",
            "charts/x.png/",
            "C:/x.png",
            "",
            "   ",
        ):
            with self.subTest(path=path):
                data = make_record(
                    charts=[{"id": "x", "title": "x", "path": path}]
                )
                self._assert_schema_rejects(data)
                self._assert_invalid(data, "charts")

    def test_chart_unknown_field_is_rejected(self) -> None:
        self._assert_invalid(
            make_record(charts=[{
                "id": "x", "title": "x", "path": "charts/x.png", "extra": 1,
            }]),
            "extra",
        )

    # 来源与额外字段 ---------------------------------------------------------------

    def test_source_fields_are_rejected(self) -> None:
        for field in ("source", "result_source", "import_time"):
            with self.subTest(field=field):
                self._assert_invalid(make_record(**{field: "presets"}), field)

    def test_unknown_top_level_field_is_rejected(self) -> None:
        self._assert_invalid(make_record(extra_field=1), "extra_field")

    def test_non_object_input_is_rejected(self) -> None:
        for data in ([], "record", 1, None):
            with self.subTest(data=data):
                with self.assertRaises(RunRecordError):
                    parse_run_record(data)  # type: ignore[arg-type]

    def test_nan_metric_is_rejected(self) -> None:
        self._assert_invalid(
            make_record(metrics={"MRR": math.nan}), "MRR"
        )

    def test_error_details(self) -> None:
        error = self._assert_invalid(make_record(run_id=".."), "run_id")
        self.assertTrue(error.errors)


if __name__ == "__main__":
    unittest.main()
