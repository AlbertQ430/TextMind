"""T-009 预置结果构建测试。"""

from __future__ import annotations

import hashlib
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.plans import RunVariant, load_plan_catalog  # noqa: E402
from records.schema import parse_run_record  # noqa: E402
from scripts import build_presets  # noqa: E402

CATALOG = load_plan_catalog()
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"


def run_variants():
    return [
        (plan, variant)
        for plan in CATALOG.plans
        for variant in plan.variants
        if isinstance(variant, RunVariant)
    ]


def evaluation_hashes() -> dict[str, str]:
    return {
        path.name: hashlib.md5(path.read_bytes()).hexdigest()
        for path in sorted(EVALUATION_DIR.glob("*.json"))
    }


class TimingMappingTests(unittest.TestCase):
    def test_bm25_mapping(self) -> None:
        timing = build_presets.map_timing(
            "bm25",
            {
                "load_seconds": 0.2,
                "index_seconds": 31.3,
                "model_load_seconds": 0.0,
                "doc_cache_load_seconds": 0.0,
                "corpus_encode_seconds": 31.3,
                "query_cache_load_seconds": 0.0,
                "query_encode_seconds": 0.0,
                "vector_search_total_seconds": 33.1,
                "search_total_seconds": 33.1,
                "search_mean_seconds": 0.016,
                "search_median_seconds": 0.009,
            },
        )
        self.assertEqual(timing["dataset_load_seconds"], 0.2)
        self.assertEqual(timing["index_build_seconds"], 31.3)
        self.assertEqual(timing["bm25_search_total_seconds"], 33.1)
        self.assertEqual(timing["retrieval_total_seconds"], 33.1)
        self.assertEqual(timing["retrieval_mean_seconds"], 0.016)
        for field in (
            "model_load_seconds",
            "doc_cache_load_seconds",
            "corpus_encode_seconds",
            "query_cache_load_seconds",
            "query_encode_seconds",
            "dense_search_total_seconds",
            "rrf_fusion_seconds",
            "evaluation_seconds",
        ):
            self.assertIsNone(timing[field], field)

    def test_dense_mapping_ignores_zero(self) -> None:
        timing = build_presets.map_timing(
            "dense",
            {
                "load_seconds": 0.01,
                "model_load_seconds": 4.6,
                "doc_cache_load_seconds": 0.009,
                "corpus_encode_seconds": 0.0,
                "query_cache_load_seconds": 0.0006,
                "query_encode_seconds": 0.9,
                "index_seconds": 0.009,
                "vector_search_total_seconds": 0.16,
                "search_total_seconds": 0.16,
                "search_mean_seconds": 0.0005,
                "search_median_seconds": 0.0004,
            },
        )
        self.assertEqual(timing["model_load_seconds"], 4.6)
        self.assertEqual(timing["dense_search_total_seconds"], 0.16)
        self.assertEqual(timing["retrieval_total_seconds"], 0.16)
        self.assertIsNone(timing["corpus_encode_seconds"])
        self.assertEqual(timing["query_encode_seconds"], 0.9)

    def test_hybrid_mapping_keeps_subtotals(self) -> None:
        timing = build_presets.map_timing(
            "hybrid",
            {
                "load_seconds": 0.2,
                "model_load_seconds": 1.7,
                "index_seconds": 31.9,
                "doc_cache_load_seconds": 0.15,
                "corpus_encode_seconds": 0.0,
                "query_cache_load_seconds": 0.005,
                "query_encode_seconds": 0.005,
                "bm25_search_seconds": 36.1,
                "dense_search_seconds": 15.6,
                "rrf_fusion_seconds": 0.07,
                "vector_search_total_seconds": 51.8,
                "search_total_seconds": 51.8,
                "search_mean_seconds": 0.025,
                "search_median_seconds": 0.017,
            },
        )
        self.assertEqual(timing["bm25_search_total_seconds"], 36.1)
        self.assertEqual(timing["dense_search_total_seconds"], 15.6)
        self.assertEqual(timing["rrf_fusion_seconds"], 0.07)
        self.assertEqual(timing["retrieval_total_seconds"], 51.8)
        self.assertIsNone(timing["corpus_encode_seconds"])

    def test_archived_start_uses_decomposable_timing(self) -> None:
        from datetime import datetime, timezone

        finished = datetime(2026, 11, 20, 12, 0, 0, tzinfo=timezone.utc)
        timing = {
            "dataset_load_seconds": 1.0,
            "model_load_seconds": 2.0,
            "index_build_seconds": 4.0,
            "query_cache_load_seconds": 0.5,
            "query_encode_seconds": 0.5,
            "retrieval_total_seconds": 10.0,
            "evaluation_seconds": 100.0,
        }
        started = build_presets.archived_start("bm25", timing, finished)
        self.assertEqual((finished - started).total_seconds(), 18.0)


class SourceCheckTests(unittest.TestCase):
    def test_all_sources_match_plans(self) -> None:
        for plan, variant in run_variants():
            with self.subTest(variant=variant.variant_id):
                source = json.loads(
                    (PROJECT_ROOT / variant.preset_source).read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(
                    build_presets.check_source(plan, variant, source), []
                )

    def test_mismatch_reported(self) -> None:
        plan, variant = run_variants()[0]
        source = json.loads(
            (PROJECT_ROOT / variant.preset_source).read_text(encoding="utf-8")
        )
        source["config"]["method"] = "dense"
        errors = build_presets.check_source(plan, variant, source)
        self.assertTrue(any("method" in error for error in errors))

    def test_rrf_weight_mismatch(self) -> None:
        sweep = CATALOG.plans[2]
        variant = next(
            item
            for item in sweep.variants
            if item.variant_id == "duretrieval-hybrid-bge-small-prefix-w0.5"
        )
        source = json.loads(
            (PROJECT_ROOT / variant.preset_source).read_text(encoding="utf-8")
        )
        source["config"]["rrf_weights"] = [0.1, 1.0]
        errors = build_presets.check_source(sweep, variant, source)
        self.assertTrue(any("rrf_weights" in error for error in errors))


class BuildRecordTests(unittest.TestCase):
    def test_all_records_validate(self) -> None:
        for plan, variant in run_variants():
            with self.subTest(variant=variant.variant_id):
                source_path = PROJECT_ROOT / variant.preset_source
                source = json.loads(source_path.read_text(encoding="utf-8"))
                record, execution = build_presets.build_record(
                    plan, variant, source, source_path
                )
                parsed = parse_run_record(record)
                self.assertEqual(parsed.run_id, variant.preset_run_id)
                self.assertEqual(parsed.status, "Completed")
                self.assertEqual(parsed.plan_id, plan.plan_id)
                if plan.dataset.name == "duretrieval":
                    self.assertEqual(parsed.dataset.split, "dev")
                self.assertIsNone(record["charts"] or None)
                self.assertEqual(len(execution.timing), 14)

    def test_metrics_and_environment_copied(self) -> None:
        plan, variant = run_variants()[0]
        source_path = PROJECT_ROOT / variant.preset_source
        source = json.loads(source_path.read_text(encoding="utf-8"))
        record, _ = build_presets.build_record(
            plan, variant, source, source_path
        )
        self.assertEqual(record["metrics"], source["metrics"])
        self.assertEqual(record["environment"], source["environment"])
        self.assertEqual(record["variant_config"]["top_k"], 100)


class HybridTimingTests(unittest.TestCase):
    def test_hybrid_retrieval_total_matches_subtotals(self) -> None:
        checked = 0
        for plan, variant in run_variants():
            if variant.method != "hybrid":
                continue
            checked += 1
            source_path = PROJECT_ROOT / variant.preset_source
            source = json.loads(source_path.read_text(encoding="utf-8"))
            record, _ = build_presets.build_record(
                plan, variant, source, source_path
            )
            timing = record["timing"]
            total = timing["retrieval_total_seconds"]
            subtotal = sum(
                timing[field] or 0.0
                for field in (
                    "bm25_search_total_seconds",
                    "dense_search_total_seconds",
                    "rrf_fusion_seconds",
                )
            )
            with self.subTest(variant=variant.variant_id):
                self.assertIsNotNone(total)
                self.assertAlmostEqual(total, subtotal, delta=1e-3)
        self.assertGreaterEqual(checked, 7)


class ReleaseDocsTests(unittest.TestCase):
    def test_release_documents_are_consistent(self) -> None:
        notice = (PROJECT_ROOT / "NOTICE.md").read_text(encoding="utf-8")
        source = (
            PROJECT_ROOT / "resources" / "scifact" / "SOURCE.md"
        ).read_text(encoding="utf-8")
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("随软件提供", notice)
        for text in (notice, source):
            for forbidden in (
                "不重新分发",
                "不随二进制包分发",
                "requirements.txt",
            ):
                self.assertNotIn(forbidden, text)
        self.assertIn("requirements.lock", notice)
        self.assertIn("CC-BY-SA-4.0", source)
        self.assertIn("data/raw/scifact", source)

        for url in (
            "huggingface.co/Qwen/Qwen3-Embedding-0.6B",
            "huggingface.co/BAAI/bge-m3",
            "huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
            "huggingface.co/BAAI/bge-small-zh-v1.5",
        ):
            self.assertIn(url, notice)

        self.assertIn("C-MTEB/DuRetrieval", notice)
        self.assertIn("Apache License 2.0", notice)
        self.assertIn("a1a333e290fe30b10f3f56498e3a0d911a693ced", notice)
        self.assertIn("497b7bd1bbb25cb3757ff34d95a8be50a3de2279", notice)
        self.assertIn("NOTICE.md", readme)


class AssetCheckTests(unittest.TestCase):
    def test_real_assets_pass(self) -> None:
        self.assertEqual(build_presets.check_assets(), [])

    def test_missing_root_notice_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "NOTICE.md"
            with mock.patch.object(build_presets, "ROOT_NOTICE", missing):
                errors = build_presets.check_assets()
        self.assertIn("缺少非空的根 NOTICE.md", errors)

    def test_missing_assets_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp)
            with mock.patch.object(build_presets, "SCIFACT_DIR", empty), \
                    mock.patch.object(build_presets, "RESOURCES_DIR", empty):
                errors = build_presets.check_assets()
        self.assertGreaterEqual(len(errors), 6)


class EndToEndBuildTests(unittest.TestCase):
    def test_build_all_produces_17_valid_records(self) -> None:
        before = evaluation_hashes()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "presets"
            published, failures = build_presets.build_all(
                CATALOG, output, force=False
            )
            self.assertEqual(failures, [])
            self.assertEqual(len(published), 17)
            self.assertEqual(
                len([child for child in output.iterdir() if child.is_dir()]),
                17,
            )
            self.assertFalse(
                any(child.name.startswith(".tmp-") for child in output.iterdir())
            )
            self.assertTrue(
                (output / "preset-duretrieval-hybrid-bge-small-prefix-equal").is_dir()
            )
            for target in published:
                record = parse_run_record(
                    json.loads(
                        (target / "run.json").read_text(encoding="utf-8")
                    )
                )
                self.assertEqual(record.status, "Completed")
                self.assertIn("metrics", record.model_dump())
                for chart in record.charts:
                    self.assertTrue((target / chart.path).is_file())
        self.assertEqual(before, evaluation_hashes())

    def test_rebuild_without_force_reports_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "presets"
            (output / "preset-scifact-bm25").mkdir(parents=True)
            published, failures = build_presets.build_all(
                CATALOG, output, force=False
            )
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0][0], "scifact-bm25")
            self.assertEqual(len(published), 16)
            self.assertTrue(
                (output / "preset-scifact-dense-qwen3-base").is_dir()
            )

    def test_failure_rolls_back_published_records(self) -> None:
        original = build_presets._build_one
        calls = {"count": 0}

        def failing(plan, variant, output, force):
            calls["count"] += 1
            if calls["count"] == 3:
                raise ValueError("模拟映射失败")
            return original(plan, variant, output, force)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "presets"
            with mock.patch.object(build_presets, "_build_one", failing):
                code = build_presets.main(["--output", str(output)])
            self.assertEqual(code, 1)
            self.assertEqual(
                [child for child in output.iterdir() if child.is_dir()], []
            )


class CliTests(unittest.TestCase):
    def test_check_assets_only(self) -> None:
        self.assertEqual(build_presets.main(["--check-assets-only"]), 0)

    def test_existing_output_requires_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "presets"
            (output / "preset-scifact-bm25").mkdir(parents=True)
            self.assertEqual(
                build_presets.main(["--output", str(output)]), 1
            )

    def test_failure_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "presets"
            with mock.patch.object(
                build_presets,
                "_build_one",
                side_effect=ValueError("模拟失败"),
            ):
                self.assertEqual(
                    build_presets.main(["--output", str(output)]), 1
                )


if __name__ == "__main__":
    unittest.main()
