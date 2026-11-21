"""T-003 单变体执行器测试。"""

from __future__ import annotations

import getpass
import hashlib
import socket
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from document.dataset import Document, Query, RetrievalDataset  # noqa: E402
from experiment.executor import (  # noqa: E402
    ExperimentExecutor,
    RunCancelled,
)
from experiment.plans import (  # noqa: E402
    DatasetSpec,
    EvaluationSpec,
    ModelSpec,
    PlanDefinition,
    RunVariant,
    load_plan_catalog,
)
from experiment.stages import STAGE_LABELS, STAGES, applicable_stages  # noqa: E402
from experiment.timing import TIMING_FIELDS, hybrid_total_matches  # noqa: E402

MODEL = ModelSpec(
    name="fake/model",
    directory_name="fake-model",
    revision="revision-1",
)


class FakeEncoder:
    """确定性伪编码器：按文本哈希生成向量，不加载真实模型。"""

    def __init__(self) -> None:
        self.encode_calls = 0
        self.seen_texts: list[str] = []

    def encode(self, texts, *, batch_size, show_progress_bar, normalize_embeddings):
        self.encode_calls += 1
        self.seen_texts.extend(texts)
        vectors = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            vectors.append(rng.normal(size=8).astype(np.float32))
        return np.array(vectors, dtype=np.float32)


def make_dataset() -> RetrievalDataset:
    return RetrievalDataset(
        documents=[
            Document("d1", "", "alpha beta"),
            Document("d2", "", "beta gamma"),
            Document("d3", "", "gamma delta"),
        ],
        queries=[
            Query("q1", "alpha"),
            Query("q2", "beta"),
            Query("q3", "unlabeled query"),
        ],
        qrels={"q1": {"d1": 1}, "q2": {"d2": 1}, "q3": {}},
        name="scifact",
    )


def make_plan(*variants: RunVariant, top_k: int = 10) -> PlanDefinition:
    return PlanDefinition(
        plan_id="test-plan",
        plan_version="v1",
        name="测试方案",
        description="",
        dataset=DatasetSpec(name="scifact", language="en", split="test"),
        evaluation=EvaluationSpec(top_k=top_k, k_values=(1, 5)),
        variants=tuple(variants),
    )


def make_bm25_variant(variant_id: str = "test-bm25") -> RunVariant:
    return RunVariant(
        variant_id=variant_id,
        name="BM25",
        kind="run",
        method="bm25",
        preset_run_id=f"preset-{variant_id}",
        preset_source="data/evaluation/test.json",
        k1=1.2,
        b=0.75,
        tokenizer="simple",
    )


def make_dense_variant(variant_id: str = "test-dense") -> RunVariant:
    return RunVariant(
        variant_id=variant_id,
        name="Dense",
        kind="run",
        method="dense",
        preset_run_id=f"preset-{variant_id}",
        preset_source="data/evaluation/test.json",
        model=MODEL,
        query_prefix="P: ",
        batch_size=2,
        normalize_embeddings=True,
    )


def make_hybrid_variant(variant_id: str = "test-hybrid") -> RunVariant:
    return RunVariant(
        variant_id=variant_id,
        name="Hybrid",
        kind="run",
        method="hybrid",
        preset_run_id=f"preset-{variant_id}",
        preset_source="data/evaluation/test.json",
        k1=1.2,
        b=0.75,
        tokenizer="simple",
        model=MODEL,
        query_prefix="P: ",
        batch_size=2,
        normalize_embeddings=True,
        rrf_k=60,
        rrf_weights=(1.0, 1.0),
    )


class RecordingExecutor:
    """构造执行器并记录注入调用。"""

    def __init__(self, *, device: str = "cpu", cache_dir=None):
        self.encoder = FakeEncoder()
        self.dataset = make_dataset()
        self.model_calls: list[tuple[ModelSpec, str]] = []
        self.dataset_calls = 0
        self.executor = ExperimentExecutor(
            model_loader=self._model_loader,
            dataset_loader=self._dataset_loader,
            device_resolver=lambda: device,
            cache_dir=cache_dir,
            show_progress=False,
        )

    def _model_loader(self, model, device):
        self.model_calls.append((model, device))
        return self.encoder

    def _dataset_loader(self, plan):
        self.dataset_calls += 1
        return self.dataset


def stage_sequence(events) -> list[tuple[str, str]]:
    return [(event.stage, event.event) for event in events]


class ExecutorStageTests(unittest.TestCase):
    def test_bm25_stage_sequence_and_progress(self) -> None:
        harness = RecordingExecutor()
        events = []
        plan = make_plan(make_bm25_variant())
        result = harness.executor.execute(plan, plan.variants[0], events.append)

        self.assertEqual(
            stage_sequence(events),
            [
                ("prepare", "started"),
                ("prepare", "completed"),
                ("load_dataset", "started"),
                ("load_dataset", "completed"),
                ("load_model", "skipped"),
                ("build_index", "started"),
                ("build_index", "completed"),
                ("encode_queries", "skipped"),
                ("retrieve", "started"),
                ("retrieve", "completed"),
                ("evaluate", "started"),
                ("evaluate", "completed"),
            ],
        )
        self.assertEqual(
            result.stage_progress.completed, applicable_stages("bm25")
        )
        self.assertEqual(
            result.stage_progress.not_applicable,
            ("load_model", "encode_queries"),
        )
        self.assertIsNone(result.stage_progress.current)
        self.assertEqual(harness.model_calls, [])

    def test_dense_stage_sequence(self) -> None:
        harness = RecordingExecutor()
        events = []
        plan = make_plan(make_dense_variant())
        result = harness.executor.execute(plan, plan.variants[0], events.append)

        self.assertEqual(
            stage_sequence(events),
            [(stage, event) for stage in STAGES for event in ("started", "completed")],
        )
        self.assertEqual(result.stage_progress.not_applicable, ())
        self.assertEqual(harness.model_calls[0][0], MODEL)
        self.assertEqual(harness.model_calls[0][1], "cpu")

    def test_hybrid_stage_sequence(self) -> None:
        harness = RecordingExecutor()
        events = []
        plan = make_plan(make_hybrid_variant())
        harness.executor.execute(plan, plan.variants[0], events.append)
        completed = [stage for stage, event in stage_sequence(events) if event == "completed"]
        self.assertEqual(tuple(completed), STAGES)

    def test_stage_events_are_timezone_aware(self) -> None:
        harness = RecordingExecutor()
        events = []
        plan = make_plan(make_bm25_variant())
        harness.executor.execute(plan, plan.variants[0], events.append)
        for event in events:
            self.assertIsNotNone(event.timestamp.tzinfo)
            self.assertIn(event.stage, STAGES)
            self.assertIn(event.stage, STAGE_LABELS)

    def test_report_stage_failure_propagates(self) -> None:
        harness = RecordingExecutor()
        plan = make_plan(make_bm25_variant())

        def failing_report(event):
            raise RuntimeError("callback failed")

        with self.assertRaises(RuntimeError):
            harness.executor.execute(plan, plan.variants[0], failing_report)

    def test_only_requested_variant_runs(self) -> None:
        harness = RecordingExecutor()
        bm25 = make_bm25_variant()
        dense = make_dense_variant()
        plan = make_plan(bm25, dense)
        result = harness.executor.execute(plan, bm25, lambda event: None)

        self.assertEqual(result.variant_config["method"], "bm25")
        self.assertEqual(harness.model_calls, [])
        self.assertEqual(harness.dataset_calls, 1)

        single_plan = make_plan(bm25)
        single = harness.executor.execute(single_plan, bm25, lambda event: None)
        self.assertEqual(result.metrics, single.metrics)
        self.assertEqual(harness.dataset_calls, 2)


class ExecutorTimingTests(unittest.TestCase):
    def test_bm25_timing_applicability(self) -> None:
        harness = RecordingExecutor()
        plan = make_plan(make_bm25_variant())
        result = harness.executor.execute(plan, plan.variants[0], lambda event: None)

        for field in (
            "model_load_seconds",
            "doc_cache_load_seconds",
            "corpus_encode_seconds",
            "query_cache_load_seconds",
            "query_encode_seconds",
            "dense_search_total_seconds",
            "rrf_fusion_seconds",
        ):
            self.assertIsNone(result.timing[field], field)

        for field in (
            "dataset_load_seconds",
            "index_build_seconds",
            "bm25_search_total_seconds",
            "retrieval_total_seconds",
            "retrieval_mean_seconds",
            "retrieval_median_seconds",
            "evaluation_seconds",
        ):
            self.assertIsNotNone(result.timing[field], field)
            self.assertGreaterEqual(result.timing[field], 0.0, field)

        self.assertEqual(set(result.timing), set(TIMING_FIELDS))

    def test_dense_timing_applicability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = RecordingExecutor(cache_dir=tmp)
            plan = make_plan(make_dense_variant())
            result = harness.executor.execute(plan, plan.variants[0], lambda event: None)

        self.assertIsNotNone(result.timing["model_load_seconds"])
        self.assertIsNotNone(result.timing["doc_cache_load_seconds"])
        self.assertGreater(result.timing["corpus_encode_seconds"], 0.0)
        self.assertIsNotNone(result.timing["query_cache_load_seconds"])
        self.assertGreater(result.timing["query_encode_seconds"], 0.0)
        self.assertIsNotNone(result.timing["dense_search_total_seconds"])
        self.assertIsNone(result.timing["bm25_search_total_seconds"])
        self.assertIsNone(result.timing["rrf_fusion_seconds"])

    def test_dense_cache_hit_uses_zero_encode_times(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(make_dense_variant())
            first = RecordingExecutor(cache_dir=tmp)
            first.executor.execute(plan, plan.variants[0], lambda event: None)
            second = RecordingExecutor(cache_dir=tmp)
            result = second.executor.execute(plan, plan.variants[0], lambda event: None)

        self.assertEqual(second.encoder.encode_calls, 0)
        self.assertEqual(result.timing["corpus_encode_seconds"], 0.0)
        self.assertEqual(result.timing["query_encode_seconds"], 0.0)
        self.assertIsNotNone(result.timing["doc_cache_load_seconds"])
        self.assertIsNotNone(result.timing["query_cache_load_seconds"])

    def test_hybrid_timing_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            harness = RecordingExecutor(cache_dir=tmp)
            plan = make_plan(make_hybrid_variant())
            result = harness.executor.execute(plan, plan.variants[0], lambda event: None)

        timing = result.timing
        for field in (
            "bm25_search_total_seconds",
            "dense_search_total_seconds",
            "rrf_fusion_seconds",
            "retrieval_total_seconds",
        ):
            self.assertIsNotNone(timing[field], field)
        self.assertTrue(hybrid_total_matches(timing))
        self.assertAlmostEqual(
            timing["retrieval_mean_seconds"] * result.dataset["query_count"],
            timing["retrieval_total_seconds"],
            places=9,
        )
        self.assertLessEqual(
            timing["retrieval_median_seconds"], timing["retrieval_total_seconds"]
        )


class ExecutorCancellationTests(unittest.TestCase):
    def test_cancel_before_stage_does_not_enter_it(self) -> None:
        harness = RecordingExecutor()
        plan = make_plan(make_bm25_variant())
        events = []

        class CancelAt:
            def raise_if_cancelled(self, stage: str) -> None:
                if stage == "retrieve":
                    raise RunCancelled(stage)

        with self.assertRaises(RunCancelled) as context:
            harness.executor.execute(plan, plan.variants[0], events.append, CancelAt())

        self.assertEqual(context.exception.stage, "retrieve")
        sequence = stage_sequence(events)
        self.assertNotIn(("retrieve", "started"), sequence)
        self.assertIn(("build_index", "completed"), sequence)
        self.assertNotIn("evaluate", [stage for stage, _ in sequence])

    def test_cancel_after_stage_end_skips_completion_event(self) -> None:
        harness = RecordingExecutor()
        plan = make_plan(make_bm25_variant())
        events = []

        class CancelAfterFirstCheck:
            def __init__(self) -> None:
                self.hits = 0

            def raise_if_cancelled(self, stage: str) -> None:
                if stage == "retrieve":
                    self.hits += 1
                    if self.hits == 2:
                        raise RunCancelled(stage)

        with self.assertRaises(RunCancelled) as context:
            harness.executor.execute(
                plan, plan.variants[0], events.append, CancelAfterFirstCheck()
            )

        sequence = stage_sequence(events)
        self.assertEqual(context.exception.stage, "retrieve")
        self.assertIn(("retrieve", "started"), sequence)
        self.assertNotIn(("retrieve", "completed"), sequence)
        self.assertNotIn("evaluate", [stage for stage, _ in sequence])

    def test_never_cancel_completes(self) -> None:
        harness = RecordingExecutor()
        plan = make_plan(make_bm25_variant())
        result = harness.executor.execute(
            plan, plan.variants[0], lambda event: None, None
        )
        self.assertEqual(result.stage_progress.completed, applicable_stages("bm25"))


class ExecutorResultTests(unittest.TestCase):
    def test_result_shape(self) -> None:
        harness = RecordingExecutor()
        plan = make_plan(make_bm25_variant())
        result = harness.executor.execute(plan, plan.variants[0], lambda event: None)

        self.assertEqual(result.dataset["name"], "scifact")
        self.assertEqual(result.dataset["split"], "test")
        self.assertEqual(result.dataset["corpus_size"], 3)
        self.assertEqual(result.dataset["query_count"], 3)
        self.assertEqual(result.dataset["qrels_pairs"], 2)
        self.assertEqual(result.variant_config["method"], "bm25")
        self.assertEqual(result.num_queries, 2)
        self.assertEqual(result.num_skipped, 1)
        self.assertIn("MRR", result.metrics)
        self.assertNotIn("source", result.variant_config)
        self.assertNotIn("run_id", result.variant_config)

    def test_environment_has_no_private_data(self) -> None:
        harness = RecordingExecutor()
        plan = make_plan(make_bm25_variant())
        result = harness.executor.execute(plan, plan.variants[0], lambda event: None)

        values = " ".join(
            str(value) for value in result.environment.values() if value is not None
        )
        self.assertNotIn(socket.gethostname(), values)
        self.assertNotIn(getpass.getuser(), values)
        self.assertNotIn(str(PROJECT_ROOT), values)
        self.assertNotIn(str(Path.cwd()), values)
        self.assertIn("python", result.environment)

    def test_results_directory_is_not_written(self) -> None:
        results_dir = PROJECT_ROOT / "results"
        before = sorted(results_dir.iterdir()) if results_dir.exists() else None

        harness = RecordingExecutor()
        plan = make_plan(make_bm25_variant())
        harness.executor.execute(plan, plan.variants[0], lambda event: None)

        after = sorted(results_dir.iterdir()) if results_dir.exists() else None
        self.assertEqual(before, after)


class ExecutorIntegrationTests(unittest.TestCase):
    def test_scifact_bm25_with_default_loader(self) -> None:
        catalog = load_plan_catalog()
        plan = next(
            plan for plan in catalog.plans if plan.plan_id == "scifact-comparison"
        )
        variant = next(
            variant for variant in plan.variants if variant.variant_id == "scifact-bm25"
        )
        executor = ExperimentExecutor(show_progress=False)
        events = []
        result = executor.execute(plan, variant, events.append)

        self.assertEqual(result.num_queries, 300)
        self.assertEqual(result.dataset["corpus_size"], 5183)
        self.assertAlmostEqual(result.metrics["P@1"], 0.5333, places=3)
        self.assertAlmostEqual(result.metrics["MRR"], 0.6312, places=3)
        self.assertEqual(result.stage_progress.not_applicable, ("load_model", "encode_queries"))


if __name__ == "__main__":
    unittest.main()
