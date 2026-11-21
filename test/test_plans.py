"""T-002 方案目录契约测试。"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from jsonschema import Draft202012Validator  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from experiment.plans import (  # noqa: E402
    CATALOG_SCHEMA_PATH,
    DEFAULT_CATALOG_PATH,
    EXPECTED_PLAN_VARIANTS,
    EXPECTED_REFERENCE_VARIANTS,
    REQUIRED_PLAN_IDS,
    REQUIRED_VARIANT_IDS,
    PlanCatalogError,
    ReferenceVariant,
    RunVariant,
    load_plan_catalog,
)

Q64 = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
BGE_M3 = "5617a9f61b028005a4858fdac845db406aefb181"
BGE_SMALL = "7999e1d3359715c523056ef9478215996d62a620"
MINILM = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

QWEN_INSTRUCT = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery:"
)
BGE_M3_PREFIX = "Represent this sentence for searching relevant passages: "
BGE_SMALL_PREFIX = "为这个句子生成表示以用于检索相关文章："


def _raw_catalog() -> dict:
    return json.loads(DEFAULT_CATALOG_PATH.read_text(encoding="utf-8"))


class PlanCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_plan_catalog()

    def _plan(self, plan_id: str):
        for plan in self.catalog.plans:
            if plan.plan_id == plan_id:
                return plan
        raise AssertionError(f"缺少方案 {plan_id}")

    def _variant(self, plan_id: str, variant_id: str):
        for variant in self._plan(plan_id).variants:
            if variant.variant_id == variant_id:
                return variant
        raise AssertionError(f"缺少变体 {variant_id}")

    def _run_variant(self, plan_id: str, variant_id: str) -> RunVariant:
        variant = self._variant(plan_id, variant_id)
        self.assertIsInstance(variant, RunVariant)
        return variant

    def _load_invalid(self, mutate) -> PlanCatalogError:
        data = copy.deepcopy(_raw_catalog())
        mutate(data)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as handle:
            json.dump(data, handle, ensure_ascii=False)
            path = Path(handle.name)
        self.addCleanup(path.unlink)
        with self.assertRaises(PlanCatalogError) as context:
            load_plan_catalog(path)
        return context.exception

    def _find_variants(self, data: dict, variant_id: str) -> list[dict]:
        found = []
        for plan in data["plans"]:
            for variant in plan["variants"]:
                if variant["variant_id"] == variant_id:
                    found.append(variant)
        return found

    # AC-1 / AC-2 -----------------------------------------------------------------

    def test_three_plans_and_versions(self) -> None:
        self.assertEqual(
            {(plan.plan_id, plan.plan_version) for plan in self.catalog.plans},
            {(plan_id, "v1") for plan_id in REQUIRED_PLAN_IDS},
        )

    def test_seventeen_unique_variants(self) -> None:
        run_ids = {
            variant.variant_id
            for plan in self.catalog.plans
            for variant in plan.variants
            if isinstance(variant, RunVariant)
        }
        self.assertEqual(run_ids, REQUIRED_VARIANT_IDS)
        self.assertEqual(len(run_ids), 17)

    def test_variant_counts_per_plan(self) -> None:
        counts = {
            plan.plan_id: (len(plan.variants), sum(
                isinstance(variant, RunVariant) for variant in plan.variants
            ))
            for plan in self.catalog.plans
        }
        self.assertEqual(counts["scifact-comparison"], (9, 9))
        self.assertEqual(counts["duretrieval-comparison"], (4, 4))
        self.assertEqual(counts["duretrieval-rrf-weight-sweep"], (5, 4))

    def test_dataset_evaluation_and_tokens(self) -> None:
        scifact = self._plan("scifact-comparison")
        duretrieval = self._plan("duretrieval-comparison")
        self.assertEqual(
            (scifact.dataset.name, scifact.dataset.language, scifact.dataset.split),
            ("scifact", "en", "test"),
        )
        self.assertEqual(
            (duretrieval.dataset.name, duretrieval.dataset.language,
             duretrieval.dataset.split),
            ("duretrieval", "zh", "dev"),
        )
        for plan in self.catalog.plans:
            self.assertEqual(plan.evaluation.top_k, 100)
            self.assertEqual(plan.evaluation.k_values, (1, 5, 10, 100))

    def test_bm25_parameters(self) -> None:
        scifact = self._run_variant("scifact-comparison", "scifact-bm25")
        duretrieval = self._run_variant("duretrieval-comparison", "duretrieval-bm25")
        self.assertEqual((scifact.k1, scifact.b, scifact.tokenizer), (1.2, 0.75, "simple"))
        self.assertEqual(
            (duretrieval.k1, duretrieval.b, duretrieval.tokenizer), (1.2, 0.75, "jieba")
        )

    def test_dense_model_and_prefixes(self) -> None:
        expectations = {
            ("scifact-comparison", "scifact-dense-qwen3-base"):
                ("Qwen/Qwen3-Embedding-0.6B", "Qwen3-Embedding-0.6B", Q64, ""),
            ("scifact-comparison", "scifact-dense-qwen3-instruct"):
                ("Qwen/Qwen3-Embedding-0.6B", "Qwen3-Embedding-0.6B", Q64, QWEN_INSTRUCT),
            ("scifact-comparison", "scifact-dense-bge-m3-base"):
                ("BAAI/bge-m3", "bge-m3", BGE_M3, ""),
            ("scifact-comparison", "scifact-dense-bge-m3-prefix"):
                ("BAAI/bge-m3", "bge-m3", BGE_M3, BGE_M3_PREFIX),
            ("scifact-comparison", "scifact-dense-minilm"):
                ("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2", MINILM, ""),
            ("duretrieval-comparison", "duretrieval-dense-bge-small-base"):
                ("BAAI/bge-small-zh-v1.5", "bge-small-zh-v1.5", BGE_SMALL, ""),
            ("duretrieval-comparison", "duretrieval-dense-bge-small-prefix"):
                ("BAAI/bge-small-zh-v1.5", "bge-small-zh-v1.5", BGE_SMALL, BGE_SMALL_PREFIX),
        }
        for (plan_id, variant_id), expected in expectations.items():
            variant = self._run_variant(plan_id, variant_id)
            self.assertEqual(
                (variant.model.name, variant.model.directory_name,
                 variant.model.revision, variant.query_prefix),
                expected,
            )
            self.assertEqual(variant.batch_size, 32)
            self.assertTrue(variant.normalize_embeddings)

    def test_prefix_preserves_whitespace_and_newline(self) -> None:
        instruct = self._run_variant(
            "scifact-comparison", "scifact-dense-qwen3-instruct"
        ).query_prefix
        self.assertEqual(instruct, QWEN_INSTRUCT)
        self.assertIn("\n", instruct)
        self.assertTrue(instruct.endswith("Query:"))

        bge_m3 = self._run_variant(
            "scifact-comparison", "scifact-dense-bge-m3-prefix"
        ).query_prefix
        self.assertTrue(bge_m3.endswith(": "))

        bge_small = self._run_variant(
            "duretrieval-comparison", "duretrieval-dense-bge-small-prefix"
        ).query_prefix
        self.assertEqual(bge_small, BGE_SMALL_PREFIX)
        self.assertTrue(bge_small.endswith("："))

    def test_hybrid_rrf_configuration(self) -> None:
        expectations = {
            "scifact-hybrid-qwen3-base": [1.0, 1.0],
            "scifact-hybrid-qwen3-instruct": [1.0, 1.0],
            "scifact-hybrid-qwen3-instruct-w0.5": [0.5, 1.0],
            "duretrieval-hybrid-bge-small-prefix-equal": [1.0, 1.0],
            "duretrieval-hybrid-bge-small-prefix-w0.5": [0.5, 1.0],
            "duretrieval-hybrid-bge-small-prefix-w0.3": [0.3, 1.0],
            "duretrieval-hybrid-bge-small-prefix-w0.2": [0.2, 1.0],
            "duretrieval-hybrid-bge-small-prefix-w0.1": [0.1, 1.0],
        }
        for variant_id, weights in expectations.items():
            for plan in self.catalog.plans:
                for variant in plan.variants:
                    if variant.variant_id == variant_id and isinstance(variant, RunVariant):
                        self.assertEqual(variant.method, "hybrid")
                        self.assertEqual(variant.rrf_k, 60)
                        self.assertEqual(list(variant.rrf_weights), weights)
                        return
        self.fail("未找到 Hybrid 变体")

    def test_preset_mapping(self) -> None:
        expectations = {
            "scifact-bm25": "scifact_bm25_20261120_002601.json",
            "scifact-dense-qwen3-instruct":
                "scifact_dense_prefix_eec03840_20261120_131037.json",
            "scifact-hybrid-qwen3-instruct-w0.5":
                "scifact_hybrid_prefix_eec03840_w0.5_1_20261120_213538.json",
            "duretrieval-bm25": "duretrieval_bm25_20261120_134645.json",
            "duretrieval-dense-bge-small-base":
                "duretrieval_dense_20261120_224242.json",
            "duretrieval-hybrid-bge-small-prefix-equal":
                "duretrieval_hybrid_prefix_802902f6_20261120_140609.json",
            "duretrieval-hybrid-bge-small-prefix-w0.1":
                "duretrieval_hybrid_prefix_802902f6_w0.1_1_20261120_213527.json",
        }
        for variant_id, filename in expectations.items():
            for plan in self.catalog.plans:
                for variant in plan.variants:
                    if variant.variant_id == variant_id and isinstance(variant, RunVariant):
                        self.assertEqual(variant.preset_run_id, f"preset-{variant_id}")
                        self.assertEqual(
                            variant.preset_source, f"data/evaluation/{filename}"
                        )
                        self.assertNotIn(filename, ("scifact_bm25_20261120_004326.json",))
                        return
        self.fail("未找到预置变体")

    def test_reference_variant_reuses_equal_result(self) -> None:
        sweep = self._plan("duretrieval-rrf-weight-sweep")
        references = [
            variant for variant in sweep.variants
            if isinstance(variant, ReferenceVariant)
        ]
        self.assertEqual(len(references), 1)
        reference = references[0]
        self.assertEqual(
            reference.variant_id, "duretrieval-hybrid-bge-small-prefix-equal"
        )
        self.assertEqual(reference.source_plan_id, "duretrieval-comparison")
        self.assertEqual(reference.source_plan_version, "v1")
        self.assertEqual(reference.source_variant_id, reference.variant_id)

    def test_catalog_is_immutable(self) -> None:
        with self.assertRaises(ValidationError):
            self.catalog.plans[0].name = "changed"
        with self.assertRaises(ValidationError):
            self.catalog.plans[0].variants[0].variant_id = "changed"

    def test_schema_is_draft_2020_12_and_accepts_catalog(self) -> None:
        schema = json.loads(CATALOG_SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["$schema"], "https://json-schema.org/draft/2020-12/schema"
        )
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(_raw_catalog())

    # AC-3 ------------------------------------------------------------------------

    def test_plan_membership_matches_expected(self) -> None:
        for plan in self.catalog.plans:
            run_ids = {
                variant.variant_id
                for variant in plan.variants
                if isinstance(variant, RunVariant)
            }
            reference_ids = {
                variant.variant_id
                for variant in plan.variants
                if isinstance(variant, ReferenceVariant)
            }
            self.assertEqual(run_ids, EXPECTED_PLAN_VARIANTS[plan.plan_id])
            self.assertEqual(
                reference_ids, EXPECTED_REFERENCE_VARIANTS[plan.plan_id]
            )

    def test_duplicate_plan_is_rejected(self) -> None:
        def mutate(data: dict) -> None:
            data["plans"][1]["plan_id"] = data["plans"][0]["plan_id"]

        error = self._load_invalid(mutate)
        self.assertIn("重复", str(error))

    def test_swapped_variants_between_plans_are_rejected(self) -> None:
        """数量不变但把变体放进错误方案，也必须拒绝。"""

        def mutate(data: dict) -> None:
            scifact = data["plans"][0]["variants"]
            duretrieval = data["plans"][1]["variants"]
            scifact[0], duretrieval[0] = duretrieval[0], scifact[0]

        error = self._load_invalid(mutate)
        self.assertIn("运行变体集合不符", str(error))

    def test_reference_in_wrong_plan_is_rejected(self) -> None:
        def mutate(data: dict) -> None:
            sweep = data["plans"][2]["variants"]
            reference = sweep.pop()
            data["plans"][0]["variants"].append(reference)

        error = self._load_invalid(mutate)
        self.assertIn("参照项集合不符", str(error))

    def test_duplicate_variant_in_plan_is_rejected(self) -> None:
        def mutate(data: dict) -> None:
            plan = data["plans"][0]
            plan["variants"].append(copy.deepcopy(plan["variants"][0]))

        error = self._load_invalid(mutate)
        self.assertIn("方案内重复变体", str(error))

    def test_missing_required_variant_is_rejected(self) -> None:
        def mutate(data: dict) -> None:
            data["plans"][0]["variants"].pop()

        error = self._load_invalid(mutate)
        self.assertIn("缺少必需变体", str(error))

    def test_invalid_bm25_b_is_rejected(self) -> None:
        def mutate(data: dict) -> None:
            self._find_variants(data, "scifact-bm25")[0]["b"] = 1.5

        error = self._load_invalid(mutate)
        self.assertIn("plans[0].variants[0].b", str(error))

    def test_missing_dense_model_is_rejected(self) -> None:
        def mutate(data: dict) -> None:
            del self._find_variants(data, "scifact-dense-qwen3-base")[0]["model"]

        error = self._load_invalid(mutate)
        self.assertIn("model", str(error))

    def test_method_incompatible_field_is_rejected(self) -> None:
        def mutate(data: dict) -> None:
            self._find_variants(data, "scifact-bm25")[0]["model"] = {
                "name": "x", "directory_name": "x", "revision": None,
            }

        error = self._load_invalid(mutate)
        self.assertIn("model", str(error))

    def test_invalid_rrf_weights_are_rejected(self) -> None:
        for weights in ([0.5], [0.0, 0.0], [-0.1, 1.0]):
            with self.subTest(weights=weights):
                def mutate(data: dict) -> None:
                    variant = self._find_variants(
                        data, "scifact-hybrid-qwen3-instruct-w0.5"
                    )[0]
                    variant["rrf_weights"] = weights

                self._load_invalid(mutate)

    def test_top_k_smaller_than_max_k_is_rejected(self) -> None:
        def mutate(data: dict) -> None:
            data["plans"][0]["evaluation"]["top_k"] = 10

        error = self._load_invalid(mutate)
        self.assertIn("top_k", str(error))

    def test_reference_to_missing_variant_is_rejected(self) -> None:
        def mutate(data: dict) -> None:
            data["plans"][2]["variants"][-1]["source_variant_id"] = "missing-variant"

        error = self._load_invalid(mutate)
        self.assertIn("参照来源不存在", str(error))

    def test_extra_field_is_rejected(self) -> None:
        def mutate(data: dict) -> None:
            data["plans"][0]["variants"][0]["unknown_field"] = 1

        error = self._load_invalid(mutate)
        self.assertIn("unknown_field", str(error))

    def test_missing_file_and_invalid_json_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.json"
            with self.assertRaises(PlanCatalogError):
                load_plan_catalog(missing)

            broken = Path(tmp) / "broken.json"
            broken.write_text("{not json", encoding="utf-8")
            with self.assertRaises(PlanCatalogError) as context:
                load_plan_catalog(broken)
            self.assertIn("JSON", str(context.exception))

    def test_default_path_is_project_config(self) -> None:
        self.assertEqual(DEFAULT_CATALOG_PATH.name, "plans.json")
        self.assertEqual(DEFAULT_CATALOG_PATH.parent.name, "config")


if __name__ == "__main__":
    unittest.main()
