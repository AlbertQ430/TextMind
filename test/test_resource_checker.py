"""T-009 资源检查测试。"""

from __future__ import annotations

import json
import shutil
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
from resources.checker import (  # noqa: E402
    ResourceChecker,
    ResourceUnavailableError,
    environment_summary,
)
from resources.settings import Settings  # noqa: E402

CATALOG = load_plan_catalog()
QWEEN = "Qwen3-Embedding-0.6B"


class CheckerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.scifact = self.root / "scifact"
        self.write_scifact()

    def write_scifact(self) -> None:
        for relative in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
            path = self.scifact / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("data\n", encoding="utf-8")

    def make_model(self, directory_name: str = QWEEN, **files) -> Path:
        directory = self.root / "models" / directory_name
        directory.mkdir(parents=True, exist_ok=True)
        defaults = {
            "config.json": "{}",
            "model.safetensors": "weights",
            "tokenizer.json": "{}",
        }
        defaults.update(files)
        for name, content in defaults.items():
            if content is None:
                continue
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return directory

    def build(
        self,
        *,
        model_root: str | None = None,
        duretrieval_root: str | None = None,
        device: str = "auto",
        device_probe=None,
    ) -> ResourceChecker:
        settings = Settings(
            model_root=model_root,
            duretrieval_root=duretrieval_root,
            device=device,
        )
        return ResourceChecker(
            CATALOG,
            lambda: settings,
            device_probe=device_probe or (lambda name: name != "cuda"),
            scifact_dir=self.scifact,
        )


class DatasetTests(CheckerTestCase):
    def test_scifact_available_and_missing(self) -> None:
        checker = self.build()
        state = checker.dataset_state("scifact")
        self.assertEqual(state.state, "available")
        self.assertIn("test", state.summary)

        (self.scifact / "queries.jsonl").unlink()
        state = checker.dataset_state("scifact")
        self.assertEqual(state.state, "missing")
        self.assertIn("scifact:queries.jsonl", state.missing[0])

    def test_empty_file_is_invalid(self) -> None:
        (self.scifact / "corpus.jsonl").write_text("", encoding="utf-8")
        state = self.build().dataset_state("scifact")
        self.assertEqual(state.state, "invalid")

    def test_duretrieval_states(self) -> None:
        checker = self.build()
        state = checker.dataset_state("duretrieval")
        self.assertEqual(state.state, "not_configured")
        self.assertIn("dev", state.summary)

        missing_root = self.root / "missing-duretrieval"
        state = self.build(duretrieval_root=str(missing_root)).dataset_state(
            "duretrieval"
        )
        self.assertEqual(state.state, "missing")

        root = self.root / "duretrieval"
        root.mkdir()
        (root / "corpus.jsonl").write_text("x", encoding="utf-8")
        state = self.build(duretrieval_root=str(root)).dataset_state(
            "duretrieval"
        )
        self.assertEqual(state.state, "missing")
        self.assertIn("dev", state.summary)
        self.assertTrue(
            any("queries.jsonl" in item for item in state.missing)
        )


class ModelTests(CheckerTestCase):
    def test_not_configured_and_missing(self) -> None:
        checker = self.build()
        state = checker.model_state(CATALOG.plans[0].variants[1].model)
        self.assertEqual(state.state, "not_configured")

        state = self.build(model_root=str(self.root / "models")).model_state(
            CATALOG.plans[0].variants[1].model
        )
        self.assertEqual(state.state, "missing")

    def test_available_and_reason_without_absolute_path(self) -> None:
        self.make_model()
        checker = self.build(model_root=str(self.root / "models"))
        model = CATALOG.plans[0].variants[1].model
        state = checker.model_state(model)
        self.assertEqual(state.state, "available")

        (self.root / "models" / QWEEN / "tokenizer.json").unlink()
        state = checker.model_state(model)
        self.assertEqual(state.state, "missing")
        dumped = json.dumps(state.as_dict(), ensure_ascii=False)
        self.assertNotIn(str(self.root), dumped)

    def test_empty_config_and_weights(self) -> None:
        self.make_model(**{"config.json": ""})
        checker = self.build(model_root=str(self.root / "models"))
        model = CATALOG.plans[0].variants[1].model
        state = checker.model_state(model)
        self.assertEqual(state.state, "invalid")

        shutil.rmtree(self.root / "models" / QWEEN)
        self.make_model(**{"model.safetensors": None, "tokenizer.json": None})
        state = checker.model_state(model)
        self.assertEqual(state.state, "missing")

    def test_sharded_index_requires_shards(self) -> None:
        self.make_model(
            **{
                "model.safetensors": None,
                "model.safetensors.index.json": json.dumps(
                    {"weight_map": {"a": "shard-1.safetensors"}}
                ),
            }
        )
        checker = self.build(model_root=str(self.root / "models"))
        model = CATALOG.plans[0].variants[1].model
        state = checker.model_state(model)
        self.assertEqual(state.state, "missing")
        self.assertTrue(
            any("shard-1.safetensors" in item for item in state.missing)
        )

        (self.root / "models" / QWEEN / "shard-1.safetensors").write_text(
            "w", encoding="utf-8"
        )
        self.assertEqual(checker.model_state(model).state, "available")

    def test_zero_byte_weight_not_available(self) -> None:
        self.make_model(**{"model.safetensors": ""})
        checker = self.build(model_root=str(self.root / "models"))
        model = CATALOG.plans[0].variants[1].model
        state = checker.model_state(model)
        self.assertNotEqual(state.state, "available")
        self.assertTrue(
            any("model.safetensors" in item for item in state.missing)
        )

    def test_unreadable_weight_not_available(self) -> None:
        self.make_model()
        checker = self.build(model_root=str(self.root / "models"))
        model = CATALOG.plans[0].variants[1].model
        from resources import checker as checker_module

        real = checker_module._file_reason

        def fake(path):
            if path.name == "model.safetensors":
                return "不可读"
            return real(path)

        with mock.patch("resources.checker._file_reason", side_effect=fake):
            state = checker.model_state(model)
        self.assertNotEqual(state.state, "available")

    def test_tokenizer_config_alternative(self) -> None:
        self.make_model(
            **{
                "tokenizer.json": None,
                "tokenizer_config.json": "{}",
                "vocab.txt": "a\nb\n",
            }
        )
        checker = self.build(model_root=str(self.root / "models"))
        model = CATALOG.plans[0].variants[1].model
        self.assertEqual(checker.model_state(model).state, "available")


class DeviceTests(CheckerTestCase):
    def test_auto_and_cpu_available(self) -> None:
        checker = self.build(device="auto")
        device = checker.device_state()
        self.assertTrue(device.available)
        self.assertIn(device.effective, {"cpu", "cuda", "mps"})

        device = self.build(device="cpu", device_probe=lambda name: False).device_state()
        self.assertTrue(device.available)
        self.assertEqual(device.effective, "cpu")

    def test_explicit_unavailable_does_not_fallback(self) -> None:
        checker = self.build(device="cuda", device_probe=lambda name: False)
        device = checker.device_state()
        self.assertFalse(device.available)
        self.assertEqual(device.effective, "cuda")
        self.assertIn("CUDA", device.summary)

    def test_explicit_available(self) -> None:
        checker = self.build(device="mps", device_probe=lambda name: True)
        self.assertTrue(checker.device_state().available)


class VariantTests(CheckerTestCase):
    def plan(self):
        return CATALOG.plans[0]

    def variant(self, variant_id: str):
        return next(
            item for item in self.plan().variants if item.variant_id == variant_id
        )

    def test_bm25_needs_only_dataset(self) -> None:
        checker = self.build()
        state = checker.variant_state(self.plan(), self.variant("scifact-bm25"))
        self.assertEqual(state["state"], "available")
        self.assertEqual(state["summary"], "资源可用")

    def test_dense_requires_model_and_device(self) -> None:
        self.make_model()
        checker = self.build(model_root=str(self.root / "models"))
        state = checker.variant_state(
            self.plan(), self.variant("scifact-dense-qwen3-base")
        )
        self.assertEqual(state["state"], "available")

        state = self.build(device="cuda", device_probe=lambda name: False).variant_state(
            self.plan(), self.variant("scifact-dense-qwen3-base")
        )
        self.assertEqual(state["state"], "unavailable")
        self.assertIn("device:cuda", state["missing"])

    def test_reference_variant(self) -> None:
        sweep = CATALOG.plans[2]
        reference = next(
            item for item in sweep.variants if item.kind == "reference"
        )
        state = self.build().variant_state(sweep, reference)
        self.assertEqual(state["state"], "not_applicable")
        self.assertEqual(state["summary"], "参照结果，无需运行资源")

    def test_runnable_count_and_require(self) -> None:
        checker = self.build()
        self.assertEqual(checker.plan_runnable_count(self.plan()), 1)

        state = checker.variant_state(
            self.plan(), self.variant("scifact-dense-qwen3-base")
        )
        self.assertEqual(state["state"], "unavailable")
        with self.assertRaises(ResourceUnavailableError) as context:
            checker.require_available(
                self.plan(), self.variant("scifact-dense-qwen3-base")
            )
        self.assertEqual(context.exception.missing, state["missing"])

    def test_snapshot_structure(self) -> None:
        self.make_model()
        snapshot = self.build(model_root=str(self.root / "models")).snapshot()
        self.assertEqual(
            [dataset["id"] for dataset in snapshot["datasets"]],
            ["scifact", "duretrieval"],
        )
        self.assertEqual(
            {model["id"] for model in snapshot["models"]},
            {QWEEN, "bge-m3", "all-MiniLM-L6-v2", "bge-small-zh-v1.5"},
        )
        self.assertIn("device", snapshot)


class EnvironmentTests(unittest.TestCase):
    def test_environment_summary(self) -> None:
        summary = environment_summary()
        self.assertIn("python", summary)
        self.assertIn("platform", summary)
        self.assertIn("fastapi", summary["dependencies"])
        self.assertNotIn("hostname", summary)


if __name__ == "__main__":
    unittest.main()
