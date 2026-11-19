"""对比汇总脚本测试。"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts.compare_experiments import build_table, describe, load_results  # noqa: E402


def write_result(directory: Path, name: str, config: dict, metrics: dict) -> None:
    data = {
        "config": config,
        "metrics": metrics,
        "timing": {"index_seconds": 1.5, "search_mean_seconds": 0.01},
    }
    (directory / name).write_text(json.dumps(data), encoding="utf-8")


class TestCompareExperiments(unittest.TestCase):
    def test_load_and_build_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_result(
                directory,
                "a.json",
                {"method": "bm25", "k1": 1.2, "b": 0.75},
                {"P@1": 0.5, "NDCG@10": 0.6, "R@100": 0.9, "MRR": 0.6},
            )
            write_result(
                directory,
                "b.json",
                {"method": "dense", "model": "model-x"},
                {"P@1": 0.6, "NDCG@10": 0.7, "R@100": 0.95, "MRR": 0.7},
            )
            results = load_results(directory)
            self.assertEqual(len(results), 2)
            table = build_table(results)
            self.assertIn("BM25", table)
            self.assertIn("Dense: model-x", table)
            self.assertIn("0.7000", table)

    def test_skips_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "broken.json").write_text("{not json", encoding="utf-8")
            self.assertEqual(load_results(directory), [])

    def test_describe(self):
        self.assertIn("BM25", describe({"method": "bm25", "k1": 1.2, "b": 0.75}))
        self.assertEqual(describe({"method": "dense", "model": "m"}), "Dense: m")


if __name__ == "__main__":
    unittest.main(verbosity=2)
