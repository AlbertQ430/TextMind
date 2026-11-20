"""图表标签测试。"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts.plot_results import short_label  # noqa: E402


class TestShortLabel(unittest.TestCase):
    def test_bm25(self):
        self.assertEqual(short_label({"method": "bm25"}), "BM25")

    def test_dense(self):
        self.assertEqual(
            short_label({"method": "dense", "model": "Qwen/Qwen3-Embedding-0.6B"}),
            "Dense (Qwen3-0.6B)",
        )
        self.assertEqual(
            short_label({"method": "dense", "model": "BAAI/bge-m3"}),
            "Dense (bge-m3)",
        )

    def test_hybrid_with_prefix(self):
        self.assertEqual(
            short_label({"method": "hybrid", "model": "BAAI/bge-m3", "query_prefix": "x"}),
            "Hybrid (bge-m3) +prefix",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
