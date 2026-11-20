"""错误分析诊断函数测试。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.error_analysis import diagnose, snippet  # noqa: E402


class TestDiagnose(unittest.TestCase):
    def test_hit_at_1(self):
        self.assertEqual(diagnose([("a", 1.0), ("b", 2.0)], {"a"}), "hit_at_1")

    def test_ranked_low(self):
        self.assertEqual(diagnose([("x", 1.0), ("a", 2.0)], {"a"}), "ranked_low")

    def test_miss_top_k(self):
        self.assertEqual(diagnose([("x", 1.0), ("y", 2.0)], {"a"}), "miss_top_k")

    def test_empty(self):
        self.assertEqual(diagnose([], {"a"}), "empty")

    def test_k_boundary(self):
        ranked = [("x", 1.0)] * 10 + [("a", 9.0)]
        self.assertEqual(diagnose(ranked, {"a"}, k=10), "miss_top_k")

    def test_snippet_truncation(self):
        self.assertEqual(snippet("a" * 150), "a" * 100 + "…")


if __name__ == "__main__":
    unittest.main(verbosity=2)
