"""评价指标手算样例测试。"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.metrics import (  # noqa: E402
    evaluate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestPrecisionRecall(unittest.TestCase):
    def test_precision_at_k(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "c"}
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, 1), 1.0)
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, 2), 0.5)
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, 4), 0.5)
        # 检索结果不足 K 时按 TREC 口径以 K 为分母
        self.assertAlmostEqual(precision_at_k(retrieved[:2], relevant, 4), 0.25)

    def test_recall_at_k(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = {"a", "c", "e"}
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, 1), 1 / 3)
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, 4), 2 / 3)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            precision_at_k(["a"], {"a"}, 0)


class TestMRR(unittest.TestCase):
    def test_first_hit_at_first_position(self):
        self.assertAlmostEqual(reciprocal_rank(["a", "b"], {"a"}), 1.0)

    def test_first_hit_at_third_position(self):
        self.assertAlmostEqual(reciprocal_rank(["x", "y", "a"], {"a"}), 1 / 3)

    def test_no_hit(self):
        self.assertAlmostEqual(reciprocal_rank(["x", "y"], {"a"}), 0.0)


class TestNDCG(unittest.TestCase):
    def test_binary_gain(self):
        # dcg = 1；idcg = 1 + 1/log2(3)
        value = ndcg_at_k(["a", "b"], {"a": 1, "c": 1}, 2)
        expected = 1.0 / (1.0 + 1.0 / math.log2(3))
        self.assertAlmostEqual(value, expected, places=6)

    def test_perfect_ranking(self):
        self.assertAlmostEqual(ndcg_at_k(["a", "b"], {"a": 1, "b": 1}, 2), 1.0)

    def test_no_relevant(self):
        self.assertAlmostEqual(ndcg_at_k(["a"], {}, 2), 0.0)


class TestEvaluate(unittest.TestCase):
    def test_aggregate_and_skip(self):
        retrieved = {
            "q1": ["a", "b"],
            "q2": ["x"],
            "q3": ["m", "n"],
        }
        qrels = {
            "q1": {"a": 1},
            "q2": {"x": 1},
            "q3": {},  # 无相关标注，应被跳过
        }
        result = evaluate(retrieved, qrels, ks=(1, 2))
        self.assertEqual(result["num_queries"], 2)
        self.assertEqual(result["num_skipped"], 1)
        self.assertAlmostEqual(result["metrics"]["P@1"], 1.0)
        self.assertAlmostEqual(result["metrics"]["MRR"], 1.0)

    def test_empty_retrieved_raises(self):
        with self.assertRaises(ValueError):
            evaluate({}, {})

    def test_invalid_ks_rejected(self):
        with self.assertRaises(ValueError):
            evaluate({"q": ["a"]}, {"q": {"a": 1}}, ks=())
        with self.assertRaises(ValueError):
            evaluate({"q": ["a"]}, {"q": {"a": 1}}, ks=(0, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
