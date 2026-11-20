"""HybridRetriever 测试：RRF 融合逻辑。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retrieval.base import Retriever  # noqa: E402
from retrieval.hybrid import HybridRetriever  # noqa: E402


class FakeRetriever(Retriever):
    def __init__(self, rankings):
        self.rankings = rankings
        self.fit_calls = 0

    def fit(self, documents):
        self.fit_calls += 1
        return self

    def search(self, query, top_k=10):
        return self.rankings.get(query, [])[:top_k]


class TestHybridRetriever(unittest.TestCase):
    def test_rrf_fusion_order(self):
        first = FakeRetriever({"q": [("a", 10.0), ("b", 8.0), ("c", 5.0)]})
        second = FakeRetriever({"q": [("b", 0.9), ("a", 0.8), ("d", 0.7)]})
        hybrid = HybridRetriever([first, second], rrf_k=60)
        results = hybrid.search("q", top_k=4)
        # a: 1/61+1/62 ≈ b: 1/62+1/61；并列时按 doc_id 升序
        self.assertEqual([doc_id for doc_id, _ in results], ["a", "b", "c", "d"])
        self.assertAlmostEqual(results[0][1], 1 / 61 + 1 / 62)

    def test_doc_only_in_one_list(self):
        first = FakeRetriever({"q": [("a", 1.0), ("b", 1.0)]})
        second = FakeRetriever({"q": [("c", 1.0)]})
        results = HybridRetriever([first, second]).search("q", top_k=3)
        scores = {doc_id: score for doc_id, score in results}
        self.assertAlmostEqual(scores["a"], 1 / 61)
        self.assertAlmostEqual(scores["c"], 1 / 61)
        self.assertAlmostEqual(scores["a"], scores["c"])

    def test_top_k_truncation(self):
        first = FakeRetriever({"q": [("a", 1.0), ("b", 1.0), ("c", 1.0)]})
        hybrid = HybridRetriever([first])
        self.assertEqual(len(hybrid.search("q", top_k=2)), 2)

    def test_fit_propagates(self):
        first, second = FakeRetriever({}), FakeRetriever({})
        HybridRetriever([first, second]).fit([object()])
        self.assertEqual((first.fit_calls, second.fit_calls), (1, 1))

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            HybridRetriever([])
        with self.assertRaises(ValueError):
            HybridRetriever([FakeRetriever({})], rrf_k=0)
        with self.assertRaises(ValueError):
            HybridRetriever([FakeRetriever({})], candidate_depth=0)
        with self.assertRaises(ValueError):
            HybridRetriever([FakeRetriever({})]).search("q", top_k=0)

    def test_missing_query_returns_empty(self):
        hybrid = HybridRetriever([FakeRetriever({}), FakeRetriever({})])
        self.assertEqual(hybrid.search("unknown", top_k=5), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
