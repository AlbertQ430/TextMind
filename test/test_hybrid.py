"""HybridRetriever 测试：RRF 融合、查询前缀分发、批量计时。"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retrieval.base import Retriever  # noqa: E402
from retrieval.hybrid import HybridRetriever  # noqa: E402


class FakeRetriever(Retriever):
    def __init__(self, rankings):
        self.rankings = rankings
        self.fit_calls = 0
        self.received_queries = []

    def fit(self, documents):
        self.fit_calls += 1
        return self

    def search(self, query, top_k=10):
        self.received_queries.append(query)
        return self.rankings.get(query, [])[:top_k]


class FakeDense(FakeRetriever):
    """模拟 Dense 检索器：支持批量编码与向量检索。"""

    def encode_queries(self, queries):
        return np.ones((len(queries), 2), dtype=np.float32)

    def search_vector(self, vector, top_k=10):
        return [("v", 1.0)][:top_k]


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

    def test_query_transforms_per_retriever(self):
        bm25 = FakeRetriever({"claim": [("a", 1.0)]})
        dense = FakeRetriever({"PREFIX claim": [("b", 1.0)]})
        hybrid = HybridRetriever(
            [bm25, dense],
            query_transforms=[None, lambda query: "PREFIX " + query],
        )
        results = hybrid.search("claim", top_k=2)
        # BM25 收到原始 query，Dense 收到加前缀的 query
        self.assertEqual(bm25.received_queries, ["claim"])
        self.assertEqual(dense.received_queries, ["PREFIX claim"])
        self.assertEqual([doc_id for doc_id, _ in results], ["a", "b"])

    def test_transform_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            HybridRetriever([FakeRetriever({})], query_transforms=[None, None])

    def test_search_batch_matches_single_search(self):
        first = FakeRetriever({"q1": [("a", 1.0), ("b", 1.0)], "q2": [("c", 1.0)]})
        second = FakeRetriever({"q1": [("b", 1.0)], "q2": [("c", 1.0), ("d", 1.0)]})
        hybrid = HybridRetriever([first, second], rrf_k=60)
        single = [hybrid.search("q1", 2), hybrid.search("q2", 2)]
        batch = hybrid.search_batch(["q1", "q2"], top_k=2)
        self.assertEqual(batch, single)

    def test_search_batch_timing_keys(self):
        hybrid = HybridRetriever([FakeRetriever({"q": [("a", 1.0)]})])
        hybrid.search_batch(["q"], top_k=2)
        timing = hybrid.last_batch_timing
        for key in ("query_encode", "bm25_search", "dense_search", "rrf_fusion"):
            self.assertIn(key, timing)
        self.assertEqual(len(timing["per_query_end_to_end"]), 1)

    def test_search_batch_dense_path(self):
        hybrid = HybridRetriever([FakeRetriever({"q": [("a", 1.0)]}), FakeDense({})])
        results = hybrid.search_batch(["q"], top_k=2)
        self.assertEqual([doc_id for doc_id, _ in results[0]], ["a", "v"])
        self.assertGreaterEqual(hybrid.last_batch_timing["dense_search"], 0.0)

    def test_weighted_rrf(self):
        first = FakeRetriever({"q": [("a", 1.0)]})
        second = FakeRetriever({"q": [("b", 1.0)]})
        hybrid = HybridRetriever([first, second], rrf_k=60, weights=[0.5, 1.0])
        results = hybrid.search("q", top_k=2)
        scores = {doc_id: score for doc_id, score in results}
        self.assertAlmostEqual(scores["a"], 0.5 / 61)
        self.assertAlmostEqual(scores["b"], 1.0 / 61)
        self.assertEqual([doc_id for doc_id, _ in results], ["b", "a"])

    def test_invalid_weights(self):
        with self.assertRaises(ValueError):
            HybridRetriever([FakeRetriever({}), FakeRetriever({})], weights=[1.0])
        with self.assertRaises(ValueError):
            HybridRetriever([FakeRetriever({})], weights=[-1.0])
        with self.assertRaises(ValueError):
            HybridRetriever([FakeRetriever({})], weights=[float("nan")])
        with self.assertRaises(ValueError):
            HybridRetriever([FakeRetriever({})], weights=[float("inf")])
        with self.assertRaises(ValueError):
            HybridRetriever([FakeRetriever({}), FakeRetriever({})], weights=[0.0, 0.0])

    def test_rrf_fuse_weight_length_mismatch(self):
        with self.assertRaises(ValueError):
            HybridRetriever.rrf_fuse([[("a", 1.0)]], 60, 1, weights=[1.0, 1.0])

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
