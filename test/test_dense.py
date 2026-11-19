"""DenseRetriever 测试：注入假编码器，不加载真实模型。"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from document.dataset import Document  # noqa: E402
from retrieval.dense import DenseRetriever  # noqa: E402


class FakeEncoder:
    """按文本返回预设向量。"""

    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, texts, *, batch_size, show_progress_bar, normalize_embeddings):
        return np.array([self.mapping[text] for text in texts], dtype=np.float32)


def make_docs():
    return [
        Document("d1", "", "alpha"),
        Document("d2", "", "beta"),
        Document("d3", "", "gamma"),
    ]


class TestDenseRetriever(unittest.TestCase):
    def setUp(self):
        mapping = {
            "alpha": (1.0, 0.0),
            "beta": (0.6, 0.8),
            "gamma": (0.0, 1.0),
            "query": (1.0, 0.0),
        }
        self.retriever = DenseRetriever(FakeEncoder(mapping)).fit(make_docs())

    def test_ranking_by_cosine(self):
        results = self.retriever.search("query", top_k=3)
        self.assertEqual([doc_id for doc_id, _ in results], ["d1", "d2", "d3"])
        self.assertAlmostEqual(results[0][1], 1.0)
        self.assertAlmostEqual(results[1][1], 0.6)

    def test_normalizes_non_unit_vectors(self):
        mapping = {"a": (2.0, 0.0), "b": (0.0, 2.0), "q": (2.0, 0.0)}
        retriever = DenseRetriever(FakeEncoder(mapping)).fit(
            [Document("a", "", "a"), Document("b", "", "b")]
        )
        results = retriever.search("q", top_k=2)
        self.assertAlmostEqual(results[0][1], 1.0)
        self.assertAlmostEqual(results[1][1], 0.0)

    def test_top_k_truncation(self):
        self.assertEqual(len(self.retriever.search("query", top_k=1)), 1)

    def test_invalid_top_k(self):
        with self.assertRaises(ValueError):
            self.retriever.search("query", top_k=0)

    def test_search_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            DenseRetriever(FakeEncoder({})).search("query")

    def test_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            DenseRetriever(FakeEncoder({}), batch_size=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
