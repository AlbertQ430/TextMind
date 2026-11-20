"""BM25 检索器测试：手算分数与排序。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from document.dataset import Document  # noqa: E402
from retrieval.bm25 import BM25Retriever, jieba_tokenize, simple_tokenize  # noqa: E402


def make_corpus():
    return [
        Document(doc_id="d1", title="", text="apple apple banana"),
        Document(doc_id="d2", title="", text="apple banana"),
        Document(doc_id="d3", title="", text="cherry"),
    ]


class TestBM25(unittest.TestCase):
    def setUp(self):
        # N=3, avgdl=2, k1=1.2, b=0.75
        self.retriever = BM25Retriever().fit(make_corpus())

    def test_term_weights_and_order(self):
        # "apple" df=2, idf=ln(1.6)；d1 词频更高但更长，仍应排在 d2 之前
        results = self.retriever.search("apple", top_k=2)
        self.assertEqual([doc_id for doc_id, _ in results], ["d1", "d2"])
        # 手算: denom(d1) = 2 + 1.2*(0.25 + 0.75*1.5) = 3.65 → 0.4700*4.4/3.65
        self.assertAlmostEqual(results[0][1], 0.4700 * 4.4 / 3.65, places=4)
        self.assertAlmostEqual(results[1][1], 0.4700, places=4)

    def test_document_length_normalization(self):
        # "banana" 在 d2（更短）上应高于 d1（更长）
        results = self.retriever.search("banana", top_k=2)
        self.assertEqual([doc_id for doc_id, _ in results], ["d2", "d1"])

    def test_rare_term_scores_higher(self):
        # "cherry" 只在 d3 出现，idf 更高
        results = self.retriever.search("cherry", top_k=1)
        self.assertEqual(results[0][0], "d3")
        self.assertAlmostEqual(results[0][1], 0.9808 * 2.2 / 1.75, places=3)

    def test_top_k_truncation(self):
        results = self.retriever.search("apple banana", top_k=1)
        self.assertEqual(len(results), 1)

    def test_no_match_returns_empty(self):
        self.assertEqual(self.retriever.search("zebra", top_k=5), [])

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            BM25Retriever(k1=-1)
        with self.assertRaises(ValueError):
            BM25Retriever(b=1.5)
        with self.assertRaises(ValueError):
            self.retriever.search("apple", top_k=0)

    def test_search_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            BM25Retriever().search("apple")

    def test_tokenizer(self):
        self.assertEqual(simple_tokenize("Hello, World! 2026"), ["hello", "world", "2026"])

    def test_jieba_tokenize(self):
        tokens = jieba_tokenize("人工智能正在改变自然语言处理技术。")
        self.assertTrue(tokens)
        self.assertIn("人工智能", tokens)

    def test_jieba_tokenizer_injected(self):
        corpus = [
            Document("d1", "", "苹果手机很好用"),
            Document("d2", "", "苹果是一种水果"),
        ]
        retriever = BM25Retriever(tokenizer=jieba_tokenize).fit(corpus)
        results = retriever.search("苹果手机", top_k=2)
        self.assertEqual(results[0][0], "d1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
