"""DenseRetriever 测试：注入假编码器，不加载真实模型。"""

import json
import sys
import tempfile
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


class CountingEncoder(FakeEncoder):
    """记录编码调用次数，用于验证缓存。"""

    def __init__(self, mapping):
        super().__init__(mapping)
        self.calls = 0

    def encode(self, *args, **kwargs):
        self.calls += 1
        return super().encode(*args, **kwargs)


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

    def test_cache_hit_avoids_reencode(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping = {"alpha": (1.0, 0.0), "beta": (0.6, 0.8), "gamma": (0.0, 1.0)}
            encoder = CountingEncoder(mapping)
            first = DenseRetriever(encoder, cache_dir=tmp, cache_key="fake").fit(make_docs())
            self.assertEqual(encoder.calls, 1)
            second = DenseRetriever(encoder, cache_dir=tmp, cache_key="fake").fit(make_docs())
            self.assertEqual(encoder.calls, 1)  # 命中缓存，未重新编码
            np.testing.assert_allclose(first.embeddings, second.embeddings)

    def test_cache_miss_on_changed_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping = {
                "alpha": (1.0, 0.0),
                "beta": (0.6, 0.8),
                "gamma": (0.0, 1.0),
                "delta": (1.0, 1.0),
            }
            encoder = CountingEncoder(mapping)
            DenseRetriever(encoder, cache_dir=tmp, cache_key="fake").fit(make_docs())
            extra = make_docs() + [Document("d4", "", "delta")]
            DenseRetriever(encoder, cache_dir=tmp, cache_key="fake").fit(extra)
            self.assertEqual(encoder.calls, 2)

    def test_no_cache_dir_reencodes(self):
        mapping = {"alpha": (1.0, 0.0), "beta": (0.6, 0.8), "gamma": (0.0, 1.0)}
        encoder = CountingEncoder(mapping)
        DenseRetriever(encoder).fit(make_docs())
        DenseRetriever(encoder).fit(make_docs())
        self.assertEqual(encoder.calls, 2)

    def test_cache_hit_matches_miss_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping = {
                "alpha": (1.0, 0.0),
                "beta": (0.6, 0.8),
                "gamma": (0.0, 1.0),
                "query": (1.0, 0.0),
            }
            miss = DenseRetriever(
                CountingEncoder(mapping), cache_dir=tmp, cache_key="fake"
            ).fit(make_docs())
            hit_encoder = CountingEncoder(mapping)
            hit = DenseRetriever(
                hit_encoder, cache_dir=tmp, cache_key="fake"
            ).fit(make_docs())
            self.assertEqual(hit_encoder.calls, 0)  # 文档编码完全命中缓存
            self.assertEqual(miss.search("query", top_k=3), hit.search("query", top_k=3))

    def test_query_cache_hit_matches_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping = {
                "alpha": (1.0, 0.0),
                "beta": (0.6, 0.8),
                "gamma": (0.0, 1.0),
                "q1": (1.0, 0.0),
                "q2": (0.0, 1.0),
            }
            encoder = CountingEncoder(mapping)
            retriever = DenseRetriever(
                encoder, cache_dir=tmp, cache_key="fake"
            ).fit(make_docs())
            first = retriever.encode_queries(["q1", "q2"])
            self.assertEqual(encoder.calls, 2)  # fit 1 次 + 查询编码 1 次
            second = DenseRetriever(
                encoder, cache_dir=tmp, cache_key="fake"
            ).fit(make_docs()).encode_queries(["q1", "q2"])
            self.assertEqual(encoder.calls, 2)  # 文档与查询均命中缓存
            np.testing.assert_allclose(first, second)

    def test_query_cache_miss_on_changed_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping = {
                "alpha": (1.0, 0.0),
                "beta": (0.6, 0.8),
                "gamma": (0.0, 1.0),
                "q1": (1.0, 0.0),
                "q2": (0.0, 1.0),
            }
            encoder = CountingEncoder(mapping)
            retriever = DenseRetriever(
                encoder, cache_dir=tmp, cache_key="fake"
            ).fit(make_docs())
            retriever.encode_queries(["q1"])
            retriever.encode_queries(["q2"])
            self.assertEqual(encoder.calls, 3)  # fit + 两次不同查询

    def test_cache_meta_contains_model_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping = {"alpha": (1.0, 0.0), "beta": (0.6, 0.8), "gamma": (0.0, 1.0)}
            DenseRetriever(
                CountingEncoder(mapping),
                cache_dir=tmp,
                cache_key="fake",
                model_name="fake-model",
                model_revision="abc123",
            ).fit(make_docs())
            metas = list(Path(tmp).glob("*.json"))
            self.assertEqual(len(metas), 1)
            meta = json.loads(metas[0].read_text(encoding="utf-8"))
            self.assertEqual(meta["model_name"], "fake-model")
            self.assertEqual(meta["revision"], "abc123")
            self.assertEqual(meta["version"], 2)

    def test_cache_invalidated_on_revision_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping = {"alpha": (1.0, 0.0), "beta": (0.6, 0.8), "gamma": (0.0, 1.0)}
            encoder = CountingEncoder(mapping)
            DenseRetriever(
                encoder, cache_dir=tmp, cache_key="fake", model_revision="rev1"
            ).fit(make_docs())
            self.assertEqual(encoder.calls, 1)
            DenseRetriever(
                encoder, cache_dir=tmp, cache_key="fake", model_revision="rev1"
            ).fit(make_docs())
            self.assertEqual(encoder.calls, 1)  # 同 revision 命中缓存
            DenseRetriever(
                encoder, cache_dir=tmp, cache_key="fake", model_revision="rev2"
            ).fit(make_docs())
            self.assertEqual(encoder.calls, 2)  # revision 变化 → 缓存失效

    def test_cache_invalidated_on_model_name_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping = {"alpha": (1.0, 0.0), "beta": (0.6, 0.8), "gamma": (0.0, 1.0)}
            encoder = CountingEncoder(mapping)
            DenseRetriever(
                encoder, cache_dir=tmp, cache_key="fake", model_name="m1"
            ).fit(make_docs())
            DenseRetriever(
                encoder, cache_dir=tmp, cache_key="fake", model_name="m1"
            ).fit(make_docs())
            self.assertEqual(encoder.calls, 1)
            DenseRetriever(
                encoder, cache_dir=tmp, cache_key="fake", model_name="m2"
            ).fit(make_docs())
            self.assertEqual(encoder.calls, 2)

    def test_cache_invalidated_on_version_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping = {"alpha": (1.0, 0.0), "beta": (0.6, 0.8), "gamma": (0.0, 1.0)}
            encoder = CountingEncoder(mapping)
            DenseRetriever(encoder, cache_dir=tmp, cache_key="fake").fit(make_docs())
            meta_path = next(Path(tmp).glob("*.json"))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["version"] = 1
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            DenseRetriever(encoder, cache_dir=tmp, cache_key="fake").fit(make_docs())
            self.assertEqual(encoder.calls, 2)

    def test_query_cache_invalidated_on_revision_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping = {
                "alpha": (1.0, 0.0),
                "beta": (0.6, 0.8),
                "gamma": (0.0, 1.0),
                "q1": (1.0, 0.0),
            }
            encoder = CountingEncoder(mapping)
            DenseRetriever(
                encoder, cache_dir=tmp, cache_key="fake", model_revision="rev1"
            ).fit(make_docs()).encode_queries(["q1"])
            self.assertEqual(encoder.calls, 2)  # fit + 查询编码
            DenseRetriever(
                encoder, cache_dir=tmp, cache_key="fake", model_revision="rev1"
            ).fit(make_docs()).encode_queries(["q1"])
            self.assertEqual(encoder.calls, 2)  # 文档与查询均命中
            DenseRetriever(
                encoder, cache_dir=tmp, cache_key="fake", model_revision="rev2"
            ).fit(make_docs()).encode_queries(["q1"])
            self.assertEqual(encoder.calls, 4)  # 文档与查询缓存均失效


if __name__ == "__main__":
    unittest.main(verbosity=2)
