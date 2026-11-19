"""SciFact 数据集加载与校验测试。"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from document.dataset import (  # noqa: E402
    Document,
    Query,
    RetrievalDataset,
    load_scifact,
)

DATA_DIR = ROOT / "data" / "raw" / "scifact"


class TestValidate(unittest.TestCase):
    def test_missing_doc_reference_raises(self):
        dataset = RetrievalDataset(
            documents=[Document("1", "", "a")],
            queries=[Query("1", "q")],
            qrels={"1": {"999": 1}},
        )
        with self.assertRaises(ValueError):
            dataset.validate()

    def test_missing_query_reference_raises(self):
        dataset = RetrievalDataset(
            documents=[Document("1", "", "a")],
            queries=[Query("1", "q")],
            qrels={"2": {"1": 1}},
        )
        with self.assertRaises(ValueError):
            dataset.validate()

    def test_duplicate_doc_id_raises(self):
        dataset = RetrievalDataset(
            documents=[Document("1", "", "a"), Document("1", "", "b")],
            queries=[Query("1", "q")],
            qrels={"1": {"1": 1}},
        )
        with self.assertRaises(ValueError):
            dataset.validate()


class TestSciFact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_scifact(
            DATA_DIR / "corpus.jsonl",
            DATA_DIR / "queries.jsonl",
            DATA_DIR / "qrels" / "test.tsv",
        )

    def test_sizes(self):
        self.assertEqual(len(self.dataset.documents), 5183)
        self.assertEqual(len(self.dataset.queries), 1109)
        self.assertEqual(len(self.dataset.qrels), 300)

    def test_all_ids_are_strings(self):
        self.assertTrue(all(isinstance(doc.doc_id, str) for doc in self.dataset.documents))
        self.assertTrue(all(isinstance(q.query_id, str) for q in self.dataset.queries))

    def test_qrels_references_resolve(self):
        doc_ids = {doc.doc_id for doc in self.dataset.documents}
        for rels in self.dataset.qrels.values():
            self.assertTrue(set(rels) <= doc_ids)

    def test_test_split_is_subset_of_queries(self):
        query_ids = {q.query_id for q in self.dataset.queries}
        self.assertTrue(set(self.dataset.qrels) <= query_ids)

    def test_document_text_not_empty(self):
        self.assertTrue(all(doc.text for doc in self.dataset.documents[:10]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
