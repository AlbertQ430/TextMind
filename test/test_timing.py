"""T-003 计时记录器测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment.timing import (  # noqa: E402
    HYBRID_SUBTOTAL_FIELDS,
    TIMING_FIELDS,
    DurationRecorder,
    hybrid_subtotal,
    hybrid_total_matches,
)


class DurationRecorderTests(unittest.TestCase):
    def test_initial_values_are_all_null(self) -> None:
        recorder = DurationRecorder()
        snapshot = recorder.snapshot()
        self.assertEqual(set(snapshot), set(TIMING_FIELDS))
        self.assertEqual(len(TIMING_FIELDS), 14)
        self.assertTrue(all(value is None for value in snapshot.values()))

    def test_record_and_add(self) -> None:
        recorder = DurationRecorder()
        recorder.record("dataset_load_seconds", 0.5)
        recorder.add("bm25_search_total_seconds", 0.1)
        recorder.add("bm25_search_total_seconds", 0.2)
        self.assertEqual(recorder.value("dataset_load_seconds"), 0.5)
        self.assertAlmostEqual(recorder.value("bm25_search_total_seconds"), 0.3)

    def test_zero_is_kept_as_applicable_value(self) -> None:
        recorder = DurationRecorder()
        recorder.record("corpus_encode_seconds", 0.0)
        self.assertEqual(recorder.value("corpus_encode_seconds"), 0.0)
        self.assertIsNone(recorder.value("model_load_seconds"))

    def test_invalid_values_are_rejected(self) -> None:
        recorder = DurationRecorder()
        with self.assertRaises(ValueError):
            recorder.record("dataset_load_seconds", -0.1)
        with self.assertRaises(ValueError):
            recorder.record("dataset_load_seconds", float("nan"))
        with self.assertRaises(ValueError):
            recorder.record("dataset_load_seconds", float("inf"))
        with self.assertRaises(TypeError):
            recorder.record("dataset_load_seconds", "0.1")

    def test_unknown_field_is_rejected(self) -> None:
        recorder = DurationRecorder()
        with self.assertRaises(KeyError):
            recorder.record("unknown_seconds", 0.1)
        with self.assertRaises(KeyError):
            recorder.value("unknown_seconds")

    def test_mark_not_applicable(self) -> None:
        recorder = DurationRecorder()
        recorder.record("model_load_seconds", 1.0)
        recorder.mark_not_applicable("model_load_seconds")
        self.assertIsNone(recorder.value("model_load_seconds"))

    def test_snapshot_is_a_copy(self) -> None:
        recorder = DurationRecorder()
        snapshot = recorder.snapshot()
        snapshot["dataset_load_seconds"] = 99.0
        self.assertIsNone(recorder.value("dataset_load_seconds"))


class HybridTimingHelperTests(unittest.TestCase):
    def _timing(self) -> dict[str, float | None]:
        return {
            field: None for field in TIMING_FIELDS
        }

    def test_subtotal_and_match(self) -> None:
        timing = self._timing()
        timing["bm25_search_total_seconds"] = 0.3
        timing["dense_search_total_seconds"] = 0.2
        timing["rrf_fusion_seconds"] = 0.1
        timing["retrieval_total_seconds"] = 0.6
        self.assertAlmostEqual(hybrid_subtotal(timing), 0.6)
        self.assertTrue(hybrid_total_matches(timing))
        self.assertEqual(len(HYBRID_SUBTOTAL_FIELDS), 3)

    def test_mismatch_and_missing_fields(self) -> None:
        timing = self._timing()
        timing["bm25_search_total_seconds"] = 0.3
        timing["dense_search_total_seconds"] = 0.2
        timing["rrf_fusion_seconds"] = 0.1
        timing["retrieval_total_seconds"] = 1.0
        self.assertFalse(hybrid_total_matches(timing))
        timing["rrf_fusion_seconds"] = None
        with self.assertRaises(ValueError):
            hybrid_subtotal(timing)
        self.assertFalse(hybrid_total_matches(timing))


if __name__ == "__main__":
    unittest.main()
