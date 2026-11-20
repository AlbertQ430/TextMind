"""实验参数校验测试。"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_experiment import parse_args, validate_args  # noqa: E402


class TestValidateArgs(unittest.TestCase):
    def test_valid_defaults(self):
        args = validate_args(parse_args([]))
        self.assertEqual(args.top_k, 100)
        self.assertEqual(args.k, [1, 5, 10, 100])

    def test_top_k_must_cover_max_k(self):
        args = parse_args(["--top-k", "5", "--k", "10"])
        with self.assertRaises(ValueError) as ctx:
            validate_args(args)
        self.assertIn("top-k", str(ctx.exception))

    def test_top_k_not_positive(self):
        args = parse_args(["--top-k", "0"])
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_k_not_positive(self):
        args = parse_args(["--k", "0", "10"])
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_empty_k_rejected(self):
        args = parse_args([])
        args.k = []
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_k_deduplicated_preserving_order(self):
        args = validate_args(parse_args(["--k", "10", "10", "5"]))
        self.assertEqual(args.k, [10, 5])

    def test_invalid_b(self):
        args = parse_args(["--b", "1.5"])
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_negative_k1(self):
        args = parse_args(["--k1", "-0.1"])
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_limit_not_positive(self):
        args = parse_args(["--limit", "0"])
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_limit_valid(self):
        args = validate_args(parse_args(["--limit", "50"]))
        self.assertEqual(args.limit, 50)

    def test_dense_batch_size_rejected(self):
        args = parse_args(["--method", "dense", "--batch-size", "0"])
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_dense_empty_model_rejected(self):
        args = parse_args(["--method", "dense", "--model", ""])
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_dense_ignores_bm25_params(self):
        args = validate_args(parse_args(["--method", "dense", "--k1", "-1", "--b", "2"]))
        self.assertEqual(args.method, "dense")

    def test_query_prefix_kept(self):
        args = validate_args(
            parse_args(["--method", "dense", "--query-prefix", "Represent: "])
        )
        self.assertEqual(args.query_prefix, "Represent: ")

    def test_hybrid_valid(self):
        args = validate_args(
            parse_args(["--method", "hybrid", "--model", "m", "--rrf-k", "10"])
        )
        self.assertEqual(args.method, "hybrid")
        self.assertEqual(args.rrf_k, 10)

    def test_hybrid_rrf_k_rejected(self):
        args = parse_args(["--method", "hybrid", "--rrf-k", "0"])
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_dataset_and_tokenizer_defaults(self):
        args = parse_args([])
        self.assertEqual(args.dataset, "scifact")
        self.assertEqual(args.tokenizer, "simple")

    def test_duretrieval_choices(self):
        args = parse_args(["--dataset", "duretrieval", "--tokenizer", "jieba"])
        self.assertEqual(args.dataset, "duretrieval")
        self.assertEqual(args.tokenizer, "jieba")

    def test_hybrid_rrf_weights(self):
        args = validate_args(
            parse_args(["--method", "hybrid", "--rrf-weights", "0.5", "1.0"])
        )
        self.assertEqual(args.rrf_weights, [0.5, 1.0])

    def test_hybrid_rrf_weights_invalid(self):
        with self.assertRaises(ValueError):
            validate_args(parse_args(["--method", "hybrid", "--rrf-weights", "0.5"]))
        with self.assertRaises(ValueError):
            validate_args(
                parse_args(["--method", "hybrid", "--rrf-weights", "-0.1", "1.0"])
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
