#!/usr/bin/env python3
"""检索实验入口：BM25 / Dense Retrieval。

用法::

    python src/scripts/run_experiment.py                      # BM25（默认）
    python src/scripts/run_experiment.py --method dense       # Dense Retrieval
    python src/scripts/run_experiment.py --method dense --model Qwen/Qwen3-Embedding-0.6B
    python src/scripts/run_experiment.py --qrels data/raw/scifact/qrels/train.tsv --limit 50

输出：终端打印指标摘要，同时把完整结果写入
``data/evaluation/<name>_<method>_<timestamp>.json``。
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from document.dataset import load_scifact  # noqa: E402
from evaluation.metrics import evaluate  # noqa: E402
from retrieval.bm25 import BM25Retriever  # noqa: E402
from retrieval.dense import DenseRetriever  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "scifact"
RESULT_DIR = PROJECT_ROOT / "data" / "evaluation"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SciFact 检索实验（BM25 / Dense）")
    parser.add_argument("--corpus", default=str(DATA_DIR / "corpus.jsonl"))
    parser.add_argument("--queries", default=str(DATA_DIR / "queries.jsonl"))
    parser.add_argument("--qrels", default=str(DATA_DIR / "qrels" / "test.tsv"))
    parser.add_argument("--name", default="scifact", help="数据集标识，用于结果命名")
    parser.add_argument("--method", choices=["bm25", "dense"], default="bm25", help="检索方法")
    parser.add_argument("--top-k", type=int, default=100, help="检索深度")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 100], help="评价的 K 值")
    parser.add_argument("--k1", type=float, default=1.2, help="BM25 参数（仅 bm25）")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 参数（仅 bm25）")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding 模型（仅 dense）")
    parser.add_argument(
        "--device", choices=["auto", "mps", "cpu"], default="auto", help="推理设备（仅 dense）"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="编码 batch 大小（仅 dense）")
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 条 query（调试用）")
    parser.add_argument("--output", default=None, help="结果 JSON 路径")
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    """校验实验参数，非法时抛出带具体数值的 ValueError。

    公共校验：
    - ``top_k`` 为正整数；
    - 评价 K 列表非空、全为正整数、去重保序；
    - ``top_k >= max(k)``，保证每个评价 K 都能取满；
    - ``limit`` 为 None 或正整数。

    方法相关校验：
    - BM25：``k1 >= 0``，``b`` 落在 ``[0, 1]``；
    - Dense：``model`` 非空，``batch_size`` 为正整数。
    """
    if args.top_k <= 0:
        raise ValueError(f"--top-k 必须为正整数，实际 {args.top_k}")

    ks = list(dict.fromkeys(args.k))
    if not ks:
        raise ValueError("--k 至少需要一个 K 值")
    if any(k <= 0 for k in ks):
        raise ValueError(f"--k 必须为正整数，实际 {args.k}")
    if args.top_k < max(ks):
        raise ValueError(f"--top-k ({args.top_k}) 必须 >= 最大评价 K ({max(ks)})")
    if args.limit is not None and args.limit <= 0:
        raise ValueError(f"--limit 必须为正整数，实际 {args.limit}")

    if args.method == "bm25":
        if args.k1 < 0:
            raise ValueError(f"--k1 必须 >= 0，实际 {args.k1}")
        if not 0.0 <= args.b <= 1.0:
            raise ValueError(f"--b 必须在 [0, 1] 内，实际 {args.b}")
    else:
        if not args.model.strip():
            raise ValueError("--model 不能为空")
        if args.batch_size <= 0:
            raise ValueError(f"--batch-size 必须为正整数，实际 {args.batch_size}")

    args.k = ks
    return args


def _resolve_device(name: str, torch) -> str:
    if name == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return name


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args = validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    start = time.perf_counter()
    dataset = load_scifact(args.corpus, args.queries, args.qrels, name=args.name)
    load_seconds = time.perf_counter() - start

    model_load_seconds = 0.0
    device = None
    torch = None
    if args.method == "bm25":
        retriever = BM25Retriever(k1=args.k1, b=args.b)
    else:
        import torch
        from sentence_transformers import SentenceTransformer

        device = _resolve_device(args.device, torch)
        start = time.perf_counter()
        model = SentenceTransformer(args.model, device=device)
        model_load_seconds = time.perf_counter() - start
        retriever = DenseRetriever(model, batch_size=args.batch_size)

    start = time.perf_counter()
    retriever.fit(dataset.documents)
    index_seconds = time.perf_counter() - start

    eval_queries = [q for q in dataset.queries if q.query_id in dataset.qrels]
    if args.limit is not None:
        eval_queries = eval_queries[: args.limit]

    retrieved = {}
    latencies = []
    for query in eval_queries:
        start = time.perf_counter()
        results = retriever.search(query.text, top_k=args.top_k)
        latencies.append(time.perf_counter() - start)
        retrieved[query.query_id] = [doc_id for doc_id, _ in results]

    subset_qrels = {query.query_id: dataset.qrels[query.query_id] for query in eval_queries}
    evaluation = evaluate(retrieved, subset_qrels, ks=args.k)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output) if args.output else (
        RESULT_DIR / f"{args.name}_{args.method}_{timestamp}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "method": args.method,
        "top_k": args.top_k,
        "k_values": list(args.k),
    }
    if args.method == "bm25":
        config.update({"k1": args.k1, "b": args.b})
    else:
        config.update(
            {
                "model": args.model,
                "device": device,
                "batch_size": args.batch_size,
                "normalize_embeddings": True,
            }
        )

    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if args.method == "dense":
        environment["torch"] = torch.__version__
        environment["device"] = device

    result = {
        "dataset": {
            "name": dataset.name,
            "corpus_size": len(dataset.documents),
            "queries_size": len(eval_queries),
            "qrels_pairs": sum(len(rels) for rels in subset_qrels.values()),
        },
        "config": config,
        "metrics": evaluation["metrics"],
        "num_queries": evaluation["num_queries"],
        "num_skipped": evaluation["num_skipped"],
        "timing": {
            "load_seconds": round(load_seconds, 4),
            "model_load_seconds": round(model_load_seconds, 4),
            "index_seconds": round(index_seconds, 4),
            "search_total_seconds": round(sum(latencies), 4),
            "search_mean_seconds": round(statistics.mean(latencies), 6),
            "search_median_seconds": round(statistics.median(latencies), 6),
        },
        "environment": environment,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"数据集: {dataset.name}  corpus={len(dataset.documents)}  "
          f"queries={len(eval_queries)}  qrels_pairs={result['dataset']['qrels_pairs']}")
    if args.method == "bm25":
        print(f"方法: BM25  k1={args.k1} b={args.b} top_k={args.top_k}")
    else:
        print(f"方法: Dense  model={args.model} device={device} "
              f"batch_size={args.batch_size} top_k={args.top_k}")
    print("指标:")
    for name, value in evaluation["metrics"].items():
        print(f"  {name:<8} {value:.4f}")
    print(f"无相关标注跳过: {evaluation['num_skipped']}")
    print(f"耗时: 加载 {load_seconds:.2f}s  模型 {model_load_seconds:.2f}s  "
          f"建索引 {index_seconds:.2f}s  "
          f"检索总计 {sum(latencies):.3f}s  "
          f"单条均值 {statistics.mean(latencies) * 1000:.2f}ms  "
          f"中位数 {statistics.median(latencies) * 1000:.2f}ms")
    print(f"结果已写入: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
