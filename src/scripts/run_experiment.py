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
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from document.dataset import load_duretrieval, load_scifact  # noqa: E402
from evaluation.metrics import evaluate  # noqa: E402
from retrieval.bm25 import BM25Retriever, jieba_tokenize, simple_tokenize  # noqa: E402
from retrieval.dense import DenseRetriever  # noqa: E402
from retrieval.hybrid import HybridRetriever  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "scifact"
RESULT_DIR = PROJECT_ROOT / "data" / "evaluation"
EMBEDDING_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "embeddings"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"

DATASET_DIRS = {
    "scifact": DATA_DIR,
    "duretrieval": PROJECT_ROOT / "data" / "raw" / "duretrieval",
}
DATASET_FILES = {
    "scifact": ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"),
    "duretrieval": ("corpus.jsonl", "queries.jsonl", "qrels_dev.tsv"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检索实验（BM25 / Dense / Hybrid）")
    parser.add_argument("--dataset", choices=["scifact", "duretrieval"], default="scifact")
    parser.add_argument("--corpus", default=None, help="默认按 --dataset 选择")
    parser.add_argument("--queries", default=None, help="默认按 --dataset 选择")
    parser.add_argument("--qrels", default=None, help="默认按 --dataset 选择")
    parser.add_argument("--name", default=None, help="数据集标识，用于结果命名（默认取 --dataset）")
    parser.add_argument(
        "--tokenizer", choices=["simple", "jieba"], default="simple", help="BM25 分词器"
    )
    parser.add_argument("--method", choices=["bm25", "dense", "hybrid"], default="bm25", help="检索方法")
    parser.add_argument("--top-k", type=int, default=100, help="检索深度")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 100], help="评价的 K 值")
    parser.add_argument("--k1", type=float, default=1.2, help="BM25 参数（仅 bm25）")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 参数（仅 bm25）")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding 模型（仅 dense）")
    parser.add_argument(
        "--device", choices=["auto", "mps", "cpu"], default="auto", help="推理设备（仅 dense）"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="编码 batch 大小（仅 dense/hybrid）")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF 融合参数（仅 hybrid）")
    parser.add_argument(
        "--query-prefix", default="", help="查询前缀（仅 dense，例如 BGE 官方检索指令）"
    )
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

    if args.method in ("bm25", "hybrid"):
        if args.k1 < 0:
            raise ValueError(f"--k1 必须 >= 0，实际 {args.k1}")
        if not 0.0 <= args.b <= 1.0:
            raise ValueError(f"--b 必须在 [0, 1] 内，实际 {args.b}")
    if args.method in ("dense", "hybrid"):
        if not args.model.strip():
            raise ValueError("--model 不能为空")
        if args.batch_size <= 0:
            raise ValueError(f"--batch-size 必须为正整数，实际 {args.batch_size}")
    if args.method == "hybrid":
        if args.rrf_k <= 0:
            raise ValueError(f"--rrf-k 必须为正整数，实际 {args.rrf_k}")

    args.k = ks
    return args


def _resolve_device(name: str, torch) -> str:
    if name == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return name


def _cache_commit_hash(model_name: str) -> str | None:
    """从 HuggingFace 本地缓存读取模型 commit hash（离线可用）。"""
    try:
        from huggingface_hub import constants

        cache_dir = Path(constants.HF_HUB_CACHE)
    except ImportError:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    ref = cache_dir / f"models--{model_name.replace('/', '--')}" / "refs" / "main"
    try:
        return ref.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _model_revision(model, model_name: str | None = None) -> str | None:
    """读取模型权重版本：优先 transformers 的 commit hash，回退到 HF 缓存 refs。"""
    try:
        commit = getattr(model[0].auto_model.config, "_commit_hash", None)
        if commit:
            return commit
    except (AttributeError, IndexError, TypeError):
        pass
    if model_name:
        return _cache_commit_hash(model_name)
    return None


def _tokenizer(name: str):
    return jieba_tokenize if name == "jieba" else simple_tokenize


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    base = DATASET_DIRS[args.dataset]
    corpus_name, queries_name, qrels_name = DATASET_FILES[args.dataset]
    args.corpus = args.corpus or str(base / corpus_name)
    args.queries = args.queries or str(base / queries_name)
    args.qrels = args.qrels or str(base / qrels_name)
    args.name = args.name or args.dataset
    try:
        args = validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    start = time.perf_counter()
    loader = load_duretrieval if args.dataset == "duretrieval" else load_scifact
    dataset = loader(args.corpus, args.queries, args.qrels, name=args.name)
    load_seconds = time.perf_counter() - start

    model_load_seconds = 0.0
    device = None
    torch = None
    if args.method == "bm25":
        retriever = BM25Retriever(
            k1=args.k1, b=args.b, tokenizer=_tokenizer(args.tokenizer)
        )
    else:
        import torch
        from sentence_transformers import SentenceTransformer

        device = _resolve_device(args.device, torch)
        start = time.perf_counter()
        model = SentenceTransformer(args.model, device=device)
        model_load_seconds = time.perf_counter() - start
        revision = _model_revision(model, args.model)
        dense = DenseRetriever(
            model,
            batch_size=args.batch_size,
            cache_dir=EMBEDDING_CACHE_DIR,
            cache_key=args.model,
            model_name=args.model,
            model_revision=revision,
        )
        if args.method == "hybrid":
            retriever = HybridRetriever(
                [
                    BM25Retriever(
                        k1=args.k1, b=args.b, tokenizer=_tokenizer(args.tokenizer)
                    ),
                    dense,
                ],
                rrf_k=args.rrf_k,
                query_transforms=[None, lambda query: args.query_prefix + query],
            )
        else:
            retriever = dense

    start = time.perf_counter()
    retriever.fit(dataset.documents)
    index_seconds = time.perf_counter() - start

    eval_queries = [q for q in dataset.queries if q.query_id in dataset.qrels]
    if args.limit is not None:
        eval_queries = eval_queries[: args.limit]

    retrieved = {}
    latencies = []
    hybrid_timing = None
    if args.method == "dense":
        query_texts = [args.query_prefix + query.text for query in eval_queries]
        query_vectors = retriever.encode_queries(query_texts)
        for query, vector in zip(eval_queries, query_vectors):
            start = time.perf_counter()
            results = retriever.search_vector(vector, top_k=args.top_k)
            latencies.append(time.perf_counter() - start)
            retrieved[query.query_id] = [doc_id for doc_id, _ in results]
    elif args.method == "hybrid":
        raw_queries = [query.text for query in eval_queries]
        fused = retriever.search_batch(raw_queries, top_k=args.top_k)
        hybrid_timing = retriever.last_batch_timing
        latencies = list(hybrid_timing["per_query_end_to_end"])
        for query, results in zip(eval_queries, fused):
            retrieved[query.query_id] = [doc_id for doc_id, _ in results]
    else:
        for query in eval_queries:
            start = time.perf_counter()
            results = retriever.search(query.text, top_k=args.top_k)
            latencies.append(time.perf_counter() - start)
            retrieved[query.query_id] = [doc_id for doc_id, _ in results]

    subset_qrels = {query.query_id: dataset.qrels[query.query_id] for query in eval_queries}
    evaluation = evaluate(retrieved, subset_qrels, ks=args.k)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = ""
    if args.query_prefix:
        digest = hashlib.sha1(args.query_prefix.encode("utf-8")).hexdigest()[:8]
        suffix = f"_prefix_{digest}"
    output = Path(args.output) if args.output else (
        RESULT_DIR / f"{args.name}_{args.method}{suffix}_{timestamp}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "dataset": args.dataset,
        "method": args.method,
        "top_k": args.top_k,
        "k_values": list(args.k),
    }
    if args.method == "bm25":
        config.update({"k1": args.k1, "b": args.b, "tokenizer": args.tokenizer})
    else:
        config.update(
            {
                "model": args.model,
                "model_revision": revision,
                "device": device,
                "batch_size": args.batch_size,
                "normalize_embeddings": True,
                "query_prefix": args.query_prefix,
            }
        )
        if args.method == "hybrid":
            config.update(
                {
                    "k1": args.k1,
                    "b": args.b,
                    "rrf_k": args.rrf_k,
                    "tokenizer": args.tokenizer,
                }
            )

    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if args.method in ("dense", "hybrid"):
        environment["torch"] = torch.__version__
        environment["device"] = device

    if args.method == "dense":
        timing = {
            "load_seconds": round(load_seconds, 4),
            "model_load_seconds": round(model_load_seconds, 4),
            "doc_cache_load_seconds": round(retriever.timing["doc_cache_load"], 4),
            "corpus_encode_seconds": round(retriever.timing["corpus_encode"], 4),
            "query_cache_load_seconds": round(retriever.timing["query_cache_load"], 4),
            "query_encode_seconds": round(retriever.timing["query_encode"], 4),
            "index_seconds": round(
                retriever.timing["doc_cache_load"] + retriever.timing["corpus_encode"], 4
            ),
            "vector_search_total_seconds": round(sum(latencies), 4),
            "search_total_seconds": round(sum(latencies), 4),
            "search_mean_seconds": round(statistics.mean(latencies), 6),
            "search_median_seconds": round(statistics.median(latencies), 6),
        }
    elif args.method == "hybrid":
        timing = {
            "load_seconds": round(load_seconds, 4),
            "model_load_seconds": round(model_load_seconds, 4),
            "doc_cache_load_seconds": round(dense.timing["doc_cache_load"], 4),
            "corpus_encode_seconds": round(dense.timing["corpus_encode"], 4),
            "query_cache_load_seconds": round(dense.timing["query_cache_load"], 4),
            "query_encode_seconds": round(hybrid_timing["query_encode"], 4),
            "index_seconds": round(index_seconds, 4),
            "bm25_search_seconds": round(hybrid_timing["bm25_search"], 4),
            "dense_search_seconds": round(hybrid_timing["dense_search"], 4),
            "rrf_fusion_seconds": round(hybrid_timing["rrf_fusion"], 4),
            "vector_search_total_seconds": round(
                hybrid_timing["bm25_search"]
                + hybrid_timing["dense_search"]
                + hybrid_timing["rrf_fusion"],
                4,
            ),
            "search_total_seconds": round(sum(latencies), 4),
            "search_mean_seconds": round(statistics.mean(latencies), 6),
            "search_median_seconds": round(statistics.median(latencies), 6),
        }
    else:
        timing = {
            "load_seconds": round(load_seconds, 4),
            "model_load_seconds": round(model_load_seconds, 4),
            "doc_cache_load_seconds": 0.0,
            "corpus_encode_seconds": round(index_seconds, 4),
            "query_cache_load_seconds": 0.0,
            "query_encode_seconds": 0.0,
            "index_seconds": round(index_seconds, 4),
            "vector_search_total_seconds": round(sum(latencies), 4),
            "search_total_seconds": round(sum(latencies), 4),
            "search_mean_seconds": round(statistics.mean(latencies), 6),
            "search_median_seconds": round(statistics.median(latencies), 6),
        }

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
        "timing": timing,
        "environment": environment,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"数据集: {dataset.name}  corpus={len(dataset.documents)}  "
          f"queries={len(eval_queries)}  qrels_pairs={result['dataset']['qrels_pairs']}")
    prefix_note = f"  query_prefix={args.query_prefix!r}" if args.query_prefix else ""
    if args.method == "bm25":
        print(f"方法: BM25  k1={args.k1} b={args.b} tokenizer={args.tokenizer} top_k={args.top_k}")
    elif args.method == "dense":
        print(f"方法: Dense  model={args.model} device={device} "
              f"batch_size={args.batch_size} top_k={args.top_k}{prefix_note}")
    else:
        print(f"方法: Hybrid (BM25 + Dense)  model={args.model} device={device} "
              f"rrf_k={args.rrf_k} top_k={args.top_k}{prefix_note}")
    print("指标:")
    for name, value in evaluation["metrics"].items():
        print(f"  {name:<8} {value:.4f}")
    print(f"无相关标注跳过: {evaluation['num_skipped']}")
    if args.method == "dense":
        print(f"耗时: 加载 {load_seconds:.2f}s  模型 {model_load_seconds:.2f}s  "
              f"文档缓存 {timing['doc_cache_load_seconds']:.2f}s  "
              f"语料编码 {timing['corpus_encode_seconds']:.2f}s")
        print(f"      查询缓存 {timing['query_cache_load_seconds']:.2f}s  "
              f"查询编码 {timing['query_encode_seconds']:.2f}s  "
              f"向量检索 {timing['vector_search_total_seconds']:.3f}s  "
              f"单条均值 {statistics.mean(latencies) * 1000:.2f}ms  "
              f"中位数 {statistics.median(latencies) * 1000:.2f}ms")
    elif args.method == "hybrid":
        print(f"耗时: 加载 {load_seconds:.2f}s  模型 {model_load_seconds:.2f}s  "
              f"文档缓存 {timing['doc_cache_load_seconds']:.2f}s  "
              f"语料编码 {timing['corpus_encode_seconds']:.2f}s  "
              f"查询编码 {timing['query_encode_seconds']:.2f}s")
        print(f"      BM25 检索 {timing['bm25_search_seconds']:.3f}s  "
              f"Dense 检索 {timing['dense_search_seconds']:.3f}s  "
              f"RRF 融合 {timing['rrf_fusion_seconds']:.3f}s  "
              f"端到端单条均值 {statistics.mean(latencies) * 1000:.2f}ms  "
              f"中位数 {statistics.median(latencies) * 1000:.2f}ms")
    else:
        print(f"耗时: 加载 {load_seconds:.2f}s  建索引 {index_seconds:.2f}s  "
              f"检索总计 {sum(latencies):.3f}s  "
              f"单条均值 {statistics.mean(latencies) * 1000:.2f}ms  "
              f"中位数 {statistics.median(latencies) * 1000:.2f}ms")
    print(f"结果已写入: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
