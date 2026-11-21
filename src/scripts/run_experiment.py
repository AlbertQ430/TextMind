#!/usr/bin/env python3
"""检索实验入口：BM25 / Dense / Hybrid。

内部复用 `experiment.executor.ExperimentExecutor` 完成数据加载、检索、
评价和精确计时；本脚本只负责参数解析、适配注入和旧格式结果输出。

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
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from document.dataset import RetrievalDataset, load_duretrieval, load_scifact  # noqa: E402
from experiment.executor import ExperimentExecutor  # noqa: E402
from experiment.plans import (  # noqa: E402
    DatasetSpec,
    EvaluationSpec,
    ModelSpec,
    PlanDefinition,
    RunVariant,
)

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
        "--rrf-weights",
        type=float,
        nargs="+",
        default=None,
        help="加权 RRF 权重，顺序为 BM25 Dense（仅 hybrid；留空为等权）",
    )
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
        if args.rrf_weights is not None:
            if len(args.rrf_weights) != 2:
                raise ValueError(
                    f"--rrf-weights 需要 2 个值（BM25, Dense），实际 {args.rrf_weights}"
                )
            if any(not math.isfinite(weight) for weight in args.rrf_weights):
                raise ValueError(f"--rrf-weights 必须是有限数值，实际 {args.rrf_weights}")
            if any(weight < 0 for weight in args.rrf_weights):
                raise ValueError(f"--rrf-weights 必须非负，实际 {args.rrf_weights}")
            if all(weight == 0 for weight in args.rrf_weights):
                raise ValueError("--rrf-weights 不能全为 0")

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


def _cli_plan_and_variant(args: argparse.Namespace) -> tuple[PlanDefinition, RunVariant]:
    """把 CLI 自由参数包装成执行器所需的方案和单变体。

    方案 ID 和预置标识只用于执行器接口校验；CLI 不写运行记录，
    因此不会产生 `results/` 内容。
    """
    method = args.method
    variant_id = f"cli-{method}"
    language, split = ("zh", "dev") if args.dataset == "duretrieval" else ("en", "test")

    variant_fields: dict[str, object] = {}
    if method in ("bm25", "hybrid"):
        variant_fields.update(
            {"k1": args.k1, "b": args.b, "tokenizer": args.tokenizer}
        )
    if method in ("dense", "hybrid"):
        variant_fields.update(
            {
                "model": ModelSpec(
                    name=args.model,
                    directory_name=args.model.rsplit("/", 1)[-1],
                    revision=None,
                ),
                "query_prefix": args.query_prefix,
                "batch_size": args.batch_size,
                "normalize_embeddings": True,
            }
        )
    if method == "hybrid":
        variant_fields.update(
            {
                "rrf_k": args.rrf_k,
                "rrf_weights": tuple(args.rrf_weights or [1.0, 1.0]),
            }
        )

    variant = RunVariant(
        variant_id=variant_id,
        name=f"CLI {method}",
        kind="run",
        method=method,
        preset_run_id=f"preset-{variant_id}",
        preset_source="data/evaluation/cli.json",
        **variant_fields,
    )
    plan = PlanDefinition(
        plan_id="cli",
        plan_version="v1",
        name=f"CLI {args.dataset}",
        description="命令行临时方案",
        dataset=DatasetSpec(name=args.dataset, language=language, split=split),
        evaluation=EvaluationSpec(top_k=args.top_k, k_values=tuple(args.k)),
        variants=(variant,),
    )
    return plan, variant


def _legacy_timing(method: str, timing: dict) -> dict:
    """把 14 项新计时映射回 CLI 历史结果字段。"""
    load_seconds = round(timing["dataset_load_seconds"] or 0.0, 4)
    index_seconds = round(timing["index_build_seconds"] or 0.0, 4)
    common = {
        "load_seconds": load_seconds,
        "model_load_seconds": round(timing["model_load_seconds"] or 0.0, 4),
        "doc_cache_load_seconds": round(timing["doc_cache_load_seconds"] or 0.0, 4),
        "corpus_encode_seconds": round(timing["corpus_encode_seconds"] or 0.0, 4),
        "query_cache_load_seconds": round(timing["query_cache_load_seconds"] or 0.0, 4),
        "query_encode_seconds": round(timing["query_encode_seconds"] or 0.0, 4),
        "index_seconds": index_seconds,
        "search_total_seconds": round(timing["retrieval_total_seconds"], 4),
        "search_mean_seconds": round(timing["retrieval_mean_seconds"], 6),
        "search_median_seconds": round(timing["retrieval_median_seconds"], 6),
    }
    if method == "bm25":
        common["vector_search_total_seconds"] = round(
            timing["bm25_search_total_seconds"], 4
        )
    elif method == "dense":
        common["vector_search_total_seconds"] = round(
            timing["dense_search_total_seconds"], 4
        )
    else:
        common["bm25_search_seconds"] = round(timing["bm25_search_total_seconds"], 4)
        common["dense_search_seconds"] = round(timing["dense_search_total_seconds"], 4)
        common["rrf_fusion_seconds"] = round(timing["rrf_fusion_seconds"], 4)
        common["vector_search_total_seconds"] = round(
            timing["bm25_search_total_seconds"]
            + timing["dense_search_total_seconds"]
            + timing["rrf_fusion_seconds"],
            4,
        )
    return common


def _legacy_config(
    args: argparse.Namespace, result, revision: str | None
) -> dict:
    config: dict = {
        "dataset": args.dataset,
        "method": args.method,
        "top_k": args.top_k,
        "k_values": list(args.k),
    }
    if args.method == "bm25":
        config.update(
            {"k1": args.k1, "b": args.b, "tokenizer": args.tokenizer}
        )
    else:
        config.update(
            {
                "model": args.model,
                "model_revision": revision,
                "device": result.environment.get("device"),
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
                    "rrf_weights": args.rrf_weights or [1.0, 1.0],
                    "tokenizer": args.tokenizer,
                }
            )
    return config


def _legacy_environment(method: str, environment: dict) -> dict:
    result = {
        "python": environment.get("python"),
        "platform": environment.get("platform"),
    }
    if method in ("dense", "hybrid"):
        result["torch"] = environment.get("torch")
        result["device"] = environment.get("device")
    return result


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

    plan, variant = _cli_plan_and_variant(args)
    loader = load_duretrieval if args.dataset == "duretrieval" else load_scifact
    revision_holder: dict[str, str | None] = {"revision": None}

    def dataset_loader(_plan: PlanDefinition) -> RetrievalDataset:
        dataset = loader(args.corpus, args.queries, args.qrels, name=args.name)
        if args.limit is not None:
            evaluation_queries = [
                query for query in dataset.queries if query.query_id in dataset.qrels
            ][: args.limit]
            subset_ids = {query.query_id for query in evaluation_queries}
            dataset.queries = evaluation_queries
            dataset.qrels = {
                query_id: rels
                for query_id, rels in dataset.qrels.items()
                if query_id in subset_ids
            }
        return dataset

    def model_loader(model_spec: ModelSpec, device: str):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_spec.name, device=device)
        revision_holder["revision"] = _model_revision(model, model_spec.name)
        return model

    def device_resolver() -> str:
        import torch

        return _resolve_device(args.device, torch)

    executor = ExperimentExecutor(
        model_loader=model_loader,
        dataset_loader=dataset_loader,
        device_resolver=device_resolver,
        cache_dir=EMBEDDING_CACHE_DIR,
        show_progress=True,
    )
    result = executor.execute(plan, variant, lambda event: None)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = ""
    if args.query_prefix:
        digest = hashlib.sha1(args.query_prefix.encode("utf-8")).hexdigest()[:8]
        suffix = f"_prefix_{digest}"
    if args.method == "hybrid" and args.rrf_weights:
        suffix += "_w" + "_".join(f"{weight:g}" for weight in args.rrf_weights)
    output = Path(args.output) if args.output else (
        RESULT_DIR / f"{args.name}_{args.method}{suffix}_{timestamp}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    timing = _legacy_timing(args.method, result.timing)
    payload = {
        "dataset": {
            "name": result.dataset["name"],
            "corpus_size": result.dataset["corpus_size"],
            "queries_size": result.dataset["query_count"],
            "qrels_pairs": result.dataset["qrels_pairs"],
        },
        "config": _legacy_config(args, result, revision_holder["revision"]),
        "metrics": result.metrics,
        "num_queries": result.num_queries,
        "num_skipped": result.num_skipped,
        "timing": timing,
        "environment": _legacy_environment(args.method, result.environment),
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"数据集: {result.dataset['name']}  corpus={result.dataset['corpus_size']}  "
        f"queries={result.dataset['query_count']}  qrels_pairs={result.dataset['qrels_pairs']}"
    )
    prefix_note = f"  query_prefix={args.query_prefix!r}" if args.query_prefix else ""
    if args.method == "bm25":
        print(
            f"方法: BM25  k1={args.k1} b={args.b} tokenizer={args.tokenizer} top_k={args.top_k}"
        )
    elif args.method == "dense":
        print(
            f"方法: Dense  model={args.model} device={payload['environment'].get('device')} "
            f"batch_size={args.batch_size} top_k={args.top_k}{prefix_note}"
        )
    else:
        weight_note = f"  rrf_weights={args.rrf_weights}" if args.rrf_weights else ""
        print(
            f"方法: Hybrid (BM25 + Dense)  model={args.model} "
            f"device={payload['environment'].get('device')} "
            f"rrf_k={args.rrf_k} top_k={args.top_k}{prefix_note}{weight_note}"
        )
    print("指标:")
    for name, value in result.metrics.items():
        print(f"  {name:<8} {value:.4f}")
    print(f"无相关标注跳过: {result.num_skipped}")
    if args.method == "dense":
        print(
            f"耗时: 加载 {timing['load_seconds']:.2f}s  模型 {timing['model_load_seconds']:.2f}s  "
            f"文档缓存 {timing['doc_cache_load_seconds']:.2f}s  "
            f"语料编码 {timing['corpus_encode_seconds']:.2f}s"
        )
        print(
            f"      查询缓存 {timing['query_cache_load_seconds']:.2f}s  "
            f"查询编码 {timing['query_encode_seconds']:.2f}s  "
            f"向量检索 {timing['vector_search_total_seconds']:.3f}s  "
            f"单条均值 {timing['search_mean_seconds'] * 1000:.2f}ms  "
            f"中位数 {timing['search_median_seconds'] * 1000:.2f}ms"
        )
    elif args.method == "hybrid":
        print(
            f"耗时: 加载 {timing['load_seconds']:.2f}s  模型 {timing['model_load_seconds']:.2f}s  "
            f"文档缓存 {timing['doc_cache_load_seconds']:.2f}s  "
            f"语料编码 {timing['corpus_encode_seconds']:.2f}s  "
            f"查询编码 {timing['query_encode_seconds']:.2f}s"
        )
        print(
            f"      BM25 检索 {timing['bm25_search_seconds']:.3f}s  "
            f"Dense 检索 {timing['dense_search_seconds']:.3f}s  "
            f"RRF 融合 {timing['rrf_fusion_seconds']:.3f}s  "
            f"端到端单条均值 {timing['search_mean_seconds'] * 1000:.2f}ms  "
            f"中位数 {timing['search_median_seconds'] * 1000:.2f}ms"
        )
    else:
        print(
            f"耗时: 加载 {timing['load_seconds']:.2f}s  建索引 {timing['index_seconds']:.2f}s  "
            f"检索总计 {timing['search_total_seconds']:.3f}s  "
            f"单条均值 {timing['search_mean_seconds'] * 1000:.2f}ms  "
            f"中位数 {timing['search_median_seconds'] * 1000:.2f}ms"
        )
    print(f"结果已写入: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
