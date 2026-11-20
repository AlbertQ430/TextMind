#!/usr/bin/env python3
"""从实验结果 JSON 生成图表（指标柱状图 + 质量-耗时散点图）。

用法::

    python src/scripts/plot_results.py
    python src/scripts/plot_results.py --output data/evaluation/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.compare_experiments import load_results  # noqa: E402

DEFAULT_DIR = PROJECT_ROOT / "data" / "evaluation"
DEFAULT_OUTPUT = DEFAULT_DIR / "figures"
METRICS = ["NDCG@10", "R@100", "MRR"]


DATASET_LABELS = {"scifact": "SciFact", "duretrieval": "DuRetrieval"}


def short_label(config: dict) -> str:
    """图表中使用的简短方法标签（含数据集前缀）。"""
    method = config.get("method", "?")
    if method == "bm25":
        label = "BM25"
        if config.get("tokenizer") == "jieba":
            label += " (jieba)"
    else:
        model = config.get("model", "?").split("/")[-1]
        model = model.replace("Qwen3-Embedding-0.6B", "Qwen3-0.6B")
        suffix = " +prefix" if config.get("query_prefix") else ""
        if method == "hybrid":
            label = f"Hybrid ({model}){suffix}"
            weights = config.get("rrf_weights")
            if weights and list(weights) != [1.0, 1.0]:
                label += f" w={weights[0]:g}"
        else:
            label = f"Dense ({model}){suffix}"
    dataset = config.get("dataset", "scifact")
    return f"{DATASET_LABELS.get(dataset, dataset)}: {label}"


def plot_metric_bars(results: list[dict], output: Path, dataset: str | None = None, title: str | None = None) -> None:
    if dataset is not None:
        results = [
            item
            for item in results
            if item.get("config", {}).get("dataset", "scifact") == dataset
        ]
    if not results:
        return
    labels = [short_label(item["config"]) for item in results]
    x = np.arange(len(labels))
    width = 0.8 / len(METRICS)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for index, metric in enumerate(METRICS):
        values = [item["metrics"].get(metric, 0.0) for item in results]
        bars = ax.bar(x + index * width, values, width, label=metric)
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=5.5,
                rotation=90,
            )
    ax.set_xticks(x + width * (len(METRICS) - 1) / 2)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(title or "Retrieval metrics (SciFact test / DuRetrieval dev)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _is_representative(config: dict) -> bool:
    """质量-延迟散点图中需要标注名称的代表性配置。"""
    dataset = config.get("dataset", "scifact")
    method = config.get("method")
    if method == "bm25":
        return True
    if method == "dense":
        model = config.get("model", "")
        if dataset == "scifact":
            return model == "Qwen/Qwen3-Embedding-0.6B" and not config.get("query_prefix")
        return model == "BAAI/bge-small-zh-v1.5"
    if method == "hybrid":
        weights = config.get("rrf_weights") or [1.0, 1.0]
        if weights[1] != 1.0:
            return False
        if dataset != "duretrieval":
            return False
        return weights[0] in (1.0, 0.1)
    return False


def plot_efficiency(results: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for item in results:
        label = short_label(item["config"])
        timing = item.get("timing", {})
        latency_ms = timing.get("search_mean_seconds", 0.0) * 1000
        score = item["metrics"].get("NDCG@10", 0.0)
        representative = _is_representative(item["config"])
        ax.scatter(latency_ms, score, s=60 if representative else 25)
        if representative:
            if "w=0.1" in label:
                xytext, ha = (-10, 6), "right"
            elif "bge-small-zh" in label and "Dense" in label:
                xytext, ha = (6, -14), "left"
            else:
                xytext, ha = (4, 3), "left"
            ax.annotate(
                label,
                (latency_ms, score),
                fontsize=7,
                xytext=xytext,
                textcoords="offset points",
                ha=ha,
            )
    ax.set_xscale("log")
    ax.set_xlabel("per-query retrieval latency (ms, log; excl. query encoding)")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Quality vs. retrieval latency (representative configs)")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_rrf_weights(results: list[dict], output: Path) -> None:
    """DuRetrieval dev：NDCG@10 随 BM25 权重变化（离散配置点）。"""
    points = {}
    for item in results:
        config = item["config"]
        if config.get("method") != "hybrid":
            continue
        if config.get("dataset") != "duretrieval":
            continue
        weights = config.get("rrf_weights") or [1.0, 1.0]
        if weights[1] != 1.0:
            continue
        points[float(weights[0])] = item["metrics"].get("NDCG@10", 0.0)
    if not points:
        return

    dense_ref = None
    for item in results:
        config = item["config"]
        if (
            config.get("method") == "dense"
            and config.get("dataset") == "duretrieval"
            and config.get("model", "").endswith("bge-small-zh-v1.5")
            and config.get("query_prefix")
        ):
            dense_ref = item["metrics"].get("NDCG@10", 0.0)
            break

    xs = sorted(points)
    ys = [points[x] for x in xs]
    positions = list(range(len(xs)))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(positions, ys, marker="o", label="Hybrid (weighted RRF)")
    if dense_ref is not None:
        ax.axhline(
            dense_ref,
            linestyle="--",
            color="gray",
            label=f"Dense single (ref {dense_ref:.4f})",
        )
    for pos, x, y in zip(positions, xs, ys):
        if pos == 0:
            offset, ha = (8, 8), "left"
        elif pos == len(positions) - 1:
            offset, ha = (-8, 8), "right"
        else:
            offset, ha = (0, 8), "center"
        ax.annotate(
            f"{y:.4f}",
            (pos, y),
            textcoords="offset points",
            xytext=offset,
            ha=ha,
            fontsize=9,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_xlabel("BM25 weight (Dense weight = 1.0)")
    ax.set_ylabel("NDCG@10")
    ax.set_title("DuRetrieval dev: weighted RRF vs BM25 weight")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_error_distribution(error_json: Path, output: Path) -> None:
    """DuRetrieval dev：query 级诊断分布。"""
    data = json.loads(error_json.read_text(encoding="utf-8"))
    methods = data["methods"]
    names = [
        ("BM25", "bm25_jieba"),
        ("Dense", "dense_bge_small_zh"),
        ("Hybrid", "hybrid_bm25_dense"),
    ]
    categories = [("hit_at_1", "hit@1"), ("ranked_low", "ranked_low"), ("miss_top_k", "miss_top_k")]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    for category, label in categories:
        values = [methods[key]["counts"][category] for _, key in names]
        ax.bar(x, values, bottom=bottom, label=label)
        bottom += np.array(values)
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in names])
    ax.set_ylabel("queries")
    ax.set_title("DuRetrieval dev: query-level diagnosis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="生成实验结果图表")
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help="结果 JSON 目录")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="图表输出目录")
    args = parser.parse_args(argv)

    results = load_results(Path(args.dir))
    if not results:
        print("没有可绘图的实验结果", file=sys.stderr)
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    bars = output_dir / "metrics_bars.png"
    scatter = output_dir / "efficiency.png"
    weights_png = output_dir / "rrf_weights.png"
    plot_metric_bars(results, bars)
    plot_metric_bars(
        results,
        output_dir / "metrics_bars_scifact.png",
        dataset="scifact",
        title="SciFact test: retrieval metrics",
    )
    plot_metric_bars(
        results,
        output_dir / "metrics_bars_duretrieval.png",
        dataset="duretrieval",
        title="DuRetrieval dev: retrieval metrics",
    )
    plot_efficiency(results, scatter)
    plot_rrf_weights(results, weights_png)
    print(f"已生成: {bars}")
    print(f"已生成: {output_dir / 'metrics_bars_scifact.png'}")
    print(f"已生成: {output_dir / 'metrics_bars_duretrieval.png'}")
    print(f"已生成: {scatter}")
    print(f"已生成: {weights_png}")

    error_json = Path(args.dir) / "error_analysis.json"
    if error_json.exists():
        error_png = output_dir / "error_distribution.png"
        plot_error_distribution(error_json, error_png)
        print(f"已生成: {error_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
