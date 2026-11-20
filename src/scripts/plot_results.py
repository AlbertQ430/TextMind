#!/usr/bin/env python3
"""从实验结果 JSON 生成图表（指标柱状图 + 质量-耗时散点图）。

用法::

    python src/scripts/plot_results.py
    python src/scripts/plot_results.py --output data/evaluation/figures
"""

from __future__ import annotations

import argparse
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


def short_label(config: dict) -> str:
    """图表中使用的简短方法标签。"""
    method = config.get("method", "?")
    if method == "bm25":
        return "BM25"
    model = config.get("model", "?").split("/")[-1]
    model = model.replace("Qwen3-Embedding-0.6B", "Qwen3-0.6B")
    suffix = " +prefix" if config.get("query_prefix") else ""
    if method == "hybrid":
        return f"Hybrid ({model}){suffix}"
    return f"Dense ({model}){suffix}"


def plot_metric_bars(results: list[dict], output: Path) -> None:
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
    ax.set_title("SciFact test split: retrieval metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_efficiency(results: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for item in results:
        label = short_label(item["config"])
        timing = item.get("timing", {})
        latency_ms = timing.get("search_mean_seconds", 0.0) * 1000
        score = item["metrics"].get("NDCG@10", 0.0)
        ax.scatter(latency_ms, score, s=60)
        ax.annotate(label, (latency_ms, score), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("per-query retrieval latency (ms, log; excl. query encoding)")
    ax.set_ylabel("NDCG@10")
    ax.set_title("SciFact test split: quality vs. retrieval latency")
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
    plot_metric_bars(results, bars)
    plot_efficiency(results, scatter)
    print(f"已生成: {bars}")
    print(f"已生成: {scatter}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
