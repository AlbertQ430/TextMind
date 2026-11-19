#!/usr/bin/env python3
"""汇总 data/evaluation/ 下的实验结果，生成模型对比表。

用法::

    python src/scripts/compare_experiments.py
    python src/scripts/compare_experiments.py --method dense --output data/evaluation/summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = PROJECT_ROOT / "data" / "evaluation"

METRIC_COLUMNS = ["P@1", "NDCG@5", "NDCG@10", "R@10", "R@100", "MRR"]


def _config_key(config: dict) -> tuple:
    """同一实验配置的标识，用于去重时保留最新结果。"""
    if config.get("method") == "bm25":
        return ("bm25", config.get("k1"), config.get("b"), config.get("top_k"))
    return (
        "dense",
        config.get("model"),
        config.get("top_k"),
        config.get("query_prefix", ""),
    )


def load_results(directory: Path) -> list[dict]:
    """读取目录下全部实验 JSON（跳过无法解析的 JSON）。

    同一方法 + 同一模型/参数重复运行时，只保留文件名排序靠后
    （即时间戳更新）的一条，避免汇总表出现重复行。
    """
    latest: dict[tuple, dict] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "metrics" not in data or "config" not in data:
            continue
        latest[_config_key(data["config"])] = data
    return list(latest.values())


def describe(config: dict) -> str:
    method = config.get("method", "?")
    if method == "bm25":
        return f"BM25 (k1={config.get('k1')}, b={config.get('b')})"
    label = f"Dense: {config.get('model', '?')}"
    if config.get("query_prefix"):
        label += " (+prefix)"
    return label


def build_table(results: list[dict]) -> str:
    lines = [
        "| 方法 | P@1 | NDCG@5 | NDCG@10 | R@10 | R@100 | MRR | 建索引(s) | 检索均值(ms) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for data in results:
        metrics = data["metrics"]
        timing = data.get("timing", {})
        row = [describe(data["config"])]
        row += [f"{metrics.get(column, 0.0):.4f}" for column in METRIC_COLUMNS]
        row.append(f"{timing.get('index_seconds', 0.0):.1f}")
        row.append(f"{timing.get('search_mean_seconds', 0.0) * 1000:.1f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="汇总实验结果生成对比表")
    parser.add_argument("--dir", default=str(DEFAULT_DIR), help="结果目录")
    parser.add_argument("--method", choices=["all", "bm25", "dense"], default="all")
    parser.add_argument("--output", default=None, help="Markdown 输出路径（默认打印到终端）")
    args = parser.parse_args(argv)

    results = load_results(Path(args.dir))
    if args.method != "all":
        results = [item for item in results if item["config"].get("method") == args.method]
    if not results:
        print("没有可汇总的实验结果", file=sys.stderr)
        return 1

    table = build_table(results)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(table + "\n", encoding="utf-8")
        print(f"已写入: {output}")
    else:
        print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
