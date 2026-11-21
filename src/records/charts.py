"""单运行图表生成。

只使用当前变体的执行结果；图表先写临时文件，确认可读后原子改名。
单图失败只记录日志，不影响 Completed 状态和其余图表。
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path

# matplotlib 需要在导入前确定可写缓存目录；避免在只读 HOME 环境中产生告警。
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="textmind-mpl-")
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp(prefix="textmind-cache-")

import matplotlib

CJK_FONT_FAMILIES: list[str] = [
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Arial Unicode MS",
    "Heiti TC",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
matplotlib.rcParams["font.sans-serif"] = CJK_FONT_FAMILIES
matplotlib.rcParams["axes.unicode_minus"] = False

from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

logger = logging.getLogger(__name__)

TIMING_LABELS: dict[str, str] = {
    "dataset_load_seconds": "加载数据集",
    "model_load_seconds": "加载模型",
    "index_build_seconds": "构建索引",
    "doc_cache_load_seconds": "文档缓存",
    "corpus_encode_seconds": "语料编码",
    "query_cache_load_seconds": "查询缓存",
    "query_encode_seconds": "查询编码",
    "bm25_search_total_seconds": "BM25 检索",
    "dense_search_total_seconds": "Dense 检索",
    "rrf_fusion_seconds": "RRF 融合",
    "retrieval_total_seconds": "检索总计",
    "retrieval_mean_seconds": "单查询均值",
    "retrieval_median_seconds": "单查询中位数",
    "evaluation_seconds": "指标评价",
}

_ACCENT = "#4C72B0"


def _variant_label(task) -> str:
    snapshot = task.config_snapshot
    parts = [
        str(snapshot.get("plan_name") or task.plan_id),
        str(snapshot.get("variant_name") or task.variant_id),
    ]
    variant = snapshot.get("variant") or {}
    model = variant.get("model") if isinstance(variant, dict) else None
    if isinstance(model, dict) and model.get("name"):
        parts.append(str(model["name"]))
    return " · ".join(parts)


def _save_figure(figure: Figure, final_path: Path) -> bool:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.parent / f".tmp-{uuid.uuid4().hex}.png"
    try:
        FigureCanvasAgg(figure)
        figure.savefig(tmp_path, format="png", dpi=120, bbox_inches="tight")
        with tmp_path.open("rb") as handle:
            if not handle.read(1):
                raise OSError("图表文件为空")
        os.replace(tmp_path, final_path)
        return True
    except Exception:  # noqa: BLE001 - 图表失败不得影响完成状态
        logger.exception("图表生成失败: %s", final_path)
        try:
            tmp_path.unlink()
        except OSError:
            pass
        return False
    finally:
        figure.clear()


def _generate_metrics(
    task, metrics: dict[str, float], charts_dir: Path
) -> dict[str, str] | None:
    if not metrics:
        return None
    figure = Figure(figsize=(7.5, 4.2))
    axes = figure.add_subplot(111)
    names = list(metrics)
    values = [metrics[name] for name in names]
    positions = list(range(len(names)))
    axes.bar(positions, values, color=_ACCENT)
    axes.set_xticks(positions)
    axes.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    top = max(values) if values else 0.0
    axes.set_ylim(0.0, max(1.0, top * 1.15))
    axes.set_ylabel("score")
    axes.set_title(f"指标 · {_variant_label(task)}", fontsize=10)
    figure.tight_layout()

    target = charts_dir / "metrics.png"
    if not _save_figure(figure, target):
        return None
    return {"id": "metrics", "title": "指标", "path": "charts/metrics.png"}


def _generate_timing(
    task, timing: dict[str, float | None], charts_dir: Path
) -> dict[str, str] | None:
    applicable = {
        name: value
        for name, value in timing.items()
        if value is not None and name in TIMING_LABELS
    }
    if not applicable:
        return None
    figure = Figure(figsize=(7.5, 4.2))
    axes = figure.add_subplot(111)
    names = list(applicable)
    values = [applicable[name] for name in names]
    labels = [TIMING_LABELS[name] for name in names]
    positions = list(range(len(names)))
    axes.bar(positions, values, color="#DD8452")
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes.set_ylabel("seconds")
    axes.set_title(f"分阶段耗时 · {_variant_label(task)}", fontsize=10)
    figure.tight_layout()

    target = charts_dir / "timing.png"
    if not _save_figure(figure, target):
        return None
    return {"id": "timing", "title": "分阶段耗时", "path": "charts/timing.png"}


def generate_charts(
    task, execution, charts_dir: Path
) -> list[dict[str, str]]:
    """生成当前变体的单运行图表，返回成功图表清单。"""
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    charts: list[dict[str, str]] = []
    metrics_chart = _generate_metrics(task, execution.metrics, charts_dir)
    if metrics_chart is not None:
        charts.append(metrics_chart)
    timing_chart = _generate_timing(task, execution.timing, charts_dir)
    if timing_chart is not None:
        charts.append(timing_chart)
    return charts
