"""同方案版本的结果比较与内存比较图。"""

from __future__ import annotations

import io
import json
import os
import re
import tempfile
from dataclasses import dataclass

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="textmind-mpl-")
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp(prefix="textmind-cache-")

import matplotlib

from records.charts import CJK_FONT_FAMILIES

matplotlib.rcParams["font.sans-serif"] = CJK_FONT_FAMILIES
matplotlib.rcParams["axes.unicode_minus"] = False

import numpy as np  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from records.catalog import CatalogError, RecordEntry, ResultCatalog  # noqa: E402


class CompareSelectionError(Exception):
    code = "COMPARE_SELECTION_INVALID"
    status_code = 422


class CompareNoMetricsError(Exception):
    code = "COMPARE_NO_METRICS"
    status_code = 422


@dataclass(frozen=True)
class CompareSelection:
    plan_id: str
    plan_version: str
    entries: list[RecordEntry]
    baseline: RecordEntry | None


def metric_sort_key(name: str) -> tuple:
    """按指标名称和 K 稳定排序，例如 P@1、P@5、NDCG@10、MRR。"""
    match = re.match(r"^(.+?)@(\d+)$", name)
    if match:
        return (match.group(1).upper(), int(match.group(2)), name)
    return (name.upper(), 0, name)


def build_selection(
    catalog: ResultCatalog, run_ids: list[str]
) -> CompareSelection:
    """校验比较选择并返回同方案版本的记录集合。"""
    unique: list[str] = []
    seen: set[str] = set()
    for run_id in run_ids:
        if run_id and run_id not in seen:
            seen.add(run_id)
            unique.append(run_id)
    if len(unique) < 2:
        raise CompareSelectionError("至少选择两条记录")

    entries: list[RecordEntry] = []
    for run_id in unique:
        # 原始读取错误（404/409）由调用方映射，不折叠为选择 422。
        entry = catalog.get(run_id)
        if entry.is_reference:
            raise CompareSelectionError(
                "等权 RRF 参照运行不能作为选中记录，应作为自动基线"
            )
        entries.append(entry)

    plans = {
        (entry.record.plan_id, entry.record.plan_version) for entry in entries
    }
    if len(plans) != 1:
        raise CompareSelectionError("所选记录必须属于同一方案和版本")
    plan_id, plan_version = next(iter(plans))

    for entry in entries:
        record = entry.record
        if record.status != "Completed" or not record.metrics:
            raise CompareSelectionError(
                f"记录 {entry.run_id} 未完成或缺少指标，不能参与比较"
            )
        if catalog.known_plan(record) is None:
            raise CompareSelectionError(
                f"记录 {entry.run_id} 的方案或变体不在当前方案目录中"
            )

    baseline = catalog.reference_baseline(plan_id, plan_version)
    return CompareSelection(plan_id, plan_version, entries, baseline)


def compare(selection: CompareSelection) -> dict:
    columns = sorted(
        {
            name
            for entry in selection.entries
            for name in (entry.record.metrics or {})
        },
        key=metric_sort_key,
    )
    if not columns:
        raise CompareNoMetricsError("没有可对齐的指标")

    rows = []
    for entry in selection.entries:
        metrics = entry.record.metrics or {}
        rows.append(
            {
                "run_id": entry.run_id,
                "source": entry.source,
                "variant_id": entry.record.variant_id,
                "variant_name": entry.record.variant_name,
                "metrics": {name: metrics.get(name) for name in columns},
                "incomparable": [
                    name for name in columns if name not in metrics
                ],
                "config": dict(entry.record.variant_config),
                "environment": dict(entry.record.environment),
            }
        )

    baseline = None
    if selection.baseline is not None:
        baseline_metrics = selection.baseline.record.metrics or {}
        baseline = {
            "run_id": selection.baseline.run_id,
            "source": selection.baseline.source,
            "variant_id": selection.baseline.record.variant_id,
            "variant_name": selection.baseline.record.variant_name,
            "metrics": {
                name: baseline_metrics.get(name) for name in columns
            },
            "incomparable": [
                name for name in columns if name not in baseline_metrics
            ],
        }

    run_ids = [entry.run_id for entry in selection.entries]
    common_columns = [
        name
        for name in columns
        if all(row["metrics"].get(name) is not None for row in rows)
    ]
    chart_available = bool(common_columns)
    chart_message = (
        None
        if chart_available
        else "所选记录没有共同可对齐的指标，无法生成比较图"
    )
    return {
        "plan_id": selection.plan_id,
        "plan_version": selection.plan_version,
        "metric_columns": columns,
        "common_metric_columns": common_columns,
        "rows": rows,
        "config_differences": _differences(
            [row["config"] for row in rows], run_ids
        ),
        "environment_differences": _differences(
            [row["environment"] for row in rows], run_ids
        ),
        "baseline": baseline,
        "chart_available": chart_available,
        "chart_message": chart_message,
        "chart_url": "/api/v1/runs/compare/chart?runs=" + ",".join(run_ids),
    }


_MISSING = "<missing>"


def _flatten(value, prefix: str = "") -> dict:
    if isinstance(value, dict):
        flattened: dict = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(child, path))
        return flattened
    return {prefix or "<root>": value}


def _differences(mappings: list[dict], run_ids: list[str]) -> dict:
    flattened = [_flatten(mapping) for mapping in mappings]
    keys = sorted(set().union(*flattened)) if flattened else []
    differences: dict = {}
    for key in keys:
        values = [flat.get(key, _MISSING) for flat in flattened]
        normalized = [
            _MISSING
            if value is _MISSING
            else json.dumps(value, sort_keys=True, ensure_ascii=False)
            for value in values
        ]
        if len(set(normalized)) > 1:
            differences[key] = dict(zip(run_ids, values))
    return differences


def build_compare_chart(selection: CompareSelection, data: dict) -> bytes:
    """在内存生成比较图 PNG；不写文件、不修改记录。

    只绘制所有选中记录都拥有的共同指标；缺失值不画成 0。
    """
    columns = data.get("common_metric_columns")
    if columns is None:
        columns = [
            name
            for name in data["metric_columns"]
            if all(
                row["metrics"].get(name) is not None for row in data["rows"]
            )
        ]
    if not columns:
        raise CompareNoMetricsError("没有共同可对齐的指标")

    series: list[tuple[str, dict]] = [
        (row["variant_name"], row["metrics"]) for row in data["rows"]
    ]
    if data.get("baseline"):
        baseline = data["baseline"]
        series.append(
            (f"{baseline['variant_name']}（参照）", baseline["metrics"])
        )

    figure = Figure(figsize=(max(7.0, len(columns) * 1.1), 4.6))
    axes = figure.add_subplot(111)
    positions = np.arange(len(columns))
    width = 0.8 / max(1, len(series))
    for index, (label, metrics) in enumerate(series):
        values = [
            metrics.get(column)
            if metrics.get(column) is not None
            else float("nan")
            for column in columns
        ]
        offset = positions - 0.4 + width / 2 + index * width
        axes.bar(offset, values, width=width, label=label)

    axes.set_xticks(positions)
    axes.set_xticklabels(columns, rotation=45, ha="right", fontsize=8)
    top = max(
        [
            value
            for _, metrics in series
            for value in metrics.values()
            if value is not None
        ]
        or [1.0]
    )
    axes.set_ylim(0.0, max(1.0, top * 1.15))
    axes.set_ylabel("score")
    axes.set_title(
        f"{data['plan_id']} / {data['plan_version']} 结果比较", fontsize=10
    )
    axes.legend(fontsize=8)
    figure.tight_layout()

    buffer = io.BytesIO()
    FigureCanvasAgg(figure)
    figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    figure.clear()
    return buffer.getvalue()
