#!/usr/bin/env python3
"""构建 17 条预置运行记录。

对 `config/plans.json` 的每个 RunVariant，读取其 `preset_source` 评估 JSON，
按 SDS §8.5.3 映射旧 timing，生成单运行图表并原子发布到输出目录。
不修改原始评估 JSON，也不把推导时间写回源文件。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from experiment.plans import (  # noqa: E402
    PlanCatalog,
    PlanDefinition,
    RunVariant,
    load_plan_catalog,
)
from experiment.stages import applicable_stages, not_applicable_stages  # noqa: E402
from experiment.timing import TIMING_FIELDS  # noqa: E402
from paths import PRESETS_DIR  # noqa: E402
from records.charts import generate_charts  # noqa: E402
from records.schema import RunRecordError, parse_run_record  # noqa: E402

SCIFACT_DIR = PROJECT_ROOT / "data" / "raw" / "scifact"
SCIFACT_DATASET_FILES = ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")
RESOURCES_DIR = PROJECT_ROOT / "resources" / "scifact"
RESULT_ASSETS = ("SOURCE.md", "NOTICE.md", "LICENSE-CC-BY-SA-4.0.txt")
ROOT_NOTICE = PROJECT_ROOT / "NOTICE.md"

_BM25_FIELDS = frozenset(
    {"dataset_load_seconds", "index_build_seconds", "bm25_search_total_seconds"}
)
_DENSE_FIELDS = frozenset(
    {
        "dataset_load_seconds",
        "model_load_seconds",
        "index_build_seconds",
        "doc_cache_load_seconds",
        "corpus_encode_seconds",
        "query_cache_load_seconds",
        "query_encode_seconds",
        "dense_search_total_seconds",
    }
)
_RETRIEVAL_FIELDS = frozenset(
    {"retrieval_total_seconds", "retrieval_mean_seconds", "retrieval_median_seconds"}
)
_HYBRID_FIELDS = frozenset(
    {"bm25_search_total_seconds", "dense_search_total_seconds", "rrf_fusion_seconds"}
)


def _positive(timing: dict, key: str) -> float | None:
    """旧字段映射：0.0 视为未知，不冒充“缓存命中”。"""
    value = timing.get(key)
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    if value <= 0:
        return None
    return float(value)


def map_timing(method: str, timing: dict) -> dict:
    """按 SDS §8.5.3 把旧 timing 映射为 14 项新口径。"""
    mapped = {field: None for field in TIMING_FIELDS}
    mapped["dataset_load_seconds"] = _positive(timing, "load_seconds")
    mapped["index_build_seconds"] = _positive(timing, "index_seconds")
    mapped["retrieval_total_seconds"] = _positive(timing, "search_total_seconds")
    mapped["retrieval_mean_seconds"] = _positive(timing, "search_mean_seconds")
    mapped["retrieval_median_seconds"] = _positive(timing, "search_median_seconds")

    if method == "bm25":
        mapped["bm25_search_total_seconds"] = _positive(
            timing, "vector_search_total_seconds"
        ) or _positive(timing, "bm25_search_seconds")
        return mapped

    mapped["model_load_seconds"] = _positive(timing, "model_load_seconds")
    mapped["doc_cache_load_seconds"] = _positive(
        timing, "doc_cache_load_seconds"
    )
    mapped["corpus_encode_seconds"] = _positive(
        timing, "corpus_encode_seconds"
    )
    mapped["query_cache_load_seconds"] = _positive(
        timing, "query_cache_load_seconds"
    )
    mapped["query_encode_seconds"] = _positive(
        timing, "query_encode_seconds"
    )

    if method == "dense":
        mapped["dense_search_total_seconds"] = _positive(
            timing, "vector_search_total_seconds"
        ) or _positive(timing, "dense_search_seconds")
        if mapped["retrieval_total_seconds"] is None:
            mapped["retrieval_total_seconds"] = _positive(
                timing, "vector_search_total_seconds"
            ) or _positive(timing, "dense_search_seconds")
        return mapped

    mapped["bm25_search_total_seconds"] = _positive(
        timing, "bm25_search_seconds"
    )
    mapped["dense_search_total_seconds"] = _positive(
        timing, "dense_search_seconds"
    )
    mapped["rrf_fusion_seconds"] = _positive(timing, "rrf_fusion_seconds")
    if mapped["retrieval_total_seconds"] is None:
        mapped["retrieval_total_seconds"] = (
            _positive(timing, "vector_search_total_seconds")
            or _positive(timing, "bm25_search_seconds")
            or _positive(timing, "dense_search_seconds")
        )
    return mapped


def archived_start(
    method: str, timing: dict, finished_at: datetime
) -> datetime:
    """按可分解 timing 推导归档开始时间。"""
    duration = sum(
        value or 0.0
        for value in (
            timing.get("dataset_load_seconds"),
            timing.get("model_load_seconds"),
            timing.get("index_build_seconds"),
            timing.get("query_cache_load_seconds"),
            timing.get("query_encode_seconds"),
            timing.get("retrieval_total_seconds"),
        )
    )
    return finished_at - timedelta(seconds=duration)


def variant_config(plan: PlanDefinition, variant: RunVariant) -> dict:
    config: dict[str, object] = {
        "method": variant.method,
        "top_k": plan.evaluation.top_k,
        "k_values": list(plan.evaluation.k_values),
    }
    if variant.method in ("bm25", "hybrid"):
        config.update(
            {"k1": variant.k1, "b": variant.b, "tokenizer": variant.tokenizer}
        )
    if variant.method in ("dense", "hybrid"):
        config.update(
            {
                "model": variant.model.name,
                "model_revision": variant.model.revision,
                "batch_size": variant.batch_size,
                "normalize_embeddings": variant.normalize_embeddings,
                "query_prefix": variant.query_prefix,
            }
        )
    if variant.method == "hybrid":
        config.update(
            {
                "rrf_k": variant.rrf_k,
                "rrf_weights": list(variant.rrf_weights),
            }
        )
    return config


def check_source(plan: PlanDefinition, variant: RunVariant, source: dict) -> list[str]:
    """校验旧评估 JSON 与方案配置一致。"""
    errors: list[str] = []
    config = source.get("config") or {}
    dataset = source.get("dataset") or {}

    if config.get("method") != variant.method:
        errors.append(
            f"method 不一致：源 {config.get('method')}，方案 {variant.method}"
        )
    if config.get("top_k") != plan.evaluation.top_k:
        errors.append(
            f"top_k 不一致：源 {config.get('top_k')}，方案 {plan.evaluation.top_k}"
        )
    if list(config.get("k_values") or []) != list(plan.evaluation.k_values):
        errors.append(
            f"k_values 不一致：源 {config.get('k_values')}，方案 {list(plan.evaluation.k_values)}"
        )
    source_dataset = dataset.get("name") or config.get("dataset")
    if source_dataset != plan.dataset.name:
        errors.append(
            f"数据集不一致：源 {source_dataset}，方案 {plan.dataset.name}"
        )

    if variant.method in ("bm25", "hybrid"):
        if config.get("k1") != variant.k1:
            errors.append(f"k1 不一致：源 {config.get('k1')}，方案 {variant.k1}")
        if config.get("b") != variant.b:
            errors.append(f"b 不一致：源 {config.get('b')}，方案 {variant.b}")
        tokenizer = config.get("tokenizer", "simple")
        if tokenizer != variant.tokenizer:
            errors.append(
                f"tokenizer 不一致：源 {tokenizer}，方案 {variant.tokenizer}"
            )

    if variant.method in ("dense", "hybrid"):
        if config.get("model") != variant.model.name:
            errors.append(
                f"模型不一致：源 {config.get('model')}，方案 {variant.model.name}"
            )
        if config.get("model_revision") != variant.model.revision:
            errors.append(
                "模型 revision 不一致："
                f"源 {config.get('model_revision')}，方案 {variant.model.revision}"
            )
        if (config.get("query_prefix") or "") != (variant.query_prefix or ""):
            errors.append("query_prefix 与方案不一致")
        if config.get("batch_size", variant.batch_size) != variant.batch_size:
            errors.append("batch_size 与方案不一致")
        if config.get(
            "normalize_embeddings", variant.normalize_embeddings
        ) != variant.normalize_embeddings:
            errors.append("normalize_embeddings 与方案不一致")

    if variant.method == "hybrid":
        if config.get("rrf_k") != variant.rrf_k:
            errors.append(
                f"rrf_k 不一致：源 {config.get('rrf_k')}，方案 {variant.rrf_k}"
            )
        weights = config.get("rrf_weights")
        # 旧等权结果不写 rrf_weights，按默认 [1.0, 1.0] 解释。
        effective_weights = (
            [1.0, 1.0] if weights is None else list(weights)
        )
        if variant.rrf_weights is not None:
            if effective_weights != list(variant.rrf_weights):
                errors.append(
                    f"rrf_weights 不一致：源 {weights}，方案 {list(variant.rrf_weights)}"
                )
        elif weights not in (None, [1.0, 1.0]):
            errors.append(f"等权变体不应带 rrf_weights：源 {weights}")

    return errors


def build_record(
    plan: PlanDefinition,
    variant: RunVariant,
    source: dict,
    source_path: Path,
) -> tuple[dict, dict]:
    """生成待校验的 RunRecord 字典与图表所需 execution。"""
    timing = map_timing(variant.method, source.get("timing") or {})
    finished_at = datetime.fromtimestamp(
        source_path.stat().st_mtime, tz=timezone.utc
    )
    started_at = archived_start(variant.method, timing, finished_at)
    dataset = source.get("dataset") or {}
    record = {
        "format_version": "1.0",
        "run_id": variant.preset_run_id,
        "plan_id": plan.plan_id,
        "plan_version": plan.plan_version,
        "variant_id": variant.variant_id,
        "plan_name": plan.name,
        "variant_name": variant.name,
        "status": "Completed",
        "stage_progress": {
            "completed": list(applicable_stages(variant.method)),
            "current": None,
            "not_applicable": list(not_applicable_stages(variant.method)),
        },
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "cancelled_at": None,
        "dataset": {
            "name": plan.dataset.name,
            "split": plan.dataset.split,
            "corpus_size": dataset.get("corpus_size"),
            "query_count": source.get("num_queries"),
            "qrels_pairs": dataset.get("qrels_pairs"),
        },
        "variant_config": variant_config(plan, variant),
        "metrics": source.get("metrics") or {},
        "num_queries": source.get("num_queries"),
        "num_skipped": source.get("num_skipped"),
        "timing": timing,
        "environment": source.get("environment") or {},
        "charts": [],
        "error": None,
    }
    execution = SimpleNamespace(
        metrics=record["metrics"], timing=record["timing"]
    )
    return record, execution


def check_assets() -> list[str]:
    """构建前检查 SciFact 数据、来源/许可文件与方案源文件。"""
    errors: list[str] = []
    for relative in SCIFACT_DATASET_FILES:
        path = SCIFACT_DIR / relative
        if not path.is_file() or path.stat().st_size <= 0:
            errors.append(f"缺少非空的 SciFact 数据文件: {relative}")
    for name in RESULT_ASSETS:
        path = RESOURCES_DIR / name
        if not path.is_file() or path.stat().st_size <= 0:
            errors.append(f"缺少非空的 SciFact 发布资产: resources/scifact/{name}")
    if not ROOT_NOTICE.is_file() or ROOT_NOTICE.stat().st_size <= 0:
        errors.append("缺少非空的根 NOTICE.md")
    return errors


def _check_sources(catalog: PlanCatalog) -> list[str]:
    errors: list[str] = []
    for plan in catalog.plans:
        for variant in plan.variants:
            if not isinstance(variant, RunVariant):
                continue
            path = PROJECT_ROOT / variant.preset_source
            if not path.is_file() or path.stat().st_size <= 0:
                errors.append(
                    f"{variant.variant_id}: 缺少非空的评估文件 {variant.preset_source}"
                )
    return errors


def _build_one(
    plan: PlanDefinition,
    variant: RunVariant,
    output: Path,
    force: bool,
) -> Path:
    source_path = PROJECT_ROOT / variant.preset_source
    source = json.loads(source_path.read_text(encoding="utf-8"))
    errors = check_source(plan, variant, source)
    if errors:
        raise ValueError("; ".join(errors))

    record, execution = build_record(plan, variant, source, source_path)
    staging = output / f".tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        task = SimpleNamespace(
            run_id=variant.preset_run_id,
            plan_id=plan.plan_id,
            variant_id=variant.variant_id,
            config_snapshot={
                "plan_name": plan.name,
                "variant_name": variant.name,
                "variant": variant.model_dump(mode="json"),
            },
        )
        charts = generate_charts(task, execution, staging / "charts")
        record["charts"] = charts
        try:
            parse_run_record(record)
        except RunRecordError as exc:
            raise ValueError(f"生成的记录不符合 Schema：{exc}") from exc

        (staging / "run.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        target = output / variant.preset_run_id
        if target.exists():
            if not force:
                raise ValueError(f"目标目录已存在: {variant.preset_run_id}")
            shutil.rmtree(target)
        staging.rename(target)
        staging = None
        return target
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def build_all(
    catalog: PlanCatalog, output: Path, force: bool
) -> tuple[list[Path], list[tuple[str, str]]]:
    variants = [
        (plan, variant)
        for plan in catalog.plans
        for variant in plan.variants
        if isinstance(variant, RunVariant)
    ]
    output.mkdir(parents=True, exist_ok=True)
    published: list[Path] = []
    failures: list[tuple[str, str]] = []
    for plan, variant in variants:
        try:
            published.append(_build_one(plan, variant, output, force))
        except Exception as exc:  # noqa: BLE001 - 单条失败继续收集
            failures.append((variant.variant_id, str(exc)))
    return published, failures


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建预置运行记录")
    parser.add_argument(
        "--output",
        default=str(PRESETS_DIR),
        help="预置结果输出目录（默认 results/presets）",
    )
    parser.add_argument(
        "--check-assets-only",
        action="store_true",
        help="只检查数据、来源与许可文件，不构建",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="替换输出目录中本脚本生成的预置记录",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    catalog = load_plan_catalog()

    errors = check_assets()
    if errors:
        for error in errors:
            print(f"资源检查失败: {error}")
        return 1
    source_errors = _check_sources(catalog)
    if source_errors:
        for error in source_errors:
            print(f"源文件检查失败: {error}")
        return 1
    if args.check_assets_only:
        print("资源检查通过：SciFact 数据、来源/许可文件与评估源文件均存在。")
        return 0

    output = Path(args.output)
    run_variants = [
        (plan, variant)
        for plan in catalog.plans
        for variant in plan.variants
        if isinstance(variant, RunVariant)
    ]
    if output.exists():
        existing = [
            child
            for child in output.iterdir()
            if child.is_dir() and child.name.startswith("preset-")
        ]
        if existing and not args.force:
            print(
                f"输出目录已存在 {len(existing)} 条预置记录：{output}；"
                "如需替换请使用 --force"
            )
            return 1

    published, failures = build_all(catalog, output, args.force)
    for variant_id, reason in failures:
        print(f"变体构建失败: {variant_id}: {reason}")
    if failures:
        for target in published:
            shutil.rmtree(target, ignore_errors=True)
        print(f"已回滚本次构建的 {len(published)} 条记录；构建失败 {len(failures)} 条。")
        return 1

    expected = len(run_variants)
    if len(published) != expected:
        print(f"预置记录数量不符：期望 {expected}，实际 {len(published)}")
        return 1

    for target in published:
        run_json = target / "run.json"
        parse_run_record(json.loads(run_json.read_text(encoding="utf-8")))
        for chart in json.loads(run_json.read_text(encoding="utf-8")).get(
            "charts", []
        ):
            if not (target / chart["path"]).is_file():
                raise SystemExit(f"图表缺失: {target.name}/{chart['path']}")

    print(
        f"已生成 {len(published)} 条预置记录到 {output}。"
        "开始时间由源文件 mtime 与可分解 timing 推导，原始评估 JSON 未修改。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
