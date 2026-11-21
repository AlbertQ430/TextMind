"""三类结果目录的统一扫描与读取。

来源由顶层目录决定：`results/presets`、`results/runs`、`results/imports`。
目录/JSON 不一致、损坏记录和跨来源重复 Run ID 只进入不可用列表，
单条记录损坏不影响其他记录。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment.plans import (
    PlanCatalog,
    ReferenceVariant,
    RunVariant,
)
from paths import IMPORTS_DIR, PRESETS_DIR, RUNS_DIR
from records.schema import RunRecord, RunRecordError, parse_run_record

SOURCE_PRESETS = "presets"
SOURCE_RUNS = "runs"
SOURCE_IMPORTS = "imports"

SOURCE_DIRS: dict[str, Path] = {
    SOURCE_PRESETS: PRESETS_DIR,
    SOURCE_RUNS: RUNS_DIR,
    SOURCE_IMPORTS: IMPORTS_DIR,
}

CODE_RECORD_MISSING = "RUN_RECORD_MISSING"
CODE_RECORD_INVALID = "RUN_RECORD_INVALID"
CODE_RECORD_ID_MISMATCH = "RUN_RECORD_ID_MISMATCH"
CODE_RECORD_UNSAFE = "RUN_RECORD_UNSAFE"
CODE_RUN_ID_CONFLICT = "RUN_ID_CONFLICT"


def _is_within(path: Path, root: Path) -> bool:
    """解析符号链接后仍位于 root 内。"""
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
    except OSError:
        return False
    return resolved == root_resolved or root_resolved in resolved.parents


class CatalogError(Exception):
    """结果目录读取错误，携带 API 错误码和状态码。"""

    code = "RUN_UNAVAILABLE"
    status_code = 409

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class RunNotFound(CatalogError):
    code = "RUN_NOT_FOUND"
    status_code = 404


class RunIdConflict(CatalogError):
    code = "RUN_ID_CONFLICT"
    status_code = 409


class RunUnavailable(CatalogError):
    code = "RUN_UNAVAILABLE"
    status_code = 409


class ChartNotFound(CatalogError):
    code = "CHART_NOT_FOUND"
    status_code = 404


class ChartUnavailable(CatalogError):
    code = "CHART_UNAVAILABLE"
    status_code = 404


class ChartUnsafe(CatalogError):
    code = "CHART_UNSAFE"
    status_code = 409


@dataclass(frozen=True)
class RecordEntry:
    source: str
    run_id: str
    directory: Path
    record: RunRecord
    is_reference: bool = False


@dataclass(frozen=True)
class UnavailableEntry:
    source: str
    run_id: str
    code: str
    message: str


@dataclass(frozen=True)
class ScanResult:
    records: list[RecordEntry]
    unavailable: list[UnavailableEntry]
    references: list[RecordEntry]


def _read_record(run_json: Path) -> RunRecord:
    data = json.loads(run_json.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RunRecordError("运行记录顶层必须是 JSON 对象")
    return parse_run_record(data)


def _is_readable(path: Path) -> bool:
    """实际打开并读取 1 字节，确认文件可读。"""
    try:
        with path.open("rb") as handle:
            handle.read(1)
    except OSError:
        return False
    return True


class ResultCatalog:
    """跨三类结果目录的只读索引。"""

    def __init__(
        self,
        *,
        presets_root: Path | str | None = None,
        runs_root: Path | str | None = None,
        imports_root: Path | str | None = None,
        catalog: PlanCatalog | None = None,
    ) -> None:
        self._roots = {
            SOURCE_PRESETS: Path(presets_root) if presets_root else PRESETS_DIR,
            SOURCE_RUNS: Path(runs_root) if runs_root else RUNS_DIR,
            SOURCE_IMPORTS: Path(imports_root) if imports_root else IMPORTS_DIR,
        }
        self._catalog = catalog

    def source_roots(self) -> dict[str, Path]:
        return dict(self._roots)

    # ------------------------------------------------------------------ 参照识别
    def reference_run_ids(self) -> set[str]:
        if self._catalog is None:
            return set()
        ids: set[str] = set()
        for plan in self._catalog.plans:
            for variant in plan.variants:
                if not isinstance(variant, ReferenceVariant):
                    continue
                for source_plan in self._catalog.plans:
                    if (
                        source_plan.plan_id != variant.source_plan_id
                        or source_plan.plan_version
                        != variant.source_plan_version
                    ):
                        continue
                    for source_variant in source_plan.variants:
                        if (
                            isinstance(source_variant, RunVariant)
                            and source_variant.variant_id
                            == variant.source_variant_id
                        ):
                            ids.add(source_variant.preset_run_id)
        return ids

    # ------------------------------------------------------------------ 扫描
    def _scan_entries(self) -> tuple[dict[str, list[tuple[str, Path, RunRecord]]], list[UnavailableEntry]]:
        occurrences: dict[str, list[tuple[str, Path, RunRecord]]] = {}
        unavailable: list[UnavailableEntry] = []
        for source, root in self._roots.items():
            if not root.is_dir():
                continue
            root_resolved = root.resolve()
            for child in sorted(root.iterdir(), key=lambda item: item.name):
                if not child.is_dir() or child.name.startswith("."):
                    continue
                if not _is_within(child, root_resolved):
                    unavailable.append(
                        UnavailableEntry(
                            source,
                            child.name,
                            CODE_RECORD_UNSAFE,
                            "运行目录解析后越出来源目录，拒绝读取",
                        )
                    )
                    continue
                run_json = child / "run.json"
                if not run_json.is_file():
                    unavailable.append(
                        UnavailableEntry(
                            source,
                            child.name,
                            CODE_RECORD_MISSING,
                            "运行记录文件缺失",
                        )
                    )
                    continue
                if not _is_within(run_json, child.resolve()):
                    unavailable.append(
                        UnavailableEntry(
                            source,
                            child.name,
                            CODE_RECORD_UNSAFE,
                            "run.json 解析后越出运行目录，拒绝读取",
                        )
                    )
                    continue
                try:
                    record = _read_record(run_json)
                except (OSError, ValueError, RunRecordError):
                    unavailable.append(
                        UnavailableEntry(
                            source,
                            child.name,
                            CODE_RECORD_INVALID,
                            "运行记录无效",
                        )
                    )
                    continue
                if record.run_id != child.name:
                    unavailable.append(
                        UnavailableEntry(
                            source,
                            child.name,
                            CODE_RECORD_ID_MISMATCH,
                            "目录名与运行记录 ID 不一致",
                        )
                    )
                    continue
                occurrences.setdefault(record.run_id, []).append(
                    (source, child, record)
                )
        return occurrences, unavailable

    def scan(self) -> ScanResult:
        occurrences, unavailable = self._scan_entries()
        reference_ids = self.reference_run_ids()
        records: list[RecordEntry] = []
        references: list[RecordEntry] = []
        for run_id, items in occurrences.items():
            if len(items) > 1:
                for source, _, _ in items:
                    unavailable.append(
                        UnavailableEntry(
                            source,
                            run_id,
                            CODE_RUN_ID_CONFLICT,
                            "同一 Run ID 出现在多个来源，无法确定唯一记录",
                        )
                    )
                continue
            source, directory, record = items[0]
            entry = RecordEntry(
                source=source,
                run_id=run_id,
                directory=directory,
                record=record,
                is_reference=run_id in reference_ids,
            )
            if entry.is_reference:
                references.append(entry)
            else:
                records.append(entry)

        records.sort(key=lambda entry: entry.run_id)
        records.sort(key=lambda entry: entry.record.started_at, reverse=True)
        references.sort(key=lambda entry: entry.run_id)
        unavailable.sort(key=lambda item: (item.run_id, item.source))
        return ScanResult(records, unavailable, references)

    # ------------------------------------------------------------------ 读取
    def get(self, run_id: str) -> RecordEntry:
        scan = self.scan()
        for entry in (*scan.records, *scan.references):
            if entry.run_id == run_id:
                return entry
        for item in scan.unavailable:
            if item.run_id != run_id:
                continue
            if item.code == CODE_RUN_ID_CONFLICT:
                raise RunIdConflict(
                    f"Run ID {run_id} 出现在多个来源，无法确定唯一记录"
                )
            raise RunUnavailable(f"运行记录不可用: {item.message}")
        raise RunNotFound(f"运行记录不存在: {run_id}")

    # ------------------------------------------------------------------ 方案与资格
    def known_plan(self, record: RunRecord):
        if self._catalog is None:
            return None
        for plan in self._catalog.plans:
            if (
                plan.plan_id != record.plan_id
                or plan.plan_version != record.plan_version
            ):
                continue
            for variant in plan.variants:
                if (
                    isinstance(variant, RunVariant)
                    and variant.variant_id == record.variant_id
                ):
                    return plan, variant
            return None
        return None

    def comparison_eligible(self, record: RunRecord) -> bool:
        return (
            record.status == "Completed"
            and bool(record.metrics)
            and self.known_plan(record) is not None
        )

    def reference_baseline(
        self, plan_id: str, plan_version: str
    ) -> RecordEntry | None:
        if self._catalog is None:
            return None
        plan = None
        for candidate in self._catalog.plans:
            if (
                candidate.plan_id == plan_id
                and candidate.plan_version == plan_version
            ):
                plan = candidate
                break
        if plan is None:
            return None
        preset_ids = set()
        for variant in plan.variants:
            if not isinstance(variant, ReferenceVariant):
                continue
            for source_plan in self._catalog.plans:
                if (
                    source_plan.plan_id != variant.source_plan_id
                    or source_plan.plan_version != variant.source_plan_version
                ):
                    continue
                for source_variant in source_plan.variants:
                    if (
                        isinstance(source_variant, RunVariant)
                        and source_variant.variant_id
                        == variant.source_variant_id
                    ):
                        preset_ids.add(source_variant.preset_run_id)
        if not preset_ids:
            return None
        for entry in self.scan().references:
            if (
                entry.run_id in preset_ids
                and entry.record.status == "Completed"
                and entry.record.metrics
            ):
                return entry
        return None

    # ------------------------------------------------------------------ 图表
    def chart_file(self, entry: RecordEntry, index: int) -> Path:
        charts = entry.record.charts
        if index < 0 or index >= len(charts):
            raise ChartNotFound(f"图表索引不存在: {index}")
        raw = charts[index].path
        relative = Path(raw)
        if relative.is_absolute():
            raise ChartUnsafe("图表路径必须是记录目录内的相对路径")
        directory = entry.directory.resolve()
        target = (directory / relative).resolve()
        if directory not in target.parents:
            raise ChartUnsafe("图表路径越出运行目录")
        if not target.is_file():
            raise ChartUnavailable("图表文件缺失或不可读")
        if not _is_readable(target):
            raise ChartUnavailable("图表文件不可读")
        return target

    def chart_available(self, entry: RecordEntry, index: int) -> bool:
        try:
            self.chart_file(entry, index)
        except CatalogError:
            return False
        return True

    # ------------------------------------------------------------------ 变化指纹
    def fingerprint(self) -> str:
        """目录集合与 run.json 的文件系统指纹，用于 SSE 变化检测。"""
        parts: list[tuple[Any, ...]] = []
        for source, root in self._roots.items():
            if not root.is_dir():
                continue
            for child in sorted(root.iterdir(), key=lambda item: item.name):
                if not child.is_dir() or child.name.startswith("."):
                    continue
                run_json = child / "run.json"
                try:
                    stat = run_json.stat()
                except OSError:
                    parts.append((source, child.name, None, None))
                    continue
                parts.append(
                    (source, child.name, stat.st_mtime_ns, stat.st_size)
                )
        return json.dumps(parts, ensure_ascii=False)
