"""单份运行记录 JSON 导入。

导入只接受一份 `.json` 上传，按“校验 → 暂存 → 原子发布”顺序写入
`results/imports/<run_id>/run.json`。写入内容与上传字节完全一致，
不修改 Schema、不补写来源字段、不生成侧车文件。
"""

from __future__ import annotations

import errno
import json
import math
import os
import re
import secrets
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

from paths import IMPORTS_DIR
from records.catalog import ResultCatalog
from records.schema import RUN_ID_PATTERN, RunRecordError, parse_run_record

MAX_IMPORT_BYTES = 5 * 1024 * 1024
_JSON_SUFFIX = ".json"
_RUN_ID_RE = re.compile(RUN_ID_PATTERN)


class ImportFailure(Exception):
    """导入失败，携带 API 错误码与状态码。"""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: list[str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = list(details or [])
        super().__init__(message)


@dataclass(frozen=True)
class ImportResult:
    run_id: str
    bytes_written: int
    missing_charts: list[dict]
    warnings: list[str]


class _DuplicateKey(Exception):
    def __init__(self, key: str) -> None:
        self.key = key


class _InvalidConstant(Exception):
    def __init__(self, value: str) -> None:
        self.value = value


def _unique_object(pairs):
    seen: dict = {}
    for key, value in pairs:
        if key in seen:
            raise _DuplicateKey(key)
        seen[key] = value
    return seen


def _reject_constant(value: str):
    raise _InvalidConstant(value)


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise _InvalidConstant(value)
    return number


def unsafe_chart_path(path: str) -> str | None:
    """返回不安全原因；安全时返回 None。"""
    if not path or not path.strip():
        return "空路径"
    if any(ord(character) < 32 or ord(character) == 127 for character in path):
        return "包含控制字符"
    if path.startswith("/"):
        return "绝对路径"
    if path[1:2] == ":":
        return "包含盘符"
    if "\\" in path:
        return "包含反斜杠"
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return "包含空段、. 或 .."
    return None


class ImportService:
    """把上传的 JSON 原子导入为 `imports` 来源记录。"""

    def __init__(
        self,
        imports_root: Path | str | None = None,
        *,
        catalog: ResultCatalog | None = None,
    ) -> None:
        self._imports_root = (
            Path(imports_root) if imports_root is not None else IMPORTS_DIR
        )
        self._catalog = catalog
        self._lock = threading.Lock()

    @property
    def imports_root(self) -> Path:
        return self._imports_root

    def _source_roots(self) -> list[tuple[str, Path]]:
        if self._catalog is not None:
            return list(self._catalog.source_roots().items())
        return [("imports", self._imports_root)]

    def _conflict_source(self, run_id: str) -> str | None:
        for source, root in self._source_roots():
            target = root / run_id
            if target.exists() or target.is_symlink():
                return source
        return None

    # ------------------------------------------------------------------ 导入
    def import_json(self, filename: str | None, content: bytes) -> ImportResult:
        if not filename:
            raise ImportFailure(
                422, "IMPORT_FILE_INVALID", "文件名缺失，请选择 .json 文件"
            )
        if Path(filename).suffix.lower() != _JSON_SUFFIX:
            raise ImportFailure(
                422, "IMPORT_FILE_INVALID", "只支持扩展名为 .json 的文件"
            )
        if not content:
            raise ImportFailure(422, "IMPORT_FILE_INVALID", "文件为空")
        if len(content) > MAX_IMPORT_BYTES:
            raise ImportFailure(
                413, "IMPORT_TOO_LARGE", "文件超过 5 MiB 上限"
            )

        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise ImportFailure(
                422, "IMPORT_INVALID_ENCODING", "文件不是有效的 UTF-8 编码"
            ) from None

        try:
            data = json.loads(
                text,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
                parse_float=_finite_float,
            )
        except _DuplicateKey as exc:
            raise ImportFailure(
                422,
                "IMPORT_DUPLICATE_KEY",
                f"JSON 中存在重复键: {exc.key}",
                [str(exc.key)],
            ) from None
        except _InvalidConstant as exc:
            raise ImportFailure(
                422,
                "IMPORT_INVALID_JSON",
                f"不支持的 JSON 常量: {exc.value}",
            ) from None
        except json.JSONDecodeError as exc:
            raise ImportFailure(
                422,
                "IMPORT_INVALID_JSON",
                f"JSON 语法错误（第 {exc.lineno} 行第 {exc.colno} 列）",
            ) from None
        except RecursionError:
            raise ImportFailure(
                422, "IMPORT_INVALID_JSON", "JSON 嵌套层级过深"
            ) from None

        if not isinstance(data, dict):
            raise ImportFailure(
                422, "IMPORT_SCHEMA_INVALID", "运行记录顶层必须是 JSON 对象"
            )
        if data.get("format_version") != "1.0":
            raise ImportFailure(
                422,
                "IMPORT_VERSION_UNSUPPORTED",
                "仅支持 format_version=1.0 的运行记录",
            )
        run_id = data.get("run_id")
        if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id):
            raise ImportFailure(
                422,
                "IMPORT_SCHEMA_INVALID",
                "run_id 缺失或格式非法",
                ["run_id"],
            )

        charts = data.get("charts")
        if isinstance(charts, list):
            for index, chart in enumerate(charts):
                if not isinstance(chart, dict):
                    continue
                path = chart.get("path")
                if not isinstance(path, str):
                    continue
                reason = unsafe_chart_path(path)
                if reason is not None:
                    raise ImportFailure(
                        422,
                        "IMPORT_PATH_INVALID",
                        f"图表路径不安全: {reason}",
                        [f"charts[{index}].path"],
                    )

        with self._lock:
            conflict = self._conflict_source(run_id)
            if conflict is not None:
                raise ImportFailure(
                    409,
                    "IMPORT_RUN_ID_CONFLICT",
                    f"Run ID 已存在于 {conflict} 来源，不能覆盖",
                )

            try:
                record = parse_run_record(data)
            except RunRecordError as exc:
                first_line = str(exc).splitlines()[0]
                raise ImportFailure(
                    422,
                    "IMPORT_SCHEMA_INVALID",
                    "运行记录不符合 Schema",
                    [first_line],
                ) from None
            except RecursionError:
                raise ImportFailure(
                    422, "IMPORT_SCHEMA_INVALID", "运行记录嵌套层级过深"
                ) from None

            self._publish(run_id, content)

        missing_charts = [
            {"id": chart.id, "title": chart.title} for chart in record.charts
        ]
        warnings = [
            f"图表 {chart['id']} 未随 JSON 导入，当前不可用"
            for chart in missing_charts
        ]
        return ImportResult(
            run_id=run_id,
            bytes_written=len(content),
            missing_charts=missing_charts,
            warnings=warnings,
        )

    def _publish(self, run_id: str, content: bytes) -> None:
        self._imports_root.mkdir(parents=True, exist_ok=True)
        staging: Path | None = None
        try:
            for _ in range(5):
                candidate = self._imports_root / f".tmp-{secrets.token_hex(8)}"
                try:
                    candidate.mkdir(parents=True, exist_ok=False)
                except FileExistsError:
                    continue
                staging = candidate
                break
            if staging is None:
                raise OSError("无法创建暂存目录")

            target = staging / "run.json"
            with target.open("wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())

            conflict = self._conflict_source(run_id)
            if conflict is not None:
                raise ImportFailure(
                    409,
                    "IMPORT_RUN_ID_CONFLICT",
                    f"Run ID 已存在于 {conflict} 来源，不能覆盖",
                )
            final = self._imports_root / run_id
            if final.exists() or final.is_symlink():
                raise ImportFailure(
                    409,
                    "IMPORT_RUN_ID_CONFLICT",
                    "导入目标目录已存在，不能覆盖",
                )
            try:
                os.rename(staging, final)
            except OSError as exc:
                if exc.errno in (
                    errno.EEXIST,
                    errno.ENOTEMPTY,
                    errno.EISDIR,
                    errno.ENOTDIR,
                ):
                    raise ImportFailure(
                        409,
                        "IMPORT_RUN_ID_CONFLICT",
                        "导入目标目录已存在，不能覆盖",
                    ) from None
                raise
            staging = None
        except ImportFailure:
            raise
        except OSError:
            raise ImportFailure(
                500,
                "IMPORT_WRITE_FAILED",
                "写入导入记录失败，已清理暂存目录",
            ) from None
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
