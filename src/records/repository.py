"""结果目录：staging、原子发布、记录读取与路径安全。

结果来源由目录决定（runs / presets / imports），记录内不保存来源字段。
"""

from __future__ import annotations

import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paths import RUNS_DIR
from records.schema import RUN_ID_PATTERN, RunRecord, RunRecordError, parse_run_record

_RUN_ID_RE = re.compile(RUN_ID_PATTERN)


class ResultRecordError(Exception):
    """运行记录损坏或不符合 Schema。"""


class ResultConflictError(Exception):
    """目标目录已存在或存在多个合法 staging 候选。"""


class ResultPathError(ValueError):
    """记录路径越界或非法。"""


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id) or run_id in (".", ".."):
        raise ResultPathError(f"非法 Run ID: {run_id!r}")


def _ensure_within(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ResultPathError(f"路径越界: {path}")
    return resolved


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat()


def generate_attempt_id(now: datetime | None = None) -> str:
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return f"{moment.strftime('%Y%m%dT%H%M%S%fZ')}-{secrets.token_hex(3)}"


class ResultRepository:
    """`results/runs` 下的记录目录管理。"""

    def __init__(self, runs_root: Path | str | None = None) -> None:
        self._root = Path(runs_root) if runs_root is not None else RUNS_DIR
        self._staging_root = self._root / ".staging"

    @property
    def root(self) -> Path:
        return self._root

    @property
    def staging_root(self) -> Path:
        return self._staging_root

    def record_directory(self, run_id: str) -> Path:
        _validate_run_id(run_id)
        return self._root / run_id

    def record_path(self, run_id: str) -> Path:
        return self.record_directory(run_id) / "run.json"

    # ------------------------------------------------------------------ staging
    def create_staging(self, run_id: str) -> Path:
        """创建本次发布独占的 staging 目录。"""
        _validate_run_id(run_id)
        self._staging_root.mkdir(parents=True, exist_ok=True)
        for _ in range(5):
            staging = self._staging_root / f"{run_id}-{generate_attempt_id()}"
            try:
                staging.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            return staging
        raise ResultConflictError(f"无法创建 staging 目录: {run_id}")

    def find_staging_candidates(self, run_id: str) -> list[Path]:
        """返回属于该 run_id 的 staging 目录（按名称排序）。"""
        _validate_run_id(run_id)
        if not self._staging_root.is_dir():
            return []
        prefix = f"{run_id}-"
        candidates = [
            path
            for path in self._staging_root.iterdir()
            if path.is_dir()
            and path.name.startswith(prefix)
            and len(path.name) > len(prefix)
        ]
        return sorted(candidates, key=lambda path: path.name)

    def publish(self, staging: Path, run_id: str) -> Path:
        """把 staging 目录原子发布为最终目录；最终目录已存在时拒绝。"""
        _validate_run_id(run_id)
        staging = _ensure_within(Path(staging), self._staging_root)
        if not staging.is_dir():
            raise ResultPathError(f"staging 目录不存在: {staging}")
        final = self.record_directory(run_id)
        if final.exists():
            raise ResultConflictError(f"最终结果目录已存在: {final}")
        final.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, final)
        return final

    # ------------------------------------------------------------------ 记录读写
    def write_record(self, directory: Path, data: dict[str, Any]) -> RunRecord:
        """校验并原子写入 `run.json`（staging 内）。"""
        try:
            record = parse_run_record(data)
        except RunRecordError as exc:
            raise ResultRecordError(f"运行记录校验失败: {exc}") from exc
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tmp_path = directory / "run.json.tmp"
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp_path, directory / "run.json")
        return record

    def read_at(self, directory: Path) -> RunRecord | None:
        """读取目录中的 `run.json`；不存在返回 None，损坏抛 ResultRecordError。"""
        directory = Path(directory)
        path = directory / "run.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ResultRecordError(f"无法读取运行记录 {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ResultRecordError(f"运行记录顶层必须是 JSON 对象: {path}")
        try:
            return parse_run_record(data)
        except RunRecordError as exc:
            raise ResultRecordError(f"运行记录无效 {path}: {exc}") from exc

    def read(self, run_id: str) -> RunRecord | None:
        """读取最终目录中的记录；不存在返回 None，损坏抛 ResultRecordError。"""
        _validate_run_id(run_id)
        return self.read_at(self.record_directory(run_id))

    # ------------------------------------------------------------------ 图表校验
    def validate_chart_paths(
        self, data: dict[str, Any], directory: Path
    ) -> None:
        """确认每个图表路径都是目录内的相对 POSIX 路径且文件存在。"""
        directory = Path(directory).resolve()
        charts = data.get("charts") or []
        if not isinstance(charts, list):
            raise ResultRecordError("charts 必须是数组")
        for chart in charts:
            if not isinstance(chart, dict) or "path" not in chart:
                raise ResultRecordError(f"非法图表条目: {chart!r}")
            raw = chart["path"]
            if not isinstance(raw, str) or not raw:
                raise ResultRecordError(f"非法图表路径: {raw!r}")
            relative = Path(raw)
            if relative.is_absolute():
                raise ResultRecordError(f"图表路径必须是相对路径: {raw}")
            resolved = (directory / relative).resolve()
            if directory not in resolved.parents:
                raise ResultRecordError(f"图表路径越界: {raw}")
            if not resolved.is_file():
                raise ResultRecordError(f"图表文件不存在: {raw}")
