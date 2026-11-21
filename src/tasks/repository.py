"""SQLite 任务与 notices 仓储。

每个操作使用独立连接，便于工作线程和请求线程共享同一仓储实例。
数据库默认位于 ``data/app/textmind.db``，启用 WAL。
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from paths import TASKS_DB_PATH
from tasks.models import (
    NOTICE_KINDS,
    Notice,
    TaskConflictError,
    TaskNotFoundError,
    TaskRecord,
    TaskStateError,
    TaskStatus,
    ensure_transition,
    utc_now,
)

_UNSET = object()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    run_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL,
    plan_version TEXT NOT NULL,
    variant_id TEXT NOT NULL,
    status TEXT NOT NULL,
    current_stage TEXT,
    completed_stages_json TEXT NOT NULL DEFAULT '[]',
    submitted_at TEXT NOT NULL,
    started_at TEXT,
    ended_at TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    config_snapshot_json TEXT NOT NULL,
    error_summary TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_status_submitted
ON tasks(status, submitted_at);

CREATE TABLE IF NOT EXISTS notices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    count INTEGER NOT NULL CHECK (count > 0),
    created_at TEXT NOT NULL
);
"""


def _to_iso(moment: datetime) -> str:
    return moment.isoformat()


def _parse_time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _to_record(row: sqlite3.Row) -> TaskRecord:
    return TaskRecord(
        run_id=row["run_id"],
        plan_id=row["plan_id"],
        plan_version=row["plan_version"],
        variant_id=row["variant_id"],
        status=TaskStatus(row["status"]),
        current_stage=row["current_stage"],
        completed_stages=tuple(
            json.loads(row["completed_stages_json"] or "[]")
        ),
        submitted_at=_parse_time(row["submitted_at"]),
        started_at=_parse_time(row["started_at"]),
        ended_at=_parse_time(row["ended_at"]),
        cancel_requested=bool(row["cancel_requested"]),
        config_snapshot=json.loads(row["config_snapshot_json"]),
        error_summary=row["error_summary"],
    )


class TaskRepository:
    """任务状态和 notices 的唯一持久化入口。"""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._path = Path(db_path) if db_path is not None else TASKS_DB_PATH

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------ 连接与初始化
    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self._path, timeout=10.0, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """显式 BEGIN IMMEDIATE，写事务串行化，避免基于过期状态提交。"""
        connection = self._connect()
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """幂等建表并启用 WAL。"""
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(_SCHEMA)
            connection.commit()
        finally:
            connection.close()

    # ------------------------------------------------------------------ 任务写入
    def create_pending(self, record: TaskRecord) -> TaskRecord:
        try:
            with self._write() as connection:
                connection.execute(
                    """
                    INSERT INTO tasks (
                        run_id, plan_id, plan_version, variant_id, status,
                        current_stage, completed_stages_json, submitted_at,
                        started_at, ended_at, cancel_requested,
                        config_snapshot_json, error_summary
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.run_id,
                        record.plan_id,
                        record.plan_version,
                        record.variant_id,
                        record.status.value,
                        record.current_stage,
                        json.dumps(
                            list(record.completed_stages), ensure_ascii=False
                        ),
                        _to_iso(record.submitted_at),
                        _to_iso(record.started_at) if record.started_at else None,
                        _to_iso(record.ended_at) if record.ended_at else None,
                        1 if record.cancel_requested else 0,
                        json.dumps(record.config_snapshot, ensure_ascii=False),
                        record.error_summary,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise TaskConflictError(f"任务已存在: {record.run_id}") from exc
        return record

    def transition(
        self,
        run_id: str,
        target: TaskStatus,
        *,
        current_stage: str | None | object = _UNSET,
        started_at: datetime | None | object = _UNSET,
        ended_at: datetime | None | object = _UNSET,
        cancel_requested: bool | object = _UNSET,
        error_summary: str | None | object = _UNSET,
    ) -> TaskRecord:
        with self._write() as connection:
            record = self._require(connection, run_id)
            ensure_transition(record.status, target)

            updates = ["status = ?"]
            params: list[Any] = [target.value]
            if current_stage is not _UNSET:
                updates.append("current_stage = ?")
                params.append(current_stage)
            if started_at is not _UNSET:
                updates.append("started_at = ?")
                params.append(_to_iso(started_at) if started_at else None)
            if ended_at is not _UNSET:
                updates.append("ended_at = ?")
                params.append(_to_iso(ended_at) if ended_at else None)
            if cancel_requested is not _UNSET:
                updates.append("cancel_requested = ?")
                params.append(1 if cancel_requested else 0)
            if error_summary is not _UNSET:
                updates.append("error_summary = ?")
                params.append(error_summary)
            params.append(run_id)

            cursor = connection.execute(
                f"UPDATE tasks SET {', '.join(updates)}"
                " WHERE run_id = ? AND status = ?",
                params + [record.status.value],
            )
            if cursor.rowcount != 1:
                latest = self._require(connection, run_id)
                raise TaskStateError(
                    "任务状态已被并发修改: "
                    f"期望 {record.status.value}，实际 {latest.status.value}"
                )
            return self._require(connection, run_id)

    def update_stage(
        self,
        run_id: str,
        *,
        current_stage: str | None,
        completed_stages: tuple[str, ...] | list[str],
    ) -> TaskRecord:
        """更新阶段进度；终态任务的阶段事件被忽略。"""
        with self._write() as connection:
            record = self._require(connection, run_id)
            if record.is_terminal:
                return record
            connection.execute(
                """
                UPDATE tasks
                SET current_stage = ?, completed_stages_json = ?
                WHERE run_id = ? AND status IN (?, ?, ?)
                """,
                (
                    current_stage,
                    json.dumps(list(completed_stages), ensure_ascii=False),
                    run_id,
                    TaskStatus.PREPARING.value,
                    TaskStatus.RUNNING.value,
                    TaskStatus.CANCELLING.value,
                ),
            )
            return self._require(connection, run_id)

    def set_error(self, run_id: str, message: str) -> None:
        with self._write() as connection:
            self._require(connection, run_id)
            connection.execute(
                "UPDATE tasks SET error_summary = ? WHERE run_id = ?",
                (message, run_id),
            )

    # ------------------------------------------------------------------ 任务读取
    @staticmethod
    def _require(
        connection: sqlite3.Connection, run_id: str
    ) -> TaskRecord:
        row = connection.execute(
            "SELECT * FROM tasks WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise TaskNotFoundError(run_id)
        return _to_record(row)

    def get(self, run_id: str) -> TaskRecord | None:
        with self._read() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE run_id = ?", (run_id,)
            ).fetchone()
        return _to_record(row) if row is not None else None

    def require(self, run_id: str) -> TaskRecord:
        record = self.get(run_id)
        if record is None:
            raise TaskNotFoundError(run_id)
        return record

    def list_tasks(self) -> list[TaskRecord]:
        with self._read() as connection:
            rows = connection.execute(
                "SELECT * FROM tasks ORDER BY submitted_at, rowid"
            ).fetchall()
        return [_to_record(row) for row in rows]

    def next_pending(self) -> TaskRecord | None:
        with self._read() as connection:
            row = connection.execute(
                """
                SELECT * FROM tasks
                WHERE status = ?
                ORDER BY submitted_at, rowid
                LIMIT 1
                """,
                (TaskStatus.PENDING.value,),
            ).fetchone()
        return _to_record(row) if row is not None else None

    def active_tasks(self) -> list[TaskRecord]:
        with self._read() as connection:
            rows = connection.execute(
                """
                SELECT * FROM tasks
                WHERE status IN (?, ?, ?)
                ORDER BY submitted_at, rowid
                """,
                (
                    TaskStatus.PREPARING.value,
                    TaskStatus.RUNNING.value,
                    TaskStatus.CANCELLING.value,
                ),
            ).fetchall()
        return [_to_record(row) for row in rows]

    def count_pending(self) -> int:
        with self._read() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM tasks WHERE status = ?",
                (TaskStatus.PENDING.value,),
            ).fetchone()
        return int(row["total"])

    # ------------------------------------------------------------------ 取消与关闭
    def delete_pending(self, run_id: str) -> None:
        with self._write() as connection:
            cursor = connection.execute(
                "DELETE FROM tasks WHERE run_id = ? AND status = ?",
                (run_id, TaskStatus.PENDING.value),
            )
            if cursor.rowcount == 0:
                latest = self._require(connection, run_id)
                raise TaskConflictError(
                    f"只能取消 Pending 任务，当前状态 {latest.status.value}"
                )

    def delete_all_pending_with_notice(self, kind: str) -> int:
        """单事务删除全部 Pending，并在有删除时插入一条 notice。"""
        if kind not in NOTICE_KINDS:
            raise ValueError(f"未知 notice 类型: {kind}")
        with self._write() as connection:
            cursor = connection.execute(
                "DELETE FROM tasks WHERE status = ?",
                (TaskStatus.PENDING.value,),
            )
            count = cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0
            if count:
                connection.execute(
                    "INSERT INTO notices (kind, count, created_at) VALUES (?, ?, ?)",
                    (kind, count, _to_iso(utc_now())),
                )
            return count

    # ------------------------------------------------------------------ notices
    def add_notice(self, kind: str, count: int) -> int:
        if kind not in NOTICE_KINDS:
            raise ValueError(f"未知 notice 类型: {kind}")
        if count <= 0:
            raise ValueError(f"notice 数量必须为正整数，实际 {count}")
        with self._write() as connection:
            cursor = connection.execute(
                "INSERT INTO notices (kind, count, created_at) VALUES (?, ?, ?)",
                (kind, count, _to_iso(utc_now())),
            )
            return int(cursor.lastrowid)

    def list_notices(self) -> list[Notice]:
        with self._read() as connection:
            rows = connection.execute(
                "SELECT * FROM notices ORDER BY created_at, id"
            ).fetchall()
        return [
            Notice(
                id=row["id"],
                kind=row["kind"],
                count=row["count"],
                created_at=_parse_time(row["created_at"]),
            )
            for row in rows
        ]

    def ack_notice(self, notice_id: int) -> bool:
        with self._write() as connection:
            cursor = connection.execute(
                "DELETE FROM notices WHERE id = ?", (notice_id,)
            )
            return cursor.rowcount > 0
