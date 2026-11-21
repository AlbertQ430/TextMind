"""单实例文件锁。

同一数据目录只允许一个任务服务实例；第二个实例获取锁失败时拒绝启动。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import IO


class InstanceLockError(RuntimeError):
    """无法获取实例锁。"""


class InstanceLock:
    """进程级独占文件锁（POSIX flock / Windows msvcrt）。"""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._handle: IO[str] | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def locked(self) -> bool:
        return self._handle is not None

    def acquire(self) -> None:
        if self._handle is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._path.open("a+")
        try:
            self._lock(handle)
        except OSError as exc:
            handle.close()
            raise InstanceLockError(
                f"无法获取实例锁 {self._path}: {exc}"
            ) from exc
        self._handle = handle

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            self._unlock(self._handle)
        finally:
            self._handle.close()
            self._handle = None

    @staticmethod
    def _lock(handle: IO[str]) -> None:
        if os.name == "nt":  # pragma: no cover - Windows 分支
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write("0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock(handle: IO[str]) -> None:
        if os.name == "nt":  # pragma: no cover - Windows 分支
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def __enter__(self) -> "InstanceLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()
