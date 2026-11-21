"""本机会话与 CSRF Token 存储（内存）。

服务重启后会话自然失效；Token 只保存在当前会话内存中，
不写入 Cookie、URL、日志或结果 JSON。
"""

from __future__ import annotations

import hmac
import secrets
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

COOKIE_NAME = "textmind_session"
SESSION_SCOPE_LOCAL = "local"
SESSION_TOKEN_BYTES = 32


@dataclass(frozen=True)
class Session:
    session_id: str
    csrf_token: str
    scope: str
    created_at: datetime


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SessionStore:
    """进程内会话表。"""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, scope: str = SESSION_SCOPE_LOCAL) -> Session:
        session = Session(
            session_id=secrets.token_urlsafe(SESSION_TOKEN_BYTES),
            csrf_token=secrets.token_urlsafe(SESSION_TOKEN_BYTES),
            scope=scope,
            created_at=_now(),
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str | None) -> Session | None:
        if not session_id:
            return None
        with self._lock:
            return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    @staticmethod
    def csrf_matches(session: Session, token: str | None) -> bool:
        if not isinstance(token, str) or not token:
            return False
        return hmac.compare_digest(session.csrf_token, token)
