"""本机会话与 LAN 会话存储（内存）。

会话只保存 session ID、CSRF Token、scope、创建时间和过期时间；
Token 不写入 Cookie、URL、日志或结果 JSON。Local 会话在进程生命周期内有效，
LAN 会话按 scope 具有固定 TTL，过期后自动清理。
"""

from __future__ import annotations

import hmac
import secrets
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

COOKIE_NAME = "textmind_session"
SESSION_TOKEN_BYTES = 32

SESSION_SCOPE_LOCAL = "local_authenticated"
SESSION_SCOPE_LAN_PREAUTH = "lan_preauth"
SESSION_SCOPE_LAN_AUTHENTICATED = "lan_authenticated"

LOCAL_SCOPES: frozenset[str] = frozenset({SESSION_SCOPE_LOCAL})
LAN_PREAUTH_SCOPES: frozenset[str] = frozenset({SESSION_SCOPE_LAN_PREAUTH})
LAN_SCOPES: frozenset[str] = frozenset(
    {SESSION_SCOPE_LAN_PREAUTH, SESSION_SCOPE_LAN_AUTHENTICATED}
)

LAN_PREAUTH_TTL_SECONDS = 10 * 60
LAN_AUTHENTICATED_TTL_SECONDS = 8 * 60 * 60

# 预认证会话的全局上限；超过时淘汰最旧的 preauth，不影响 Local/已认证会话。
LAN_PREAUTH_MAX_SESSIONS = 256


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Session:
    session_id: str
    csrf_token: str
    scope: str
    created_at: datetime
    expires_at: datetime | None = None

    def is_expired(self, now: datetime | None = None) -> bool:
        if self.expires_at is None:
            return False
        return (now or _now()) >= self.expires_at


class SessionStore:
    """线程安全的进程内会话表。"""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(
        self,
        scope: str = SESSION_SCOPE_LOCAL,
        ttl_seconds: float | None = None,
    ) -> Session:
        now = _now()
        expires_at = (
            now + timedelta(seconds=ttl_seconds)
            if ttl_seconds is not None
            else None
        )
        session = Session(
            session_id=secrets.token_urlsafe(SESSION_TOKEN_BYTES),
            csrf_token=secrets.token_urlsafe(SESSION_TOKEN_BYTES),
            scope=scope,
            created_at=now,
            expires_at=expires_at,
        )
        with self._lock:
            self._sessions = {
                key: value
                for key, value in self._sessions.items()
                if not value.is_expired(now)
            }
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str | None) -> Session | None:
        if not session_id:
            return None
        now = _now()
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.is_expired(now):
                self._sessions.pop(session_id, None)
                return None
            return session

    def rotate_csrf(self, session_id: str) -> Session | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            rotated = Session(
                session_id=session.session_id,
                csrf_token=secrets.token_urlsafe(SESSION_TOKEN_BYTES),
                scope=session.scope,
                created_at=session.created_at,
                expires_at=session.expires_at,
            )
            self._sessions[session_id] = rotated
            return rotated

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def create_with_scope_limit(
        self,
        scope: str,
        limit: int,
        ttl_seconds: float | None = None,
    ) -> Session:
        """原子地清理过期会话、淘汰最旧的同 scope 会话并创建新会话。

        不裁剪其他 scope；创建后指定 scope 的会话数量不超过 `limit`。
        """
        if limit < 1:
            raise ValueError("limit 必须为正整数")
        now = _now()
        expires_at = (
            now + timedelta(seconds=ttl_seconds)
            if ttl_seconds is not None
            else None
        )
        with self._lock:
            self._sessions = {
                key: value
                for key, value in self._sessions.items()
                if not value.is_expired(now)
            }
            targets = sorted(
                (
                    session
                    for session in self._sessions.values()
                    if session.scope == scope
                ),
                key=lambda session: session.created_at,
            )
            excess = len(targets) - (limit - 1)
            for session in targets[: max(0, excess)]:
                self._sessions.pop(session.session_id, None)
            session = Session(
                session_id=secrets.token_urlsafe(SESSION_TOKEN_BYTES),
                csrf_token=secrets.token_urlsafe(SESSION_TOKEN_BYTES),
                scope=scope,
                created_at=now,
                expires_at=expires_at,
            )
            self._sessions[session.session_id] = session
        return session

    def delete_by_scope(self, scope: str) -> int:
        with self._lock:
            targets = [
                session_id
                for session_id, session in self._sessions.items()
                if session.scope == scope
            ]
            for session_id in targets:
                self._sessions.pop(session_id, None)
            return len(targets)

    def delete_scopes(self, scopes: frozenset[str]) -> int:
        with self._lock:
            targets = [
                session_id
                for session_id, session in self._sessions.items()
                if session.scope in scopes
            ]
            for session_id in targets:
                self._sessions.pop(session_id, None)
            return len(targets)

    def cleanup(self) -> int:
        now = _now()
        with self._lock:
            expired = [
                session_id
                for session_id, session in self._sessions.items()
                if session.is_expired(now)
            ]
            for session_id in expired:
                self._sessions.pop(session_id, None)
            return len(expired)

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
