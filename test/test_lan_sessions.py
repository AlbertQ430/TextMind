"""T-010 会话 scope、TTL、轮换与线程安全测试。"""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from web.session import (  # noqa: E402
    COOKIE_NAME,
    LAN_AUTHENTICATED_TTL_SECONDS,
    LAN_PREAUTH_TTL_SECONDS,
    SESSION_SCOPE_LAN_AUTHENTICATED,
    SESSION_SCOPE_LAN_PREAUTH,
    SESSION_SCOPE_LOCAL,
    SessionStore,
)


class SessionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SessionStore()

    def test_create_scopes_and_ttl(self) -> None:
        local = self.store.create()
        self.assertEqual(local.scope, SESSION_SCOPE_LOCAL)
        self.assertIsNone(local.expires_at)

        preauth = self.store.create(
            SESSION_SCOPE_LAN_PREAUTH, ttl_seconds=LAN_PREAUTH_TTL_SECONDS
        )
        self.assertEqual(preauth.scope, SESSION_SCOPE_LAN_PREAUTH)
        ttl = (preauth.expires_at - preauth.created_at).total_seconds()
        self.assertAlmostEqual(ttl, LAN_PREAUTH_TTL_SECONDS, delta=1)

        authenticated = self.store.create(
            SESSION_SCOPE_LAN_AUTHENTICATED,
            ttl_seconds=LAN_AUTHENTICATED_TTL_SECONDS,
        )
        ttl = (authenticated.expires_at - authenticated.created_at).total_seconds()
        self.assertAlmostEqual(ttl, LAN_AUTHENTICATED_TTL_SECONDS, delta=1)

    def test_expired_session_removed(self) -> None:
        session = self.store.create(
            SESSION_SCOPE_LAN_PREAUTH, ttl_seconds=0
        )
        self.assertIsNone(self.store.get(session.session_id))
        self.assertEqual(len(self.store), 0)

    def test_get_unknown_and_missing_id(self) -> None:
        self.assertIsNone(self.store.get(None))
        self.assertIsNone(self.store.get("missing"))

    def test_rotate_csrf_changes_token_only(self) -> None:
        session = self.store.create()
        rotated = self.store.rotate_csrf(session.session_id)
        self.assertIsNotNone(rotated)
        self.assertEqual(rotated.session_id, session.session_id)
        self.assertNotEqual(rotated.csrf_token, session.csrf_token)
        self.assertEqual(rotated.scope, session.scope)
        self.assertIs(self.store.get(session.session_id).csrf_token, rotated.csrf_token)

    def test_delete_by_scope_and_scopes(self) -> None:
        local = self.store.create()
        preauth = self.store.create(SESSION_SCOPE_LAN_PREAUTH)
        authenticated = self.store.create(SESSION_SCOPE_LAN_AUTHENTICATED)
        self.assertEqual(self.store.delete_by_scope(SESSION_SCOPE_LAN_PREAUTH), 1)
        self.assertIsNotNone(self.store.get(local.session_id))
        self.assertIsNone(self.store.get(preauth.session_id))
        self.assertIsNotNone(self.store.get(authenticated.session_id))

        removed = self.store.delete_scopes(
            frozenset(
                {SESSION_SCOPE_LAN_PREAUTH, SESSION_SCOPE_LAN_AUTHENTICATED}
            )
        )
        self.assertEqual(removed, 1)
        self.assertEqual(len(self.store), 1)
        self.assertIsNotNone(self.store.get(local.session_id))

    def test_csrf_matches_constant_time(self) -> None:
        session = self.store.create()
        self.assertTrue(SessionStore.csrf_matches(session, session.csrf_token))
        self.assertFalse(SessionStore.csrf_matches(session, "wrong"))
        self.assertFalse(SessionStore.csrf_matches(session, ""))
        self.assertFalse(SessionStore.csrf_matches(session, None))

    def test_thread_safety(self) -> None:
        created: list[str] = []
        lock = threading.Lock()

        def worker() -> None:
            for _ in range(20):
                session = self.store.create()
                with lock:
                    created.append(session.session_id)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(created), 200)
        self.assertEqual(len(self.store), 200)
        self.assertEqual(len(set(created)), 200)

    def test_create_with_scope_limit_is_atomic(self) -> None:
        for _ in range(256):
            self.store.create(SESSION_SCOPE_LAN_PREAUTH)
        local = self.store.create()
        authenticated = self.store.create(
            SESSION_SCOPE_LAN_AUTHENTICATED, ttl_seconds=3600
        )
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait()
            self.store.create_with_scope_limit(
                SESSION_SCOPE_LAN_PREAUTH, 256, ttl_seconds=600
            )

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        preauth = sum(
            1
            for session in self.store._sessions.values()
            if session.scope == SESSION_SCOPE_LAN_PREAUTH
        )
        self.assertLessEqual(preauth, 256)
        self.assertIsNotNone(self.store.get(local.session_id))
        self.assertIsNotNone(self.store.get(authenticated.session_id))

    def test_create_with_scope_limit_rejects_invalid_limit(self) -> None:
        with self.assertRaises(ValueError):
            self.store.create_with_scope_limit(
                SESSION_SCOPE_LAN_PREAUTH, 0
            )

    def test_cleanup_removes_expired(self) -> None:
        self.store.create(SESSION_SCOPE_LAN_PREAUTH, ttl_seconds=0)
        self.assertEqual(self.store.cleanup(), 1)

    def test_create_cleans_expired_sessions(self) -> None:
        self.store.create(SESSION_SCOPE_LAN_PREAUTH, ttl_seconds=0)
        self.assertEqual(len(self.store), 1)
        self.store.create()
        self.assertEqual(len(self.store), 1)

    def test_cookie_name(self) -> None:
        self.assertEqual(COOKIE_NAME, "textmind_session")


if __name__ == "__main__":
    unittest.main()
