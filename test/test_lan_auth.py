"""T-010 LAN 登录、CSRF/Origin、限流与登出测试。"""

from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from experiment.plans import load_plan_catalog  # noqa: E402
from resources.settings import SettingsStore  # noqa: E402
from runtime.network import NetworkController  # noqa: E402
from tasks.models import TaskNotFoundError, TaskSubmitError  # noqa: E402
from web.app import create_app  # noqa: E402
from web.context import AppContext  # noqa: E402
from web.session import (  # noqa: E402
    COOKIE_NAME,
    SESSION_SCOPE_LAN_AUTHENTICATED,
    SESSION_SCOPE_LAN_PREAUTH,
    SessionStore,
)

CATALOG = load_plan_catalog()
ACCESS_CODE = "test-access-code"


class FakeTaskService:
    def start(self) -> None:
        return None

    def stop_normal(self, reason: str = "服务关闭") -> None:
        return None

    def list_tasks(self):
        return []

    def list_notices(self):
        return []

    def ack_notice(self, notice_id):
        return False

    def get(self, run_id):
        raise TaskNotFoundError(run_id)

    def submit(self, *args):
        raise TaskSubmitError("测试上下文不提交任务")

    def cancel(self, run_id):
        raise TaskNotFoundError(run_id)


class FakeServer:
    def __init__(self) -> None:
        self.started = False
        self.should_exit = False

    def run(self) -> None:
        self.started = True


def extract_csrf(html: str) -> str:
    match = re.search(r'name="_csrf" value="([^"]+)"', html)
    assert match is not None, "登录页缺少 CSRF 字段"
    return match.group(1)


class LanAuthTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.store = SettingsStore(self.root / "settings.json")
        self.sessions = SessionStore()
        self.context = AppContext(
            catalog=CATALOG,
            task_service=FakeTaskService(),
            session_store=self.sessions,
            settings_store=self.store,
        )
        self.controller = NetworkController(
            app_factory=lambda **kwargs: create_app(self.context, **kwargs),
            settings_store=self.store,
            session_store=self.sessions,
            local_port=8765,
            server_factory=lambda app, host, port: FakeServer(),
            port_checker=lambda host, port: True,
            address_provider=lambda: ["192.168.1.10"],
            code_factory=lambda: ACCESS_CODE,
            start_timeout=2.0,
            join_timeout=2.0,
        )
        self.controller.enable(8766)
        self.app = create_app(self.context, mode="lan", network=self.controller)
        self.client = TestClient(self.app, base_url="http://testserver")
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def open_login(self):
        response = self.client.get("/lan/login")
        self.assertEqual(response.status_code, 200)
        return response, extract_csrf(response.text)

    def submit(
        self,
        csrf: str,
        code: str = ACCESS_CODE,
        *,
        referer: str | None = "http://testserver/lan/login",
        origin: str | None = None,
        follow_redirects: bool = False,
    ):
        headers = {}
        if referer is not None:
            headers["Referer"] = referer
        if origin is not None:
            headers["Origin"] = origin
        return self.client.post(
            "/lan/login",
            data={"_csrf": csrf, "access_code": code},
            headers=headers,
            follow_redirects=follow_redirects,
        )

    def login(self):
        _, csrf = self.open_login()
        response = self.submit(csrf)
        self.assertEqual(response.status_code, 303)
        return response


class LoginPageTests(LanAuthTestCase):
    def test_get_creates_preauth_and_page(self) -> None:
        response, csrf = self.open_login()
        self.assertIn("局域网登录", response.text)
        self.assertIn("HTTP", response.text)
        self.assertIn(COOKIE_NAME, response.headers.get("set-cookie", ""))
        session = self.sessions.get(response.cookies.get(COOKIE_NAME))
        self.assertIsNotNone(session)
        self.assertEqual(session.scope, SESSION_SCOPE_LAN_PREAUTH)
        self.assertTrue(csrf)

    def test_authenticated_redirects_to_plans(self) -> None:
        self.login()
        response = self.client.get("/lan/login", follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/plans")


class FailureCleanupTests(LanAuthTestCase):
    def test_success_clears_old_session_failures(self) -> None:
        _, csrf = self.open_login()
        preauth_id = self.client.cookies.get(COOKIE_NAME)
        self.submit(csrf, code="wrong")
        self.assertEqual(
            self.controller._session_failures.get(preauth_id), 1
        )
        response = self.submit(extract_csrf(self.client.get("/lan/login").text))
        self.assertEqual(response.status_code, 303)
        self.assertIsNone(
            self.controller._session_failures.get(preauth_id)
        )

    def test_logout_clears_session_failures(self) -> None:
        self.login()
        session_id = self.client.cookies.get(COOKIE_NAME)
        self.controller.login_failure(session_id, "testclient")
        self.assertIn(session_id, self.controller._session_failures)
        csrf = self.client.get("/api/v1/session").json()["session"]["csrf_token"]
        response = self.client.post(
            "/lan/logout",
            headers={
                "X-CSRF-Token": csrf,
                "Referer": "http://testserver/plans",
            },
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 303)
        self.assertNotIn(session_id, self.controller._session_failures)
        self.assertEqual(len(self.sessions), 0)


class LoginSuccessTests(LanAuthTestCase):
    def test_login_creates_new_authenticated_session(self) -> None:
        self.open_login()
        old_id = self.client.cookies.get(COOKIE_NAME)
        old_session = self.sessions.get(old_id)
        self.assertEqual(old_session.scope, SESSION_SCOPE_LAN_PREAUTH)

        response = self.login()
        new_id = self.client.cookies.get(COOKIE_NAME)
        self.assertNotEqual(new_id, old_id)
        self.assertIsNone(self.sessions.get(old_id))
        new_session = self.sessions.get(new_id)
        self.assertEqual(new_session.scope, SESSION_SCOPE_LAN_AUTHENTICATED)
        self.assertIsNotNone(new_session.expires_at)
        ttl = (new_session.expires_at - new_session.created_at).total_seconds()
        self.assertAlmostEqual(ttl, 8 * 3600, delta=1)

        page = self.client.get("/plans")
        self.assertEqual(page.status_code, 200)

    def test_login_with_origin_header(self) -> None:
        _, csrf = self.open_login()
        response = self.submit(csrf, origin="http://testserver")
        self.assertEqual(response.status_code, 303)


class LoginFailureTests(LanAuthTestCase):
    def test_wrong_access_code_rotates_csrf(self) -> None:
        _, csrf = self.open_login()
        response = self.submit(csrf, code="wrong")
        self.assertEqual(response.status_code, 401)
        self.assertIn("访问码无效", response.text)
        rotated = extract_csrf(response.text)
        self.assertNotEqual(rotated, csrf)
        self.assertEqual(self.submit(rotated, code="wrong").status_code, 401)

    def test_wrong_csrf_rejected(self) -> None:
        self.open_login()
        response = self.submit("wrong-csrf", code=ACCESS_CODE)
        self.assertEqual(response.status_code, 401)

    def test_missing_origin_and_referer_rejected(self) -> None:
        _, csrf = self.open_login()
        response = self.submit(csrf, referer=None, origin=None)
        self.assertEqual(response.status_code, 401)

    def test_cross_origin_rejected(self) -> None:
        _, csrf = self.open_login()
        response = self.submit(csrf, origin="http://evil.example")
        self.assertEqual(response.status_code, 401)

    def test_five_failures_end_session(self) -> None:
        _, csrf = self.open_login()
        preauth_id = self.client.cookies.get(COOKIE_NAME)
        response = None
        for _ in range(5):
            response = self.submit(csrf, code="wrong")
            if response.status_code == 401:
                csrf = extract_csrf(response.text)
        self.assertEqual(response.status_code, 429)
        self.assertIn("重新打开登录页", response.text)
        self.assertIsNone(self.sessions.get(preauth_id))
        self.assertEqual(len(self.sessions), 0)
        self.assertNotIn(
            preauth_id, self.controller._session_failures
        )

    def test_source_throttle(self) -> None:
        for _ in range(20):
            self.controller.login_failure("other-session", "testclient")
        _, csrf = self.open_login()
        response = self.submit(csrf, code=ACCESS_CODE)
        self.assertEqual(response.status_code, 429)


class LogoutTests(LanAuthTestCase):
    def test_logout_deletes_current_session(self) -> None:
        self.login()
        session_id = self.client.cookies.get(COOKIE_NAME)
        csrf = self.client.get("/api/v1/session").json()["session"]["csrf_token"]
        response = self.client.post(
            "/lan/logout",
            headers={
                "X-CSRF-Token": csrf,
                "Referer": "http://testserver/plans",
            },
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/lan/login")
        self.assertIsNone(self.sessions.get(session_id))

    def test_logout_requires_authenticated(self) -> None:
        self.open_login()
        response = self.client.post("/lan/logout", follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/lan/login")


if __name__ == "__main__":
    unittest.main()
