"""T-010 队列取消提示 API、页面与跨进程保留测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from experiment.plans import load_plan_catalog  # noqa: E402
from resources.settings import SettingsStore  # noqa: E402
from runtime.network import NetworkController  # noqa: E402
from tasks.models import (  # noqa: E402
    NOTICE_QUEUE_CANCELLED_ON_RESTART,
    NOTICE_QUEUE_CANCELLED_ON_SHUTDOWN,
    Notice,
    TaskNotFoundError,
)
from tasks.repository import TaskRepository  # noqa: E402
from web.app import create_app  # noqa: E402
from web.context import AppContext  # noqa: E402
from web.session import (  # noqa: E402
    COOKIE_NAME,
    SESSION_SCOPE_LAN_AUTHENTICATED,
    SessionStore,
)

CATALOG = load_plan_catalog()


class FakeNoticeService:
    def __init__(self) -> None:
        self.notices = [
            Notice(
                id=1,
                kind=NOTICE_QUEUE_CANCELLED_ON_RESTART,
                count=2,
                created_at=datetime(2026, 11, 21, 10, 0, tzinfo=timezone.utc),
            ),
            Notice(
                id=2,
                kind=NOTICE_QUEUE_CANCELLED_ON_SHUTDOWN,
                count=1,
                created_at=datetime(2026, 11, 21, 10, 5, tzinfo=timezone.utc),
            ),
        ]

    def start(self) -> None:
        return None

    def stop_normal(self, reason: str = "服务关闭") -> None:
        return None

    def list_tasks(self):
        return []

    def list_notices(self):
        return list(self.notices)

    def ack_notice(self, notice_id):
        for index, notice in enumerate(self.notices):
            if notice.id == notice_id:
                self.notices.pop(index)
                return True
        return False

    def get(self, run_id):
        raise TaskNotFoundError(run_id)

    def submit(self, *args):
        raise TaskNotFoundError("no")

    def cancel(self, run_id):
        raise TaskNotFoundError(run_id)


class FakeServer:
    def __init__(self) -> None:
        self.started = False
        self.should_exit = False

    def run(self) -> None:
        self.started = True


class NoticesApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.store = SettingsStore(self.root / "settings.json")
        self.service = FakeNoticeService()
        self.context = AppContext(
            catalog=CATALOG,
            task_service=self.service,
            session_store=SessionStore(),
            settings_store=self.store,
        )
        self.app = create_app(self.context, mode="local")
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.token = self.client.get("/api/v1/session").json()["session"][
            "csrf_token"
        ]


class LocalNoticeTests(NoticesApiTestCase):
    def test_list_and_ack(self) -> None:
        response = self.client.get("/api/v1/notices")
        self.assertEqual(response.status_code, 200)
        notices = response.json()["notices"]
        self.assertEqual(len(notices), 2)
        self.assertEqual(notices[0]["kind"], NOTICE_QUEUE_CANCELLED_ON_RESTART)
        self.assertEqual(notices[0]["count"], 2)
        self.assertEqual(
            notices[0]["message"], "服务重启时取消了 2 个等待任务"
        )
        self.assertIn("created_at", notices[0])
        self.assertNotIn("config", response.text)
        self.assertNotIn("path", response.text)

        response = self.client.post(
            "/api/v1/notices/1/ack",
            headers={"X-CSRF-Token": self.token},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["acknowledged"])

        remaining = self.client.get("/api/v1/notices").json()["notices"]
        self.assertEqual([notice["id"] for notice in remaining], [2])

        response = self.client.post(
            "/api/v1/notices/1/ack",
            headers={"X-CSRF-Token": self.token},
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json()["error"]["code"], "NOTICE_NOT_FOUND"
        )

    def test_ack_requires_csrf(self) -> None:
        response = self.client.post("/api/v1/notices/1/ack")
        self.assertEqual(response.status_code, 403)
        self.assertEqual(len(self.client.get("/api/v1/notices").json()["notices"]), 2)

    def test_empty_notices(self) -> None:
        self.service.notices.clear()
        self.assertEqual(
            self.client.get("/api/v1/notices").json(), {"notices": []}
        )

    def test_pages_have_notice_region(self) -> None:
        history = self.client.get("/history").text
        self.assertIn('id="notices"', history)
        login_template = (
            PROJECT_ROOT / "src" / "web" / "templates" / "login.html"
        ).read_text(encoding="utf-8")
        self.assertNotIn('id="notices"', login_template)


class LanNoticeTests(NoticesApiTestCase):
    def test_authenticated_lan_can_read_and_ack(self) -> None:
        sessions = self.context.session_store
        controller = NetworkController(
            app_factory=lambda **kwargs: create_app(
                self.context, **kwargs
            ),
            settings_store=self.store,
            session_store=sessions,
            server_factory=lambda app, host, port: FakeServer(),
            port_checker=lambda host, port: True,
            address_provider=lambda: ["192.168.1.10"],
            code_factory=lambda: "notice-code",
        )
        controller.enable(8766)
        self.addCleanup(controller.shutdown)
        lan_app = create_app(
            self.context, mode="lan", network=controller
        )
        session = sessions.create(
            SESSION_SCOPE_LAN_AUTHENTICATED, ttl_seconds=3600
        )
        with TestClient(lan_app, base_url="http://testserver") as client:
            client.cookies.set(COOKIE_NAME, session.session_id)
            response = client.get("/api/v1/notices")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.json()["notices"]), 2)

            response = client.post(
                "/api/v1/notices/2/ack",
                headers={
                    "X-CSRF-Token": session.csrf_token,
                    "Referer": "http://testserver/plans",
                },
            )
            self.assertEqual(response.status_code, 200)
            remaining = client.get("/api/v1/notices").json()["notices"]
            self.assertEqual([notice["id"] for notice in remaining], [1])


class RepositoryPersistenceTests(unittest.TestCase):
    def test_unconfirmed_notices_survive_new_repository(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "tasks.db"
            repository = TaskRepository(db_path)
            repository.initialize()
            repository.add_notice(NOTICE_QUEUE_CANCELLED_ON_RESTART, 2)

            reopened = TaskRepository(db_path)
            reopened.initialize()
            notices = reopened.list_notices()
            self.assertEqual(len(notices), 1)
            self.assertTrue(reopened.ack_notice(notices[0].id))
            self.assertEqual(reopened.list_notices(), [])


if __name__ == "__main__":
    unittest.main()
