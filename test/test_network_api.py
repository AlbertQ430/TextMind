"""T-010 网络管理 API 测试（Local-only）。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from experiment.plans import load_plan_catalog  # noqa: E402
from resources.settings import (  # noqa: E402
    NetworkSettings,
    Settings,
    SettingsStore,
    SettingsWriteError,
)
from runtime.network import NetworkController  # noqa: E402
from tasks.models import TaskNotFoundError, TaskSubmitError  # noqa: E402
from web.app import create_app  # noqa: E402
from web.context import AppContext  # noqa: E402
from web.session import SessionStore  # noqa: E402

CATALOG = load_plan_catalog()


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
    instances: list["FakeServer"] = []

    def __init__(self, *, started: bool = True) -> None:
        self.started = False
        self._should_start = started
        self.should_exit = False
        FakeServer.instances.append(self)

    def run(self) -> None:
        if self._should_start:
            self.started = True


class NetworkApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.store = SettingsStore(self.root / "settings.json")
        self.context = AppContext(
            catalog=CATALOG,
            task_service=FakeTaskService(),
            session_store=SessionStore(),
            settings_store=self.store,
        )
        FakeServer.instances = []
        self.server = FakeServer()
        self.port_available = True
        self.controller = NetworkController(
            app_factory=lambda **kwargs: create_app(self.context, **kwargs),
            settings_store=self.store,
            session_store=self.context.session_store,
            local_port=8765,
            server_factory=lambda app, host, port: self.server,
            port_checker=lambda host, port: self.port_available,
            address_provider=lambda: ["192.168.1.10"],
            code_factory=lambda: "api-test-code",
            start_timeout=2.0,
            join_timeout=2.0,
        )
        self.app = create_app(
            self.context, mode="local", network=self.controller
        )
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.token = self.client.get("/api/v1/session").json()["session"][
            "csrf_token"
        ]

    def post(self, path: str, payload=None, token=None):
        return self.client.post(
            path,
            json=payload,
            headers={"X-CSRF-Token": token or self.token},
        )


class NetworkStatusTests(NetworkApiTestCase):
    def test_initial_status(self) -> None:
        response = self.client.get("/api/v1/network")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["enabled"])
        self.assertIsNone(payload["port"])
        self.assertIsNone(payload["access_code"])
        self.assertIn("HTTP", payload["tls_warning"])
        self.assertIsNone(payload["last_error"])
        self.assertNotIn("0.0.0.0", response.text)

    def test_enable_and_disable(self) -> None:
        response = self.post("/api/v1/network/enable", {"lan_port": 8766})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["port"], 8766)
        self.assertEqual(
            payload["addresses"], ["http://192.168.1.10:8766"]
        )
        self.assertEqual(payload["access_code"], "api-test-code")
        self.assertNotIn("0.0.0.0", response.text)

        saved = json.loads(
            (self.root / "settings.json").read_text(encoding="utf-8")
        )
        self.assertTrue(saved["network"]["lan_enabled"])

        response = self.post("/api/v1/network/disable")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["enabled"])
        self.assertIsNone(payload["access_code"])
        self.assertIsNone(self.controller.access_code)

    def test_lan_app_has_no_network_route(self) -> None:
        lan_app = create_app(self.context, mode="lan", network=self.controller)
        paths = {getattr(route, "path", None) for route in lan_app.routes}
        self.assertNotIn("/api/v1/network", paths)
        self.assertNotIn("/api/v1/settings", paths)
        self.assertNotIn("/api/v1/imports", paths)


class NetworkErrorTests(NetworkApiTestCase):
    def test_invalid_ports(self) -> None:
        for port in (0, 70000, 8765):
            with self.subTest(port=port):
                response = self.post(
                    "/api/v1/network/enable", {"lan_port": port}
                )
                self.assertEqual(response.status_code, 422)
                self.assertEqual(
                    response.json()["error"]["code"], "NETWORK_PORT_INVALID"
                )

    def test_port_in_use(self) -> None:
        self.port_available = False
        response = self.post("/api/v1/network/enable", {"lan_port": 8766})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json()["error"]["code"], "NETWORK_PORT_IN_USE"
        )

    def test_start_failed(self) -> None:
        self.server = FakeServer(started=False)
        response = self.post("/api/v1/network/enable", {"lan_port": 8766})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["error"]["code"], "NETWORK_START_FAILED"
        )

    def test_write_failed(self) -> None:
        with mock.patch.object(
            self.store, "save", side_effect=SettingsWriteError("boom")
        ):
            response = self.post(
                "/api/v1/network/enable", {"lan_port": 8766}
            )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["error"]["code"], "NETWORK_WRITE_FAILED"
        )
        self.assertFalse(self.controller.enabled)

    def test_port_type_is_strict(self) -> None:
        for value in (True, 1.5, "8766"):
            with self.subTest(value=value):
                response = self.post(
                    "/api/v1/network/enable", {"lan_port": value}
                )
                self.assertEqual(response.status_code, 422)
                self.assertEqual(
                    response.json()["error"]["code"], "VALIDATION_ERROR"
                )
        self.assertFalse(self.controller.enabled)

    def test_disable_persists_off_when_not_running(self) -> None:
        self.store.save(
            Settings(
                network=NetworkSettings(
                    lan_enabled=True, local_port=8765, lan_port=8766
                )
            )
        )
        self.assertFalse(self.controller.enabled)
        response = self.post("/api/v1/network/disable")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["enabled"])
        saved = json.loads(
            (self.root / "settings.json").read_text(encoding="utf-8")
        )
        self.assertFalse(saved["network"]["lan_enabled"])

    def test_start_failure_reports_last_error(self) -> None:
        self.server = FakeServer(started=False)
        response = self.post("/api/v1/network/enable", {"lan_port": 8766})
        self.assertEqual(response.status_code, 500)
        status = self.client.get("/api/v1/network").json()
        self.assertIsNotNone(status["last_error"])
        self.assertFalse(status["enabled"])

    def test_csrf_required(self) -> None:
        response = self.client.post(
            "/api/v1/network/enable", json={"lan_port": 8766}
        )
        self.assertEqual(response.status_code, 403)
        self.assertFalse(self.controller.enabled)

    def test_controller_missing(self) -> None:
        app = create_app(self.context, mode="local", network=None)
        self.context.network_controller = None
        with TestClient(app) as client:
            response = client.get("/api/v1/network")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["error"]["code"], "NETWORK_UNAVAILABLE"
        )


if __name__ == "__main__":
    unittest.main()
