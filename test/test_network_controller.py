"""T-010 网络控制器生命周期、访问码与限流测试。"""

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

from resources.settings import (  # noqa: E402
    NetworkSettings,
    Settings,
    SettingsStore,
    SettingsWriteError,
)
from runtime.network import (  # noqa: E402
    ACCESS_CODE_LENGTH,
    LOGIN_WINDOW_SECONDS,
    SOURCE_FAILURE_LIMIT,
    NetworkController,
    NetworkError,
    generate_access_code,
    local_ipv4_addresses,
)
from web.session import (  # noqa: E402
    SESSION_SCOPE_LAN_AUTHENTICATED,
    SESSION_SCOPE_LAN_PREAUTH,
    SESSION_SCOPE_LOCAL,
    SessionStore,
)


class FakeServer:
    instances: list["FakeServer"] = []

    def __init__(self, *, started: bool = True) -> None:
        self.started = False
        self._should_start = started
        self.should_exit = False
        self.run_calls = 0
        FakeServer.instances.append(self)

    def run(self) -> None:
        self.run_calls += 1
        if self._should_start:
            self.started = True


class ControllerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.settings_store = SettingsStore(self.root / "settings.json")
        self.sessions = SessionStore()
        FakeServer.instances = []
        self.server = FakeServer()
        self.controller = NetworkController(
            app_factory=lambda **kwargs: object(),
            settings_store=self.settings_store,
            session_store=self.sessions,
            local_port=8765,
            server_factory=lambda app, host, port: self.server,
            port_checker=lambda host, port: True,
            address_provider=lambda: ["192.168.1.10"],
            code_factory=lambda: "test-access-code",
            start_timeout=2.0,
            join_timeout=2.0,
        )

    def settings(self) -> dict:
        return json.loads((self.root / "settings.json").read_text(encoding="utf-8"))


class EnableTests(ControllerTestCase):
    def test_enable_success(self) -> None:
        status = self.controller.enable(8766)
        self.assertTrue(status["enabled"])
        self.assertEqual(status["port"], 8766)
        self.assertEqual(
            status["addresses"], ["http://192.168.1.10:8766"]
        )
        self.assertEqual(status["access_code"], "test-access-code")
        self.assertIn("HTTP", status["tls_warning"])
        self.assertNotIn("0.0.0.0", json.dumps(status))
        self.assertTrue(self.settings()["network"]["lan_enabled"])
        self.assertEqual(self.settings()["network"]["lan_port"], 8766)
        self.assertTrue(self.server.started)

    def test_invalid_ports(self) -> None:
        for port in (0, 70000, 8765):
            with self.subTest(port=port):
                with self.assertRaises(NetworkError) as context:
                    self.controller.enable(port)
                self.assertEqual(context.exception.code, "NETWORK_PORT_INVALID")
                self.assertEqual(context.exception.status_code, 422)
        self.assertFalse(self.controller.enabled)

    def test_port_in_use(self) -> None:
        controller = NetworkController(
            app_factory=lambda **kwargs: object(),
            settings_store=self.settings_store,
            session_store=self.sessions,
            port_checker=lambda host, port: False,
            server_factory=lambda app, host, port: self.server,
        )
        with self.assertRaises(NetworkError) as context:
            controller.enable(8766)
        self.assertEqual(context.exception.code, "NETWORK_PORT_IN_USE")
        self.assertEqual(context.exception.status_code, 409)
        self.assertFalse(controller.enabled)
        self.assertFalse(self.controller.access_code)

    def test_start_failure_rolls_back(self) -> None:
        self.server = FakeServer(started=False)
        with self.assertRaises(NetworkError) as context:
            self.controller.enable(8766)
        self.assertEqual(
            context.exception.code, "NETWORK_START_FAILED"
        )
        self.assertFalse(self.controller.enabled)
        self.assertIsNone(self.controller.access_code)
        if (self.root / "settings.json").exists():
            self.assertFalse(self.settings()["network"]["lan_enabled"])
        self.assertTrue(self.server.should_exit)

    def test_write_failure_rolls_back(self) -> None:
        with mock.patch.object(
            self.settings_store, "save", side_effect=SettingsWriteError("boom")
        ):
            with self.assertRaises(NetworkError) as context:
                self.controller.enable(8766)
        self.assertEqual(context.exception.code, "NETWORK_WRITE_FAILED")
        self.assertEqual(context.exception.status_code, 500)
        self.assertFalse(self.controller.enabled)
        self.assertIsNone(self.controller.access_code)
        self.assertTrue(self.server.should_exit)


class EarlyFailureTests(ControllerTestCase):
    def test_last_error_on_invalid_port(self) -> None:
        with self.assertRaises(NetworkError):
            self.controller.enable(8765)
        status = self.controller.status()
        self.assertIsNotNone(status["last_error"])
        self.assertFalse(status["enabled"])

    def test_last_error_on_port_in_use(self) -> None:
        controller = NetworkController(
            app_factory=lambda **kwargs: object(),
            settings_store=self.settings_store,
            session_store=self.sessions,
            port_checker=lambda host, port: False,
            server_factory=lambda app, host, port: self.server,
        )
        with self.assertRaises(NetworkError):
            controller.enable(8766)
        status = controller.status()
        self.assertIsNotNone(status["last_error"])
        self.assertFalse(status["enabled"])


class DisableTests(ControllerTestCase):
    def test_disable_clears_lan_sessions_and_saves_off(self) -> None:
        self.controller.enable(8766)
        local = self.sessions.create()
        preauth = self.sessions.create(SESSION_SCOPE_LAN_PREAUTH)
        authenticated = self.sessions.create(SESSION_SCOPE_LAN_AUTHENTICATED)

        status = self.controller.disable()
        self.assertFalse(status["enabled"])
        self.assertIsNone(status["access_code"])
        self.assertTrue(self.server.should_exit)
        self.assertIsNotNone(self.sessions.get(local.session_id))
        self.assertIsNone(self.sessions.get(preauth.session_id))
        self.assertIsNone(self.sessions.get(authenticated.session_id))
        self.assertFalse(self.settings()["network"]["lan_enabled"])

    def test_disable_when_off_is_safe(self) -> None:
        status = self.controller.disable()
        self.assertFalse(status["enabled"])
        self.assertFalse(self.settings()["network"]["lan_enabled"])

    def test_disable_persists_off_when_settings_enabled_but_not_running(
        self,
    ) -> None:
        self.settings_store.save(
            Settings(
                network=NetworkSettings(
                    lan_enabled=True, local_port=8765, lan_port=8766
                )
            )
        )
        self.assertFalse(self.controller.enabled)
        status = self.controller.disable()
        self.assertFalse(status["enabled"])
        self.assertFalse(self.settings()["network"]["lan_enabled"])

    def test_last_error_is_reported_and_cleared(self) -> None:
        self.server = FakeServer(started=False)
        with self.assertRaises(NetworkError):
            self.controller.enable(8766)
        status = self.controller.status()
        self.assertIsNotNone(status["last_error"])
        self.assertNotIn("0.0.0.0", status["last_error"])

        self.server = FakeServer(started=True)
        status = self.controller.enable(8766)
        self.assertIsNone(status["last_error"])

    def test_disable_clears_failure_counts(self) -> None:
        self.controller.enable(8766)
        session = self.sessions.create(SESSION_SCOPE_LAN_PREAUTH)
        self.controller.login_failure(session.session_id, "192.168.1.20")
        self.assertTrue(self.controller._session_failures)
        self.controller.disable()
        self.assertEqual(self.controller._session_failures, {})
        self.assertEqual(self.controller._source_failures, {})

    def test_shutdown_stops_without_saving(self) -> None:
        self.controller.enable(8766)
        self.sessions.create(SESSION_SCOPE_LAN_PREAUTH)
        self.controller.shutdown()
        self.assertFalse(self.controller.enabled)
        self.assertIsNone(self.controller.access_code)
        self.assertTrue(self.server.should_exit)
        self.assertEqual(len(self.sessions), 0)
        self.assertTrue(self.settings()["network"]["lan_enabled"])


class AccessCodeTests(ControllerTestCase):
    def test_generated_access_code_is_six_digits(self) -> None:
        with mock.patch("runtime.network.secrets.randbelow", return_value=42):
            self.assertEqual(generate_access_code(), "000042")
        for _ in range(100):
            code = generate_access_code()
            self.assertEqual(len(code), ACCESS_CODE_LENGTH)
            self.assertTrue(code.isdigit())

    def test_verify_access_code(self) -> None:
        self.assertFalse(self.controller.verify_access_code("test-access-code"))
        self.controller.enable(8766)
        self.assertTrue(self.controller.verify_access_code("test-access-code"))
        self.assertFalse(self.controller.verify_access_code("wrong"))
        self.assertFalse(self.controller.verify_access_code(None))
        self.controller.disable()
        self.assertFalse(self.controller.verify_access_code("test-access-code"))

    def test_login_failure_limits(self) -> None:
        session = self.sessions.create(SESSION_SCOPE_LAN_PREAUTH)
        session_id = session.session_id
        counts = []
        for _ in range(SOURCE_FAILURE_LIMIT):
            count, throttled = self.controller.login_failure(
                session_id, "192.168.1.20"
            )
            counts.append(count)
        self.assertEqual(counts[-1], SOURCE_FAILURE_LIMIT)
        self.assertTrue(throttled)
        self.assertTrue(self.controller.source_throttled("192.168.1.20"))
        self.assertFalse(self.controller.source_throttled("192.168.1.21"))

        self.controller.clear_source_failures("192.168.1.20")
        self.assertFalse(self.controller.source_throttled("192.168.1.20"))

    def test_session_failure_counter(self) -> None:
        session = self.sessions.create(SESSION_SCOPE_LAN_PREAUTH)
        session_id = session.session_id
        count, _ = self.controller.login_failure(session_id, "ip")
        self.assertEqual(count, 1)
        count, _ = self.controller.login_failure(session_id, "ip")
        self.assertEqual(count, 2)
        self.controller.clear_session_failures(session_id)
        count, _ = self.controller.login_failure(session_id, "ip")
        self.assertEqual(count, 1)

    def test_failure_tables_are_bounded(self) -> None:
        for index in range(1200):
            self.sessions.create(SESSION_SCOPE_LAN_PREAUTH)
            self.controller.login_failure(f"session-{index}", f"ip-{index}")
        self.assertLessEqual(len(self.controller._session_failures), 1024)
        self.assertLessEqual(len(self.controller._source_failures), 1024)

    def test_deleted_session_failure_is_forgotten(self) -> None:
        session = self.sessions.create(SESSION_SCOPE_LAN_PREAUTH)
        self.controller.login_failure(session.session_id, "ip")
        self.sessions.delete(session.session_id)
        count, _ = self.controller.login_failure(session.session_id, "ip")
        self.assertEqual(count, 0)
        self.assertIsNone(
            self.controller._session_failures.get(session.session_id)
        )

    def test_login_window_constant(self) -> None:
        self.assertEqual(LOGIN_WINDOW_SECONDS, 600)


class AddressTests(unittest.TestCase):
    def test_addresses_exclude_loopback_and_any(self) -> None:
        addresses = local_ipv4_addresses()
        self.assertNotIn("0.0.0.0", addresses)
        for address in addresses:
            self.assertFalse(address.startswith("127."))


if __name__ == "__main__":
    unittest.main()
