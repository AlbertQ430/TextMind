"""T-010 LAN 权限矩阵、共享状态与 SSE 失效测试。"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
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
    TaskNotFoundError,
    TaskSubmitError,
    build_task_record,
    generate_run_id,
)
from web.app import create_app  # noqa: E402
from web.context import AppContext  # noqa: E402
from web.session import (  # noqa: E402
    COOKIE_NAME,
    SESSION_SCOPE_LAN_AUTHENTICATED,
    SESSION_SCOPE_LAN_PREAUTH,
    SESSION_SCOPE_LOCAL,
    SessionStore,
)

CATALOG = load_plan_catalog()
ACCESS_CODE = "permissions-code"


class FakeTaskService:
    def __init__(self) -> None:
        self.tasks = {}
        self.started = 0
        self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def stop_normal(self, reason: str = "服务关闭") -> None:
        self.stopped += 1

    def list_tasks(self):
        return sorted(
            self.tasks.values(),
            key=lambda task: (task.submitted_at, task.run_id),
        )

    def list_notices(self):
        return []

    def ack_notice(self, notice_id):
        return False

    def get(self, run_id):
        task = self.tasks.get(run_id)
        if task is None:
            raise TaskNotFoundError(run_id)
        return task

    def submit(self, plan_id, plan_version, variant_id):
        run_id = generate_run_id()
        self.tasks[run_id] = build_task_record(
            run_id,
            plan_id,
            plan_version,
            variant_id,
            {"plan_name": "测试方案", "variant_name": variant_id},
            submitted_at=datetime.now(timezone.utc),
        )
        return run_id

    def cancel(self, run_id):
        task = self.get(run_id)
        if task.is_terminal:
            raise TaskSubmitError("任务已结束")


class FakeServer:
    def __init__(self) -> None:
        self.started = False
        self.should_exit = False

    def run(self) -> None:
        self.started = True


def extract_csrf(html: str) -> str:
    match = re.search(r'name="_csrf" value="([^"]+)"', html)
    assert match is not None
    return match.group(1)


def read_event(iterator, limit: int = 400):
    event = None
    data_lines: list[str] = []
    for _ in range(limit):
        line = next(iterator)
        if line.startswith("event: "):
            event = line[len("event: "):]
        elif line.startswith("data: "):
            data_lines.append(line[len("data: "):])
        elif line.startswith(":"):
            continue
        elif line == "":
            if event is not None:
                return event, json.loads("\n".join(data_lines))
    raise AssertionError("未读到完整事件")


class LanPermissionsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.store = SettingsStore(self.root / "settings.json")
        self.sessions = SessionStore()
        self.service = FakeTaskService()
        self.context = AppContext(
            catalog=CATALOG,
            task_service=self.service,
            session_store=self.sessions,
            settings_store=self.store,
            sse_poll_interval=0.02,
            sse_heartbeat_interval=0.05,
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
        self.lan_app = create_app(
            self.context, mode="lan", network=self.controller
        )
        self.client = TestClient(self.lan_app, base_url="http://testserver")
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def login(self) -> None:
        response = self.client.get("/lan/login")
        csrf = extract_csrf(response.text)
        response = self.client.post(
            "/lan/login",
            data={"_csrf": csrf, "access_code": ACCESS_CODE},
            headers={"Referer": "http://testserver/lan/login"},
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 303)

    def csrf(self) -> str:
        return self.client.get("/api/v1/session").json()["session"][
            "csrf_token"
        ]


class UnauthenticatedTests(LanPermissionsTestCase):
    def test_new_client_redirected_and_api_401(self) -> None:
        response = self.client.get("/plans", follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/lan/login")

        response = self.client.get("/api/v1/plans")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "UNAUTHORIZED")

    def test_preauth_cannot_access_business_pages(self) -> None:
        self.client.get("/lan/login")
        response = self.client.get("/plans", follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        response = self.client.get("/api/v1/tasks")
        self.assertEqual(response.status_code, 401)
        response = self.client.get("/api/v1/session")
        self.assertEqual(response.status_code, 401)


class AuthenticatedTests(LanPermissionsTestCase):
    def test_business_pages_and_apis_allowed(self) -> None:
        self.login()
        for path in ("/plans", "/monitor", "/history", "/compare"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)
        for path in (
            "/api/v1/plans",
            "/api/v1/plans/scifact-comparison",
            "/api/v1/tasks",
            "/api/v1/runs",
            "/api/v1/notices",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

    def test_local_only_routes_are_absent(self) -> None:
        self.login()
        for path in (
            "/settings",
            "/api/v1/settings",
            "/api/v1/network",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 404)
        same_origin = {
            "X-CSRF-Token": self.csrf(),
            "Referer": "http://testserver/plans",
        }
        response = self.client.post(
            "/api/v1/imports",
            files={"file": ("run.json", b"{}", "application/json")},
            headers=same_origin,
        )
        self.assertEqual(response.status_code, 404)
        response = self.client.put(
            "/api/v1/settings",
            json={"model_root": None},
            headers=same_origin,
        )
        self.assertEqual(response.status_code, 404)

    def test_pages_hide_local_only_entries(self) -> None:
        self.login()
        plans = self.client.get("/plans").text
        self.assertNotIn('href="/settings"', plans)
        self.assertIn("lan-logout-button", plans)
        history = self.client.get("/history").text
        self.assertNotIn('id="import-form"', history)

    def test_submit_task_shared_with_local(self) -> None:
        self.login()
        response = self.client.post(
            "/api/v1/tasks",
            json={
                "plan_id": "scifact-comparison",
                "plan_version": "v1",
                "variant_id": "scifact-bm25",
            },
            headers={
                "X-CSRF-Token": self.csrf(),
                "Referer": "http://testserver/plans",
            },
        )
        self.assertEqual(response.status_code, 201)
        run_id = response.json()["run_id"]
        self.assertIn(run_id, self.service.tasks)

        local_app = create_app(
            self.context, mode="local", network=self.controller
        )
        with TestClient(local_app) as local_client:
            snapshot = local_client.get("/api/v1/tasks").json()
        self.assertEqual(snapshot["counts"]["total"], 1)


class LanSseTests(LanPermissionsTestCase):
    def test_authenticated_sse_requires_origin(self) -> None:
        self.login()
        response = self.client.get("/api/v1/events")
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"]["code"], "ORIGIN_INVALID")

    def test_preauth_sse_rejected(self) -> None:
        self.client.get("/lan/login")
        response = self.client.get(
            "/api/v1/events", headers={"Origin": "http://testserver"}
        )
        self.assertEqual(response.status_code, 401)

    def test_disable_lan_sends_session_expired(self) -> None:
        self.login()
        with self.client.stream(
            "GET", "/api/v1/events", headers={"Origin": "http://testserver"}
        ) as response:
            iterator = response.iter_lines()
            self.assertEqual(read_event(iterator)[0], "snapshot")
            self.controller.disable()
            event, data = read_event(iterator)
            self.assertEqual(event, "session.expired")
            self.assertIn("reason", data)

        response = self.client.get("/plans", follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        response = self.client.get("/api/v1/tasks")
        self.assertEqual(response.status_code, 401)


class LanDisableTests(LanPermissionsTestCase):
    def test_disable_clears_lan_sessions_keeps_local(self) -> None:
        self.login()
        lan_id = self.client.cookies.get(COOKIE_NAME)
        local = self.sessions.create(SESSION_SCOPE_LOCAL)

        self.controller.disable()

        self.assertIsNone(self.sessions.get(lan_id))
        self.assertIsNotNone(self.sessions.get(local.session_id))
        self.assertEqual(
            sum(
                1
                for session in [
                    self.sessions.get(session_id)
                    for session_id in [
                        lan_id,
                        local.session_id,
                    ]
                ]
                if session is not None
            ),
            1,
        )


class SessionCreationTests(LanPermissionsTestCase):
    def test_unauthenticated_requests_do_not_create_sessions(self) -> None:
        before = len(self.sessions)
        for _ in range(5):
            self.client.get("/api/v1/plans")
            self.client.get("/api/v1/health")
            self.client.get("/plans", follow_redirects=False)
        self.assertEqual(len(self.sessions), before)

    def test_preauth_only_created_by_login_get(self) -> None:
        self.client.get("/lan/login")
        self.assertEqual(len(self.sessions), 1)
        self.client.get("/lan/login")
        self.assertEqual(len(self.sessions), 1)

        with TestClient(self.lan_app, base_url="http://testserver") as client:
            response = client.post(
                "/lan/login",
                data={"_csrf": "x", "access_code": "x"},
            )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(len(self.sessions), 1)

    def test_preauth_sessions_are_capped(self) -> None:
        for _ in range(300):
            self.sessions.create(SESSION_SCOPE_LAN_PREAUTH)
        local = self.sessions.create(SESSION_SCOPE_LOCAL)
        authenticated = self.sessions.create(
            SESSION_SCOPE_LAN_AUTHENTICATED, ttl_seconds=3600
        )
        with TestClient(self.lan_app, base_url="http://testserver") as client:
            response = client.get("/lan/login")
        self.assertEqual(response.status_code, 200)
        # 300 个旧 preauth 被裁剪到 255，再新增 1 个；Local/认证会话保留。
        preauth_count = sum(
            1
            for session in self.sessions._sessions.values()
            if session.scope == SESSION_SCOPE_LAN_PREAUTH
        )
        self.assertEqual(preauth_count, 256)
        self.assertEqual(len(self.sessions), 258)
        self.assertIsNotNone(self.sessions.get(local.session_id))
        self.assertIsNotNone(self.sessions.get(authenticated.session_id))

    def test_authenticated_write_requires_origin_or_referer(self) -> None:
        self.login()
        headers = {"X-CSRF-Token": self.csrf()}
        requests = [
            (
                "POST",
                "/api/v1/tasks",
                {
                    "json": {
                        "plan_id": "scifact-comparison",
                        "plan_version": "v1",
                        "variant_id": "scifact-bm25",
                    }
                },
            ),
            (
                "POST",
                "/api/v1/runs/compare",
                {"json": {"run_ids": ["a", "b"]}},
            ),
            ("POST", "/api/v1/notices/1/ack", {}),
            ("POST", "/lan/logout", {}),
        ]
        for method, path, kwargs in requests:
            with self.subTest(path=path):
                response = self.client.request(
                    method, path, headers=headers, **kwargs
                )
                self.assertEqual(response.status_code, 403)
                self.assertEqual(
                    response.json()["error"]["code"], "ORIGIN_INVALID"
                )


NODE_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const appPath = process.argv[2];
const mode = process.argv[3];
const code = fs.readFileSync(appPath, "utf8");
const calls = { fetch: [], logout: 0 };

function makeEl(id) {
  return {
    id, dataset: {}, style: {}, children: [], textContent: "", hidden: true,
    disabled: false, firstChild: null,
    classList: { contains: () => false, add: () => {}, remove: () => {} },
    appendChild(el) { this.children.push(el); },
    removeChild() {},
    remove() {},
    addEventListener(name, cb) { this.listeners = this.listeners || {}; this.listeners[name] = cb; },
    setAttribute() {}, querySelector: () => null, querySelectorAll: () => [],
  };
}

const ids = mode === "monitor" ? ["task-list"] : ["plans-grid", "notices", "lan-logout-button"];
const elements = {};
ids.forEach((id) => { elements[id] = makeEl(id); });
if (elements["lan-logout-button"]) {
  const target = elements["lan-logout-button"];
  const original = target.addEventListener;
  target.addEventListener = function (name, cb) {
    if (name === "click") calls.logout += 1;
    return original.call(this, name, cb);
  };
}

const documentMock = {
  querySelector: (sel) => (sel.indexOf("csrf-token") >= 0 ? { content: "token" } : null),
  querySelectorAll: () => [],
  getElementById: (id) => elements[id] || null,
  createElement: () => makeEl("created"),
};

class FakeEventSource {
  constructor(url) { this.listeners = {}; FakeEventSource.instances.push(this); }
  addEventListener(name, cb) { this.listeners[name] = cb; }
  close() { this.closed = true; }
}
FakeEventSource.instances = [];

const sandbox = {
  document: documentMock,
  window: {
    EventSource: FakeEventSource,
    setInterval: () => 1,
    setTimeout: () => 1,
    location: { href: "" },
  },
  fetch: (url) => {
    calls.fetch.push(url);
    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve(
        url.indexOf("notices") >= 0
          ? { notices: [{ id: 1, message: "提示" }] }
          : { current: null, queue: [], tasks: [] }
      ),
    });
  },
  console, Map, Array, Object, JSON, Number, isNaN, Promise, Date,
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(code, sandbox);

if (mode === "monitor") {
  const source = FakeEventSource.instances[0];
  source.listeners["session.expired"]();
  console.log(JSON.stringify({ href: sandbox.window.location.href }));
} else {
  Promise.resolve()
    .then(() => new Promise((resolve) => setTimeout(resolve, 20)))
    .then(() => {
      console.log(JSON.stringify({
        fetch: calls.fetch,
        logout: calls.logout,
        notices: elements["notices"].children.length,
      }));
    });
}
"""


class FrontendExecutionTests(unittest.TestCase):
    APP_JS = PROJECT_ROOT / "src" / "web" / "static" / "app.js"

    def _run(self, mode: str) -> dict:
        node = shutil.which("node")
        if not node:
            self.skipTest("node 不可用，无法执行前端脚本验证")
        with tempfile.TemporaryDirectory() as tmp:
            harness = Path(tmp) / "harness.js"
            harness.write_text(NODE_HARNESS, encoding="utf-8")
            result = subprocess.run(
                [node, str(harness), str(self.APP_JS), mode],
                capture_output=True,
                text=True,
                timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout.strip().splitlines()[-1])

    def test_plans_script_executes_and_initializes(self) -> None:
        data = self._run("plans")
        self.assertIn("/api/v1/notices", data["fetch"])
        self.assertGreaterEqual(data["logout"], 1)
        self.assertEqual(data["notices"], 1)

    def test_monitor_session_expired_redirects(self) -> None:
        data = self._run("monitor")
        self.assertEqual(data["href"], "/lan/login")


if __name__ == "__main__":
    unittest.main()
