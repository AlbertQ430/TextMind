"""T-006 Web 应用、页面与任务 API 测试。"""

from __future__ import annotations

import re
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from experiment.plans import RunVariant, load_plan_catalog  # noqa: E402
from resources.settings import SettingsStore  # noqa: E402
from tasks.models import (  # noqa: E402
    TaskConflictError,
    TaskNotFoundError,
    TaskStatus,
    TaskSubmitError,
    build_task_record,
    generate_run_id,
)
from web.app import DEFAULT_PORT, LOCAL_HOST, create_app  # noqa: E402
from web.context import AppContext  # noqa: E402
from web.routes import find_plan  # noqa: E402
from web.session import SessionStore  # noqa: E402

CATALOG = load_plan_catalog()
SETTINGS_PATH = Path(tempfile.mkdtemp()) / "settings.json"


class FakeTaskService:
    def __init__(self) -> None:
        self.tasks = {}
        self.started = 0
        self.stopped = 0
        self.submit_calls = 0

    def start(self) -> None:
        self.started += 1

    def stop_normal(self, reason: str = "服务关闭") -> None:
        self.stopped += 1

    def list_tasks(self):
        return sorted(
            self.tasks.values(),
            key=lambda task: (task.submitted_at, task.run_id),
        )

    def get(self, run_id):
        task = self.tasks.get(run_id)
        if task is None:
            raise TaskNotFoundError(run_id)
        return task

    def submit(self, plan_id, plan_version, variant_id):
        self.submit_calls += 1
        plan = find_plan(CATALOG, plan_id)
        if plan is None or plan.plan_version != plan_version:
            raise TaskSubmitError("未知方案")
        variant = next(
            (item for item in plan.variants if item.variant_id == variant_id), None
        )
        if variant is None:
            raise TaskSubmitError("方案中没有该变体")
        if not isinstance(variant, RunVariant):
            raise TaskSubmitError("参照结果不可提交")
        run_id = generate_run_id()
        self.tasks[run_id] = build_task_record(
            run_id,
            plan_id,
            plan_version,
            variant_id,
            {
                "plan_name": plan.name,
                "variant_name": variant.name,
                "variant": {"method": variant.method},
            },
            submitted_at=datetime.now(timezone.utc),
        )
        return run_id

    def cancel(self, run_id):
        task = self.get(run_id)
        if task.is_terminal:
            raise TaskConflictError("任务已处于终态，无法取消")
        if task.status == TaskStatus.PENDING:
            del self.tasks[run_id]
            return
        raise TaskConflictError("测试服务不支持运行中取消")

    def add_terminal_task(self) -> str:
        run_id = generate_run_id()
        self.tasks[run_id] = build_task_record(
            run_id,
            "scifact-comparison",
            "v1",
            "scifact-bm25",
            {"plan_name": "SciFact", "variant_name": "BM25"},
            submitted_at=datetime.now(timezone.utc),
            status=TaskStatus.COMPLETED,
        )
        return run_id


def build_app(service: FakeTaskService):
    context = AppContext(
        catalog=CATALOG,
        task_service=service,
        session_store=SessionStore(),
        settings_store=SettingsStore(SETTINGS_PATH),
    )
    return create_app(context), context


class AppLifecycleTests(unittest.TestCase):
    def test_create_app_has_no_side_effects(self) -> None:
        service = FakeTaskService()
        before = threading.active_count()
        app = build_app(service)[0]
        self.assertEqual(service.started, 0)
        self.assertEqual(threading.active_count(), before)
        self.assertEqual(app.version, "0.1.0")

    def test_lifespan_starts_and_stops_once(self) -> None:
        service = FakeTaskService()
        app, _ = build_app(service)
        with TestClient(app) as client:
            self.assertEqual(service.started, 1)
            self.assertEqual(client.get("/plans").status_code, 200)
        self.assertEqual(service.stopped, 1)

    def test_loopback_listener_constants(self) -> None:
        self.assertEqual(LOCAL_HOST, "127.0.0.1")
        self.assertEqual(DEFAULT_PORT, 8765)


class PageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = FakeTaskService()
        self.app, self.context = build_app(self.service)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_root_redirects_to_plans(self) -> None:
        response = self.client.get("/", follow_redirects=False)
        self.assertIn(response.status_code, (302, 303, 307))
        self.assertEqual(response.headers["location"], "/plans")

    def test_main_pages_and_settings_are_accessible(self) -> None:
        for path in ("/plans", "/monitor", "/results", "/history", "/settings"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn("TextMind", response.text)

    def test_navigation_contains_all_entries(self) -> None:
        html = self.client.get("/plans").text
        for label in ("实验中心", "运行监控", "实验结果", "历史记录", "设置"):
            self.assertIn(label, html)
        self.assertIn('href="/plans"', html)
        self.assertIn('href="/monitor"', html)

    def test_plans_page_and_api_are_consistent(self) -> None:
        api = self.client.get("/api/v1/plans").json()["plans"]
        self.assertEqual(len(api), 3)
        html = self.client.get("/plans").text
        for plan in api:
            self.assertIn(plan["name"], html)
            self.assertIn(f'/plans/{plan["plan_id"]}', html)

    def test_plan_detail_page_and_api_are_consistent(self) -> None:
        detail = self.client.get("/api/v1/plans/scifact-comparison").json()["plan"]
        html = self.client.get("/plans/scifact-comparison").text
        self.assertEqual(len(detail["variants"]), 9)
        for variant in detail["variants"]:
            self.assertIn(variant["variant_id"], html)
        self.assertIn("待资源检查", html)

    def test_unknown_plan(self) -> None:
        self.assertEqual(
            self.client.get("/api/v1/plans/missing").status_code, 404
        )
        page = self.client.get("/plans/missing")
        self.assertEqual(page.status_code, 404)
        self.assertIn("未找到方案", page.text)

    def test_placeholders_and_real_history(self) -> None:
        redirect = self.client.get("/results", follow_redirects=False)
        self.assertEqual(redirect.status_code, 307)
        self.assertEqual(redirect.headers["location"], "/history")

        history = self.client.get("/history")
        self.assertEqual(history.status_code, 200)
        self.assertIn("暂无运行记录", history.text)

        settings = self.client.get("/settings")
        self.assertEqual(settings.status_code, 200)
        self.assertIn("资源与设备", settings.text)
        self.assertIn("后续任务", settings.text)
        self.assertNotIn("P@1", settings.text)

    def test_static_assets_are_local(self) -> None:
        css = self.client.get("/static/app.css")
        js = self.client.get("/static/app.js")
        self.assertEqual(css.status_code, 200)
        self.assertEqual(js.status_code, 200)
        for body in (css.text, js.text):
            self.assertNotIn("cdn.", body.lower())
            self.assertNotIn("https://", body)

    def test_elapsed_time_updates_locally(self) -> None:
        js = self.client.get("/static/app.js").text
        self.assertIn("function tickElapsed", js)
        self.assertIn("setInterval(tickElapsed, 1000)", js)
        self.assertIn("formatElapsed", js)
        self.assertIn('className = "elapsed"', js)
        self.assertIn("textContent", js)


class TaskApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = FakeTaskService()
        self.app, self.context = build_app(self.service)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.token = self.client.get("/api/v1/session").json()["session"][
            "csrf_token"
        ]

    def _post(self, path, payload=None):
        return self.client.post(
            path,
            json=payload or {},
            headers={"X-CSRF-Token": self.token},
        )

    def _submit(self, variant_id="scifact-bm25", plan_id="scifact-comparison"):
        return self._post(
            "/api/v1/tasks",
            {
                "plan_id": plan_id,
                "plan_version": "v1",
                "variant_id": variant_id,
            },
        )

    def test_submit_and_list(self) -> None:
        response = self._submit()
        self.assertEqual(response.status_code, 201)
        run_id = response.json()["run_id"]
        self.assertTrue(run_id)

        listed = self.client.get("/api/v1/tasks").json()
        self.assertEqual(listed["counts"]["total"], 1)
        self.assertEqual(listed["tasks"][0]["run_id"], run_id)
        single = self.client.get(f"/api/v1/tasks/{run_id}").json()["task"]
        self.assertEqual(single["status"], "Pending")
        self.assertEqual(single["plan_name"], "SciFact 英文检索方法与模型对比")

    def test_submit_rejects_unknown_plan_and_variant(self) -> None:
        unknown_plan = self._submit(plan_id="missing")
        self.assertEqual(unknown_plan.status_code, 422)
        self.assertEqual(unknown_plan.json()["error"]["code"], "VARIANT_INVALID")

        unknown_variant = self._submit(variant_id="missing")
        self.assertEqual(unknown_variant.status_code, 422)

        reference = self._submit(
            plan_id="duretrieval-rrf-weight-sweep",
            variant_id="duretrieval-hybrid-bge-small-prefix-equal",
        )
        self.assertEqual(reference.status_code, 422)

    def test_cancel_pending(self) -> None:
        run_id = self._submit().json()["run_id"]
        response = self._post(f"/api/v1/tasks/{run_id}/cancel")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "Deleted")
        self.assertEqual(self.client.get("/api/v1/tasks").json()["counts"]["total"], 0)

    def test_cancel_terminal_conflicts(self) -> None:
        run_id = self.service.add_terminal_task()
        response = self._post(f"/api/v1/tasks/{run_id}/cancel")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "TASK_CONFLICT")

    def test_unknown_task_returns_404(self) -> None:
        response = self.client.get("/api/v1/tasks/missing")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "TASK_NOT_FOUND")

    def test_error_payload_shape(self) -> None:
        response = self._submit(plan_id="missing")
        error = response.json()["error"]
        self.assertEqual(set(error), {"code", "message", "request_id"})
        self.assertTrue(error["request_id"])


class SharedStateTests(unittest.TestCase):
    def test_two_clients_share_task_state(self) -> None:
        service = FakeTaskService()
        app, _ = build_app(service)
        with TestClient(app) as first, TestClient(app) as second:
            token = first.get("/api/v1/session").json()["session"]["csrf_token"]
            run_id = first.post(
                "/api/v1/tasks",
                json={
                    "plan_id": "scifact-comparison",
                    "plan_version": "v1",
                    "variant_id": "scifact-bm25",
                },
                headers={"X-CSRF-Token": token},
            ).json()["run_id"]
            seen = second.get("/api/v1/tasks").json()["tasks"]
        self.assertEqual([task["run_id"] for task in seen], [run_id])


if __name__ == "__main__":
    unittest.main()
