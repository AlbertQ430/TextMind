"""T-006 本机会话与 CSRF 测试。"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from experiment.plans import RunVariant, load_plan_catalog  # noqa: E402
from tasks.models import (  # noqa: E402
    TaskConflictError,
    TaskNotFoundError,
    TaskStatus,
    TaskSubmitError,
    build_task_record,
    generate_run_id,
)
from web.app import create_app  # noqa: E402
from web.context import AppContext  # noqa: E402
from web.routes import find_plan  # noqa: E402
from web.session import COOKIE_NAME, SessionStore  # noqa: E402

CATALOG = load_plan_catalog()


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
        return list(self.tasks.values())

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
            raise TaskConflictError("任务已处于终态")
        if task.status == TaskStatus.PENDING:
            del self.tasks[run_id]
            return
        raise TaskConflictError("测试服务不支持运行中取消")


def build_app(service: FakeTaskService, store: SessionStore | None = None):
    context = AppContext(
        catalog=CATALOG,
        task_service=service,
        session_store=store or SessionStore(),
    )
    return create_app(context), context


class SessionCookieTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = FakeTaskService()
        self.app, self.context = build_app(self.service)

    def test_first_visit_sets_session_cookie(self) -> None:
        with TestClient(self.app) as client:
            response = client.get("/plans")
        set_cookie = response.headers.get("set-cookie", "")
        self.assertIn(f"{COOKIE_NAME}=", set_cookie)
        self.assertIn("HttpOnly", set_cookie)
        self.assertIn("SameSite=strict", set_cookie)
        self.assertIn("Path=/", set_cookie)
        self.assertNotIn("Secure", set_cookie)

    def test_session_endpoint_returns_token_with_no_store(self) -> None:
        with TestClient(self.app) as client:
            response = client.get("/api/v1/session")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("cache-control"), "no-store")
        session = response.json()["session"]
        self.assertTrue(session["csrf_token"])
        self.assertEqual(session["scope"], "local_authenticated")
        self.assertTrue(session["authenticated"])

    def test_cookie_does_not_contain_token(self) -> None:
        with TestClient(self.app) as client:
            token = client.get("/api/v1/session").json()["session"]["csrf_token"]
            cookie_value = client.cookies.get(COOKIE_NAME)
        self.assertNotEqual(cookie_value, token)
        self.assertNotIn(token, cookie_value or "")

    def test_new_sessions_get_distinct_tokens(self) -> None:
        with TestClient(self.app) as first:
            first_token = first.get("/api/v1/session").json()["session"]["csrf_token"]
        with TestClient(self.app) as second:
            second_token = second.get("/api/v1/session").json()["session"]["csrf_token"]
        self.assertNotEqual(first_token, second_token)

    def test_old_cookie_is_invalid_after_restart(self) -> None:
        with TestClient(self.app) as client:
            client.get("/api/v1/session")
            old_cookie = client.cookies.get(COOKIE_NAME)

        # 新进程等价于新的 SessionStore。
        new_service = FakeTaskService()
        new_app, _ = build_app(new_service, SessionStore())
        with TestClient(new_app, cookies={COOKIE_NAME: old_cookie}) as restarted:
            response = restarted.post(
                "/api/v1/tasks",
                json={
                    "plan_id": "scifact-comparison",
                    "plan_version": "v1",
                    "variant_id": "scifact-bm25",
                },
                headers={"X-CSRF-Token": "stale-token"},
            )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(new_service.submit_calls, 0)


class CsrfTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = FakeTaskService()
        self.app, self.context = build_app(self.service)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.token = self.client.get("/api/v1/session").json()["session"][
            "csrf_token"
        ]

    def _submit(self, **kwargs):
        return self.client.post(
            "/api/v1/tasks",
            json={
                "plan_id": "scifact-comparison",
                "plan_version": "v1",
                "variant_id": "scifact-bm25",
            },
            **kwargs,
        )

    def test_get_requests_do_not_require_csrf(self) -> None:
        self.assertEqual(self.client.get("/api/v1/plans").status_code, 200)

    def test_missing_csrf_is_rejected(self) -> None:
        response = self._submit()
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"]["code"], "CSRF_INVALID")
        self.assertTrue(response.json()["error"]["request_id"])
        self.assertEqual(self.service.submit_calls, 0)

    def test_wrong_csrf_is_rejected(self) -> None:
        response = self._submit(headers={"X-CSRF-Token": "wrong"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.service.submit_calls, 0)

    def test_cross_origin_is_rejected(self) -> None:
        response = self._submit(
            headers={
                "X-CSRF-Token": self.token,
                "Origin": "http://evil.example",
            }
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.service.submit_calls, 0)

    def test_valid_csrf_and_origin_are_accepted(self) -> None:
        response = self._submit(
            headers={
                "X-CSRF-Token": self.token,
                "Origin": "http://testserver",
            }
        )
        self.assertEqual(response.status_code, 201)
        self.assertTrue(response.json()["run_id"])
        self.assertEqual(self.service.submit_calls, 1)


class ApiErrorFormatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = FakeTaskService()
        self.app, _ = build_app(self.service)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.token = self.client.get("/api/v1/session").json()["session"][
            "csrf_token"
        ]

    def _assert_error(self, response, status: int, code: str):
        self.assertEqual(response.status_code, status)
        self.assertTrue(
            response.headers["content-type"].startswith("application/json")
        )
        payload = response.json()
        self.assertEqual(set(payload), {"error"})
        error = payload["error"]
        self.assertEqual(error["code"], code)
        self.assertTrue(error["message"])
        self.assertTrue(error["request_id"])
        return error

    def test_missing_field_returns_structured_error(self) -> None:
        response = self.client.post(
            "/api/v1/tasks",
            json={"plan_id": "scifact-comparison", "plan_version": "v1"},
            headers={"X-CSRF-Token": self.token},
        )
        error = self._assert_error(response, 422, "VALIDATION_ERROR")
        self.assertTrue(any("variant_id" in item for item in error["details"]))
        self.assertEqual(self.service.submit_calls, 0)

    def test_broken_json_returns_structured_error(self) -> None:
        response = self.client.post(
            "/api/v1/tasks",
            content="{bad json",
            headers={
                "Content-Type": "application/json",
                "X-CSRF-Token": self.token,
            },
        )
        self._assert_error(response, 422, "VALIDATION_ERROR")
        self.assertEqual(self.service.submit_calls, 0)

    def test_unknown_api_path_returns_structured_error(self) -> None:
        self._assert_error(
            self.client.get("/api/v1/unknown"), 404, "NOT_FOUND"
        )

    def test_method_not_allowed_returns_structured_error(self) -> None:
        self._assert_error(
            self.client.delete(
                "/api/v1/plans", headers={"X-CSRF-Token": self.token}
            ),
            405,
            "METHOD_NOT_ALLOWED",
        )

    def test_unknown_page_is_not_json(self) -> None:
        response = self.client.get("/nope")
        self.assertEqual(response.status_code, 404)
        self.assertNotIn(
            "application/json", response.headers.get("content-type", "")
        )


class OriginTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = FakeTaskService()
        self.app, _ = build_app(self.service)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.token = self.client.get("/api/v1/session").json()["session"][
            "csrf_token"
        ]

    def _submit(self, origin: str):
        return self.client.post(
            "/api/v1/tasks",
            json={
                "plan_id": "scifact-comparison",
                "plan_version": "v1",
                "variant_id": "scifact-bm25",
            },
            headers={"X-CSRF-Token": self.token, "Origin": origin},
        )

    def test_https_origin_is_rejected_on_http_request(self) -> None:
        response = self._submit("https://testserver")
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"]["code"], "CSRF_INVALID")
        self.assertEqual(self.service.submit_calls, 0)

    def test_port_mismatch_is_rejected(self) -> None:
        response = self._submit("http://testserver:8080")
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.service.submit_calls, 0)

    def test_default_port_is_accepted(self) -> None:
        response = self._submit("http://testserver:80")
        self.assertEqual(response.status_code, 201)
        self.assertEqual(self.service.submit_calls, 1)

    def test_invalid_port_is_rejected(self) -> None:
        response = self._submit("http://testserver:bad")
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"]["code"], "CSRF_INVALID")
        self.assertEqual(self.service.submit_calls, 0)

    def test_out_of_range_port_is_rejected(self) -> None:
        response = self._submit("http://testserver:70000")
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"]["code"], "CSRF_INVALID")
        self.assertEqual(self.service.submit_calls, 0)


if __name__ == "__main__":
    unittest.main()
