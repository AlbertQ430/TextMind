"""T-006 SSE 快照、事件与心跳测试。"""

from __future__ import annotations

import json
import sys
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from experiment.plans import load_plan_catalog  # noqa: E402
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
from web.session import SessionStore  # noqa: E402

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
        plan = next(
            (p for p in CATALOG.plans if p.plan_id == plan_id), None
        )
        if plan is None:
            raise TaskSubmitError("未知方案")
        variant = next(
            (v for v in plan.variants if v.variant_id == variant_id), None
        )
        if variant is None:
            raise TaskSubmitError("没有该变体")
        run_id = generate_run_id()
        self.tasks[run_id] = build_task_record(
            run_id,
            plan_id,
            plan_version,
            variant_id,
            {
                "plan_name": plan.name,
                "variant_name": variant.name,
                "variant": {"method": getattr(variant, "method", None)},
            },
            submitted_at=datetime.now(timezone.utc),
        )
        return run_id

    def cancel(self, run_id):
        raise TaskConflictError("测试服务不支持取消")

    def add_running_task(self) -> str:
        run_id = generate_run_id()
        started = datetime.now(timezone.utc)
        record = build_task_record(
            run_id,
            "scifact-comparison",
            "v1",
            "scifact-bm25",
            {
                "plan_name": "SciFact 英文检索方法与模型对比",
                "variant_name": "BM25",
                "variant": {"method": "bm25"},
            },
            submitted_at=started,
            status=TaskStatus.RUNNING,
        )
        self.tasks[run_id] = replace(record, started_at=started)
        return run_id

    def set_status(self, run_id: str, status: TaskStatus) -> None:
        task = self.tasks[run_id]
        self.tasks[run_id] = build_task_record(
            task.run_id,
            task.plan_id,
            task.plan_version,
            task.variant_id,
            dict(task.config_snapshot),
            submitted_at=task.submitted_at,
            status=status,
        )


def build_app(service: FakeTaskService):
    context = AppContext(
        catalog=CATALOG,
        task_service=service,
        session_store=SessionStore(),
        sse_poll_interval=0.02,
        sse_heartbeat_interval=0.05,
    )
    return create_app(context), context


def read_event(iterator, limit: int = 400):
    """读取一个完整事件；跳过心跳注释行。"""
    event = None
    data_lines: list[str] = []
    for _ in range(limit):
        line = next(iterator)
        if line.startswith("event: "):
            event = line[len("event: "):]
        elif line.startswith("data: "):
            data_lines.append(line[len("data: "):])
        elif line == "":
            if event is not None:
                return event, json.loads("\n".join(data_lines))
        elif line.startswith(":"):
            continue
    raise AssertionError("在限定行数内没有读到完整事件")


class SseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = FakeTaskService()
        self.app, self.context = build_app(self.service)

    def test_snapshot_on_connect(self) -> None:
        with TestClient(self.app) as client:
            with client.stream("GET", "/api/v1/events") as response:
                self.assertEqual(response.status_code, 200)
                self.assertTrue(
                    response.headers["content-type"].startswith(
                        "text/event-stream"
                    )
                )
                event, data = read_event(response.iter_lines())
        self.assertEqual(event, "snapshot")
        self.assertIn("tasks", data)
        self.assertIn("queue", data)
        self.assertIn("current", data)
        self.assertIsNone(data["current"])

    def test_task_updated_event_on_change(self) -> None:
        with TestClient(self.app) as client:
            with client.stream("GET", "/api/v1/events") as response:
                iterator = response.iter_lines()
                event, data = read_event(iterator)
                self.assertEqual(event, "snapshot")

                run_id = self.service.add_running_task()
                event, data = read_event(iterator)

        self.assertEqual(event, "task.updated")
        self.assertIsNotNone(data["current"])
        self.assertEqual(data["current"]["run_id"], run_id)
        self.assertEqual(data["current"]["status"], "Running")

    def test_heartbeat_comments(self) -> None:
        with TestClient(self.app) as client:
            with client.stream("GET", "/api/v1/events") as response:
                iterator = response.iter_lines()
                read_event(iterator)  # snapshot
                found = False
                for _ in range(400):
                    line = next(iterator)
                    if line.startswith(": ping"):
                        found = True
                        break
        self.assertTrue(found)

    def test_stream_does_not_leak_secrets(self) -> None:
        with TestClient(self.app) as client:
            token = client.get("/api/v1/session").json()["session"][
                "csrf_token"
            ]
            with client.stream("GET", "/api/v1/events") as response:
                _, data = read_event(response.iter_lines())
        text = json.dumps(data)
        self.assertNotIn(token, text)
        self.assertNotIn(str(PROJECT_ROOT), text)
        self.assertNotIn("csrf", text.lower())
        self.assertNotIn("cookie", text.lower())

    def test_disconnect_does_not_affect_tasks(self) -> None:
        with TestClient(self.app) as client:
            with client.stream("GET", "/api/v1/events") as response:
                read_event(response.iter_lines())
            run_id = self.service.submit(
                "scifact-comparison", "v1", "scifact-bm25"
            )
            self.assertTrue(run_id)
            self.assertEqual(self.service.submit_calls, 1)

    def test_sse_requires_and_uses_session_cookie(self) -> None:
        with TestClient(self.app) as client:
            with client.stream("GET", "/api/v1/events") as response:
                self.assertEqual(response.status_code, 200)
                self.assertIn("textmind_session=", response.headers.get(
                    "set-cookie", ""
                ))
                read_event(response.iter_lines())

    def test_queue_updated_on_pending_submit(self) -> None:
        with TestClient(self.app) as client:
            with client.stream("GET", "/api/v1/events") as response:
                iterator = response.iter_lines()
                read_event(iterator)  # snapshot
                run_id = self.service.submit(
                    "scifact-comparison", "v1", "scifact-bm25"
                )
                event, data = read_event(iterator)
        self.assertEqual(event, "queue.updated")
        self.assertEqual([task["run_id"] for task in data["queue"]], [run_id])

    def test_queue_updated_on_pending_cancel(self) -> None:
        with TestClient(self.app) as client:
            with client.stream("GET", "/api/v1/events") as response:
                iterator = response.iter_lines()
                read_event(iterator)  # snapshot
                run_id = self.service.submit(
                    "scifact-comparison", "v1", "scifact-bm25"
                )
                # 提交同时改变队列和任务列表，先收 queue 再收 task。
                self.assertEqual(read_event(iterator)[0], "queue.updated")
                self.assertEqual(read_event(iterator)[0], "task.updated")

                del self.service.tasks[run_id]
                event, data = read_event(iterator)
        self.assertEqual(event, "queue.updated")
        self.assertEqual(data["queue"], [])

    def test_pending_to_running_sends_queue_then_task(self) -> None:
        with TestClient(self.app) as client:
            with client.stream("GET", "/api/v1/events") as response:
                iterator = response.iter_lines()
                read_event(iterator)  # snapshot
                run_id = self.service.submit(
                    "scifact-comparison", "v1", "scifact-bm25"
                )
                self.assertEqual(read_event(iterator)[0], "queue.updated")
                self.assertEqual(read_event(iterator)[0], "task.updated")

                self.service.set_status(run_id, TaskStatus.RUNNING)
                first_event, _ = read_event(iterator)
                second_event, data = read_event(iterator)

        self.assertEqual(first_event, "queue.updated")
        self.assertEqual(second_event, "task.updated")
        self.assertIsNotNone(data["current"])
        self.assertEqual(data["current"]["run_id"], run_id)
        self.assertEqual(data["current"]["status"], "Running")
        self.assertEqual(data["queue"], [])

    def test_cross_origin_sse_is_rejected(self) -> None:
        with TestClient(self.app) as client:
            response = client.get(
                "/api/v1/events",
                headers={"Origin": "http://evil.example"},
            )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"]["code"], "ORIGIN_INVALID")

    def test_no_origin_is_allowed_for_local_clients(self) -> None:
        # 本地命令行或 SSE 客户端不带 Origin；浏览器跨源请求一定会带 Origin。
        with TestClient(self.app) as client:
            with client.stream("GET", "/api/v1/events") as response:
                self.assertEqual(response.status_code, 200)
                event, _ = read_event(response.iter_lines())
        self.assertEqual(event, "snapshot")

    def test_invalid_origin_port_is_rejected(self) -> None:
        with TestClient(self.app) as client:
            response = client.get(
                "/api/v1/events",
                headers={"Origin": "http://testserver:bad"},
            )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"]["code"], "ORIGIN_INVALID")

    def test_elapsed_seconds_do_not_trigger_task_updates(self) -> None:
        with TestClient(self.app) as client:
            with client.stream("GET", "/api/v1/events") as response:
                iterator = response.iter_lines()
                read_event(iterator)  # snapshot
                self.service.add_running_task()
                event, _ = read_event(iterator)
                self.assertEqual(event, "task.updated")

                # 运行时间随墙钟增长，但只应收到心跳，不产生新事件。
                heartbeat_count = 0
                for _ in range(80):
                    line = next(iterator)
                    if line.startswith("event: "):
                        self.fail(f"运行时间变化不应产生事件: {line}")
                    if line.startswith(": ping"):
                        heartbeat_count += 1
                        if heartbeat_count >= 3:
                            break
        self.assertGreaterEqual(heartbeat_count, 3)


if __name__ == "__main__":
    unittest.main()
