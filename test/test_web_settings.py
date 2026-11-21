"""T-009 设置 API、资源状态、提交门禁与 settings.changed 测试。"""

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

from experiment.executor import ExperimentExecutor  # noqa: E402
from experiment.plans import load_plan_catalog  # noqa: E402
from records.publisher import RunRecordPublisher  # noqa: E402
from resources.checker import ResourceChecker  # noqa: E402
from resources.settings import (  # noqa: E402
    NetworkSettings,
    Settings,
    SettingsStore,
)
from tasks.models import TaskNotFoundError, TaskSubmitError  # noqa: E402
from tasks.repository import TaskRepository  # noqa: E402
from tasks.service import TaskService  # noqa: E402
from web.app import create_app  # noqa: E402
from web.context import AppContext  # noqa: E402
from web.session import SessionStore  # noqa: E402

CATALOG = load_plan_catalog()
MODEL_NAME = "Qwen3-Embedding-0.6B"


class FakeLock:
    def acquire(self) -> None:
        return None

    def release(self) -> None:
        return None


class FakeTaskService:
    def start(self) -> None:
        return None

    def stop_normal(self, reason: str = "服务关闭") -> None:
        return None

    def list_tasks(self):
        return []

    def get(self, run_id):
        raise TaskNotFoundError(run_id)

    def submit(self, *args):
        raise TaskSubmitError("测试上下文不提交任务")

    def cancel(self, run_id):
        raise TaskNotFoundError(run_id)


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


class SettingsWebTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.settings_path = self.root / "settings.json"
        self.store = SettingsStore(self.settings_path)
        self.scifact = self.root / "scifact"
        self._write_scifact()
        self.models = self.root / "models"
        self.models.mkdir()
        self.apply_settings(Settings())

    def _write_scifact(self) -> None:
        for relative in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
            path = self.scifact / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("data\n", encoding="utf-8")

    def apply_settings(self, settings: Settings) -> None:
        self.store.save(settings)
        self.checker = ResourceChecker(
            CATALOG,
            self.store.load,
            device_probe=lambda name: name != "cuda",
            scifact_dir=self.scifact,
        )

    def build_client(self, *, task_service=None, settings_store=None):
        context = AppContext(
            catalog=CATALOG,
            task_service=task_service or FakeTaskService(),
            session_store=SessionStore(),
            settings_store=settings_store or self.store,
            resource_checker=self.checker,
            sse_poll_interval=0.02,
            sse_heartbeat_interval=0.05,
        )
        app = create_app(context)
        client = TestClient(app)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        token = client.get("/api/v1/session").json()["session"]["csrf_token"]
        return client, token

    def make_model(self, directory_name: str = MODEL_NAME) -> None:
        directory = self.models / directory_name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text("{}", encoding="utf-8")
        (directory / "model.safetensors").write_text("w", encoding="utf-8")
        (directory / "tokenizer.json").write_text("{}", encoding="utf-8")


class SettingsApiTests(SettingsWebTestCase):
    def test_get_defaults(self) -> None:
        client, _ = self.build_client()
        response = client.get("/api/v1/settings")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["settings"]["device"], "auto")
        self.assertIsNone(payload["settings"]["model_root"])
        self.assertEqual(
            [dataset["id"] for dataset in payload["resources"]["datasets"]],
            ["scifact", "duretrieval"],
        )
        self.assertEqual(
            {model["id"] for model in payload["resources"]["models"]},
            {MODEL_NAME, "bge-m3", "all-MiniLM-L6-v2", "bge-small-zh-v1.5"},
        )
        self.assertIn("python", payload["environment"])
        self.assertNotIn("hostname", json.dumps(payload))

    def test_put_success_and_network_preserved(self) -> None:
        self.store.save(
            Settings(
                network=NetworkSettings(
                    lan_enabled=True, local_port=9000, lan_port=9001
                )
            )
        )
        client, token = self.build_client()
        response = client.put(
            "/api/v1/settings",
            json={
                "model_root": str(self.models),
                "duretrieval_root": None,
                "device": "cpu",
            },
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["settings"]["device"], "cpu")
        self.assertTrue(
            Path(payload["settings"]["model_root"]).is_absolute()
        )
        self.assertTrue(payload["settings"]["network"]["lan_enabled"])
        self.assertEqual(payload["settings"]["network"]["local_port"], 9000)

    def test_put_network_is_read_only(self) -> None:
        client, token = self.build_client()
        response = client.put(
            "/api/v1/settings",
            json={
                "model_root": None,
                "duretrieval_root": None,
                "device": "auto",
                "network": {"lan_enabled": True},
            },
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "SETTINGS_READ_ONLY")

    def test_put_network_message_points_to_network_api(self) -> None:
        client, token = self.build_client()
        response = client.put(
            "/api/v1/settings",
            json={
                "model_root": None,
                "duretrieval_root": None,
                "device": "auto",
                "network": {"lan_enabled": True},
            },
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 409)
        message = response.json()["error"]["message"]
        self.assertIn("/api/v1/network/disable", message)
        self.assertNotIn("后续任务", message)

    def test_put_missing_or_invalid_fields(self) -> None:
        client, token = self.build_client()
        response = client.put(
            "/api/v1/settings",
            json={"model_root": None},
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "SETTINGS_INVALID")

        response = client.put(
            "/api/v1/settings",
            json={
                "model_root": None,
                "duretrieval_root": None,
                "device": "tpu",
            },
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "SETTINGS_INVALID")

    def test_put_requires_csrf(self) -> None:
        client, _ = self.build_client()
        before = self.settings_path.read_bytes()
        response = client.put(
            "/api/v1/settings",
            json={
                "model_root": None,
                "duretrieval_root": None,
                "device": "auto",
            },
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.settings_path.read_bytes(), before)

    def test_corrupt_settings_does_not_break_plans_or_tasks(self) -> None:
        repository = TaskRepository(self.root / "tasks.db")
        publisher = RunRecordPublisher(self.root / "results")
        executor = ExperimentExecutor(
            dataset_loader=lambda plan: None,
            model_loader=lambda model, device: None,
            device_resolver=lambda: "cpu",
            settings_provider=self.store.load,
        )
        service = TaskService(
            CATALOG,
            repository,
            executor,
            publisher,
            reconciler=publisher,
            resource_checker=self.checker,
            lock=FakeLock(),
            poll_interval=0.01,
        )
        client, token = self.build_client(task_service=service)
        self.settings_path.write_text("{bad", encoding="utf-8")

        response = client.get("/api/v1/plans")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "SETTINGS_INVALID")

        response = client.get("/api/v1/plans/scifact-comparison")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "SETTINGS_INVALID")

        plans_page = client.get("/plans")
        self.assertEqual(plans_page.status_code, 200)
        self.assertIn("设置读取失败", plans_page.text)
        self.assertIn("SciFact", plans_page.text)

        detail_page = client.get("/plans/scifact-comparison")
        self.assertEqual(detail_page.status_code, 200)
        self.assertIn("设置读取失败", detail_page.text)

        self.assertEqual(client.get("/history").status_code, 200)

        response = client.post(
            "/api/v1/tasks",
            json={
                "plan_id": "scifact-comparison",
                "plan_version": "v1",
                "variant_id": "scifact-bm25",
            },
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "SETTINGS_INVALID")
        self.assertEqual(service.list_tasks(), [])

    def test_put_write_failure_returns_500(self) -> None:
        client, token = self.build_client()
        with mock.patch(
            "resources.settings.Path.mkdir", side_effect=OSError("denied")
        ):
            response = client.put(
                "/api/v1/settings",
                json={
                    "model_root": None,
                    "duretrieval_root": None,
                    "device": "cpu",
                },
                headers={"X-CSRF-Token": token},
            )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["error"]["code"], "SETTINGS_WRITE_FAILED"
        )

    def test_corrupt_settings_reports_error(self) -> None:
        self.settings_path.write_text("{bad", encoding="utf-8")
        client, _ = self.build_client()
        response = client.get("/api/v1/settings")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "SETTINGS_INVALID")

        page = client.get("/settings")
        self.assertEqual(page.status_code, 200)
        self.assertIn("设置文件", page.text)
        self.assertEqual(
            self.settings_path.read_text(encoding="utf-8"), "{bad"
        )


class ResourceGateTests(SettingsWebTestCase):
    def test_plans_api_and_page_reflect_resources(self) -> None:
        client, _ = self.build_client()
        plans = client.get("/api/v1/plans").json()["plans"]
        scifact = next(
            plan for plan in plans if plan["plan_id"] == "scifact-comparison"
        )
        self.assertEqual(scifact["runnable_variant_count"], 1)
        self.assertEqual(scifact["run_variant_count"], 9)

        detail = client.get("/api/v1/plans/scifact-comparison").json()["plan"]
        variants = {item["variant_id"]: item for item in detail["variants"]}
        self.assertTrue(variants["scifact-bm25"]["runnable"])
        self.assertFalse(variants["scifact-dense-qwen3-base"]["runnable"])
        self.assertEqual(
            variants["scifact-dense-qwen3-base"]["resource_state"],
            "unavailable",
        )
        self.assertTrue(
            variants["scifact-dense-qwen3-base"]["resource_missing"]
        )

        page = client.get("/plans").text
        self.assertIn("1 / 9", page)
        detail_page = client.get("/plans/scifact-comparison").text
        self.assertIn("不可运行", detail_page)
        self.assertIn("模型根目录", detail_page)

    def test_model_available_enables_dense(self) -> None:
        self.make_model()
        self.apply_settings(Settings(model_root=str(self.models), device="cpu"))
        client, _ = self.build_client()
        detail = client.get("/api/v1/plans/scifact-comparison").json()["plan"]
        variants = {item["variant_id"]: item for item in detail["variants"]}
        self.assertTrue(variants["scifact-dense-qwen3-base"]["runnable"])
        self.assertEqual(
            variants["scifact-dense-qwen3-base"]["resource_state"], "available"
        )

    def test_plans_page_watches_settings_changes(self) -> None:
        client, _ = self.build_client()
        page = client.get("/plans").text
        self.assertIn('id="plans-grid"', page)

        app_js = (
            PROJECT_ROOT / "src" / "web" / "static" / "app.js"
        ).read_text(encoding="utf-8")
        self.assertIn('document.getElementById("plans-grid")', app_js)
        self.assertIn("function startSettingsWatcher()", app_js)
        self.assertIn("failures > 3", app_js)
        self.assertIn("setInterval", app_js)
        self.assertIn("settings.changed", app_js)

    def test_settings_page_content(self) -> None:
        client, _ = self.build_client()
        page = client.get("/settings")
        self.assertEqual(page.status_code, 200)
        self.assertIn("资源与设备", page.text)
        self.assertIn("数据集", page.text)
        self.assertIn("计算设备", page.text)
        self.assertIn("局域网访问", page.text)
        self.assertIn("访问码", page.text)
        self.assertIn("HTTP", page.text)

    def test_submit_without_model_returns_409(self) -> None:
        repository = TaskRepository(self.root / "tasks.db")
        publisher = RunRecordPublisher(self.root / "results")
        executor = ExperimentExecutor(
            dataset_loader=lambda plan: None,
            model_loader=lambda model, device: None,
            device_resolver=lambda: "cpu",
            settings_provider=self.store.load,
        )
        service = TaskService(
            CATALOG,
            repository,
            executor,
            publisher,
            reconciler=publisher,
            resource_checker=self.checker,
            lock=FakeLock(),
            poll_interval=0.01,
        )
        client, token = self.build_client(task_service=service)
        response = client.post(
            "/api/v1/tasks",
            json={
                "plan_id": "scifact-comparison",
                "plan_version": "v1",
                "variant_id": "scifact-dense-qwen3-base",
            },
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(
            response.json()["error"]["code"], "RESOURCE_UNAVAILABLE"
        )
        self.assertEqual(service.list_tasks(), [])

    def test_submit_bm25_still_allowed(self) -> None:
        repository = TaskRepository(self.root / "tasks.db")
        publisher = RunRecordPublisher(self.root / "results")
        executor = ExperimentExecutor(
            dataset_loader=lambda plan: None,
            model_loader=lambda model, device: None,
            device_resolver=lambda: "cpu",
            settings_provider=self.store.load,
        )
        service = TaskService(
            CATALOG,
            repository,
            executor,
            publisher,
            reconciler=publisher,
            resource_checker=self.checker,
            lock=FakeLock(),
            poll_interval=0.01,
        )
        client, token = self.build_client(task_service=service)
        response = client.post(
            "/api/v1/tasks",
            json={
                "plan_id": "scifact-comparison",
                "plan_version": "v1",
                "variant_id": "scifact-bm25",
            },
            headers={"X-CSRF-Token": token},
        )
        self.assertEqual(response.status_code, 201)
        self.assertTrue(service.list_tasks())


class SettingsSseTests(SettingsWebTestCase):
    def test_settings_changed_sent_once(self) -> None:
        client, _ = self.build_client()
        with client.stream("GET", "/api/v1/events") as response:
            iterator = response.iter_lines()
            self.assertEqual(read_event(iterator)[0], "snapshot")

            self.store.save(Settings(device="cpu"))
            event, data = read_event(iterator)
            self.assertEqual(event, "settings.changed")
            self.assertTrue(data["changed"])

            heartbeats = 0
            for _ in range(160):
                line = next(iterator)
                if line.startswith("event: "):
                    self.fail(f"设置未变化却收到事件: {line}")
                if line.startswith(": ping"):
                    heartbeats += 1
                    if heartbeats >= 2:
                        break
            self.assertGreaterEqual(heartbeats, 2)


if __name__ == "__main__":
    unittest.main()
