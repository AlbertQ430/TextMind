"""T-001 工程骨架测试：应用工厂、健康检查、路径初始化和启动约束。"""

from __future__ import annotations

import re
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for extra_path in (SRC_DIR, SCRIPTS_DIR):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from fastapi.testclient import TestClient  # noqa: E402

import check_env  # noqa: E402
import paths  # noqa: E402
import web.app as web_app  # noqa: E402
from experiment.plans import load_plan_catalog  # noqa: E402
from tasks.models import TaskNotFoundError, TaskSubmitError  # noqa: E402
from web.session import SessionStore  # noqa: E402


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


class FakeContext:
    def __init__(self) -> None:
        self.catalog = load_plan_catalog()
        self.task_service = FakeTaskService()
        self.session_store = SessionStore()
        self.sse_poll_interval = 0.05
        self.sse_heartbeat_interval = 15.0
        self.started = 0
        self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1


class HealthCheckTests(unittest.TestCase):
    def test_health_returns_contract_json(self) -> None:
        context = FakeContext()
        app = web_app.create_app(context)
        with TestClient(app) as client:
            response = client.get("/api/v1/health")
            self.assertEqual(context.started, 1)
        self.assertEqual(context.stopped, 1)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("application/json"))
        self.assertEqual(response.json(), {"status": "ok", "version": "0.1.0"})

    def test_health_does_not_leak_host_or_paths(self) -> None:
        app = web_app.create_app(FakeContext())
        with TestClient(app) as client:
            text = client.get("/api/v1/health").text

        self.assertNotIn(socket.gethostname(), text)
        self.assertNotIn(str(PROJECT_ROOT), text)
        self.assertNotIn(str(Path.home()), text)
        self.assertNotIn("PATH", text)

    def test_create_app_starts_no_background_thread(self) -> None:
        before = threading.active_count()
        app = web_app.create_app(FakeContext())
        self.assertEqual(threading.active_count(), before)
        self.assertEqual(app.version, "0.1.0")


class LaunchConstraintTests(unittest.TestCase):
    def test_default_listener_is_loopback(self) -> None:
        self.assertEqual(web_app.LOCAL_HOST, "127.0.0.1")
        self.assertEqual(web_app.DEFAULT_PORT, 8765)
        self.assertNotEqual(web_app.LOCAL_HOST, "0.0.0.0")


class PathInitializationTests(unittest.TestCase):
    def _patched_tree(self, root: Path) -> dict[str, object]:
        layout = {
            "DATA_DIR": root / "data",
            "APP_DATA_DIR": root / "data" / "app",
            "RESULTS_DIR": root / "results",
            "PRESETS_DIR": root / "results" / "presets",
            "RUNS_DIR": root / "results" / "runs",
            "IMPORTS_DIR": root / "results" / "imports",
            "LOGS_DIR": root / "logs",
        }
        layout["APP_DIRS"] = tuple(layout[name] for name in (
            "APP_DATA_DIR", "RESULTS_DIR", "PRESETS_DIR",
            "RUNS_DIR", "IMPORTS_DIR", "LOGS_DIR",
        ))
        return layout

    def test_paths_are_absolute_and_inside_project(self) -> None:
        self.assertTrue(paths.PROJECT_ROOT.is_absolute())
        for name in (
            "DATA_DIR", "APP_DATA_DIR", "RESULTS_DIR", "PRESETS_DIR",
            "RUNS_DIR", "IMPORTS_DIR", "LOGS_DIR",
        ):
            self.assertTrue(getattr(paths, name).is_absolute(), name)
        self.assertEqual(paths.PRESETS_DIR.parent, paths.RESULTS_DIR)

    def test_ensure_app_dirs_is_idempotent_and_keeps_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = self._patched_tree(Path(tmp))
            with mock.patch.multiple(paths, **layout):
                paths.ensure_app_dirs()
                marker = paths.RUNS_DIR / "keep.json"
                marker.write_text("{}", encoding="utf-8")

                paths.ensure_app_dirs()

                self.assertEqual(marker.read_text(encoding="utf-8"), "{}")
                for directory in paths.APP_DIRS:
                    self.assertTrue(directory.is_dir(), directory)

    def test_ensure_app_dirs_reports_creation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            layout = self._patched_tree(Path(tmp))
            blocker = layout["LOGS_DIR"]
            blocker.write_text("not a directory", encoding="utf-8")
            with mock.patch.multiple(paths, **layout):
                with self.assertRaises(OSError):
                    paths.ensure_app_dirs()


class LockFileTests(unittest.TestCase):
    """requirements.lock 必须是全部精确版本的完整解析集。"""

    LOCK_FILE = PROJECT_ROOT / "requirements.lock"
    PINNED_LINE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*==[0-9][0-9A-Za-z.+_-]*$")
    DIRECT_DEPENDENCIES = (
        "fastapi", "uvicorn", "jinja2", "python-multipart", "jsonschema",
        "httpx", "numpy", "torch", "transformers", "sentence-transformers",
        "jieba", "matplotlib",
    )
    CRITICAL_TRANSITIVE_DEPENDENCIES = (
        "starlette", "pydantic", "pydantic-core", "tokenizers", "anyio", "h11",
    )

    @staticmethod
    def _normalize(name: str) -> str:
        return name.lower().replace("_", "-")

    def _entries(self) -> list[str]:
        entries: list[str] = []
        for raw in self.LOCK_FILE.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(line)
        return entries

    def _package_names(self) -> set[str]:
        return {self._normalize(line.split("==")[0]) for line in self._entries()}

    def test_every_entry_is_pinned(self) -> None:
        entries = self._entries()
        self.assertTrue(entries, "requirements.lock 不能为空")
        for line in entries:
            self.assertRegex(line, self.PINNED_LINE)

    def test_no_duplicate_entries(self) -> None:
        names = [self._normalize(line.split("==")[0]) for line in self._entries()]
        self.assertEqual(len(names), len(set(names)))

    def test_covers_direct_and_critical_transitive_dependencies(self) -> None:
        names = self._package_names()
        for dependency in self.DIRECT_DEPENDENCIES:
            self.assertIn(dependency, names)
        for dependency in self.CRITICAL_TRANSITIVE_DEPENDENCIES:
            self.assertIn(dependency, names)

    def test_env_check_dependencies_are_locked(self) -> None:
        names = self._package_names()
        for distribution, _purpose in check_env.DEPENDENCIES:
            self.assertIn(self._normalize(distribution), names)


class EnvironmentCheckTests(unittest.TestCase):
    def test_installed_environment_passes(self) -> None:
        self.assertTrue(check_env.run_checks(verbose=False))

    def test_missing_distribution_is_reported(self) -> None:
        with mock.patch.object(
            check_env, "_installed_version", return_value=None
        ):
            self.assertFalse(check_env.run_checks(verbose=False))


if __name__ == "__main__":
    unittest.main()
