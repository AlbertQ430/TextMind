"""T-011 发布包解压后的真实进程端到端冒烟测试。"""

from __future__ import annotations

import http.cookiejar
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for extra_path in (SRC_DIR, SCRIPTS_DIR):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

import package_release  # noqa: E402
from records.schema import parse_run_record  # noqa: E402


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Client:
    def __init__(self, base: str) -> None:
        self.base = base
        jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(jar)
        )
        self.csrf: str | None = None

    def status(self, path: str, timeout: float = 30) -> int:
        response = self.opener.open(self.base + path, timeout=timeout)
        try:
            return response.status
        finally:
            response.close()

    def get_json(self, path: str):
        response = self.opener.open(self.base + path, timeout=30)
        try:
            return json.loads(response.read())
        finally:
            response.close()

    def text(self, path: str) -> str:
        response = self.opener.open(self.base + path, timeout=30)
        try:
            return response.read().decode("utf-8")
        finally:
            response.close()

    def post_json(self, path: str, payload: dict):
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-CSRF-Token": self.csrf or "",
            },
        )
        response = self.opener.open(request, timeout=60)
        try:
            return response.status, json.loads(response.read())
        finally:
            response.close()


class ReleaseSmokeTests(unittest.TestCase):
    process: subprocess.Popen | None = None
    client: _Client
    release_root: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        zip_path = cls.root / "textmind-0.1.0.zip"
        package_release.build_release(zip_path)

        extract = cls.root / "extract"
        extract.mkdir()
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract)
        cls.release_root = extract / package_release.TOP_DIR

        # 代理指向关闭端口，阻止应用主动访问外部资源；本机请求绕过代理。
        env = os.environ.copy()
        env.update(
            {
                "HTTP_PROXY": "http://127.0.0.1:9",
                "HTTPS_PROXY": "http://127.0.0.1:9",
                "ALL_PROXY": "http://127.0.0.1:9",
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        cls.port = _free_port()
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.process = subprocess.Popen(
            [
                sys.executable,
                str(cls.release_root / "scripts" / "start.py"),
                "--port",
                str(cls.port),
            ],
            cwd=str(cls.release_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        cls.client = _Client(cls.base)
        deadline = time.monotonic() + 120
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if cls.process.poll() is not None:
                stdout, stderr = cls.process.communicate(timeout=5)
                raise RuntimeError(
                    "服务提前退出:\n"
                    + (stdout or "")[-2000:]
                    + "\n"
                    + (stderr or "")[-2000:]
                )
            try:
                if cls.client.status("/api/v1/health") == 200:
                    break
            except (urllib.error.URLError, ConnectionError) as exc:
                last_error = exc
            time.sleep(0.3)
        else:
            raise RuntimeError(f"服务未在超时内就绪: {last_error}")

        cls.client.csrf = cls.client.get_json("/api/v1/session")["session"][
            "csrf_token"
        ]

    @classmethod
    def tearDownClass(cls) -> None:
        process = cls.process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
        cls._tmp.cleanup()

    def test_health_and_pages_from_release(self) -> None:
        self.assertEqual(self.client.get_json("/api/v1/health")["version"], "0.1.0")
        for path in ("/plans", "/monitor", "/history", "/settings"):
            with self.subTest(path=path):
                self.assertEqual(self.client.status(path), 200)

    def test_missing_assets_only_disable_affected_variants(self) -> None:
        plans = self.client.get_json("/api/v1/plans")["plans"]
        scifact = next(
            plan
            for plan in plans
            if plan["plan_id"] == "scifact-comparison"
        )
        self.assertEqual(scifact["runnable_variant_count"], 1)

        with self.assertRaises(urllib.error.HTTPError) as context:
            self.client.post_json(
                "/api/v1/tasks",
                {
                    "plan_id": "scifact-comparison",
                    "plan_version": "v1",
                    "variant_id": "scifact-dense-qwen3-base",
                },
            )
        error = context.exception
        try:
            self.assertEqual(error.code, 409)
            payload = json.loads(error.read())
        finally:
            error.close()
        self.assertEqual(
            payload["error"]["code"], "RESOURCE_UNAVAILABLE"
        )

    def test_run_bm25_and_validate_record(self) -> None:
        status, payload = self.client.post_json(
            "/api/v1/tasks",
            {
                "plan_id": "scifact-comparison",
                "plan_version": "v1",
                "variant_id": "scifact-bm25",
            },
        )
        run_id = payload["run_id"]
        self.assertEqual(status, 201)

        task = None
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            task = self.client.get_json(f"/api/v1/tasks/{run_id}")["task"]
            if task["status"] in (
                "Completed",
                "Failed",
                "Cancelled",
                "Interrupted",
            ):
                break
            time.sleep(0.5)
        self.assertIsNotNone(task)
        self.assertEqual(task["status"], "Completed")

        run_json = (
            self.release_root / "results" / "runs" / run_id / "run.json"
        )
        self.assertTrue(run_json.is_file())
        record = parse_run_record(
            json.loads(run_json.read_text(encoding="utf-8"))
        )
        self.assertEqual(record.status, "Completed")
        self.assertEqual(task["status"], record.status)
        self.assertNotIn("source", record.model_dump())
        self.assertNotIn("import_time", record.model_dump())
        self.assertTrue(record.charts)
        for chart in record.charts:
            self.assertTrue(
                (run_json.parent / chart.path).is_file(), chart.path
            )

        # 记录只写入发布包解压目录，不写入当前工作区。
        self.assertFalse(
            (PROJECT_ROOT / "results" / "runs" / run_id).exists()
        )

    def test_all_presets_are_openable_from_release(self) -> None:
        preset_dir = self.release_root / "results" / "presets"
        directories = sorted(
            path for path in preset_dir.iterdir() if path.is_dir()
        )
        self.assertEqual(len(directories), 17)
        for directory in directories:
            run_id = directory.name
            with self.subTest(run_id=run_id):
                view = self.client.get_json(f"/api/v1/runs/{run_id}")
                self.assertEqual(view["source"], "presets")
                self.assertEqual(view["run"]["run_id"], run_id)
                self.assertTrue(view["charts"])
                for chart in view["charts"]:
                    self.assertTrue(chart["available"], chart["id"])
                self.assertEqual(
                    self.client.status(f"/results/{run_id}"), 200
                )

        history = self.client.get_json("/api/v1/runs")["records"]
        self.assertEqual(len(history), 16)
        history_ids = {record["run_id"] for record in history}
        self.assertNotIn(
            "preset-duretrieval-hybrid-bge-small-prefix-equal",
            history_ids,
        )
        reference = self.client.get_json(
            "/api/v1/runs/preset-duretrieval-hybrid-bge-small-prefix-equal"
        )
        self.assertTrue(reference["is_reference"])

    def test_reference_result_link_in_plan_detail(self) -> None:
        page = self.client.text("/plans/duretrieval-rrf-weight-sweep")
        self.assertIn("查看参照结果", page)
        self.assertIn(
            'href="/results/preset-duretrieval-hybrid-bge-small-prefix-equal"',
            page,
        )
        self.assertEqual(
            self.client.status(
                "/results/preset-duretrieval-hybrid-bge-small-prefix-equal"
            ),
            200,
        )

    def test_rerun_creates_new_run_id(self) -> None:
        def submit_and_wait() -> str:
            status, payload = self.client.post_json(
                "/api/v1/tasks",
                {
                    "plan_id": "scifact-comparison",
                    "plan_version": "v1",
                    "variant_id": "scifact-bm25",
                },
            )
            self.assertEqual(status, 201)
            run_id = payload["run_id"]
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                task = self.client.get_json(
                    f"/api/v1/tasks/{run_id}"
                )["task"]
                if task["status"] in (
                    "Completed",
                    "Failed",
                    "Cancelled",
                    "Interrupted",
                ):
                    self.assertEqual(task["status"], "Completed")
                    return run_id
                time.sleep(0.5)
            self.fail("运行未在超时内结束")

        first = submit_and_wait()
        second = submit_and_wait()
        self.assertNotEqual(first, second)
        for run_id in (first, second):
            record_path = (
                self.release_root
                / "results"
                / "runs"
                / run_id
                / "run.json"
            )
            self.assertTrue(record_path.is_file(), run_id)

    def test_release_sources_keep_local_only_loading(self) -> None:
        source = (
            self.release_root / "src" / "experiment" / "executor.py"
        ).read_text(encoding="utf-8")
        self.assertIn("local_files_only=True", source)


if __name__ == "__main__":
    unittest.main()
