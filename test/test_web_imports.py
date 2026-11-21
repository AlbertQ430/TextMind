"""T-008 Web 导入接口与历史页控件测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from experiment.plans import load_plan_catalog  # noqa: E402
from experiment.timing import TIMING_FIELDS  # noqa: E402
from records.catalog import ResultCatalog  # noqa: E402
from records.importer import ImportService  # noqa: E402
from tasks.models import TaskNotFoundError, TaskSubmitError  # noqa: E402
from web.app import create_app  # noqa: E402
from web.context import AppContext  # noqa: E402
from web.session import SessionStore  # noqa: E402

CATALOG = load_plan_catalog()
RUN_ID = "20261121T120000000000Z-web001"


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


def make_record(run_id: str = RUN_ID) -> dict:
    return {
        "format_version": "1.0",
        "run_id": run_id,
        "plan_id": "scifact-comparison",
        "plan_version": "v1",
        "variant_id": "scifact-bm25",
        "plan_name": "SciFact 英文检索方法与模型对比",
        "variant_name": "BM25",
        "status": "Completed",
        "stage_progress": {
            "completed": ["prepare", "load_dataset"],
            "current": None,
            "not_applicable": ["load_model", "encode_queries"],
        },
        "started_at": "2026-11-21T10:00:00Z",
        "finished_at": "2026-11-21T10:00:05Z",
        "dataset": {
            "name": "scifact",
            "split": "test",
            "corpus_size": 3,
            "query_count": 1,
            "qrels_pairs": 1,
        },
        "variant_config": {"method": "bm25"},
        "metrics": {"MRR": 0.5},
        "num_queries": 1,
        "num_skipped": 0,
        "timing": {field: None for field in TIMING_FIELDS},
        "environment": {"python": "3.11.18"},
        "charts": [
            {"id": "metrics", "title": "指标", "path": "charts/metrics.png"}
        ],
        "error": None,
    }


def dump(record: dict) -> bytes:
    return json.dumps(record, ensure_ascii=False).encode("utf-8")


class WebImportTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.runs = root / "runs"
        self.imports = root / "imports"
        result_catalog = ResultCatalog(
            runs_root=self.runs,
            imports_root=self.imports,
            catalog=CATALOG,
        )
        context = AppContext(
            catalog=CATALOG,
            task_service=FakeTaskService(),
            session_store=SessionStore(),
            result_catalog=result_catalog,
            import_service=ImportService(
                self.imports, catalog=result_catalog
            ),
        )
        self.app = create_app(context)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.token = self.client.get("/api/v1/session").json()["session"][
            "csrf_token"
        ]

    def upload(self, content: bytes, *, filename="run.json", token=None):
        return self.client.post(
            "/api/v1/imports",
            files={"file": (filename, content, "application/json")},
            headers={"X-CSRF-Token": token or self.token},
        )

    def assert_error(self, response, status: int, code: str) -> dict:
        self.assertEqual(response.status_code, status)
        payload = response.json()
        self.assertEqual(set(payload), {"error"})
        error = payload["error"]
        self.assertEqual(error["code"], code)
        self.assertTrue(error["message"])
        self.assertTrue(error["request_id"])
        return error


class ImportApiSuccessTests(WebImportTestCase):
    def test_import_success_and_history_visibility(self) -> None:
        content = dump(make_record())
        response = self.upload(content)

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["run_id"], RUN_ID)
        self.assertEqual(payload["source"], "imports")
        self.assertEqual(payload["record_url"], f"/results/{RUN_ID}")
        self.assertEqual(payload["bytes_written"], len(content))
        self.assertEqual(payload["charts_imported"], 0)
        self.assertEqual(len(payload["missing_charts"]), 1)
        self.assertTrue(payload["warnings"])

        stored = (self.imports / RUN_ID / "run.json").read_bytes()
        self.assertEqual(stored, content)

        history = self.client.get("/api/v1/runs").json()
        self.assertEqual(
            [(record["run_id"], record["source"]) for record in history["records"]],
            [(RUN_ID, "imports")],
        )
        page = self.client.get(f"/results/{RUN_ID}")
        self.assertEqual(page.status_code, 200)
        self.assertIn("chart", page.text)

    def test_import_response_has_no_absolute_path(self) -> None:
        response = self.upload(dump(make_record()))
        self.assertNotIn(str(self.imports), response.text)
        self.assertNotIn("source_path", response.text)

    def test_failed_import_record_is_visible(self) -> None:
        record = make_record()
        record["status"] = "Failed"
        record["metrics"] = None
        record["num_queries"] = None
        record["num_skipped"] = None
        record["error"] = {"message": "失败", "detail": None, "stage": "retrieve"}
        response = self.upload(dump(record))
        self.assertEqual(response.status_code, 201)
        history = self.client.get("/api/v1/runs").json()
        self.assertEqual(history["records"][0]["status"], "Failed")


class ImportApiValidationTests(WebImportTestCase):
    def test_missing_file(self) -> None:
        response = self.client.post(
            "/api/v1/imports", headers={"X-CSRF-Token": self.token}
        )
        self.assert_error(response, 422, "IMPORT_FILE_REQUIRED")

    def test_multiple_files_rejected(self) -> None:
        response = self.client.post(
            "/api/v1/imports",
            files=[
                ("file", ("a.json", dump(make_record()), "application/json")),
                ("file", ("b.json", dump(make_record("second")), "application/json")),
            ],
            headers={"X-CSRF-Token": self.token},
        )
        self.assert_error(response, 422, "IMPORT_FILE_REQUIRED")

    def test_wrong_extension_and_empty_file(self) -> None:
        self.assert_error(
            self.upload(dump(make_record()), filename="run.txt"),
            422,
            "IMPORT_FILE_INVALID",
        )
        self.assert_error(self.upload(b""), 422, "IMPORT_FILE_INVALID")

    def test_too_large(self) -> None:
        response = self.upload(b" " * (5 * 1024 * 1024 + 1))
        self.assert_error(response, 413, "IMPORT_TOO_LARGE")

    def test_duplicate_key(self) -> None:
        content = b'{"format_version": "1.0", "run_id": "a", "run_id": "b"}'
        self.assert_error(
            self.upload(content), 422, "IMPORT_DUPLICATE_KEY"
        )

    def test_invalid_json_and_schema(self) -> None:
        self.assert_error(self.upload(b"{bad"), 422, "IMPORT_INVALID_JSON")
        record = make_record()
        record["unknown_field"] = 1
        self.assert_error(
            self.upload(dump(record)), 422, "IMPORT_SCHEMA_INVALID"
        )

    def test_conflict(self) -> None:
        content = dump(make_record())
        self.assertEqual(self.upload(content).status_code, 201)
        self.assert_error(
            self.upload(content), 409, "IMPORT_RUN_ID_CONFLICT"
        )

    def test_control_character_path_rejected(self) -> None:
        record = make_record()
        record["charts"] = [
            {"id": "metrics", "title": "指标", "path": "charts/bad\u0000.png"}
        ]
        self.assert_error(
            self.upload(dump(record)), 422, "IMPORT_PATH_INVALID"
        )
        self.assertFalse((self.imports / RUN_ID).exists())

    def test_exponent_overflow_rejected(self) -> None:
        text = dump(make_record()).decode("utf-8").replace(
            '"environment": {"python": "3.11.18"}',
            '"environment": {"python": 1e999}',
        )
        self.assert_error(
            self.upload(text.encode("utf-8")), 422, "IMPORT_INVALID_JSON"
        )

    def test_deeply_nested_json_rejected(self) -> None:
        content = b"[" * 2000 + b"]" * 2000
        self.assert_error(self.upload(content), 422, "IMPORT_INVALID_JSON")

    def test_concurrent_imports_same_run_id(self) -> None:
        content = dump(make_record())
        barrier = threading.Barrier(2)
        statuses: list[int] = []

        def worker() -> None:
            barrier.wait()
            response = self.upload(content)
            statuses.append(response.status_code)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)

        self.assertCountEqual(statuses, [201, 409])
        self.assertTrue((self.imports / RUN_ID / "run.json").is_file())
        leftovers = [
            path.name
            for path in self.imports.iterdir()
            if path.name.startswith(".tmp-")
        ]
        self.assertEqual(leftovers, [])

    def test_csrf_required(self) -> None:
        response = self.client.post(
            "/api/v1/imports",
            files={"file": ("run.json", dump(make_record()), "application/json")},
        )
        self.assert_error(response, 403, "CSRF_INVALID")
        self.assertFalse((self.imports / RUN_ID).exists())

        response = self.upload(dump(make_record()), token="wrong")
        self.assert_error(response, 403, "CSRF_INVALID")
        self.assertFalse((self.imports / RUN_ID).exists())


class ImportPageTests(WebImportTestCase):
    def test_history_page_has_import_controls_without_delete(self) -> None:
        html = self.client.get("/history").text
        self.assertIn('id="import-form"', html)
        self.assertIn('type="file"', html)
        self.assertIn('accept=".json,application/json"', html)
        self.assertNotIn("导出", html)
        self.assertNotIn("删除记录", html)
        self.assertNotIn("delete", html.lower())

    def test_no_delete_or_export_api(self) -> None:
        response = self.client.delete(
            "/api/v1/imports", headers={"X-CSRF-Token": self.token}
        )
        self.assertEqual(response.status_code, 405)
        self.client.delete(
            f"/api/v1/runs/{RUN_ID}", headers={"X-CSRF-Token": self.token}
        )


if __name__ == "__main__":
    unittest.main()
