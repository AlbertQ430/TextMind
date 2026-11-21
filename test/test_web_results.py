"""T-007 结果、历史、比较 API 与页面测试。"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from experiment.plans import load_plan_catalog  # noqa: E402
from experiment.timing import TIMING_FIELDS  # noqa: E402
from records.catalog import ResultCatalog  # noqa: E402
from tasks.models import TaskNotFoundError, TaskSubmitError  # noqa: E402
from web.app import create_app  # noqa: E402
from web.context import AppContext  # noqa: E402
from web.session import SessionStore  # noqa: E402

CATALOG = load_plan_catalog()
REFERENCE_RUN_ID = "preset-duretrieval-hybrid-bge-small-prefix-equal"


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


def make_record(
    run_id: str,
    *,
    variant_id: str = "scifact-bm25",
    plan_id: str = "scifact-comparison",
    plan_version: str = "v1",
    status: str = "Completed",
    metrics: dict | None = None,
    charts: list | None = None,
    started_at: str = "2026-11-21T10:00:00Z",
) -> dict:
    record = {
        "format_version": "1.0",
        "run_id": run_id,
        "plan_id": plan_id,
        "plan_version": plan_version,
        "variant_id": variant_id,
        "plan_name": "SciFact 英文检索方法与模型对比",
        "variant_name": variant_id,
        "status": status,
        "stage_progress": {
            "completed": ["prepare", "load_dataset"],
            "current": None,
            "not_applicable": ["load_model", "encode_queries"],
        },
        "started_at": started_at,
        "finished_at": "2026-11-21T10:00:05Z",
        "dataset": {
            "name": "scifact",
            "split": "test",
            "corpus_size": 3,
            "query_count": 1,
            "qrels_pairs": 1,
        },
        "variant_config": {"method": "bm25", "k1": 1.2},
        "metrics": {"MRR": 0.5} if metrics is None else metrics,
        "num_queries": 1,
        "num_skipped": 0,
        "timing": {field: None for field in TIMING_FIELDS},
        "environment": {"python": "3.11.18"},
        "charts": charts or [],
        "error": None,
    }
    if status != "Completed":
        record["metrics"] = None
        record["num_queries"] = None
        record["num_skipped"] = None
        record["error"] = {"message": "失败", "detail": None, "stage": "retrieve"}
    return record


def write_record(
    runs_root: Path,
    record: dict,
    *,
    chart_bytes: bytes | None = None,
) -> Path:
    directory = runs_root / record["run_id"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run.json").write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf-8"
    )
    if record.get("charts"):
        charts_dir = directory / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        if chart_bytes is not None:
            for chart in record["charts"]:
                (directory / chart["path"]).write_bytes(chart_bytes)
    return directory


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


class ResultWebTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs_root = Path(self._tmp.name) / "runs"
        self.context = AppContext(
            catalog=CATALOG,
            task_service=FakeTaskService(),
            session_store=SessionStore(),
            result_catalog=ResultCatalog(runs_root=self.runs_root, catalog=CATALOG),
            sse_poll_interval=0.02,
            sse_heartbeat_interval=0.05,
        )
        self.app = create_app(self.context)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.token = self.client.get("/api/v1/session").json()["session"][
            "csrf_token"
        ]

    def post_compare(self, run_ids):
        return self.client.post(
            "/api/v1/runs/compare",
            json={"run_ids": run_ids},
            headers={"X-CSRF-Token": self.token},
        )


class HistoryApiTests(ResultWebTestCase):
    def test_empty_history(self) -> None:
        response = self.client.get("/api/v1/runs")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"records": [], "unavailable": []})

    def test_history_records_and_unavailable(self) -> None:
        write_record(self.runs_root, make_record("run-a", metrics={"MRR": 0.5}))
        write_record(
            self.runs_root,
            make_record(
                "run-b", started_at="2026-11-21T11:00:00Z", metrics={"MRR": 0.6}
            ),
        )
        write_record(self.runs_root, make_record("run-bad"))
        (self.runs_root / "run-bad" / "run.json").write_text(
            "{bad", encoding="utf-8"
        )

        payload = self.client.get("/api/v1/runs").json()
        self.assertEqual(
            [record["run_id"] for record in payload["records"]],
            ["run-b", "run-a"],
        )
        self.assertEqual(payload["records"][0]["source"], "runs")
        self.assertTrue(payload["records"][0]["comparison_eligible"])
        self.assertEqual(len(payload["unavailable"]), 1)
        self.assertEqual(
            payload["unavailable"][0]["code"], "RUN_RECORD_INVALID"
        )
        self.assertNotIn(str(self.runs_root), json.dumps(payload))

    def test_reference_hidden_from_history(self) -> None:
        write_record(
            self.runs_root,
            make_record(
                REFERENCE_RUN_ID,
                plan_id="duretrieval-comparison",
                variant_id="duretrieval-hybrid-bge-small-prefix-equal",
            ),
        )
        payload = self.client.get("/api/v1/runs").json()
        self.assertEqual(payload["records"], [])


class RunDetailApiTests(ResultWebTestCase):
    def test_run_view_and_charts(self) -> None:
        write_record(
            self.runs_root,
            make_record(
                "run-a",
                charts=[
                    {"id": "metrics", "title": "指标", "path": "charts/metrics.png"},
                    {"id": "timing", "title": "耗时", "path": "charts/timing.png"},
                ],
            ),
            chart_bytes=b"png",
        )
        payload = self.client.get("/api/v1/runs/run-a").json()
        self.assertEqual(payload["run"]["run_id"], "run-a")
        self.assertEqual(payload["source"], "runs")
        self.assertTrue(payload["known_plan"])
        self.assertTrue(payload["comparison_eligible"])
        self.assertEqual(len(payload["charts"]), 2)
        self.assertTrue(payload["charts"][0]["available"])
        self.assertEqual(
            payload["charts"][0]["url"], "/api/v1/runs/run-a/charts/0"
        )
        self.assertNotIn("source", payload["run"])

    def test_missing_chart_marks_unavailable(self) -> None:
        write_record(
            self.runs_root,
            make_record(
                "run-a",
                charts=[
                    {"id": "metrics", "title": "指标", "path": "charts/metrics.png"}
                ],
            ),
        )
        payload = self.client.get("/api/v1/runs/run-a").json()
        self.assertFalse(payload["charts"][0]["available"])
        self.assertIsNone(payload["charts"][0]["url"])

    def test_run_errors(self) -> None:
        response = self.client.get("/api/v1/runs/missing")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "RUN_NOT_FOUND")

        bad = self.runs_root / "run-bad"
        bad.mkdir(parents=True)
        (bad / "run.json").write_text("{bad", encoding="utf-8")
        response = self.client.get("/api/v1/runs/run-bad")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "RUN_UNAVAILABLE")

    def test_conflict_error(self) -> None:
        record = make_record("run-x")
        write_record(self.runs_root, record)
        imports = Path(self._tmp.name) / "imports"
        write_record(imports, record)
        context = AppContext(
            catalog=CATALOG,
            task_service=FakeTaskService(),
            session_store=SessionStore(),
            result_catalog=ResultCatalog(
                runs_root=self.runs_root,
                imports_root=imports,
                catalog=CATALOG,
            ),
        )
        app = create_app(context)
        with TestClient(app) as client:
            response = client.get("/api/v1/runs/run-x")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "RUN_ID_CONFLICT")


class ChartApiTests(ResultWebTestCase):
    def test_chart_returns_png(self) -> None:
        write_record(
            self.runs_root,
            make_record(
                "run-a",
                charts=[
                    {"id": "metrics", "title": "指标", "path": "charts/metrics.png"}
                ],
            ),
            chart_bytes=b"\x89PNG\r\n\x1a\nfake",
        )
        response = self.client.get("/api/v1/runs/run-a/charts/0")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(response.content[:8], b"\x89PNG\r\n\x1a\n")

    def test_chart_index_out_of_range(self) -> None:
        write_record(
            self.runs_root,
            make_record(
                "run-a",
                charts=[
                    {"id": "metrics", "title": "指标", "path": "charts/metrics.png"}
                ],
            ),
            chart_bytes=b"png",
        )
        response = self.client.get("/api/v1/runs/run-a/charts/9")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "CHART_NOT_FOUND")

    def test_malformed_chart_path_returns_409(self) -> None:
        write_record(
            self.runs_root,
            make_record(
                "run-nul",
                charts=[
                    {
                        "id": "metrics",
                        "title": "指标",
                        "path": "charts/bad\u0000.png",
                    }
                ],
            ),
        )
        view = self.client.get("/api/v1/runs/run-nul")
        self.assertEqual(view.status_code, 200)
        self.assertFalse(view.json()["charts"][0]["available"])
        response = self.client.get("/api/v1/runs/run-nul/charts/0")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "CHART_UNSAFE")

    def test_unreadable_chart_returns_404(self) -> None:
        write_record(
            self.runs_root,
            make_record(
                "run-unreadable",
                charts=[
                    {
                        "id": "metrics",
                        "title": "指标",
                        "path": "charts/metrics.png",
                    }
                ],
            ),
            chart_bytes=b"png",
        )
        with mock.patch("records.catalog._is_readable", return_value=False):
            view = self.client.get("/api/v1/runs/run-unreadable").json()
            self.assertFalse(view["charts"][0]["available"])
            self.assertIsNone(view["charts"][0]["url"])
            page = self.client.get("/results/run-unreadable")
            self.assertIn("图表不可用", page.text)
            response = self.client.get("/api/v1/runs/run-unreadable/charts/0")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            response.json()["error"]["code"], "CHART_UNAVAILABLE"
        )

    def test_chart_symlink_escape(self) -> None:
        outside = Path(self._tmp.name) / "outside.png"
        outside.write_bytes(b"secret")
        directory = write_record(
            self.runs_root,
            make_record(
                "run-a",
                charts=[
                    {"id": "metrics", "title": "指标", "path": "charts/metrics.png"}
                ],
            ),
        )
        (directory / "charts").mkdir(parents=True, exist_ok=True)
        (directory / "charts" / "metrics.png").symlink_to(outside)
        response = self.client.get("/api/v1/runs/run-a/charts/0")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "CHART_UNSAFE")


class CompareApiTests(ResultWebTestCase):
    def setUp(self) -> None:
        super().setUp()
        write_record(
            self.runs_root,
            make_record("run-a", metrics={"P@1": 0.5, "MRR": 0.7}),
        )
        write_record(
            self.runs_root,
            make_record(
                "run-b",
                variant_id="scifact-dense-qwen3-base",
                metrics={"P@5": 0.3, "MRR": 0.6},
                started_at="2026-11-21T11:00:00Z",
            ),
        )

    def test_compare_requires_csrf(self) -> None:
        response = self.client.post(
            "/api/v1/runs/compare", json={"run_ids": ["run-a", "run-b"]}
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"]["code"], "CSRF_INVALID")

    def test_compare_success(self) -> None:
        response = self.post_compare(["run-a", "run-b"])
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["plan_id"], "scifact-comparison")
        self.assertEqual(payload["metric_columns"], ["MRR", "P@1", "P@5"])
        self.assertEqual([row["run_id"] for row in payload["rows"]], ["run-a", "run-b"])
        self.assertTrue(payload["chart_url"].startswith("/api/v1/runs/compare/chart?runs="))

    def test_compare_selection_errors(self) -> None:
        for run_ids in ([], ["run-a"], ["run-a", "run-a"]):
            with self.subTest(run_ids=run_ids):
                response = self.post_compare(run_ids)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(
                    response.json()["error"]["code"], "COMPARE_SELECTION_INVALID"
                )

    def test_compare_missing_run_keeps_404(self) -> None:
        response = self.post_compare(["run-a", "missing"])
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "RUN_NOT_FOUND")

    def test_compare_unavailable_run_is_409(self) -> None:
        bad = self.runs_root / "run-bad"
        bad.mkdir(parents=True)
        (bad / "run.json").write_text("{bad", encoding="utf-8")
        response = self.post_compare(["run-a", "run-bad"])
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "RUN_UNAVAILABLE")

    def test_compare_conflict_run_is_409(self) -> None:
        record = make_record("run-x")
        write_record(self.runs_root, record)
        imports = Path(self._tmp.name) / "imports"
        write_record(imports, record)
        context = AppContext(
            catalog=CATALOG,
            task_service=FakeTaskService(),
            session_store=SessionStore(),
            result_catalog=ResultCatalog(
                runs_root=self.runs_root,
                imports_root=imports,
                catalog=CATALOG,
            ),
        )
        app = create_app(context)
        with TestClient(app) as client:
            token = client.get("/api/v1/session").json()["session"][
                "csrf_token"
            ]
            response = client.post(
                "/api/v1/runs/compare",
                json={"run_ids": ["run-x", "run-a"]},
                headers={"X-CSRF-Token": token},
            )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "RUN_ID_CONFLICT")

    def test_compare_cross_plan_rejected(self) -> None:
        write_record(
            self.runs_root,
            make_record(
                "run-duretrieval",
                plan_id="duretrieval-comparison",
                variant_id="duretrieval-bm25",
            ),
        )
        response = self.post_compare(["run-a", "run-duretrieval"])
        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()["error"]["code"], "COMPARE_SELECTION_INVALID"
        )

    def test_compare_chart_returns_png(self) -> None:
        response = self.client.get(
            "/api/v1/runs/compare/chart?runs=run-a,run-b"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        image = Image.open(io.BytesIO(response.content))
        self.assertGreater(image.width, 0)

    def test_compare_chart_selection_error(self) -> None:
        response = self.client.get("/api/v1/runs/compare/chart?runs=run-a")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json()["error"]["code"], "COMPARE_SELECTION_INVALID"
        )


class ResultPageTests(ResultWebTestCase):
    def setUp(self) -> None:
        super().setUp()
        write_record(
            self.runs_root,
            make_record(
                "run-a",
                charts=[
                    {"id": "metrics", "title": "指标", "path": "charts/metrics.png"}
                ],
            ),
            chart_bytes=b"png",
        )
        write_record(
            self.runs_root,
            make_record(
                "run-b",
                variant_id="scifact-dense-qwen3-base",
                metrics={"P@5": 0.3, "MRR": 0.6},
                started_at="2026-11-21T11:00:00Z",
            ),
        )

    def test_results_redirects_to_history(self) -> None:
        response = self.client.get("/results", follow_redirects=False)
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["location"], "/history")

    def test_result_page_content(self) -> None:
        html = self.client.get("/results/run-a").text
        self.assertIn("SciFact 英文检索方法与模型对比", html)
        self.assertIn("MRR", html)
        self.assertIn("分阶段耗时", html)
        self.assertIn("图表", html)
        self.assertIn("/api/v1/runs/run-a/charts/0", html)
        self.assertIn("配置", html)
        self.assertIn("环境", html)
        self.assertIn("不适用", html)
        self.assertIn("数据集", html)
        self.assertIn("scifact", html)
        self.assertIn("test", html)
        self.assertIn("3", html)
        self.assertIn("python", html)

    def test_result_page_failed_record(self) -> None:
        record = make_record("run-failed", status="Failed", metrics=None)
        record["environment"] = {}
        write_record(self.runs_root, record)
        html = self.client.get("/results/run-failed").text
        self.assertIn("Failed", html)
        self.assertIn("错误详情", html)
        self.assertIn("未记录", html)
        self.assertIn("scifact", html)

    def test_result_page_missing(self) -> None:
        response = self.client.get("/results/missing")
        self.assertEqual(response.status_code, 404)
        self.assertIn("不存在", response.text)

    def test_history_page(self) -> None:
        html = self.client.get("/history").text
        self.assertIn("run-a", html)
        self.assertIn("run-b", html)
        self.assertIn('class="history-select"', html)

    def test_compare_page(self) -> None:
        html = self.client.get(
            "/compare?runs=run-a,run-b", follow_redirects=True
        ).text
        self.assertIn("结果比较", html)
        self.assertIn("不可比较", html)
        self.assertIn("/api/v1/runs/compare/chart?runs=run-a,run-b", html)

    def test_compare_page_error(self) -> None:
        html = self.client.get("/compare").text
        self.assertIn("至少选择两条记录", html)
        self.assertIn("返回历史记录", html)

    def test_compare_page_without_common_metrics(self) -> None:
        write_record(self.runs_root, make_record("run-c", metrics={"P@1": 0.5}))
        write_record(
            self.runs_root,
            make_record(
                "run-d",
                variant_id="scifact-dense-qwen3-base",
                metrics={"MRR": 0.7},
                started_at="2026-11-21T12:00:00Z",
            ),
        )
        html = self.client.get("/compare?runs=run-c,run-d").text
        self.assertIn("没有共同可对齐的指标", html)

    def test_no_delete_route(self) -> None:
        response = self.client.delete(
            "/api/v1/runs/run-a", headers={"X-CSRF-Token": self.token}
        )
        self.assertEqual(response.status_code, 405)


def assert_only_heartbeats(testcase, iterator, count: int = 2) -> None:
    heartbeats = 0
    for _ in range(160):
        line = next(iterator)
        if line.startswith("event: "):
            testcase.fail(f"结果集合未变化却发送了事件: {line}")
        if line.startswith(": ping"):
            heartbeats += 1
            if heartbeats >= count:
                return
    testcase.fail("未收到预期心跳")


class RecordsChangedSseTests(ResultWebTestCase):
    def test_records_changed_on_add_delete_and_corrupt(self) -> None:
        with TestClient(self.app) as client:
            with client.stream("GET", "/api/v1/events") as response:
                iterator = response.iter_lines()
                self.assertEqual(read_event(iterator)[0], "snapshot")

                write_record(self.runs_root, make_record("run-new"))
                event, data = read_event(iterator)
                self.assertEqual(event, "records.changed")
                self.assertTrue(data["changed"])
                assert_only_heartbeats(self, iterator)

                shutil.rmtree(self.runs_root / "run-new")
                event, _ = read_event(iterator)
                self.assertEqual(event, "records.changed")
                assert_only_heartbeats(self, iterator)

                broken = self.runs_root / "run-broken"
                broken.mkdir(parents=True)
                (broken / "run.json").write_text("{bad", encoding="utf-8")
                event, _ = read_event(iterator)
                self.assertEqual(event, "records.changed")


NODE_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const appPath = process.argv[2];
const mode = process.argv[3];
const code = fs.readFileSync(appPath, "utf8");

function makeEl() {
  return {
    dataset: {}, style: {}, children: [], textContent: "", disabled: false, href: "",
    classList: { contains: () => false, add: () => {}, remove: () => {} },
    appendChild: () => {}, addEventListener: () => {}, setAttribute: () => {},
    querySelectorAll: () => [],
  };
}
const elements = {
  "history-body": makeEl(),
  "compare-button": makeEl(),
  "compare-hint": makeEl(),
  "history-empty": makeEl(),
  "unavailable-list": makeEl(),
};
const intervals = [];
const timeouts = [];
class FakeEventSource {
  constructor() { this.listeners = {}; FakeEventSource.instances.push(this); }
  addEventListener(name, cb) { this.listeners[name] = cb; }
  close() {}
}
FakeEventSource.instances = [];
const sandbox = {
  document: {
    querySelector: () => null,
    querySelectorAll: () => [],
    getElementById: (id) => elements[id] || null,
    createElement: () => makeEl(),
  },
  window: {
    EventSource: mode === "no-eventsource" ? undefined : FakeEventSource,
    setInterval: (fn, ms) => { intervals.push(ms); return intervals.length; },
    setTimeout: (fn, ms) => { timeouts.push(fn); return timeouts.length; },
    location: { href: "" },
  },
  fetch: () => Promise.resolve({
    ok: true,
    json: () => Promise.resolve({ records: [], unavailable: [] }),
  }),
  console, Map, Array, Object, JSON, Number, isNaN, Promise,
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(code, sandbox);

if (mode === "no-eventsource") {
  console.log(JSON.stringify({ intervals, timeouts: timeouts.length, sources: 0 }));
} else {
  for (let i = 0; i < 4; i++) {
    const source = FakeEventSource.instances[FakeEventSource.instances.length - 1];
    source.onerror();
    const next = timeouts.shift();
    if (next && i < 3) {
      next();
    }
  }
  console.log(JSON.stringify({
    intervals,
    timeouts: timeouts.length,
    sources: FakeEventSource.instances.length,
  }));
}
"""


class HistorySseFallbackTests(unittest.TestCase):
    APP_JS = PROJECT_ROOT / "src" / "web" / "static" / "app.js"

    def _run(self, mode: str) -> dict:
        node = shutil.which("node")
        if not node:
            self.skipTest("node 不可用，无法执行前端降级脚本验证")
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

    def test_no_eventsource_starts_polling(self) -> None:
        data = self._run("no-eventsource")
        self.assertEqual(data["intervals"], [5000])
        self.assertEqual(data["sources"], 0)

    def test_reconnect_three_times_then_poll(self) -> None:
        data = self._run("eventsource")
        self.assertEqual(data["sources"], 4)
        self.assertEqual(data["intervals"], [5000])


if __name__ == "__main__":
    unittest.main()
