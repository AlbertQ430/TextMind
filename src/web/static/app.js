(function () {
  "use strict";

  var csrfMeta = document.querySelector('meta[name="csrf-token"]');
  var csrfToken = csrfMeta ? csrfMeta.content : "";
  var STAGES = [
    "prepare",
    "load_dataset",
    "load_model",
    "build_index",
    "encode_queries",
    "retrieve",
    "evaluate"
  ];
  var STAGE_LABELS = {
    prepare: "准备",
    load_dataset: "加载数据集",
    load_model: "加载模型",
    build_index: "构建索引",
    encode_queries: "编码查询",
    retrieve: "检索",
    evaluate: "评价"
  };
  var STATUS_LABELS = {
    Pending: "等待中",
    Preparing: "准备中",
    Running: "运行中",
    Cancelling: "取消中",
    Completed: "已完成",
    Failed: "失败",
    Cancelled: "已取消",
    Interrupted: "已中断"
  };

  function showNotice(message, kind) {
    var el = document.getElementById("notice");
    if (!el) {
      return;
    }
    el.textContent = message;
    el.className = "notice " + (kind || "");
    el.hidden = false;
    window.clearTimeout(el._hideTimer);
    el._hideTimer = window.setTimeout(function () {
      el.hidden = true;
    }, 6000);
  }

  async function postJson(url, body) {
    var response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRF-Token": csrfToken
      },
      body: JSON.stringify(body),
      credentials: "same-origin"
    });
    var data = await response.json().catch(function () {
      return {};
    });
    if (!response.ok) {
      var message =
        data && data.error && data.error.message
          ? data.error.message
          : "请求失败，请稍后重试";
      throw new Error(message);
    }
    return data;
  }

  document.querySelectorAll("[data-run-variant]").forEach(function (button) {
    button.addEventListener("click", async function () {
      if (button.dataset.busy) {
        return;
      }
      button.dataset.busy = "1";
      button.disabled = true;
      try {
        await postJson("/api/v1/tasks", {
          plan_id: button.dataset.runPlan,
          plan_version: button.dataset.runVersion,
          variant_id: button.dataset.runVariant
        });
        showNotice("已提交运行：" + button.dataset.runVariant + "，可在运行监控查看", "ok");
      } catch (error) {
        showNotice(error.message, "error");
      } finally {
        button.disabled = false;
        delete button.dataset.busy;
      }
    });
  });

  function formatTime(value) {
    if (!value) {
      return "—";
    }
    var date = new Date(value);
    if (isNaN(date.getTime())) {
      return value;
    }
    return date.toLocaleString("zh-CN", { hour12: false });
  }

  function formatElapsed(startedAt, endedAt) {
    if (!startedAt) {
      return "未开始";
    }
    var start = new Date(startedAt).getTime();
    if (isNaN(start)) {
      return "—";
    }
    var end = endedAt ? new Date(endedAt).getTime() : Date.now();
    if (isNaN(end)) {
      end = Date.now();
    }
    return Math.max(0, (end - start) / 1000).toFixed(1) + " 秒";
  }

  function createElapsed(task) {
    var elapsed = document.createElement("span");
    elapsed.className = "elapsed";
    elapsed.dataset.started = task.started_at || "";
    elapsed.dataset.ended = task.ended_at || "";
    elapsed.textContent =
      "已运行：" + formatElapsed(task.started_at, task.ended_at);
    return elapsed;
  }

  function tickElapsed() {
    document.querySelectorAll(".elapsed").forEach(function (el) {
      el.textContent =
        "已运行：" + formatElapsed(el.dataset.started, el.dataset.ended);
    });
  }

  function renderStages(task) {
    var wrap = document.createElement("div");
    wrap.className = "stage-list";
    var done = task.completed_stages || [];
    var notApplicable = task.not_applicable_stages || [];
    STAGES.forEach(function (stage) {
      var el = document.createElement("span");
      el.className = "stage";
      el.textContent = STAGE_LABELS[stage] || stage;
      if (notApplicable.indexOf(stage) >= 0) {
        el.classList.add("na");
        el.textContent += "（不适用）";
      } else if (task.current_stage === stage) {
        el.classList.add("current");
      } else if (done.indexOf(stage) >= 0) {
        el.classList.add("done");
      }
      wrap.appendChild(el);
    });
    return wrap;
  }

  async function cancelTask(runId, button) {
    button.disabled = true;
    try {
      var data = await postJson(
        "/api/v1/tasks/" + encodeURIComponent(runId) + "/cancel",
        {}
      );
      showNotice("取消请求已记录，当前阶段结束后停止；状态：" + data.status, "ok");
      button.textContent = "无法立即取消";
    } catch (error) {
      showNotice(error.message, "error");
      button.disabled = false;
    }
  }

  function renderTaskCard(task) {
    var card = document.createElement("div");
    card.className = "task-card";

    var title = document.createElement("div");
    var name = document.createElement("strong");
    name.textContent = task.variant_name;
    title.appendChild(name);
    title.appendChild(document.createTextNode(" "));
    var status = document.createElement("span");
    status.className = "status " + task.status;
    status.textContent = STATUS_LABELS[task.status] || task.status;
    title.appendChild(status);
    card.appendChild(title);

    var row = document.createElement("div");
    row.className = "row";
    [
      ["方案", task.plan_name],
      ["运行 ID", task.run_id],
      ["提交时间", formatTime(task.submitted_at)]
    ].forEach(function (pair) {
      var span = document.createElement("span");
      span.textContent = pair[0] + "：" + pair[1];
      row.appendChild(span);
    });
    row.appendChild(createElapsed(task));
    card.appendChild(row);
    card.appendChild(renderStages(task));

    if (task.error_summary) {
      var error = document.createElement("div");
      error.className = "muted";
      error.textContent = "错误：" + task.error_summary;
      card.appendChild(error);
    }

    if (task.status === "Pending" || task.status === "Preparing" || task.status === "Running") {
      var cancel = document.createElement("button");
      cancel.className = "danger";
      cancel.textContent = "取消";
      cancel.addEventListener("click", function () {
        cancelTask(task.run_id, cancel);
      });
      card.appendChild(cancel);
    }
    return card;
  }

  function clear(el) {
    while (el.firstChild) {
      el.removeChild(el.firstChild);
    }
  }

  function renderMonitor(snapshot) {
    var current = document.getElementById("current-task");
    var queue = document.getElementById("queue");
    var list = document.getElementById("task-list");

    clear(current);
    if (snapshot.current) {
      current.classList.remove("empty");
      current.appendChild(renderTaskCard(snapshot.current));
    } else {
      current.classList.add("empty");
      current.textContent = "暂无运行中的任务";
    }

    clear(queue);
    if (snapshot.queue && snapshot.queue.length) {
      queue.classList.remove("empty");
      snapshot.queue.forEach(function (task) {
        queue.appendChild(renderTaskCard(task));
      });
    } else {
      queue.classList.add("empty");
      queue.textContent = "队列为空";
    }

    clear(list);
    var tasks = (snapshot.tasks || []).slice().sort(function (a, b) {
      return String(b.submitted_at).localeCompare(String(a.submitted_at));
    });
    if (!tasks.length) {
      list.classList.add("empty");
      list.textContent = "暂无任务";
      return;
    }
    list.classList.remove("empty");
    tasks.forEach(function (task) {
      list.appendChild(renderTaskCard(task));
    });
  }

  function startMonitor() {
    var failures = 0;
    var polling = false;

    function applySnapshot(snapshot) {
      renderMonitor(snapshot);
    }

    function poll() {
      fetch("/api/v1/tasks", { credentials: "same-origin" })
        .then(function (response) {
          if (!response.ok) {
            throw new Error("任务状态读取失败");
          }
          return response.json();
        })
        .then(applySnapshot)
        .catch(function () {
          showNotice("无法读取任务状态，正在重试", "error");
        });
      window.setTimeout(poll, 2000);
    }

    function connect() {
      if (!window.EventSource) {
        polling = true;
        poll();
        return;
      }
      var source = new window.EventSource("/api/v1/events");
      source.addEventListener("snapshot", function (event) {
        applySnapshot(JSON.parse(event.data));
        failures = 0;
      });
      source.addEventListener("task.updated", function (event) {
        applySnapshot(JSON.parse(event.data));
        failures = 0;
      });
      source.addEventListener("queue.updated", function (event) {
        applySnapshot(JSON.parse(event.data));
        failures = 0;
      });
      source.onerror = function () {
        source.close();
        failures += 1;
        if (failures >= 3) {
          if (!polling) {
            polling = true;
            poll();
            showNotice("实时连接不可用，已切换为轮询", "error");
          }
          return;
        }
        window.setTimeout(connect, 2000);
      };
    }

    connect();
  }

  if (document.getElementById("task-list")) {
    startMonitor();
    window.setInterval(tickElapsed, 1000);
  }

  // ---------------------------------------------------------------- 历史与比较
  function initHistory() {
    var body = document.getElementById("history-body");
    if (!body) {
      return;
    }
    var selected = new Map();
    var compareButton = document.getElementById("compare-button");
    var compareHint = document.getElementById("compare-hint");

    function metricSummary(metrics) {
      if (!metrics) {
        return "—";
      }
      return Object.keys(metrics)
        .slice(0, 2)
        .map(function (name) {
          return name + "=" + Number(metrics[name]).toFixed(3);
        })
        .join(" ");
    }

    function cell(text, className) {
      var td = document.createElement("td");
      if (className) {
        td.className = className;
      }
      td.textContent = text;
      return td;
    }

    function renderRows(data) {
      clear(body);
      (data.records || []).forEach(function (record) {
        var tr = document.createElement("tr");

        var selectCell = document.createElement("td");
        if (record.comparison_eligible) {
          var checkbox = document.createElement("input");
          checkbox.type = "checkbox";
          checkbox.className = "history-select";
          checkbox.dataset.runId = record.run_id;
          checkbox.dataset.planId = record.plan_id;
          checkbox.dataset.planVersion = record.plan_version;
          selectCell.appendChild(checkbox);
        }
        tr.appendChild(selectCell);

        var planCell = document.createElement("td");
        var link = document.createElement("a");
        link.href = "/results/" + encodeURIComponent(record.run_id);
        link.textContent = record.plan_name;
        planCell.appendChild(link);
        tr.appendChild(planCell);
        tr.appendChild(cell(record.variant_name));
        tr.appendChild(cell(record.plan_version));

        var sourceCell = document.createElement("td");
        var tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = record.source;
        sourceCell.appendChild(tag);
        tr.appendChild(sourceCell);

        var statusCell = document.createElement("td");
        var status = document.createElement("span");
        status.className = "status " + record.status;
        status.textContent = record.status;
        statusCell.appendChild(status);
        tr.appendChild(statusCell);

        tr.appendChild(cell(formatTime(record.started_at)));
        tr.appendChild(cell(metricSummary(record.metrics), "metric-cell"));
        tr.appendChild(cell(record.run_id, "mono"));
        body.appendChild(tr);
      });

      var empty = document.getElementById("history-empty");
      if (empty) {
        empty.hidden = (data.records || []).length > 0;
      }
      renderUnavailable(data.unavailable || []);
      selected.clear();
      updateCompareButton();
    }

    function renderUnavailable(items) {
      var container = document.getElementById("unavailable-list");
      if (!container) {
        return;
      }
      clear(container);
      if (!items.length) {
        var empty = document.createElement("p");
        empty.className = "muted";
        empty.textContent = "无不可用记录。";
        container.appendChild(empty);
        return;
      }
      var list = document.createElement("ul");
      list.className = "unavailable";
      items.forEach(function (item) {
        var li = document.createElement("li");
        var tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = item.source;
        var runId = document.createElement("span");
        runId.className = "mono";
        runId.textContent = item.run_id;
        li.appendChild(tag);
        li.appendChild(document.createTextNode(" "));
        li.appendChild(runId);
        li.appendChild(
          document.createTextNode(" — " + item.code + "：" + item.message)
        );
        list.appendChild(li);
      });
      container.appendChild(list);
    }

    function updateCompareButton() {
      if (!compareButton) {
        return;
      }
      compareButton.disabled = selected.size < 2;
      compareButton.textContent = "比较所选（" + selected.size + "）";
      if (compareHint) {
        compareHint.textContent =
          selected.size >= 2
            ? "仅比较同一方案版本内已完成的记录"
            : "勾选同一方案版本下至少两条已完成的记录";
      }
    }

    body.addEventListener("change", function (event) {
      var checkbox = event.target;
      if (!checkbox.classList || !checkbox.classList.contains("history-select")) {
        return;
      }
      var runId = checkbox.dataset.runId;
      var planKey = checkbox.dataset.planId + "|" + checkbox.dataset.planVersion;
      if (checkbox.checked) {
        var currentPlan = null;
        selected.forEach(function (value) {
          currentPlan = value;
        });
        if (currentPlan && currentPlan !== planKey) {
          checkbox.checked = false;
          showNotice("只能选择同一方案和版本的记录", "error");
          return;
        }
        selected.set(runId, planKey);
      } else {
        selected.delete(runId);
      }
      updateCompareButton();
    });

    if (compareButton) {
      compareButton.addEventListener("click", function () {
        if (selected.size < 2) {
          return;
        }
        var ids = Array.from(selected.keys()).map(encodeURIComponent);
        window.location.href = "/compare?runs=" + ids.join(",");
      });
    }

    var importForm = document.getElementById("import-form");
    if (importForm) {
      importForm.addEventListener("submit", async function (event) {
        event.preventDefault();
        var fileInput = document.getElementById("import-file");
        var button = document.getElementById("import-button");
        var result = document.getElementById("import-result");
        if (!fileInput || !fileInput.files || !fileInput.files.length) {
          showNotice("请选择要导入的 JSON 文件", "error");
          return;
        }
        if (button.dataset.busy) {
          return;
        }
        button.dataset.busy = "1";
        button.disabled = true;
        try {
          var formData = new FormData();
          formData.append("file", fileInput.files[0]);
          var response = await fetch("/api/v1/imports", {
            method: "POST",
            headers: { "X-CSRF-Token": csrfToken },
            body: formData,
            credentials: "same-origin",
          });
          var data = await response.json().catch(function () {
            return {};
          });
          if (!response.ok) {
            throw new Error(
              data && data.error && data.error.message
                ? data.error.message
                : "导入失败，请检查文件后重试"
            );
          }
          if (result) {
            clear(result);
            result.hidden = false;
            result.className = "notice ok";
            var lines = [
              "导入成功：Run ID " + data.run_id,
              "已写入 " +
                data.bytes_written +
                " 字节；缺失图表 " +
                (data.missing_charts || []).length +
                " 个。"
            ].concat(
              (data.warnings || []).map(function (item) {
                return "提示：" + item;
              })
            );
            lines.forEach(function (line) {
              var paragraph = document.createElement("p");
              paragraph.textContent = line;
              result.appendChild(paragraph);
            });
          }
          fileInput.value = "";
          reload();
        } catch (error) {
          showNotice(error.message, "error");
        } finally {
          button.disabled = false;
          delete button.dataset.busy;
        }
      });
    }

    function reload() {
      fetch("/api/v1/runs", { credentials: "same-origin" })
        .then(function (response) {
          if (!response.ok) {
            throw new Error("历史记录读取失败");
          }
          return response.json();
        })
        .then(renderRows)
        .catch(function () {
          showNotice("历史记录刷新失败，请稍后重试", "error");
        });
    }

    var pollingStarted = false;

    function startPolling() {
      if (pollingStarted) {
        return;
      }
      pollingStarted = true;
      reload();
      window.setInterval(reload, 5000);
    }

    if (window.EventSource) {
      var failures = 0;
      function connect() {
        var source = new window.EventSource("/api/v1/events");
        var failed = false;
        source.addEventListener("records.changed", function () {
          failures = 0;
          reload();
        });
        source.onerror = function () {
          if (failed) {
            return;
          }
          failed = true;
          source.close();
          failures += 1;
          if (failures > 3) {
            startPolling();
          } else {
            window.setTimeout(connect, 2000);
          }
        };
      }
      connect();
    } else {
      startPolling();
    }

    updateCompareButton();
  }

  if (document.getElementById("history-body")) {
    initHistory();
  }
})();
