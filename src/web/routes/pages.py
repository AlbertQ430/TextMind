"""页面路由：实验中心、方案详情、运行监控、结果、历史与比较。"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from records.catalog import CatalogError
from records.compare import (
    CompareNoMetricsError,
    CompareSelectionError,
    build_selection,
    compare,
)
from web.routes import (
    find_plan,
    plan_detail,
    plan_summary,
    record_summary,
    record_view,
    timing_rows,
)

router = APIRouter()


def _result_catalog(request: Request):
    return getattr(request.app.state.context, "result_catalog", None)


def _render(request: Request, template_name: str, *, status_code: int = 200, **context):
    templates = request.app.state.templates
    context.setdefault("csrf_token", request.state.session.csrf_token)
    return templates.TemplateResponse(
        request, template_name, context, status_code=status_code
    )


def _placeholder(
    request: Request,
    title: str,
    message: str,
    *,
    back: str | None = None,
    back_label: str | None = None,
    active: str = "",
    status_code: int = 200,
):
    return _render(
        request,
        "placeholder.html",
        status_code=status_code,
        title=title,
        message=message,
        back=back,
        back_label=back_label,
        active=active,
    )


@router.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/plans", status_code=307)


@router.get("/plans")
def plans_page(request: Request):
    catalog = request.app.state.context.catalog
    return _render(
        request,
        "plans.html",
        title="实验中心",
        active="plans",
        plans=[plan_summary(plan) for plan in catalog.plans],
    )


@router.get("/plans/{plan_id}")
def plan_page(plan_id: str, request: Request):
    plan = find_plan(request.app.state.context.catalog, plan_id)
    if plan is None:
        return _render(
            request,
            "placeholder.html",
            status_code=404,
            title="方案不存在",
            message=f"未找到方案 {plan_id}，请返回实验中心重新选择。",
            back="/plans",
            back_label="返回实验中心",
            active="plans",
        )
    return _render(
        request,
        "plan_detail.html",
        title=plan.name,
        active="plans",
        plan=plan_detail(plan),
    )


@router.get("/monitor")
def monitor_page(request: Request):
    return _render(request, "monitor.html", title="运行监控", active="monitor")


@router.get("/results")
def results_page() -> RedirectResponse:
    return RedirectResponse(url="/history", status_code=307)


@router.get("/results/{run_id}")
def result_page(run_id: str, request: Request):
    catalog = _result_catalog(request)
    if catalog is None:
        return _placeholder(
            request,
            "实验结果",
            "结果目录不可用，请返回历史记录。",
            back="/history",
            back_label="返回历史记录",
            active="results",
        )
    try:
        entry = catalog.get(run_id)
    except CatalogError as exc:
        return _placeholder(
            request,
            "实验结果",
            exc.message,
            back="/history",
            back_label="返回历史记录",
            active="results",
            status_code=exc.status_code,
        )
    return _render(
        request,
        "results.html",
        title="实验结果",
        active="results",
        record=entry.record,
        view=record_view(entry, catalog),
        source=entry.source,
        timing_rows=timing_rows(entry.record),
    )


@router.get("/history")
def history_page(request: Request):
    catalog = _result_catalog(request)
    if catalog is None:
        return _render(
            request,
            "history.html",
            title="历史记录",
            active="history",
            records=[],
            unavailable=[],
        )
    scan = catalog.scan()
    return _render(
        request,
        "history.html",
        title="历史记录",
        active="history",
        records=[record_summary(entry, catalog) for entry in scan.records],
        unavailable=[
            {
                "run_id": item.run_id,
                "source": item.source,
                "code": item.code,
                "message": item.message,
            }
            for item in scan.unavailable
        ],
    )


@router.get("/compare")
def compare_page(request: Request, runs: str = ""):
    catalog = _result_catalog(request)
    run_ids = [item for item in runs.split(",") if item]
    comparison = None
    error = None
    if catalog is None:
        error = "结果目录不可用，请返回历史记录。"
    else:
        try:
            selection = build_selection(catalog, run_ids)
            comparison = compare(selection)
        except CompareSelectionError as exc:
            error = str(exc)
        except CompareNoMetricsError as exc:
            error = str(exc)
        except CatalogError as exc:
            error = exc.message
    return _render(
        request,
        "compare.html",
        title="结果比较",
        active="history",
        comparison=comparison,
        error=error,
        run_ids=run_ids,
    )


@router.get("/settings")
def settings_page(request: Request):
    return _placeholder(
        request,
        "设置",
        "资源路径、环境检查和局域网开关将在后续任务实现。本页仅提供入口。",
        active="settings",
    )
