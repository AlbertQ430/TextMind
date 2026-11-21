"""页面路由：实验中心、方案详情、运行监控与占位页。"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from web.routes import find_plan, plan_detail, plan_summary

router = APIRouter()


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
):
    return _render(
        request,
        "placeholder.html",
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
def results_page(request: Request):
    return _placeholder(
        request,
        "实验结果",
        "结果详情将在后续任务实现。本页仅提供入口，不展示任何结果数据。",
        active="results",
    )


@router.get("/results/{run_id}")
def result_page(run_id: str, request: Request):
    return _placeholder(
        request,
        "实验结果",
        f"运行 {run_id} 的结果详情将在后续任务实现，当前不展示伪造或占位数据。",
        back="/results",
        back_label="返回实验结果",
        active="results",
    )


@router.get("/history")
def history_page(request: Request):
    return _placeholder(
        request,
        "历史记录",
        "历史记录与同方案比较将在后续任务实现。本页仅提供入口。",
        active="history",
    )


@router.get("/settings")
def settings_page(request: Request):
    return _placeholder(
        request,
        "设置",
        "资源路径、环境检查和局域网开关将在后续任务实现。本页仅提供入口。",
        active="settings",
    )
