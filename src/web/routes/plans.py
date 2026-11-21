"""方案列表与详情 API。"""

from __future__ import annotations

from fastapi import APIRouter, Request

from web.routes import find_plan, plan_detail, plan_summary
from web.security import ApiError

router = APIRouter(prefix="/api/v1/plans")


@router.get("")
def list_plans(request: Request) -> dict:
    catalog = request.app.state.context.catalog
    return {"plans": [plan_summary(plan) for plan in catalog.plans]}


@router.get("/{plan_id}")
def get_plan(plan_id: str, request: Request) -> dict:
    plan = find_plan(request.app.state.context.catalog, plan_id)
    if plan is None:
        raise ApiError(404, "PLAN_NOT_FOUND", "方案不存在，请返回实验中心重新选择")
    return {"plan": plan_detail(plan)}
