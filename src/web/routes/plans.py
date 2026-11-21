"""方案列表与详情 API。"""

from __future__ import annotations

from fastapi import APIRouter, Request

from resources.settings import SettingsInvalidError, SettingsReadError
from web.routes import find_plan, plan_detail, plan_summary
from web.security import ApiError


def _settings_api_error(exc: Exception) -> ApiError:
    return ApiError(422, "SETTINGS_INVALID", str(exc))

router = APIRouter(prefix="/api/v1/plans")


@router.get("")
def list_plans(request: Request) -> dict:
    catalog = request.app.state.context.catalog
    checker = getattr(request.app.state.context, "resource_checker", None)
    try:
        return {
            "plans": [plan_summary(plan, checker) for plan in catalog.plans]
        }
    except (SettingsReadError, SettingsInvalidError) as exc:
        raise _settings_api_error(exc) from None


@router.get("/{plan_id}")
def get_plan(plan_id: str, request: Request) -> dict:
    plan = find_plan(request.app.state.context.catalog, plan_id)
    if plan is None:
        raise ApiError(404, "PLAN_NOT_FOUND", "方案不存在，请返回实验中心重新选择")
    checker = getattr(request.app.state.context, "resource_checker", None)
    try:
        return {"plan": plan_detail(plan, checker)}
    except (SettingsReadError, SettingsInvalidError) as exc:
        raise _settings_api_error(exc) from None
