"""主机设置与资源检查 API。"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import ValidationError

from resources.checker import environment_summary
from resources.settings import (
    SettingsInvalidError,
    SettingsReadError,
    SettingsUpdate,
    SettingsWriteError,
)
from web.security import ApiError

router = APIRouter(prefix="/api/v1")


def _store(request: Request):
    return getattr(request.app.state.context, "settings_store", None)


def _checker(request: Request):
    return getattr(request.app.state.context, "resource_checker", None)


def _snapshot(store, checker) -> dict:
    settings = store.load()
    resources = (
        checker.snapshot()
        if checker is not None
        else {
            "datasets": [],
            "models": [],
            "device": {
                "configured": settings.device,
                "effective": settings.device,
                "available": True,
                "summary": "",
            },
        }
    )
    return {
        "settings": settings.model_dump(mode="json"),
        "resources": resources,
        "environment": environment_summary(),
    }


def _read_settings(store):
    try:
        return store.load()
    except SettingsReadError as exc:
        raise ApiError(422, "SETTINGS_INVALID", str(exc)) from None
    except SettingsInvalidError as exc:
        raise ApiError(422, "SETTINGS_INVALID", str(exc)) from None


@router.get("/settings")
def get_settings(request: Request) -> dict:
    store = _store(request)
    if store is None:
        raise ApiError(500, "SETTINGS_WRITE_FAILED", "设置存储不可用")
    _read_settings(store)
    return _snapshot(store, _checker(request))


@router.put("/settings")
async def put_settings(request: Request) -> dict:
    store = _store(request)
    if store is None:
        raise ApiError(500, "SETTINGS_WRITE_FAILED", "设置存储不可用")
    _read_settings(store)

    try:
        raw = await request.json()
    except Exception:
        raise ApiError(422, "SETTINGS_INVALID", "请求体必须是 JSON 对象") from None
    if not isinstance(raw, dict):
        raise ApiError(422, "SETTINGS_INVALID", "请求体必须是 JSON 对象")
    if "network" in raw:
        raise ApiError(
            409,
            "SETTINGS_READ_ONLY",
            "网络字段由服务端保留；局域网请通过 "
            "/api/v1/network/enable 或 /api/v1/network/disable 管理",
        )

    try:
        update = SettingsUpdate.model_validate(raw)
    except ValidationError as exc:
        lines = [
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in exc.errors()
        ]
        raise ApiError(
            422, "SETTINGS_INVALID", "设置字段非法：" + "; ".join(lines)
        ) from None

    try:
        store.save_update(update)
    except SettingsWriteError as exc:
        raise ApiError(500, "SETTINGS_WRITE_FAILED", str(exc)) from None
    except SettingsInvalidError as exc:
        raise ApiError(422, "SETTINGS_INVALID", str(exc)) from None

    return _snapshot(store, _checker(request))
