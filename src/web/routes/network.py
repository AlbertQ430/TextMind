"""Local-only 网络管理 API。"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, StrictInt

from runtime.network import NetworkError
from web.security import ApiError

router = APIRouter(prefix="/api/v1")


class EnableNetworkRequest(BaseModel):
    lan_port: StrictInt


def _controller(request: Request):
    return getattr(request.app.state, "network", None)


def _require_controller(request: Request):
    controller = _controller(request)
    if controller is None:
        raise ApiError(
            503, "NETWORK_UNAVAILABLE", "网络控制器未装配，无法管理局域网"
        )
    return controller


@router.get("/network")
def network_status(request: Request) -> dict:
    controller = _require_controller(request)
    return controller.status()


@router.post("/network/enable")
def network_enable(payload: EnableNetworkRequest, request: Request) -> dict:
    controller = _require_controller(request)
    try:
        return controller.enable(payload.lan_port)
    except NetworkError as exc:
        raise ApiError(
            exc.status_code, exc.code, exc.message, exc.details
        ) from None


@router.post("/network/disable")
def network_disable(request: Request) -> dict:
    controller = _require_controller(request)
    try:
        return controller.disable()
    except NetworkError as exc:
        raise ApiError(
            exc.status_code, exc.code, exc.message, exc.details
        ) from None
