"""FastAPI 应用工厂。

启动方式：uvicorn web.app:create_app --factory --app-dir src
Local 与 LAN 两个 App 共享同一 AppContext；LAN App 只注册受限路由，
不注册导入、设置和网络管理接口。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from experiment.plans import PlanCatalogError
from paths import STATIC_DIR, TEMPLATES_DIR
from tasks.models import (
    ServiceNotRunningError,
    TaskConflictError,
    TaskNotFoundError,
    TaskStateError,
    TaskSubmitError,
)
from web.context import AppContext, build_default_context
from web.routes import (
    imports,
    lan_auth,
    network as network_routes,
    notices,
    pages,
    plans,
    runs,
    settings as settings_routes,
    tasks,
)
from web.routes import session as session_routes
from web.security import (
    ApiError,
    error_response,
    install_session_middleware,
    same_origin,
)
from web.session import SESSION_SCOPE_LAN_AUTHENTICATED
from web.sse import event_stream

LOGGER = logging.getLogger(__name__)

APP_VERSION = "0.1.1"

LOCAL_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
LAN_HOST = "0.0.0.0"
DEFAULT_LAN_PORT = 8766

_HTTP_ERROR_CODES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
}


def create_app(
    context: AppContext | None = None,
    *,
    mode: str = "local",
    network=None,
) -> FastAPI:
    """创建 Local 或 LAN 应用实例；不启动线程、不访问任务数据库。"""
    if mode not in ("local", "lan"):
        raise ValueError(f"未知应用模式: {mode}")
    ctx = context or build_default_context()
    network_controller = (
        network if network is not None else getattr(ctx, "network_controller", None)
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # 只有 Local App 管理共享 AppContext 的启动与关闭。
        if mode == "local":
            ctx.start()
        try:
            yield
        finally:
            if mode == "local":
                ctx.stop()

    app = FastAPI(title="TextMind", version=APP_VERSION, lifespan=lifespan)
    app.state.context = ctx
    app.state.mode = mode
    app.state.network = network_controller
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    install_session_middleware(
        app, ctx.session_store, mode=mode, network=network_controller
    )

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": APP_VERSION}

    @app.get("/api/v1/events")
    async def events(request: Request):
        session = request.state.session
        if mode == "lan":
            # 局域网 SSE 必须携带显式同源 Origin，且只接受已认证会话。
            if not request.headers.get("origin") or not same_origin(request):
                return error_response(
                    403,
                    "ORIGIN_INVALID",
                    "局域网事件流需要同源 Origin",
                    request,
                )
            if (
                session is None
                or session.scope != SESSION_SCOPE_LAN_AUTHENTICATED
            ):
                return error_response(
                    401, "UNAUTHORIZED", "请先使用访问码登录", request
                )

            def still_valid() -> bool:
                current = ctx.session_store.get(session.session_id)
                return (
                    current is not None
                    and current.scope == SESSION_SCOPE_LAN_AUTHENTICATED
                    and (
                        network_controller is None
                        or network_controller.enabled
                    )
                )

            return StreamingResponse(
                event_stream(
                    request,
                    ctx,
                    still_valid=still_valid,
                    expired_reason="lan_session_invalid",
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-store",
                    "X-Accel-Buffering": "no",
                },
            )

        if not same_origin(request):
            return error_response(
                403, "ORIGIN_INVALID", "跨源请求被拒绝", request
            )
        return StreamingResponse(
            event_stream(request, ctx),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
            },
        )

    app.include_router(session_routes.router)
    app.include_router(plans.router)
    app.include_router(tasks.router)
    app.include_router(runs.router)
    app.include_router(notices.router)
    if mode == "local":
        app.include_router(imports.router)
        app.include_router(settings_routes.router)
        app.include_router(network_routes.router)
    else:
        app.include_router(lan_auth.router)
    app.include_router(pages.router)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return error_response(
            exc.status_code, exc.code, exc.message, request, exc.details
        )

    @app.exception_handler(TaskNotFoundError)
    async def _task_not_found(request: Request, exc: TaskNotFoundError):
        return error_response(
            404, "TASK_NOT_FOUND", "任务不存在或已被清理", request
        )

    @app.exception_handler(TaskConflictError)
    async def _task_conflict(request: Request, exc: TaskConflictError):
        return error_response(409, "TASK_CONFLICT", str(exc), request)

    @app.exception_handler(TaskStateError)
    async def _task_state(request: Request, exc: TaskStateError):
        return error_response(409, "TASK_STATE_CONFLICT", str(exc), request)

    @app.exception_handler(TaskSubmitError)
    async def _task_submit(request: Request, exc: TaskSubmitError):
        return error_response(422, "VARIANT_INVALID", str(exc), request)

    @app.exception_handler(ServiceNotRunningError)
    async def _service_not_running(
        request: Request, exc: ServiceNotRunningError
    ):
        return error_response(409, "SERVICE_NOT_RUNNING", str(exc), request)

    @app.exception_handler(PlanCatalogError)
    async def _plan_catalog(request: Request, exc: PlanCatalogError):
        return error_response(422, "PLAN_CATALOG_INVALID", str(exc), request)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        details = []
        for error in exc.errors():
            location = ".".join(
                str(part)
                for part in error.get("loc", [])
                if part != "body"
            )
            details.append(f"{location or 'body'}: {error.get('msg', '')}")
        return error_response(
            422, "VALIDATION_ERROR", "请求字段校验失败", request, details
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_exception(request: Request, exc: StarletteHTTPException):
        if request.url.path.startswith("/api/"):
            code = _HTTP_ERROR_CODES.get(exc.status_code, "HTTP_ERROR")
            return error_response(
                exc.status_code, code, str(exc.detail), request
            )
        return PlainTextResponse(str(exc.detail), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        LOGGER.exception("未处理的服务错误", exc_info=exc)
        if request.url.path.startswith("/api/"):
            return error_response(
                500,
                "INTERNAL_ERROR",
                "服务内部错误，请查看应用日志",
                request,
            )
        return PlainTextResponse("服务内部错误", status_code=500)

    return app
