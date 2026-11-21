"""FastAPI 应用工厂。

启动方式：uvicorn web.app:create_app --factory --app-dir src
应用只监听回环地址；LAN、结果详情、导入和设置资源由后续任务实现。
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
from web.routes import pages, plans, tasks
from web.routes import session as session_routes
from web.security import (
    ApiError,
    error_response,
    install_session_middleware,
    same_origin,
)
from web.sse import event_stream

LOGGER = logging.getLogger(__name__)

APP_VERSION = "0.1.0"

# 只监听回环地址；本任务不提供局域网监听。
LOCAL_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

_HTTP_ERROR_CODES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
}


def create_app(context: AppContext | None = None) -> FastAPI:
    """创建应用实例；不启动线程、不访问任务数据库。"""
    ctx = context or build_default_context()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        ctx.start()
        try:
            yield
        finally:
            ctx.stop()

    app = FastAPI(title="TextMind", version=APP_VERSION, lifespan=lifespan)
    app.state.context = ctx
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    install_session_middleware(app, ctx.session_store)

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": APP_VERSION}

    @app.get("/api/v1/events")
    async def events(request: Request):
        # SSE 是只读 GET，但仍需校验显式跨源 Origin；无 Origin 视为本地客户端。
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
                str(part) for part in error.get("loc", []) if part != "body"
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
                500, "INTERNAL_ERROR", "服务内部错误，请查看应用日志", request
            )
        return PlainTextResponse("服务内部错误", status_code=500)

    return app
