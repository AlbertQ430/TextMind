"""本机会话中间件、CSRF 与同源校验。"""

from __future__ import annotations

import uuid
from urllib.parse import urlsplit

from fastapi import Request
from fastapi.responses import JSONResponse

from web.session import COOKIE_NAME, SessionStore

WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
CSRF_HEADER = "x-csrf-token"
CSRF_ERROR_CODE = "CSRF_INVALID"


class ApiError(Exception):
    """业务 API 错误，统一转换为 JSON 错误结构。"""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: list[str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = list(details or [])
        super().__init__(message)


def error_payload(
    code: str,
    message: str,
    request_id: str,
    details: list[str] | None = None,
) -> dict:
    payload: dict = {
        "code": code,
        "message": message,
        "request_id": request_id,
    }
    if details:
        payload["details"] = list(details)
    return {"error": payload}


def error_response(
    status_code: int,
    code: str,
    message: str,
    request: Request,
    details: list[str] | None = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
    return JSONResponse(
        status_code=status_code,
        content=error_payload(code, message, request_id, details),
    )


DEFAULT_PORTS = {"http": 80, "https": 443}


def _effective_port(scheme: str, port: int | None) -> int:
    if port is not None:
        return port
    return DEFAULT_PORTS.get(scheme, 80)


def same_origin(request: Request) -> bool:
    """校验 Origin 与请求的 Scheme、Host 和有效端口一致。

    无 Origin 表示非浏览器客户端（本地命令行、SSE 客户端），视为同源；
    浏览器跨源请求一定会携带 Origin，因此不会被放过。
    非法或越界端口一律视为不同源，不抛出未处理异常。
    """
    origin = request.headers.get("origin")
    if not origin:
        return True
    try:
        parsed = urlsplit(origin)
        origin_port = parsed.port
    except ValueError:
        return False
    if parsed.scheme not in DEFAULT_PORTS or not parsed.hostname:
        return False

    request_url = request.url
    request_scheme = request_url.scheme
    try:
        request_port = request_url.port
    except ValueError:
        return False
    if parsed.scheme != request_scheme:
        return False
    if parsed.hostname.lower() != (request_url.hostname or "").lower():
        return False
    return _effective_port(parsed.scheme, origin_port) == _effective_port(
        request_scheme, request_port
    )


def csrf_valid(request: Request, session) -> bool:
    token = request.headers.get(CSRF_HEADER)
    return SessionStore.csrf_matches(session, token) and same_origin(request)


def install_session_middleware(app, store: SessionStore) -> None:
    """为每个请求建立本机会话并校验修改请求。"""

    @app.middleware("http")
    async def _session_middleware(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex
        session = store.get(request.cookies.get(COOKIE_NAME))
        new_session = session is None
        if new_session:
            session = store.create()
        request.state.session = session
        request.state.new_session = new_session

        if request.method in WRITE_METHODS and not csrf_valid(request, session):
            return error_response(
                403,
                CSRF_ERROR_CODE,
                "请求校验失败，请刷新页面后重试",
                request,
            )

        response = await call_next(request)
        if new_session:
            response.set_cookie(
                key=COOKIE_NAME,
                value=session.session_id,
                httponly=True,
                samesite="strict",
                path="/",
                secure=False,
            )
        response.headers.setdefault("Cache-Control", "no-store")
        return response
