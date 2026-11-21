"""会话中间件、CSRF 与同源校验；Local 与 LAN App 共用实现。"""

from __future__ import annotations

import uuid
from urllib.parse import urlsplit

from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse

from web.session import (
    COOKIE_NAME,
    LAN_AUTHENTICATED_TTL_SECONDS,
    LAN_PREAUTH_MAX_SESSIONS,
    LAN_PREAUTH_TTL_SECONDS,
    LAN_SCOPES,
    SESSION_SCOPE_LAN_AUTHENTICATED,
    SESSION_SCOPE_LAN_PREAUTH,
    SESSION_SCOPE_LOCAL,
    SessionStore,
)

WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
CSRF_HEADER = "x-csrf-token"
CSRF_ERROR_CODE = "CSRF_INVALID"

LAN_PUBLIC_PATHS = frozenset({"/lan/login", "/api/v1/health", "/favicon.ico"})
LAN_STATIC_PREFIX = "/static/"


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


def origin_value_matches(request: Request, value: str) -> bool:
    """校验给定的 Origin/Referer 值与请求 Scheme、Host、有效端口一致。"""
    try:
        parsed = urlsplit(value)
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


def same_origin(request: Request) -> bool:
    """校验 Origin 与请求同源；缺失 Origin 视为非浏览器客户端。

    浏览器跨源请求一定携带 Origin，因此不会被放过；非法或越界端口一律
    视为不同源，不抛出未处理异常。
    """
    origin = request.headers.get("origin")
    if not origin:
        return True
    return origin_value_matches(request, origin)


def referer_matches(request: Request) -> bool:
    referer = request.headers.get("referer")
    if not referer:
        return False
    return origin_value_matches(request, referer)


def csrf_valid(request: Request, session) -> bool:
    token = request.headers.get(CSRF_HEADER)
    return SessionStore.csrf_matches(session, token) and same_origin(request)


def lan_origin_ok(request: Request) -> bool:
    """LAN 写请求必须携带显式 Origin，或至少 Referer 同源。"""
    origin = request.headers.get("origin")
    if origin:
        return origin_value_matches(request, origin)
    return referer_matches(request)


def lan_csrf_valid(request: Request, session) -> bool:
    token = request.headers.get(CSRF_HEADER)
    return (
        SessionStore.csrf_matches(session, token)
        and lan_origin_ok(request)
    )


def _set_session_cookie(response, session) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=session.session_id,
        httponly=True,
        samesite="strict",
        path="/",
        secure=False,
    )


def _clear_session_cookie(response) -> None:
    response.delete_cookie(
        key=COOKIE_NAME,
        path="/",
        httponly=True,
        samesite="strict",
    )


def install_session_middleware(
    app,
    store: SessionStore,
    *,
    mode: str = "local",
    network=None,
) -> None:
    """安装会话中间件。

    - local：每个请求保证存在进程级 Local 会话，写方法校验 CSRF 与同源。
    - lan：预认证只能访问登录页；未认证页面重定向登录、API 返回 401；
      已认证会话在写方法上校验 CSRF 与同源，LAN 关闭后立即失效。
    """

    @app.middleware("http")
    async def _session_middleware(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex
        request.state.app_mode = mode
        request.state.is_local = mode == "local"
        path = request.url.path
        session = store.get(request.cookies.get(COOKIE_NAME))

        if mode == "local":
            new_session = session is None or session.scope != SESSION_SCOPE_LOCAL
            if new_session:
                session = store.create(SESSION_SCOPE_LOCAL)
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
                _set_session_cookie(response, session)
            response.headers.setdefault("Cache-Control", "no-store")
            return response

        # ------------------------------------------------------------ LAN
        if session is not None and session.scope not in LAN_SCOPES:
            # Local 会话 Cookie 与 LAN 端口共享名称，跨模式一律视为无效。
            session = None
        is_api = path.startswith("/api/")
        is_public = path in LAN_PUBLIC_PATHS or path.startswith(
            LAN_STATIC_PREFIX
        )

        if path == "/lan/login":
            authenticated = (
                session is not None
                and session.scope == SESSION_SCOPE_LAN_AUTHENTICATED
            )
            if authenticated and request.method == "GET":
                return RedirectResponse("/plans", status_code=303)
            new_session = False
            if session is None or session.scope != SESSION_SCOPE_LAN_PREAUTH:
                if request.method != "GET":
                    return error_response(
                        401,
                        "UNAUTHORIZED",
                        "请先打开登录页获取会话",
                        request,
                    )
                session = store.create_with_scope_limit(
                    SESSION_SCOPE_LAN_PREAUTH,
                    LAN_PREAUTH_MAX_SESSIONS,
                    ttl_seconds=LAN_PREAUTH_TTL_SECONDS,
                )
                new_session = True
            request.state.session = session
            request.state.new_session = new_session
            response = await call_next(request)
            if new_session:
                _set_session_cookie(response, session)
            response.headers.setdefault("Cache-Control", "no-store")
            return response

        if is_public:
            # 健康检查、静态资源不创建会话。
            request.state.session = session
            request.state.new_session = False
            response = await call_next(request)
            response.headers.setdefault("Cache-Control", "no-store")
            return response

        if (
            session is not None
            and session.scope == SESSION_SCOPE_LAN_AUTHENTICATED
        ):
            if network is not None and not network.enabled:
                store.delete(session.session_id)
                if is_api:
                    return error_response(
                        401,
                        "SESSION_EXPIRED",
                        "局域网会话已失效，请重新登录",
                        request,
                    )
                return RedirectResponse("/lan/login", status_code=303)
            if request.method in WRITE_METHODS:
                if not SessionStore.csrf_matches(
                    session, request.headers.get(CSRF_HEADER)
                ):
                    return error_response(
                        403,
                        CSRF_ERROR_CODE,
                        "请求校验失败，请刷新页面后重试",
                        request,
                    )
                if not lan_origin_ok(request):
                    return error_response(
                        403,
                        "ORIGIN_INVALID",
                        "局域网写请求必须来自同源页面",
                        request,
                    )
            request.state.session = session
            request.state.new_session = False
            response = await call_next(request)
            response.headers.setdefault("Cache-Control", "no-store")
            return response

        # 未认证或仅预认证：不创建业务会话。
        request.state.session = session
        request.state.new_session = False
        if is_api:
            return error_response(
                401, "UNAUTHORIZED", "请先使用访问码登录", request
            )
        return RedirectResponse("/lan/login", status_code=303)


def rotate_login_csrf(store: SessionStore, session) -> None:
    """登录失败时轮换 preauth CSRF。"""
    store.rotate_csrf(session.session_id)


LAN_TTL = {
    SESSION_SCOPE_LAN_PREAUTH: LAN_PREAUTH_TTL_SECONDS,
    SESSION_SCOPE_LAN_AUTHENTICATED: LAN_AUTHENTICATED_TTL_SECONDS,
}
