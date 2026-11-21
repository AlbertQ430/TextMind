"""LAN 登录、退出与会话轮换。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from runtime.network import SESSION_FAILURE_LIMIT
from web.security import origin_value_matches, referer_matches
from web.session import (
    COOKIE_NAME,
    LAN_AUTHENTICATED_TTL_SECONDS,
    SESSION_SCOPE_LAN_AUTHENTICATED,
    SESSION_SCOPE_LAN_PREAUTH,
    SessionStore,
)

router = APIRouter()


def _network(request: Request):
    return getattr(request.app.state, "network", None)


def _client_source(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _login_response(
    request: Request,
    session,
    *,
    csrf_token: str | None = None,
    error: str | None = None,
    status_code: int = 200,
):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "request": request,
            "csrf_token": csrf_token or session.csrf_token,
            "error": error,
        },
        status_code=status_code,
    )


def _clear_cookie(response) -> None:
    response.delete_cookie(
        key=COOKIE_NAME, path="/", httponly=True, samesite="strict"
    )


@router.get("/lan/login")
def login_page(request: Request):
    session = request.state.session
    if session.scope == SESSION_SCOPE_LAN_AUTHENTICATED:
        return RedirectResponse("/plans", status_code=303)
    if session.scope != SESSION_SCOPE_LAN_PREAUTH:
        return RedirectResponse("/lan/login", status_code=303)
    return _login_response(request, session)


@router.post("/lan/login")
async def login_submit(request: Request):
    session = request.state.session
    store: SessionStore = request.app.state.context.session_store
    controller = _network(request)
    if session.scope != SESSION_SCOPE_LAN_PREAUTH:
        raise HTTPException(403, "已登录会话不能重复登录")

    source = _client_source(request)
    if controller is not None and controller.source_throttled(source):
        return _login_response(
            request,
            session,
            error="登录尝试次数过多，请 10 分钟后再试",
            status_code=429,
        )

    form = await request.form()
    origin = request.headers.get("origin")
    origin_ok = (
        origin_value_matches(request, origin)
        if origin
        else referer_matches(request)
    )
    csrf_ok = SessionStore.csrf_matches(session, form.get("_csrf"))
    code = form.get("access_code")
    code_ok = (
        controller is not None
        and isinstance(code, str)
        and controller.verify_access_code(code)
    )

    if not (origin_ok and csrf_ok and code_ok):
        if controller is not None:
            session_count, source_throttled = controller.login_failure(
                session.session_id, source
            )
        else:
            session_count, source_throttled = 1, False
        rotated = store.rotate_csrf(session.session_id)
        if session_count >= SESSION_FAILURE_LIMIT or source_throttled:
            store.delete(session.session_id)
            if controller is not None:
                controller.clear_session_failures(session.session_id)
            response = _login_response(
                request,
                rotated or session,
                error="登录失败次数过多，请重新打开登录页",
                status_code=429,
            )
            _clear_cookie(response)
            return response
        return _login_response(
            request,
            rotated or session,
            error="访问码无效，请重试",
            status_code=401,
        )

    store.delete(session.session_id)
    new_session = store.create(
        SESSION_SCOPE_LAN_AUTHENTICATED,
        ttl_seconds=LAN_AUTHENTICATED_TTL_SECONDS,
    )
    if controller is not None:
        controller.clear_source_failures(source)
        controller.clear_session_failures(session.session_id)
    response = RedirectResponse("/plans", status_code=303)
    response.set_cookie(
        key=COOKIE_NAME,
        value=new_session.session_id,
        httponly=True,
        samesite="strict",
        path="/",
        secure=False,
    )
    return response


@router.post("/lan/logout")
def logout(request: Request):
    session = request.state.session
    if session.scope != SESSION_SCOPE_LAN_AUTHENTICATED:
        raise HTTPException(403, "仅局域网认证会话可以退出")
    store: SessionStore = request.app.state.context.session_store
    store.delete(session.session_id)
    controller = _network(request)
    if controller is not None:
        controller.clear_session_failures(session.session_id)
    response = RedirectResponse("/lan/login", status_code=303)
    _clear_cookie(response)
    return response
