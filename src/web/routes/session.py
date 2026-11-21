"""本机会话与 CSRF Token API。"""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/v1")


@router.get("/session")
def read_session(request: Request) -> dict:
    session = request.state.session
    return {
        "session": {
            "session_id": session.session_id,
            "csrf_token": session.csrf_token,
            "scope": session.scope,
            "authenticated": True,
        }
    }
