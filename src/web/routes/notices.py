"""队列取消提示 API。"""

from __future__ import annotations

from fastapi import APIRouter, Request

from tasks.models import (
    NOTICE_QUEUE_CANCELLED_ON_RESTART,
    NOTICE_QUEUE_CANCELLED_ON_SHUTDOWN,
)
from web.security import ApiError

router = APIRouter(prefix="/api/v1")

NOTICE_MESSAGES: dict[str, str] = {
    NOTICE_QUEUE_CANCELLED_ON_RESTART: "服务重启时取消了 {count} 个等待任务",
    NOTICE_QUEUE_CANCELLED_ON_SHUTDOWN: "服务关闭时取消了 {count} 个等待任务",
}


def _iso(moment) -> str:
    from datetime import timezone

    return moment.astimezone(timezone.utc).isoformat()


@router.get("/notices")
def list_notices(request: Request) -> dict:
    service = request.app.state.context.task_service
    notices = []
    for notice in service.list_notices():
        template = NOTICE_MESSAGES.get(notice.kind, "有 {count} 条系统提示")
        notices.append(
            {
                "id": notice.id,
                "kind": notice.kind,
                "count": notice.count,
                "message": template.format(count=notice.count),
                "created_at": _iso(notice.created_at),
            }
        )
    return {"notices": notices}


@router.post("/notices/{notice_id}/ack")
def ack_notice(notice_id: int, request: Request) -> dict:
    service = request.app.state.context.task_service
    if not service.ack_notice(notice_id):
        raise ApiError(404, "NOTICE_NOT_FOUND", "提示不存在或已确认")
    return {"acknowledged": True}
