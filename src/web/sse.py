"""SSE 任务状态快照与事件格式化。"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from fastapi import Request

from web.context import AppContext
from web.routes import task_snapshot


def format_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def heartbeat_event() -> str:
    return ": ping\n\n"


def _task_payload(context: AppContext) -> dict:
    return task_snapshot(context.task_service.list_tasks())


def _task_key(payload: dict) -> str:
    """任务状态与阶段变化键；排除随墙钟变化的 elapsed_seconds。"""
    return json.dumps(
        [
            {
                key: value
                for key, value in task.items()
                if key != "elapsed_seconds"
            }
            for task in payload["tasks"]
        ],
        sort_keys=True,
        ensure_ascii=False,
    )


def _queue_key(payload: dict) -> str:
    """等待队列成员与顺序（按提交时间）键。"""
    return json.dumps(
        [
            (task["run_id"], task["submitted_at"])
            for task in payload["queue"]
        ],
        ensure_ascii=False,
    )


def _records_key(context: AppContext) -> str:
    catalog = getattr(context, "result_catalog", None)
    if catalog is None:
        return ""
    return catalog.fingerprint()


def _settings_key(context: AppContext) -> str:
    store = getattr(context, "settings_store", None)
    if store is None:
        return ""
    return store.fingerprint()


def build_snapshot(context: AppContext) -> dict:
    payload = _task_payload(context)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    return payload


async def event_stream(
    request: Request,
    context: AppContext,
    *,
    still_valid=None,
    expired_reason: str = "session_expired",
):
    """先发送 snapshot，再分别推送队列/任务变化和心跳。

    `still_valid` 在每轮询时检查；返回 False 时发送 `session.expired`
    并结束流（LAN 会话过期或 LAN 关闭）。
    """
    initial = _task_payload(context)
    snapshot = dict(initial)
    snapshot["generated_at"] = datetime.now(timezone.utc).isoformat()
    yield format_event("snapshot", snapshot)

    last_task_key = _task_key(initial)
    last_queue_key = _queue_key(initial)
    last_records_key = _records_key(context)
    last_settings_key = _settings_key(context)
    loop = asyncio.get_event_loop()
    last_heartbeat = loop.time()
    poll_interval = max(context.sse_poll_interval, 0.01)
    heartbeat_interval = max(context.sse_heartbeat_interval, poll_interval)

    while True:
        await asyncio.sleep(poll_interval)
        if await request.is_disconnected():
            break
        if still_valid is not None and not still_valid():
            yield format_event("session.expired", {"reason": expired_reason})
            break

        payload = _task_payload(context)
        task_key = _task_key(payload)
        queue_key = _queue_key(payload)
        records_key = _records_key(context)
        settings_key = _settings_key(context)
        task_changed = task_key != last_task_key
        queue_changed = queue_key != last_queue_key
        records_changed = records_key != last_records_key
        settings_changed = settings_key != last_settings_key

        if not (task_changed or queue_changed or records_changed or settings_changed):
            if loop.time() - last_heartbeat >= heartbeat_interval:
                yield heartbeat_event()
                last_heartbeat = loop.time()
            continue

        event_payload = dict(payload)
        event_payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        if queue_changed:
            yield format_event("queue.updated", event_payload)
            last_queue_key = queue_key
        if task_changed:
            yield format_event("task.updated", event_payload)
            last_task_key = task_key
        if records_changed:
            yield format_event(
                "records.changed",
                {
                    "changed": True,
                    "generated_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                },
            )
            last_records_key = records_key
        if settings_changed:
            yield format_event(
                "settings.changed",
                {
                    "changed": True,
                    "generated_at": datetime.now(
                        timezone.utc
                    ).isoformat(),
                },
            )
            last_settings_key = settings_key
        last_heartbeat = loop.time()
