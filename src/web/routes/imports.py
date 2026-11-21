"""单份 JSON 导入 API。"""

from __future__ import annotations

from fastapi import APIRouter, Request

from records.importer import ImportFailure
from web.security import ApiError

router = APIRouter(prefix="/api/v1")

# multipart 单 part 上限放宽到 64 MiB，让导入服务按 5 MiB 契约精确拒绝并返回 413；
# 超过该上限的恶意请求由解析器丢弃文件部分，最终返回缺少文件错误。
_FORM_PART_LIMIT = 64 * 1024 * 1024


@router.post("/imports", status_code=201)
async def import_run(request: Request) -> dict:
    service = getattr(request.app.state.context, "import_service", None)
    if service is None:
        raise ApiError(500, "IMPORT_WRITE_FAILED", "导入服务不可用")

    try:
        form = await request.form(
            max_files=10,
            max_fields=10,
            max_part_size=_FORM_PART_LIMIT,
        )
    except Exception:
        raise ApiError(
            422, "IMPORT_FILE_REQUIRED", "无法解析上传请求，请使用单文件表单"
        ) from None

    uploads = [
        item for item in form.getlist("file") if hasattr(item, "filename")
    ]
    try:
        if len(uploads) != 1:
            raise ApiError(
                422, "IMPORT_FILE_REQUIRED", "必须上传恰好一个 JSON 文件"
            )
        upload = uploads[0]
        try:
            content = await upload.read()
        except Exception:
            raise ApiError(
                422, "IMPORT_FILE_INVALID", "无法读取上传文件"
            ) from None

        try:
            result = service.import_json(upload.filename, content)
        except ImportFailure as exc:
            raise ApiError(
                exc.status_code, exc.code, exc.message, exc.details
            ) from None
    finally:
        for item in uploads:
            try:
                await item.close()
            except Exception:
                pass

    return {
        "run_id": result.run_id,
        "source": "imports",
        "record_url": f"/results/{result.run_id}",
        "bytes_written": result.bytes_written,
        "charts_imported": 0,
        "missing_charts": result.missing_charts,
        "warnings": result.warnings,
    }
