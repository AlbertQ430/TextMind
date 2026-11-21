"""运行结果、历史与比较 API。"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import Response
from pydantic import BaseModel

from records.catalog import CatalogError
from records.compare import (
    CompareNoMetricsError,
    CompareSelectionError,
    build_compare_chart,
    build_selection,
    compare,
)
from web.routes import record_summary, record_view
from web.security import ApiError

router = APIRouter(prefix="/api/v1/runs")


def _result_catalog(request: Request):
    return getattr(request.app.state.context, "result_catalog", None)


def _as_api_error(exc: CatalogError) -> ApiError:
    return ApiError(exc.status_code, exc.code, exc.message)


class CompareRequest(BaseModel):
    run_ids: list[str]


@router.get("")
def list_runs(request: Request) -> dict:
    catalog = _result_catalog(request)
    if catalog is None:
        return {"records": [], "unavailable": []}
    scan = catalog.scan()
    return {
        "records": [record_summary(entry, catalog) for entry in scan.records],
        "unavailable": [
            {
                "run_id": item.run_id,
                "source": item.source,
                "code": item.code,
                "message": item.message,
            }
            for item in scan.unavailable
        ],
    }


@router.post("/compare")
def compare_runs(payload: CompareRequest, request: Request) -> dict:
    catalog = _result_catalog(request)
    if catalog is None:
        raise ApiError(422, "COMPARE_SELECTION_INVALID", "结果目录不可用")
    try:
        selection = build_selection(catalog, payload.run_ids)
        return compare(selection)
    except CompareSelectionError as exc:
        raise ApiError(422, exc.code, str(exc)) from None
    except CompareNoMetricsError as exc:
        raise ApiError(422, exc.code, str(exc)) from None
    except CatalogError as exc:
        raise _as_api_error(exc) from None


@router.get("/compare/chart")
def compare_chart(request: Request, runs: str = "") -> Response:
    catalog = _result_catalog(request)
    if catalog is None:
        raise ApiError(422, "COMPARE_SELECTION_INVALID", "结果目录不可用")
    try:
        selection = build_selection(
            catalog, [item for item in runs.split(",") if item]
        )
        data = compare(selection)
        png = build_compare_chart(selection, data)
    except CompareSelectionError as exc:
        raise ApiError(422, exc.code, str(exc)) from None
    except CompareNoMetricsError as exc:
        raise ApiError(422, exc.code, str(exc)) from None
    except CatalogError as exc:
        raise _as_api_error(exc) from None
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/{run_id}")
def get_run(run_id: str, request: Request) -> dict:
    catalog = _result_catalog(request)
    if catalog is None:
        raise ApiError(404, "RUN_NOT_FOUND", "运行记录不存在")
    try:
        entry = catalog.get(run_id)
    except CatalogError as exc:
        raise _as_api_error(exc) from None
    return record_view(entry, catalog)


@router.get("/{run_id}/charts/{chart_index}")
def get_run_chart(
    run_id: str, chart_index: int, request: Request
) -> Response:
    catalog = _result_catalog(request)
    if catalog is None:
        raise ApiError(404, "RUN_NOT_FOUND", "运行记录不存在")
    try:
        entry = catalog.get(run_id)
    except CatalogError as exc:
        raise _as_api_error(exc) from None
    try:
        path = catalog.chart_file(entry, chart_index)
    except CatalogError as exc:
        raise _as_api_error(exc) from None
    try:
        content = path.read_bytes()
    except OSError:
        raise ApiError(
            404, "CHART_UNAVAILABLE", "图表文件不可读"
        ) from None
    return Response(
        content=content,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )
