"""FastAPI 应用工厂。

启动方式：uvicorn web.app:create_app --factory --app-dir src
应用只提供服务端骨架：本任务仅注册健康检查，不启动后台任务线程。
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from paths import ensure_app_dirs

APP_VERSION = "0.1.0"

# 只监听回环地址；本任务不提供局域网监听。
LOCAL_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """启动时幂等创建运行目录；创建失败直接抛出，阻止服务启动。"""
    ensure_app_dirs()
    yield


def create_app() -> FastAPI:
    """创建应用实例。

    健康检查不读取模型、数据集或环境变量，也不返回主机信息。
    """
    app = FastAPI(
        title="TextMind",
        version=APP_VERSION,
        lifespan=_lifespan,
    )

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": APP_VERSION}

    return app
