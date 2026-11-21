#!/usr/bin/env python3
"""启动 TextMind 本地应用。

启动前依次检查核心依赖、创建运行目录，然后只监听 127.0.0.1。
本任务不打开浏览器、不启用局域网监听、不下载模型或数据集。
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# 让 `paths` 与 scripts/check_env.py 可直接导入。
for extra_path in (SRC_DIR, SCRIPTS_DIR):
    value = str(extra_path)
    if value not in sys.path:
        sys.path.insert(0, value)

import check_env  # noqa: E402  (依赖上面的 sys.path 调整)
from paths import ensure_app_dirs  # noqa: E402

from web.app import DEFAULT_PORT, LOCAL_HOST, APP_VERSION  # noqa: E402


def is_port_available(host: str, port: int) -> bool:
    """在真正启动 Uvicorn 前探测端口是否可以绑定。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="启动 TextMind 本地服务")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"监听端口（默认 {DEFAULT_PORT}）",
    )
    args = parser.parse_args(argv)

    if not 1 <= args.port <= 65535:
        print(f"[错误] 端口超出范围：{args.port}", file=sys.stderr)
        return 1

    print("[信息] 检查核心依赖……")
    if not check_env.run_checks(verbose=True):
        print("\n[错误] 核心依赖检查未通过，已阻止启动。", file=sys.stderr)
        return 1

    try:
        ensure_app_dirs()
    except OSError as exc:
        print(f"[错误] 运行目录创建失败，已阻止启动：{exc}", file=sys.stderr)
        return 1

    if not is_port_available(LOCAL_HOST, args.port):
        print(
            f"[错误] 端口 {LOCAL_HOST}:{args.port} 已被占用，请关闭占用进程或使用 --port 指定其他端口。",
            file=sys.stderr,
        )
        return 1

    print(f"[信息] TextMind {APP_VERSION} 启动中：http://{LOCAL_HOST}:{args.port}/plans")

    import uvicorn  # 延迟导入，环境检查失败时不加载服务器

    try:
        uvicorn.run(
            "web.app:create_app",
            factory=True,
            host=LOCAL_HOST,
            port=args.port,
            app_dir=str(SRC_DIR),
        )
    except SystemExit:
        raise
    except OSError as exc:
        print(f"[错误] 服务启动失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
