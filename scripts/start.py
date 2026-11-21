#!/usr/bin/env python3
"""启动 TextMind 本地应用。

启动前依次检查核心依赖、创建运行目录，然后监听 127.0.0.1。
若设置中 lan_enabled=true，则用同一 AppContext 自动恢复局域网监听并生成
新的访问码；LAN 启动失败只警告，不阻止 Local App。
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
from tasks.locking import InstanceLockError  # noqa: E402

from web.app import DEFAULT_PORT, LOCAL_HOST, APP_VERSION, create_app  # noqa: E402


def is_port_available(host: str, port: int) -> bool:
    """在真正启动 Uvicorn 前探测端口是否可以绑定。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _restore_lan(context, controller) -> None:
    from resources.settings import SettingsError
    from runtime.network import NetworkError

    try:
        settings = context.settings_store.load()
    except SettingsError as exc:
        print(
            f"[警告] 无法读取设置文件，局域网不会自动恢复：{exc}",
            file=sys.stderr,
        )
        return
    if not settings.network.lan_enabled:
        return
    try:
        status = controller.enable(settings.network.lan_port)
    except NetworkError as exc:
        print(
            f"[警告] 局域网自动启动失败，Local App 继续运行：{exc}",
            file=sys.stderr,
        )
        return
    addresses = "、".join(status["addresses"]) or "未检测到非回环 IPv4 地址"
    print(f"[信息] 已恢复局域网访问：{addresses}")
    print("[信息] 局域网访问码请在 Local 设置页的“局域网访问”中查看。")


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

    from runtime.network import NetworkController  # noqa: E402
    from web.context import build_default_context  # noqa: E402

    context = build_default_context()
    controller = NetworkController(
        app_factory=lambda **kwargs: create_app(context, **kwargs),
        settings_store=context.settings_store,
        session_store=context.session_store,
        local_port=args.port,
    )
    # 先启动共享 AppContext/TaskService，再恢复 LAN listener。
    try:
        context.start()
    except InstanceLockError:
        print(
            "[错误] 已有 TextMind 实例正在使用当前数据目录，请关闭后重试。",
            file=sys.stderr,
        )
        controller.shutdown()
        return 1
    try:
        _restore_lan(context, controller)
        app = create_app(context, mode="local", network=controller)

        import uvicorn  # 延迟导入，环境检查失败时不加载服务器

        uvicorn.run(app, host=LOCAL_HOST, port=args.port, log_level="info")
    except SystemExit:
        raise
    except OSError as exc:
        print(f"[错误] 服务启动失败：{exc}", file=sys.stderr)
        return 1
    finally:
        controller.shutdown()
        context.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
