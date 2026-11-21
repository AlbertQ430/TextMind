"""LAN 监听生命周期、访问码与登录失败限流。

只在 Local App 主动开启后启动第二个 Uvicorn 服务器线程；两个 App 共享
同一 SessionStore、TaskService 和结果目录。访问码只保存在内存中。
"""

from __future__ import annotations

import hmac
import secrets
import socket
import threading
import time
from typing import Callable

from resources.settings import (
    NetworkSettings,
    Settings,
    SettingsInvalidError,
    SettingsReadError,
    SettingsStore,
    SettingsWriteError,
)
from web.session import LAN_SCOPES, LAN_PREAUTH_TTL_SECONDS

LAN_HOST = "0.0.0.0"
LOCAL_PORT_DEFAULT = 8765
DEFAULT_LAN_PORT = 8766

TLS_WARNING = (
    "局域网使用 HTTP 明文传输，访问码和数据不加密；请只在可信网络使用，"
    "不要通过公网或不可信 Wi-Fi 开启。"
)

LOGIN_WINDOW_SECONDS = 10 * 60
SOURCE_FAILURE_LIMIT = 20
SESSION_FAILURE_LIMIT = 5

_SESSION_FAILURES_MAX = 1024
_SOURCE_FAILURES_MAX = 1024


class NetworkError(Exception):
    """网络生命周期错误，携带 API 错误码与状态码。"""

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


def local_ipv4_addresses() -> list[str]:
    """返回非回环的 IPv4 地址，供 Local 页面显示访问 URL。"""
    try:
        hostname = socket.gethostname()
        infos = socket.getaddrinfo(hostname, None, socket.AF_INET)
    except OSError:
        return []
    addresses = sorted({info[4][0] for info in infos})
    return [
        address
        for address in addresses
        if address and not address.startswith("127.") and address != "0.0.0.0"
    ]


def port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _default_server_factory(app, host: str, port: int):
    import uvicorn

    return uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_level="warning")
    )


class NetworkController:
    """LAN 监听、访问码与登录失败计数。"""

    def __init__(
        self,
        *,
        app_factory: Callable[..., object],
        settings_store: SettingsStore,
        session_store,
        local_port: int = LOCAL_PORT_DEFAULT,
        server_factory: Callable[..., object] | None = None,
        port_checker: Callable[[str, int], bool] | None = None,
        address_provider: Callable[[], list[str]] | None = None,
        code_factory: Callable[[], str] | None = None,
        start_timeout: float = 5.0,
        join_timeout: float = 5.0,
    ) -> None:
        self._app_factory = app_factory
        self._settings_store = settings_store
        self._session_store = session_store
        self._local_port = local_port
        self._server_factory = server_factory or _default_server_factory
        self._port_checker = port_checker or port_available
        self._address_provider = address_provider or local_ipv4_addresses
        self._code_factory = code_factory or (
            lambda: secrets.token_urlsafe(24)
        )
        self._start_timeout = start_timeout
        self._join_timeout = join_timeout

        self._lock = threading.RLock()
        self._enabled = False
        self._port: int | None = None
        self._access_code: str | None = None
        self._server = None
        self._thread: threading.Thread | None = None
        self._session_failures: dict[str, int] = {}
        self._source_failures: dict[str, list[float]] = {}
        self._last_error: str | None = None

    # ------------------------------------------------------------------ 状态
    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def access_code(self) -> str | None:
        with self._lock:
            return self._access_code

    @property
    def lan_port(self) -> int | None:
        with self._lock:
            return self._port

    def status(self) -> dict:
        with self._lock:
            enabled = self._enabled
            port = self._port
            code = self._access_code
            last_error = self._last_error
        if enabled and port is not None:
            addresses = [
                f"http://{address}:{port}"
                for address in self._address_provider()
            ]
        else:
            addresses = []
        return {
            "enabled": enabled,
            "port": port,
            "addresses": addresses,
            "access_code": code,
            "tls_warning": TLS_WARNING,
            "last_error": last_error,
        }

    # ------------------------------------------------------------------ 开启
    def enable(self, lan_port: int) -> dict:
        with self._lock:
            if self._enabled:
                return self.status()
            if (
                isinstance(lan_port, bool)
                or not isinstance(lan_port, int)
                or not 1 <= lan_port <= 65535
            ):
                message = "局域网端口必须是 1-65535 的整数"
                self._last_error = message
                raise NetworkError(422, "NETWORK_PORT_INVALID", message)
            if lan_port == self._local_port:
                message = "局域网端口不能与本机端口相同"
                self._last_error = message
                raise NetworkError(422, "NETWORK_PORT_INVALID", message)
            if not self._port_checker(LAN_HOST, lan_port):
                message = (
                    f"端口 {lan_port} 已被占用，请更换端口或关闭占用进程"
                )
                self._last_error = message
                raise NetworkError(409, "NETWORK_PORT_IN_USE", message)
            code = self._code_factory()
            app = self._app_factory(mode="lan", network=self)
            server = self._server_factory(app, LAN_HOST, lan_port)
            thread = threading.Thread(
                target=server.run, name="textmind-lan", daemon=True
            )
            thread.start()

        if not self._wait_ready(server, thread):
            self._stop_server(server, thread)
            message = "局域网监听启动失败，请检查端口和防火墙设置后重试"
            with self._lock:
                self._last_error = message
            raise NetworkError(500, "NETWORK_START_FAILED", message)

        try:
            current = self._settings_store.load()
            self._settings_store.save(
                Settings(
                    schema_version=current.schema_version,
                    model_root=current.model_root,
                    duretrieval_root=current.duretrieval_root,
                    device=current.device,
                    network=NetworkSettings(
                        lan_enabled=True,
                        local_port=current.network.local_port,
                        lan_port=lan_port,
                    ),
                )
            )
        except (
            SettingsReadError,
            SettingsInvalidError,
            SettingsWriteError,
        ) as exc:
            self._stop_server(server, thread)
            message = "开启局域网后保存设置失败，请检查数据目录写入权限"
            with self._lock:
                self._last_error = message
            raise NetworkError(
                500, "NETWORK_WRITE_FAILED", message
            ) from None

        with self._lock:
            self._enabled = True
            self._port = lan_port
            self._access_code = code
            self._server = server
            self._thread = thread
            self._last_error = None
        return self.status()

    # ------------------------------------------------------------------ 关闭
    def disable(self) -> dict:
        with self._lock:
            server = self._server
            thread = self._thread
            self._enabled = False
            self._access_code = None
            self._port = None
            self._server = None
            self._thread = None
        self._clear_lan_sessions()
        if server is not None and thread is not None:
            self._stop_server(server, thread)

        write_error = None
        try:
            current = self._settings_store.load()
            self._settings_store.save(
                Settings(
                    schema_version=current.schema_version,
                    model_root=current.model_root,
                    duretrieval_root=current.duretrieval_root,
                    device=current.device,
                    network=NetworkSettings(
                        lan_enabled=False,
                        local_port=current.network.local_port,
                        lan_port=current.network.lan_port,
                    ),
                )
            )
        except (
            SettingsReadError,
            SettingsInvalidError,
            SettingsWriteError,
        ) as exc:
            write_error = exc
        with self._lock:
            self._reset_login_failures()
            if write_error is None:
                self._last_error = None
        if write_error is not None:
            raise NetworkError(
                500,
                "NETWORK_WRITE_FAILED",
                "关闭局域网后保存设置失败，请检查数据目录写入权限",
            )
        return self.status()

    def shutdown(self) -> None:
        """进程退出：停止 listener、清会话与访问码，不持久化关闭状态。"""
        with self._lock:
            server = self._server
            thread = self._thread
            self._enabled = False
            self._access_code = None
            self._port = None
            self._server = None
            self._thread = None
        self._clear_lan_sessions()
        if server is not None and thread is not None:
            self._stop_server(server, thread)

    # ------------------------------------------------------------------ 访问码
    def verify_access_code(self, code: str | None) -> bool:
        with self._lock:
            if not self._enabled or not self._access_code:
                return False
            if not isinstance(code, str) or not code:
                return False
            return hmac.compare_digest(self._access_code, code)

    # ------------------------------------------------------------------ 登录限流
    def login_failure(self, session_id: str, source: str) -> tuple[int, bool]:
        now = time.monotonic()
        with self._lock:
            if self._session_store.get(session_id) is None:
                # 会话已被删除或过期：清除旧计数并立即返回，不重新创建计数。
                self._session_failures.pop(session_id, None)
                window = self._source_failures.setdefault(source, [])
                window.append(now)
                window[:] = [
                    stamp
                    for stamp in window
                    if stamp >= now - LOGIN_WINDOW_SECONDS
                ]
                self._prune_failure_tables_locked(now, keep_source=source)
                return 0, len(window) >= SOURCE_FAILURE_LIMIT
            session_count = self._session_failures.get(session_id, 0) + 1
            self._session_failures[session_id] = session_count
            window = self._source_failures.setdefault(source, [])
            window.append(now)
            window[:] = [
                stamp
                for stamp in window
                if stamp >= now - LOGIN_WINDOW_SECONDS
            ]
            self._prune_failure_tables_locked(now, keep_source=source)
            source_count = len(window)
        return session_count, source_count >= SOURCE_FAILURE_LIMIT

    def source_throttled(self, source: str) -> bool:
        now = time.monotonic()
        with self._lock:
            window = self._source_failures.get(source, [])
            window[:] = [
                stamp for stamp in window if stamp >= now - LOGIN_WINDOW_SECONDS
            ]
            return len(window) >= SOURCE_FAILURE_LIMIT

    def clear_source_failures(self, source: str) -> None:
        with self._lock:
            self._source_failures.pop(source, None)

    def clear_session_failures(self, session_id: str) -> None:
        with self._lock:
            self._session_failures.pop(session_id, None)

    # ------------------------------------------------------------------ 内部
    def _wait_ready(self, server, thread: threading.Thread) -> bool:
        deadline = time.monotonic() + self._start_timeout
        while time.monotonic() < deadline:
            if getattr(server, "started", False):
                return True
            if not thread.is_alive():
                return bool(getattr(server, "started", False))
            time.sleep(0.02)
        return bool(getattr(server, "started", False))

    def _stop_server(self, server, thread: threading.Thread) -> None:
        try:
            server.should_exit = True
        except Exception:  # noqa: BLE001 - 关闭尽力而为
            pass
        thread.join(self._join_timeout)

    def _clear_lan_sessions(self) -> int:
        return self._session_store.delete_scopes(LAN_SCOPES)

    def _reset_login_failures(self) -> None:
        self._session_failures.clear()
        self._source_failures.clear()

    def _prune_failure_tables_locked(
        self, now: float, keep_source: str | None = None
    ) -> None:
        cutoff = now - LOGIN_WINDOW_SECONDS
        self._source_failures = {
            source: [stamp for stamp in stamps if stamp >= cutoff]
            for source, stamps in self._source_failures.items()
        }
        self._source_failures = {
            source: stamps
            for source, stamps in self._source_failures.items()
            if stamps or source == keep_source
        }
        excess = len(self._session_failures) - _SESSION_FAILURES_MAX
        if excess > 0:
            for session_id in list(self._session_failures)[:excess]:
                self._session_failures.pop(session_id, None)
        while len(self._source_failures) > _SOURCE_FAILURES_MAX:
            candidates = [
                item
                for item in self._source_failures
                if item != keep_source
            ]
            if not candidates:
                break
            oldest = min(
                candidates,
                key=lambda item: min(self._source_failures[item]),
            )
            self._source_failures.pop(oldest, None)
