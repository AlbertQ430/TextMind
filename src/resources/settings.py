"""主机设置模型与原子读写。

设置保存在 `data/app/settings.json`，只包含资源根目录、计算设备和网络
端口配置；不保存访问码、会话、令牌、任务状态或资源扫描结果。
文件损坏时抛出 `SettingsReadError`，调用方必须显式报告而不能静默覆盖。
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from paths import APP_DATA_DIR

SETTINGS_FILENAME = "settings.json"
DEVICES = ("auto", "cpu", "cuda", "mps")

SettingsDevice = Literal["auto", "cpu", "cuda", "mps"]


class SettingsError(Exception):
    """设置读取/校验/写入失败。"""


class SettingsReadError(SettingsError):
    """设置文件不存在之外的读取或校验失败。"""


class SettingsWriteError(SettingsError):
    """原子写入失败。"""


class SettingsInvalidError(SettingsError):
    """设置字段非法（请求或文件内容）。"""


class _DuplicateKey(Exception):
    def __init__(self, key: str) -> None:
        self.key = key


class _InvalidConstant(Exception):
    def __init__(self, value: str) -> None:
        self.value = value


def _unique_object(pairs):
    seen: dict = {}
    for key, value in pairs:
        if key in seen:
            raise _DuplicateKey(key)
        seen[key] = value
    return seen


def _reject_constant(value: str):
    raise _InvalidConstant(value)


class NetworkSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lan_enabled: bool = False
    local_port: int = Field(default=8765, ge=1, le=65535)
    lan_port: int = Field(default=8766, ge=1, le=65535)

    @model_validator(mode="after")
    def _validate_ports(self) -> "NetworkSettings":
        if self.local_port == self.lan_port:
            raise ValueError("local_port 与 lan_port 不能相同")
        return self


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    model_root: str | None = None
    duretrieval_root: str | None = None
    device: SettingsDevice = "auto"
    network: NetworkSettings = Field(default_factory=NetworkSettings)


class SettingsUpdate(BaseModel):
    """`PUT /api/v1/settings` 的请求体：只允许资源与设备字段。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_root: str | None
    duretrieval_root: str | None
    device: SettingsDevice


def normalize_root(value: str | None) -> str | None:
    """把路径规范化为绝对路径；不存在也允许保存。"""
    if value is None:
        return None
    if not isinstance(value, str):
        raise SettingsInvalidError("路径必须是字符串或 null")
    stripped = value.strip()
    if not stripped:
        raise SettingsInvalidError("路径不能为空字符串")
    if any(ord(character) < 32 or ord(character) == 127 for character in stripped):
        raise SettingsInvalidError("路径不能包含 NUL 或其他控制字符")
    try:
        path = Path(stripped).expanduser()
        return str(path.resolve())
    except (ValueError, RuntimeError, OSError) as exc:
        raise SettingsInvalidError(f"路径无法解析: {exc}") from None


class SettingsStore:
    """`settings.json` 的读取、校验与原子保存。"""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = (
            Path(path) if path is not None else APP_DATA_DIR / SETTINGS_FILENAME
        )

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------ 读取
    def load(self) -> Settings:
        if not self._path.exists():
            return Settings()
        try:
            raw_bytes = self._path.read_bytes()
        except OSError as exc:
            raise SettingsReadError(
                f"无法读取设置文件 {SETTINGS_FILENAME}: {exc}"
            ) from exc
        try:
            raw = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise SettingsReadError(
                "设置文件不是有效的 UTF-8 编码"
            ) from None
        try:
            data = json.loads(
                raw,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        except _DuplicateKey as exc:
            raise SettingsReadError(
                f"设置文件包含重复键: {exc.key}"
            ) from None
        except _InvalidConstant as exc:
            raise SettingsReadError(
                f"设置文件包含非标准 JSON 常量: {exc.value}"
            ) from None
        except json.JSONDecodeError as exc:
            raise SettingsReadError(
                f"设置文件不是合法 JSON（第 {exc.lineno} 行第 {exc.colno} 列）"
            ) from None
        except RecursionError:
            raise SettingsReadError("设置文件嵌套层级过深") from None
        if not isinstance(data, dict):
            raise SettingsReadError("设置文件顶层必须是 JSON 对象")
        try:
            data = dict(data)
            data["model_root"] = normalize_root(data.get("model_root"))
            data["duretrieval_root"] = normalize_root(
                data.get("duretrieval_root")
            )
            return Settings.model_validate(data)
        except SettingsInvalidError:
            raise
        except (ValidationError, TypeError) as exc:
            raise SettingsInvalidError(
                "设置文件字段非法：" + _summarize_validation(exc)
            ) from exc

    # ------------------------------------------------------------------ 写入
    def save(self, settings: Settings) -> Settings:
        normalized = Settings(
            schema_version=settings.schema_version,
            model_root=normalize_root(settings.model_root),
            duretrieval_root=normalize_root(settings.duretrieval_root),
            device=settings.device,
            network=settings.network,
        )
        payload = json.dumps(
            normalized.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        ) + "\n"
        staging: Path | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            staging = self._path.with_name(
                f".{self._path.name}.tmp-{secrets.token_hex(6)}"
            )
            with staging.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(staging, self._path)
        except OSError as exc:
            raise SettingsWriteError(
                f"写入设置文件失败: {exc}"
            ) from exc
        finally:
            if staging is not None and staging.exists():
                try:
                    staging.unlink()
                except OSError:
                    pass
        return normalized

    def save_update(self, update: SettingsUpdate) -> Settings:
        """保留网络字段，只更新资源根目录与设备。"""
        current = self.load()
        return self.save(
            Settings(
                model_root=update.model_root,
                duretrieval_root=update.duretrieval_root,
                device=update.device,
                network=current.network,
            )
        )

    # ------------------------------------------------------------------ 变化检测
    def fingerprint(self) -> str:
        """设置文件存在性与 mtime/size 指纹，供 SSE 变化检测。"""
        try:
            stat = self._path.stat()
        except OSError:
            return "missing"
        return f"{stat.st_mtime_ns}:{stat.st_size}"


def _summarize_validation(exc) -> str:
    if isinstance(exc, ValidationError):
        lines = []
        for item in exc.errors():
            location = ".".join(str(part) for part in item["loc"])
            lines.append(f"{location}: {item['msg']}")
        return "; ".join(lines)
    return str(exc)
