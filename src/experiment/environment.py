"""运行环境信息采集。

只采集操作系统、CPU 架构、软件版本和计算设备；不包含主机名、用户名、
IP、设备序列号、当前工作目录或模型绝对路径。
"""

from __future__ import annotations

import platform
from importlib import metadata

VERSION_DISTRIBUTIONS: dict[str, str] = {
    "torch": "torch",
    "transformers": "transformers",
    "sentence_transformers": "sentence-transformers",
    "numpy": "numpy",
    "jieba": "jieba",
    "matplotlib": "matplotlib",
}


def _distribution_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def resolve_device(name: str = "auto") -> str:
    """把 `auto` 解析为 mps / cuda / cpu；显式设备名原样返回。"""
    if name != "auto":
        return name
    try:
        import torch
    except ImportError:  # pragma: no cover - 无 torch 的降级路径
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _gpu_name(device: str | None) -> str | None:
    if device != "cuda":
        return None
    try:
        import torch
    except ImportError:  # pragma: no cover - 无 torch 的降级路径
        return None
    try:
        return torch.cuda.get_device_name(0)
    except (RuntimeError, AssertionError):  # pragma: no cover - 无 GPU 时
        return None


def collect_environment(device: str | None = None) -> dict[str, str | None]:
    """返回脱敏的运行环境字典。"""
    environment: dict[str, str | None] = {
        "os": platform.system() or None,
        "os_version": platform.release() or None,
        "platform": platform.platform() or None,
        "python": platform.python_version(),
        "cpu_arch": platform.machine() or None,
        "cpu_model": platform.processor() or None,
        "device": device,
        "gpu": _gpu_name(device),
    }
    for field, distribution in VERSION_DISTRIBUTIONS.items():
        environment[field] = _distribution_version(distribution)
    return environment
