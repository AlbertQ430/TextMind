#!/usr/bin/env python3
"""检查 Python 版本和锁定依赖是否可用。

核心依赖缺失时返回非零退出码，启动脚本据此阻止启动。
本脚本只读取已安装包的元数据，不初始化模型、不加载数据集、不访问网络。
"""

from __future__ import annotations

import sys
from importlib import metadata

from install import install_recovery_command

REQUIRED_PYTHON = (3, 11)

# (发行包名, 用途说明)；顺序与 requirements.lock 的核心依赖一致。
DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("fastapi", "Web 框架"),
    ("uvicorn", "ASGI 服务器"),
    ("jinja2", "页面模板"),
    ("python-multipart", "表单与上传"),
    ("jsonschema", "配置校验"),
    ("httpx", "HTTP 客户端"),
    ("numpy", "数值计算"),
    ("torch", "深度学习运行时"),
    ("transformers", "模型加载"),
    ("sentence-transformers", "Embedding 模型"),
    ("jieba", "中文分词"),
    ("matplotlib", "图表生成"),
)


def _installed_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def run_checks(verbose: bool = True) -> bool:
    """输出检查结果；全部通过返回 True。"""
    ok = True

    version_ok = sys.version_info[:2] == REQUIRED_PYTHON
    if verbose:
        print(f"[{'OK' if version_ok else 'FAIL'}] Python {sys.version.split()[0]}"
              f"（需要 {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}）")
    ok = ok and version_ok

    for distribution, purpose in DEPENDENCIES:
        version = _installed_version(distribution)
        passed = version is not None
        ok = ok and passed
        if verbose:
            shown = version if passed else "未安装"
            print(f"[{'OK' if passed else 'FAIL'}] {distribution} {shown}（{purpose}）")

    return ok


def main() -> int:
    passed = run_checks(verbose=True)
    if not passed:
        print(
            "\n[错误] 环境检查未通过：请先运行 "
            f"{install_recovery_command()}",
            file=sys.stderr,
        )
        return 1
    print("\n[完成] 环境检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
