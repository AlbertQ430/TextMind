#!/usr/bin/env python3
"""创建项目根目录的 .venv 并安装 requirements.lock 中的锁定依赖。

约束：
- 只在项目内创建 .venv，不安装全局包。
- 不下载模型或数据集。
- 失败时输出可理解的错误并返回非零退出码。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = PROJECT_ROOT / ".venv"
LOCK_FILE = PROJECT_ROOT / "requirements.lock"
REQUIRED_PYTHON = (3, 11)


def _python_version(executable: str) -> tuple[int, int] | None:
    """返回解释器主次版本号；无法执行时返回 None。"""
    try:
        result = subprocess.run(
            [executable, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    try:
        major, minor = result.stdout.strip().split(".")
        return int(major), int(minor)
    except ValueError:
        return None


def missing_python_hint(platform_name: str | None = None) -> str:
    """找不到兼容解释器时，按当前平台给出可执行的安装命令。"""
    name = platform_name or sys.platform
    if name == "win32":
        return (
            "需要 Python 3.11 解释器来创建 .venv。可用以下任一方式指定：\n"
            "  py -3.11 scripts\\install.py\n"
            "  .venv\\Scripts\\python.exe scripts\\install.py --python <python3.11.exe 路径>"
        )
    return (
        "需要 Python 3.11 解释器来创建 .venv。可用以下任一方式指定：\n"
        "  TEXTMIND_PYTHON=/path/to/python3.11 python3 scripts/install.py\n"
        "  python3 scripts/install.py --python /path/to/python3.11"
    )


def _find_python(explicit: str | None) -> tuple[str | None, str]:
    """按优先级选择用于创建 .venv 的兼容解释器。"""
    candidates: list[tuple[str, str]] = []
    if explicit:
        candidates.append((explicit, "--python"))
    env_python = os.environ.get("TEXTMIND_PYTHON")
    if env_python:
        candidates.append((env_python, "TEXTMIND_PYTHON"))
    candidates.append((sys.executable, "当前解释器"))
    on_path = shutil.which("python3.11")
    if on_path:
        candidates.append((on_path, "PATH 中的 python3.11"))

    checked: list[str] = []
    for executable, source in candidates:
        if not executable:
            continue
        resolved = shutil.which(executable) or executable
        checked.append(f"{source}: {executable}")
        version = _python_version(resolved)
        if version == REQUIRED_PYTHON:
            return resolved, source

    hint = (
        "需要 Python 3.11 解释器来创建 .venv。可用以下任一方式指定：\n"
        "  TEXTMIND_PYTHON=/path/to/python3.11 python3 scripts/install.py\n"
        "  python3 scripts/install.py --python /path/to/python3.11\n"
        "已检查：\n  " + "\n  ".join(checked)
    )
    return None, hint


def venv_python(platform_name: str | None = None) -> Path:
    """返回当前平台虚拟环境解释器路径（相对项目根目录）。"""
    name = platform_name or sys.platform
    if name == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def check_env_command(platform_name: str | None = None) -> str:
    name = platform_name or sys.platform
    if name == "win32":
        return r".venv\Scripts\python.exe scripts\check_env.py"
    return ".venv/bin/python scripts/check_env.py"


def start_command(platform_name: str | None = None) -> str:
    name = platform_name or sys.platform
    if name == "win32":
        return r".venv\Scripts\python.exe scripts\start.py"
    return ".venv/bin/python scripts/start.py"


def install_recovery_command(platform_name: str | None = None) -> str:
    name = platform_name or sys.platform
    if name == "win32":
        return r".venv\Scripts\python.exe scripts\install.py"
    return ".venv/bin/python scripts/install.py"


def remove_venv_command(platform_name: str | None = None) -> str:
    name = platform_name or sys.platform
    if name == "win32":
        return "rmdir /s /q .venv"
    return "rm -rf .venv"


def _venv_python() -> Path:
    return venv_python()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="创建 .venv 并安装锁定依赖")
    parser.add_argument(
        "--python",
        metavar="PATH",
        help="用于创建 .venv 的 Python 3.11 解释器路径",
    )
    args = parser.parse_args(argv)

    if not LOCK_FILE.is_file():
        print(f"[错误] 找不到依赖锁定文件：{LOCK_FILE}", file=sys.stderr)
        return 1

    if VENV_DIR.exists():
        existing = _python_version(str(_venv_python()))
        if existing != REQUIRED_PYTHON:
            print(
                f"[错误] {VENV_DIR} 已存在，但其解释器不是 Python 3.11"
                f"（检测结果：{existing or '无法执行'}）。\n"
                f"请先手动删除该目录后重试：{remove_venv_command()}",
                file=sys.stderr,
            )
            return 1
        print(f"[信息] 复用已有虚拟环境：{VENV_DIR}")
        interpreter = str(_venv_python())
    else:
        python, note = _find_python(args.python)
        if python is None:
            print(f"[错误] {note}", file=sys.stderr)
            return 1
        print(f"[信息] 使用解释器（{note}）：{python}")
        try:
            subprocess.run(
                [python, "-m", "venv", str(VENV_DIR)],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"[错误] 创建虚拟环境失败（退出码 {exc.returncode}）。", file=sys.stderr)
            return 1
        interpreter = str(_venv_python())

    print(f"[信息] 从 {LOCK_FILE.name} 安装锁定依赖，可能需要数分钟……")
    try:
        subprocess.run(
            [
                interpreter,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-r",
                str(LOCK_FILE),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(
            f"[错误] 依赖安装失败（退出码 {exc.returncode}）。"
            "请检查网络、包源和上方 pip 输出。",
            file=sys.stderr,
        )
        return 1

    print("[完成] 依赖安装成功。后续命令：")
    print(f"  {check_env_command()}")
    print(f"  {start_command()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
