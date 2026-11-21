#!/usr/bin/env python3
"""可重复构建 TextMind 发布 ZIP。

只使用 Python 标准库；显式 allowlist 复制必需文件/目录，在暂存目录运行
预置结果构建，生成 MANIFEST.txt，并按固定顺序写入固定顶层目录的 ZIP。
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.plans import load_plan_catalog  # noqa: E402
from scripts import build_presets  # noqa: E402

VERSION = "0.1.0"
TOP_DIR = f"textmind-{VERSION}"
PRESET_COUNT = 17

REQUIRED_FILES: tuple[str, ...] = (
    "README.md",
    "NOTICE.md",
    "AGENTS.md",
    "requirements.lock",
    "pyproject.toml",
    "start.command",
    "start.bat",
    "results/README.md",
)
REQUIRED_DIRS: tuple[str, ...] = (
    "scripts",
    "src",
    "config",
    "schemas",
    "data/evaluation",
    "data/raw/scifact",
    "resources/scifact",
    "docs",
)
REQUIRED_ASSETS: tuple[str, ...] = (
    "resources/scifact/SOURCE.md",
    "resources/scifact/NOTICE.md",
    "resources/scifact/LICENSE-CC-BY-SA-4.0.txt",
)

FORBIDDEN_PREFIXES: tuple[str, ...] = (
    ".git/",
    ".venv/",
    "data/app/",
    "logs/",
    "dist/",
    "results/runs/",
    "results/imports/",
)
FORBIDDEN_NAMES: frozenset[str] = frozenset({".DS_Store"})
FORBIDDEN_DIRS: frozenset[str] = frozenset({"__pycache__"})
FORBIDDEN_SUFFIXES: tuple[str, ...] = (".pyc",)
EXCLUDED_PATHS: tuple[str, ...] = ("docs/export",)


class ReleaseError(Exception):
    """发布检查或构建失败。"""

    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _has_files(root: Path, relative: str) -> bool:
    directory = root / relative
    return directory.is_dir() and any(
        child.is_file() for child in directory.rglob("*")
    )


def check_release(root: Path | str = PROJECT_ROOT) -> list[str]:
    """只读检查发布必需项；返回缺失或为空的路径。"""
    root = Path(root)
    errors: list[str] = []
    for relative in REQUIRED_FILES:
        path = root / relative
        if not path.is_file() or path.stat().st_size <= 0:
            errors.append(f"缺少必需文件: {relative}")
    for relative in REQUIRED_DIRS:
        if not _has_files(root, relative):
            errors.append(f"缺少必需目录或目录为空: {relative}")
    for relative in REQUIRED_ASSETS:
        path = root / relative
        if not path.is_file() or path.stat().st_size <= 0:
            errors.append(f"缺少非空发布资产: {relative}")
    try:
        errors.extend(build_presets.check_assets())
    except Exception as exc:  # noqa: BLE001 - 检查失败也要转为可读错误
        errors.append(f"预置资源检查失败: {exc}")
    return errors


def _is_forbidden(relative: str) -> bool:
    if relative in FORBIDDEN_NAMES or relative.endswith(FORBIDDEN_SUFFIXES):
        return True
    if relative.split("/")[0] in FORBIDDEN_DIRS:
        return True
    if any(part in FORBIDDEN_DIRS for part in relative.split("/")):
        return True
    return any(
        relative == prefix.rstrip("/") or relative.startswith(prefix)
        for prefix in FORBIDDEN_PREFIXES
    )


def _is_excluded(relative: str) -> bool:
    return any(
        relative == excluded or relative.startswith(excluded + "/")
        for excluded in EXCLUDED_PATHS
    )


def _copy_path(root: Path, package: Path, relative: str) -> int:
    source = root / relative
    target = package / relative
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return 1
    copied = 0
    for child in sorted(source.rglob("*")):
        if child.is_dir():
            continue
        rel = _relative(source, child)
        if _is_forbidden(rel) or _is_excluded(rel):
            continue
        destination = target / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(child, destination)
        copied += 1
    return copied


def _scan_forbidden(package: Path) -> list[str]:
    errors: list[str] = []
    for path in package.rglob("*"):
        relative = _relative(package, path)
        if _is_forbidden(relative) or _is_excluded(relative):
            errors.append(f"暂存目录包含不应发布的内容: {relative}")
    return errors


def _write_manifest(package: Path) -> list[str]:
    entries = sorted(
        _relative(package, path)
        for path in package.rglob("*")
        if path.is_file()
    )
    lines = [
        f"{_sha256(package / entry)}  {entry}" for entry in entries
    ]
    (package / "MANIFEST.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return lines


def build_release(
    output: Path | str,
    root: Path | str = PROJECT_ROOT,
) -> dict:
    """检查并构建发布 ZIP，返回输出路径、SHA-256 和条目数。"""
    root = Path(root)
    errors = check_release(root)
    if errors:
        raise ReleaseError(errors)

    output = Path(output)
    staging = Path(tempfile.mkdtemp(prefix="textmind-release-"))
    package = staging / TOP_DIR
    try:
        package.mkdir(parents=True, exist_ok=True)
        for relative in REQUIRED_FILES:
            _copy_path(root, package, relative)
        for relative in REQUIRED_DIRS:
            _copy_path(root, package, relative)

        catalog = load_plan_catalog()
        presets_dir = package / "results" / "presets"
        published, failures = build_presets.build_all(
            catalog, presets_dir, force=False
        )
        if failures:
            raise ReleaseError(
                [f"预置构建失败 {variant}: {reason}" for variant, reason in failures]
            )
        if len(published) != PRESET_COUNT:
            raise ReleaseError(
                [f"预置记录数量不符：期望 {PRESET_COUNT}，实际 {len(published)}"]
            )

        forbidden = _scan_forbidden(package)
        if forbidden:
            raise ReleaseError(forbidden)

        manifest = _write_manifest(package)
        entries = sorted(
            _relative(package, path)
            for path in package.rglob("*")
            if path.is_file()
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for entry in entries:
                info = zipfile.ZipInfo(
                    f"{TOP_DIR}/{entry}", date_time=(2026, 1, 1, 0, 0, 0)
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                mode = 0o755 if entry == "start.command" else 0o644
                info.external_attr = mode << 16
                archive.writestr(info, (package / entry).read_bytes())

        return {
            "output": output,
            "sha256": _sha256(output),
            "entries": entries,
            "manifest_lines": manifest,
        }
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查并构建 TextMind 发布 ZIP")
    parser.add_argument(
        "--check", action="store_true", help="只读检查发布必需项"
    )
    parser.add_argument("--output", help="发布 ZIP 输出路径")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.check and not args.output:
        print("请指定 --check 或 --output", file=sys.stderr)
        return 2

    errors = check_release()
    if errors:
        for error in errors:
            print(f"发布检查失败: {error}")
        return 1
    if args.check:
        print("发布检查通过：必需文件、资产和预置资源均存在。")
        return 0

    try:
        result = build_release(args.output)
    except ReleaseError as exc:
        for error in exc.errors:
            print(f"发布构建失败: {error}")
        return 1
    print(f"已生成发布包: {result['output']}")
    print(f"文件数（含 MANIFEST.txt）: {len(result['entries'])}")
    print(f"SHA-256: {result['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
