"""T-011 双平台启动入口、发布包与浏览器支持测试。"""

from __future__ import annotations

import contextlib
import io
import os
import posixpath
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import package_release  # noqa: E402

TOP_DIR = package_release.TOP_DIR


class StartScriptTests(unittest.TestCase):
    def test_command_script_contract(self) -> None:
        script = PROJECT_ROOT / "start.command"
        text = script.read_text(encoding="utf-8")
        self.assertTrue(os.access(script, os.X_OK))
        self.assertIn(".venv/bin/python", text)
        self.assertIn('dirname -- "$0"', text)
        self.assertIn("scripts/start.py", text)
        self.assertIn("install.py", text)
        self.assertIn("exit 1", text)

    @unittest.skipIf(
        sys.platform == "win32", "POSIX shell 用例仅在 POSIX 平台执行"
    )
    def test_command_fails_without_venv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "start.command"
            shutil.copy2(PROJECT_ROOT / "start.command", target)
            target.chmod(0o755)
            result = subprocess.run(
                ["sh", str(target)],
                cwd=str(Path(tmp)),
                capture_output=True,
                text=True,
                timeout=30,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("install.py", result.stderr + result.stdout)

    def test_bat_script_contract(self) -> None:
        text = (PROJECT_ROOT / "start.bat").read_bytes().decode("ascii")
        self.assertIn("%~dp0", text)
        self.assertIn(".venv\\Scripts\\python.exe", text)
        self.assertIn("scripts\\start.py", text)
        self.assertIn("install.py", text)
        self.assertIn("exit /b 1", text)
        self.assertIn("if errorlevel 1", text)


class PackageCheckTests(unittest.TestCase):
    def test_repository_passes_check(self) -> None:
        self.assertEqual(package_release.check_release(), [])

    def test_missing_required_file_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                package_release.build_presets,
                "check_assets",
                return_value=[],
            ):
                errors = package_release.check_release(Path(tmp))
        self.assertTrue(any("README.md" in error for error in errors))
        self.assertTrue(any("resources/scifact" in error for error in errors))

    def test_forbidden_paths_detected(self) -> None:
        self.assertTrue(package_release._is_forbidden(".venv/bin/python"))
        self.assertTrue(package_release._is_forbidden("results/runs/x/run.json"))
        self.assertTrue(package_release._is_forbidden("data/app/settings.json"))
        self.assertTrue(package_release._is_forbidden("src/__pycache__/a.pyc"))
        self.assertTrue(package_release._is_forbidden("logs/app.log"))
        self.assertFalse(package_release._is_forbidden("results/presets/x/run.json"))
        self.assertFalse(package_release._is_forbidden("docs/development/T-011.md"))


class PackageBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.zip_path = cls.root / "textmind-0.1.0.zip"
        cls.result = package_release.build_release(cls.zip_path)
        cls.second_zip = cls.root / "textmind-0.1.0-second.zip"
        cls.second = package_release.build_release(cls.second_zip)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_zip_has_fixed_top_dir_and_required_entries(self) -> None:
        with zipfile.ZipFile(self.zip_path) as archive:
            names = archive.namelist()
        self.assertTrue(all(name.startswith(TOP_DIR + "/") for name in names))
        top = {name.split("/")[0] for name in names}
        self.assertEqual(top, {TOP_DIR})

        prefix = TOP_DIR + "/"
        for required in (
            "README.md",
            "NOTICE.md",
            "requirements.lock",
            "pyproject.toml",
            "start.command",
            "start.bat",
            "MANIFEST.txt",
            "results/README.md",
        ):
            self.assertIn(prefix + required, names)
        for directory in (
            "scripts/",
            "src/",
            "config/",
            "schemas/",
            "data/raw/scifact/",
            "data/evaluation/",
            "resources/scifact/",
            "docs/",
            "results/presets/",
        ):
            self.assertTrue(
                any(name.startswith(prefix + directory) for name in names),
                directory,
            )

    def test_zip_excludes_forbidden_content(self) -> None:
        with zipfile.ZipFile(self.zip_path) as archive:
            names = archive.namelist()
        forbidden = [
            name
            for name in names
            if any(
                marker in name
                for marker in (
                    ".git/",
                    ".venv/",
                    "data/app/",
                    "logs/",
                    "dist/",
                    "__pycache__/",
                    ".DS_Store",
                    "results/runs/",
                    "results/imports/",
                )
            )
            or name.endswith(".pyc")
        ]
        self.assertEqual(forbidden, [])

    def test_presets_and_charts_included(self) -> None:
        with zipfile.ZipFile(self.zip_path) as archive:
            names = archive.namelist()
        prefix = f"{TOP_DIR}/results/presets/"
        run_jsons = [
            name
            for name in names
            if name.startswith(prefix) and name.endswith("/run.json")
        ]
        charts = [
            name
            for name in names
            if name.startswith(prefix) and name.endswith(".png")
        ]
        self.assertEqual(len(run_jsons), 17)
        self.assertEqual(len(charts), 34)

    def test_manifest_matches_archive_content(self) -> None:
        import hashlib

        with zipfile.ZipFile(self.zip_path) as archive:
            names = archive.namelist()
            manifest_text = archive.read(f"{TOP_DIR}/MANIFEST.txt").decode(
                "utf-8"
            )
            entries = [
                name[len(TOP_DIR) + 1:] for name in names if name != f"{TOP_DIR}/MANIFEST.txt"
            ]
            digests = {
                name: hashlib.sha256(archive.read(name)).hexdigest()
                for name in names
                if name != f"{TOP_DIR}/MANIFEST.txt"
            }

        lines = [line for line in manifest_text.splitlines() if line]
        self.assertEqual(len(lines), len(entries))
        manifest_entries: list[str] = []
        for line in lines:
            digest, relative = line.split("  ", 1)
            manifest_entries.append(relative)
            self.assertEqual(
                digests[f"{TOP_DIR}/{relative}"], digest, relative
            )
        self.assertEqual(manifest_entries, sorted(entries))

    def test_repeat_build_has_same_entry_order(self) -> None:
        with zipfile.ZipFile(self.zip_path) as first:
            first_names = first.namelist()
        with zipfile.ZipFile(self.second_zip) as second:
            second_names = second.namelist()
        self.assertEqual(first_names, second_names)
        self.assertEqual(
            first_names, sorted(first_names, key=lambda name: name.split("/", 1)[1])
        )

    def test_start_command_is_executable_in_zip(self) -> None:
        with zipfile.ZipFile(self.zip_path) as archive:
            info = archive.getinfo(f"{TOP_DIR}/start.command")
        self.assertEqual((info.external_attr >> 16) & 0o777, 0o755)

    def test_result_reports_sha256(self) -> None:
        self.assertRegex(self.result["sha256"], r"^[0-9a-f]{64}$")

    def test_no_broken_local_markdown_links(self) -> None:
        pattern = re.compile(r"\]\(([^)\s]+)\)")
        with zipfile.ZipFile(self.zip_path) as archive:
            names = set(archive.namelist())
            missing: list[tuple[str, str]] = []
            for name in sorted(names):
                if not name.endswith(".md"):
                    continue
                text = archive.read(name).decode("utf-8")
                base = posixpath.dirname(name)
                for target in pattern.findall(text):
                    if target.startswith(
                        ("http://", "https://", "mailto:", "#")
                    ):
                        continue
                    path = target.split("#", 1)[0]
                    if not path:
                        continue
                    resolved = posixpath.normpath(
                        posixpath.join(base, path)
                    )
                    if resolved in names:
                        continue
                    if any(
                        item.startswith(resolved + "/") for item in names
                    ):
                        continue
                    missing.append((name, target))
        self.assertEqual(missing, [], f"断链: {missing[:10]}")


class BrowserSupportTests(unittest.TestCase):
    BROWSERS = (
        ("Safari", "17"),
        ("Edge", "120"),
    )

    def test_readme_declares_browser_matrix(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("Safari 17", readme)
        self.assertIn("Edge 120", readme)
        self.assertNotIn("Chrome 120", readme)
        self.assertNotIn("Firefox 121", readme)

    def test_sds_declares_same_browser_matrix(self) -> None:
        sds = (PROJECT_ROOT / "docs" / "design" / "SDS.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Safari 17", sds)
        self.assertIn("Edge 120", sds)
        self.assertNotIn("Chrome 120", sds)
        self.assertNotIn("Firefox 121", sds)


class PlatformCommandTests(unittest.TestCase):
    def test_platform_paths_and_commands(self) -> None:
        import install

        self.assertEqual(install.venv_python("win32").parts[-2], "Scripts")
        self.assertEqual(install.venv_python("darwin").parts[-2], "bin")
        self.assertIn("Scripts", install.check_env_command("win32"))
        self.assertIn("Scripts", install.start_command("win32"))
        self.assertIn("Scripts", install.install_recovery_command("win32"))
        self.assertIn("rmdir /s /q .venv", install.remove_venv_command("win32"))
        self.assertIn(".venv/bin/python", install.check_env_command("darwin"))
        self.assertIn("rm -rf .venv", install.remove_venv_command("darwin"))

        win_hint = install.missing_python_hint("win32")
        self.assertIn("py -3.11", win_hint)
        self.assertIn("scripts\\install.py", win_hint)
        self.assertNotIn("TEXTMIND_PYTHON=", win_hint)
        posix_hint = install.missing_python_hint("darwin")
        self.assertIn("TEXTMIND_PYTHON=", posix_hint)
        self.assertIn("python3 scripts/install.py", posix_hint)

    def test_check_env_failure_message_uses_platform_command(self) -> None:
        import check_env

        stderr = io.StringIO()
        with mock.patch.object(sys, "platform", "win32"), mock.patch.object(
            check_env, "run_checks", return_value=False
        ), contextlib.redirect_stderr(stderr):
            code = check_env.main()
        self.assertEqual(code, 1)
        self.assertIn("Scripts", stderr.getvalue())
        self.assertNotIn(".venv/bin/python", stderr.getvalue())


class StartupErrorTests(unittest.TestCase):
    def test_instance_lock_error_is_single_line(self) -> None:
        import start
        from tasks.locking import InstanceLockError

        fake_context = mock.Mock()
        fake_context.start.side_effect = InstanceLockError("locked")
        fake_context.settings_store = mock.Mock()
        fake_context.session_store = mock.Mock()

        stderr = io.StringIO()
        with mock.patch.object(start, "check_env") as env, mock.patch.object(
            start, "ensure_app_dirs"
        ), mock.patch.object(
            start, "is_port_available", return_value=True
        ), mock.patch(
            "web.context.build_default_context", return_value=fake_context
        ), mock.patch(
            "runtime.network.NetworkController"
        ) as controller_cls, contextlib.redirect_stderr(stderr):
            env.run_checks.return_value = True
            code = start.main(["--port", "8765"])

        self.assertEqual(code, 1)
        self.assertIn("已有 TextMind 实例", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        controller_cls.return_value.shutdown.assert_called_once()


class StageStatementTests(unittest.TestCase):
    def test_readme_stage_matches_current_state(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        stage = readme[readme.index("## 当前阶段说明"):]
        self.assertIn("Safari 26.6 与 Edge 158 人工验收已完成", stage)
        self.assertIn("仅 Windows 实机验证待补充", stage)
        self.assertNotIn("Safari/Chrome/Firefox", stage)

    def test_open_questions_stage_matches_current_state(self) -> None:
        text = (
            PROJECT_ROOT / "docs" / "requirements" / "open-questions.md"
        ).read_text(encoding="utf-8")
        stage = text[text.index("## 6. 阶段说明"):]
        self.assertIn("仅 Windows 实机验证待补充", stage)
        self.assertNotIn("Safari/Chrome/Firefox", stage)

    def test_sds_stage_matches_current_state(self) -> None:
        sds = (PROJECT_ROOT / "docs" / "design" / "SDS.md").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("Safari 待补充", sds)
        self.assertNotIn("Safari 人工验收记录待补充", sds)
        self.assertIn("Safari 26.6 与 Edge 158 人工验收已完成", sds)
        self.assertIn("已关闭：Safari 17 与 Edge 120", sds)


class NoAutoDownloadTests(unittest.TestCase):
    def test_readme_promises_no_auto_download(self) -> None:
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("不会自动下载", readme)

    def test_model_loader_uses_local_files_only(self) -> None:
        source = (
            PROJECT_ROOT / "src" / "experiment" / "executor.py"
        ).read_text(encoding="utf-8")
        self.assertIn("local_files_only=True", source)

    def test_install_script_uses_lock_file(self) -> None:
        source = (PROJECT_ROOT / "scripts" / "install.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("LOCK_FILE", source)
        self.assertIn("requirements.lock", source)


if __name__ == "__main__":
    unittest.main()
