"""T-009 设置模型与原子读写测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from resources.settings import (  # noqa: E402
    NetworkSettings,
    Settings,
    SettingsInvalidError,
    SettingsReadError,
    SettingsStore,
    SettingsUpdate,
    SettingsWriteError,
)


class SettingsStoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.path = self.root / "settings.json"
        self.store = SettingsStore(self.path)


class LoadTests(SettingsStoreTestCase):
    def test_missing_file_returns_defaults_without_writing(self) -> None:
        settings = self.store.load()
        self.assertEqual(settings.schema_version, 1)
        self.assertIsNone(settings.model_root)
        self.assertIsNone(settings.duretrieval_root)
        self.assertEqual(settings.device, "auto")
        self.assertFalse(settings.network.lan_enabled)
        self.assertEqual(settings.network.local_port, 8765)
        self.assertEqual(settings.network.lan_port, 8766)
        self.assertFalse(self.path.exists())

    def test_roundtrip_normalizes_roots(self) -> None:
        update = SettingsUpdate(
            model_root=str(self.root / "models"),
            duretrieval_root=str(self.root / "duretrieval"),
            device="cpu",
        )
        saved = self.store.save_update(update)
        self.assertTrue(Path(saved.model_root).is_absolute())
        loaded = self.store.load()
        self.assertEqual(loaded.model_root, saved.model_root)
        self.assertEqual(loaded.device, "cpu")

    def test_unknown_field_rejected(self) -> None:
        self.path.write_text(
            json.dumps({"schema_version": 1, "unknown": True}), encoding="utf-8"
        )
        with self.assertRaises(SettingsInvalidError):
            self.store.load()

    def test_invalid_port_and_device_rejected(self) -> None:
        self.path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "network": {"local_port": 9000, "lan_port": 9000},
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaises(SettingsInvalidError):
            self.store.load()

        self.path.write_text(
            json.dumps({"schema_version": 1, "device": "tpu"}), encoding="utf-8"
        )
        with self.assertRaises(SettingsInvalidError):
            self.store.load()

    def test_corrupt_file_reports_error_without_overwrite(self) -> None:
        self.path.write_text("{bad json", encoding="utf-8")
        with self.assertRaises(SettingsReadError):
            self.store.load()
        self.assertEqual(self.path.read_text(encoding="utf-8"), "{bad json")

    def test_non_object_and_wrong_version(self) -> None:
        self.path.write_text("[]", encoding="utf-8")
        with self.assertRaises(SettingsReadError):
            self.store.load()
        self.path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
        with self.assertRaises(SettingsInvalidError):
            self.store.load()

    def test_invalid_utf8_reports_error(self) -> None:
        self.path.write_bytes(b"\xff\xfe")
        with self.assertRaises(SettingsReadError):
            self.store.load()

    def test_deep_json_reports_error(self) -> None:
        self.path.write_text("[" * 2000 + "]" * 2000, encoding="utf-8")
        with self.assertRaises(SettingsReadError):
            self.store.load()

    def test_duplicate_key_reports_error(self) -> None:
        self.path.write_text(
            '{"schema_version": 1, "device": "cpu", "device": "mps"}',
            encoding="utf-8",
        )
        with self.assertRaises(SettingsReadError):
            self.store.load()

    def test_non_standard_constant_reports_error(self) -> None:
        self.path.write_text(
            '{"schema_version": 1, "device": NaN}', encoding="utf-8"
        )
        with self.assertRaises(SettingsReadError):
            self.store.load()

    def test_invalid_path_type(self) -> None:
        self.path.write_text(
            json.dumps({"schema_version": 1, "model_root": 123}),
            encoding="utf-8",
        )
        with self.assertRaises(SettingsInvalidError):
            self.store.load()


class SaveTests(SettingsStoreTestCase):
    def test_save_keeps_network_fields(self) -> None:
        self.store.save(
            Settings(
                network=NetworkSettings(
                    lan_enabled=True, local_port=9000, lan_port=9001
                )
            )
        )
        update = SettingsUpdate(
            model_root=None, duretrieval_root=None, device="mps"
        )
        saved = self.store.save_update(update)
        self.assertEqual(saved.network.lan_enabled, True)
        self.assertEqual(saved.network.local_port, 9000)
        self.assertEqual(saved.network.lan_port, 9001)
        self.assertEqual(saved.device, "mps")

    def test_atomic_write_failure_keeps_original(self) -> None:
        self.store.save(Settings(device="cpu"))
        before = self.path.read_text(encoding="utf-8")
        with mock.patch(
            "resources.settings.os.replace",
            side_effect=OSError("boom"),
        ):
            with self.assertRaises(SettingsWriteError):
                self.store.save(Settings(device="mps"))
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)
        leftovers = [
            child.name
            for child in self.root.iterdir()
            if child.name.startswith(".settings.json.tmp-")
        ]
        self.assertEqual(leftovers, [])

    def test_parent_mkdir_failure_raises_write_error(self) -> None:
        with mock.patch(
            "resources.settings.Path.mkdir", side_effect=OSError("denied")
        ):
            with self.assertRaises(SettingsWriteError):
                self.store.save(Settings(device="cpu"))

    def test_update_requires_all_fields(self) -> None:
        with self.assertRaises(Exception):
            SettingsUpdate.model_validate({"device": "cpu"})
        with self.assertRaises(Exception):
            SettingsUpdate.model_validate(
                {"model_root": None, "duretrieval_root": None, "device": "tpu"}
            )

    def test_empty_path_rejected(self) -> None:
        with self.assertRaises(SettingsInvalidError):
            self.store.save_update(
                SettingsUpdate(
                    model_root="   ",
                    duretrieval_root=None,
                    device="auto",
                )
            )

    def test_nul_and_control_char_paths_rejected(self) -> None:
        for path in ("/tmp/bad\u0000", "/tmp/bad\u0001", "/tmp/bad\nx"):
            with self.subTest(path=repr(path)):
                with self.assertRaises(SettingsInvalidError):
                    self.store.save_update(
                        SettingsUpdate(
                            model_root=path,
                            duretrieval_root=None,
                            device="auto",
                        )
                    )

    def test_unresolvable_path_rejected(self) -> None:
        with mock.patch(
            "resources.settings.Path.resolve",
            side_effect=RuntimeError("boom"),
        ):
            with self.assertRaises(SettingsInvalidError):
                self.store.save_update(
                    SettingsUpdate(
                        model_root="/tmp/models",
                        duretrieval_root=None,
                        device="auto",
                    )
                )


class FingerprintTests(SettingsStoreTestCase):
    def test_fingerprint_changes_after_save(self) -> None:
        self.assertEqual(self.store.fingerprint(), "missing")
        self.store.save(Settings(device="cpu"))
        first = self.store.fingerprint()
        self.assertNotEqual(first, "missing")
        self.store.save(Settings(device="mps"))
        self.assertNotEqual(first, self.store.fingerprint())


if __name__ == "__main__":
    unittest.main()
