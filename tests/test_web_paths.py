from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from inferencia_hub.web_paths import PACKAGE_WEB_DIR, resolve_web_dir


class WebPathTest(unittest.TestCase):
    def test_uses_existing_configured_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(resolve_web_dir(directory), Path(directory).resolve())

    def test_falls_back_to_packaged_assets_for_legacy_path(self) -> None:
        self.assertEqual(
            resolve_web_dir("/app/web/path-that-does-not-exist"),
            PACKAGE_WEB_DIR.resolve(),
        )
        self.assertTrue(resolve_web_dir().is_dir())
