from __future__ import annotations

import unittest

from inferencia_hub.api import history, home_assistant, layout, presence, replay, training


class RouterModuleTest(unittest.TestCase):
    def test_domain_modules_declare_owned_paths(self) -> None:
        modules = (presence, history, home_assistant, layout, training, replay)
        declared = set()
        for module in modules:
            self.assertTrue(callable(module.build_router))
            self.assertTrue(module.PATHS)
            self.assertFalse(declared.intersection(module.PATHS))
            declared.update(module.PATHS)

        self.assertIn("/api/events", declared)
        self.assertIn("/api/history/events", declared)
        self.assertIn("/api/history/alerts", declared)
        self.assertIn("/api/ha_entities", declared)
        self.assertIn("/api/replay_status", declared)
        self.assertIn("/api/train_presence_supervised", declared)
        self.assertIn("/api/training/manifests", declared)
