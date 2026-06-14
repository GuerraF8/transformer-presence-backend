from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from inferencia_hub.app_context import HAEntityCatalog
from inferencia_hub.hub_state import InferenceHubState
from inferencia_hub.profile_store import PresenceProfileStore
from inferencia_hub.runtime import profiles as profile_runtime


class ProfileReconciliationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.original_store = profile_runtime.profile_store
        self.original_catalog = profile_runtime.ha_entity_catalog
        self.original_hub = profile_runtime.hub_state
        profile_runtime.profile_store = PresenceProfileStore(
            Path(self.temporary.name) / "profiles.json"
        )
        profile_runtime.ha_entity_catalog = HAEntityCatalog()
        profile_runtime.hub_state = InferenceHubState()
        self.profile = profile_runtime.profile_store.create(
            {
                "name": "Casa",
                "source": "detected",
                "rooms": [
                    {
                        "slug": "office",
                        "name": "Oficina",
                        "area_id": "office",
                        "area_name": "Oficina",
                    }
                ],
                "areas": [
                    {
                        "area_id": "office",
                        "room_slug": "office",
                        "name": "Oficina",
                    }
                ],
                "assignments": [
                    {
                        "entity_id": "binary_sensor.old_name",
                        "room_slug": "office",
                        "sensor_type": "motion",
                        "area_id": "office",
                        "area_name": "Oficina",
                        "unique_id": "motion-1",
                        "platform": "test",
                    }
                ],
                "edges": [],
            },
            activate=True,
        )

    async def asyncTearDown(self) -> None:
        profile_runtime.profile_store = self.original_store
        profile_runtime.ha_entity_catalog = self.original_catalog
        profile_runtime.hub_state = self.original_hub
        self.temporary.cleanup()

    async def test_renames_entity_and_area_without_changing_room_slug(self) -> None:
        profile_runtime.ha_entity_catalog.replace(
            {
                "areas": [{"area_id": "office", "name": "Estudio"}],
                "entities": [
                    {
                        "entity_id": "binary_sensor.new_name",
                        "area_id": "office",
                        "area_name": "Estudio",
                        "unique_id": "motion-1",
                        "platform": "test",
                    }
                ],
            }
        )

        result = await profile_runtime.reconcile_profiles_with_catalog()
        updated = profile_runtime.profile_store.get(self.profile["id"])

        self.assertEqual(result["profiles_changed"], 1)
        self.assertEqual(updated["rooms"][0]["slug"], "office")
        self.assertEqual(updated["rooms"][0]["name"], "Estudio")
        self.assertEqual(
            updated["assignments"][0]["entity_id"],
            "binary_sensor.new_name",
        )
        self.assertTrue(updated["assignments"][0]["enabled"])

    async def test_moved_entity_is_disabled_with_warning(self) -> None:
        profile_runtime.ha_entity_catalog.replace(
            {
                "areas": [
                    {"area_id": "office", "name": "Oficina"},
                    {"area_id": "kitchen", "name": "Cocina"},
                ],
                "entities": [
                    {
                        "entity_id": "binary_sensor.old_name",
                        "area_id": "kitchen",
                        "area_name": "Cocina",
                        "unique_id": "motion-1",
                        "platform": "test",
                    }
                ],
            }
        )

        result = await profile_runtime.reconcile_profiles_with_catalog()
        assignment = profile_runtime.profile_store.get(
            self.profile["id"]
        )["assignments"][0]

        self.assertEqual(result["assignments_disabled"], 1)
        self.assertFalse(assignment["enabled"])
        self.assertEqual(assignment["status"], "moved")
        self.assertTrue(assignment["warning"])

    async def test_apply_profile_restores_loaded_model_presence_belief(self) -> None:
        class LoadedModel:
            def __init__(self) -> None:
                self.rooms: list[str] = []

            def load_state(self, _path: Path) -> dict[str, bool]:
                self.rooms = ["office", "kitchen"]
                return {"loaded": True}

        with (
            patch.object(
                profile_runtime,
                "_profile_model_status",
                return_value={"available": True, "compatible": True},
            ),
            patch.object(profile_runtime, "AIAdjacencyModel", LoadedModel),
        ):
            await profile_runtime._apply_profile(self.profile)

        self.assertTrue(
            profile_runtime.hub_state.active_profile_model_compatible
        )
        np.testing.assert_allclose(
            profile_runtime.hub_state.presence_belief,
            np.array([0.5, 0.5], dtype=np.float32),
        )


class LegacyProfileMigrationTest(unittest.TestCase):
    def test_legacy_sensor_config_is_migrated_once_and_activated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_path = root / "real_sensor_config.json"
            legacy_path.write_text(
                json.dumps(
                    {
                        "rooms": ["kitchen", "living"],
                        "assignments": [
                            {
                                "entity_id": "binary_sensor.kitchen",
                                "room": "kitchen",
                                "enabled": True,
                                "sensor_type": "motion",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            original_store = profile_runtime.profile_store
            profile_runtime.profile_store = PresenceProfileStore(
                root / "presence_profiles.json"
            )
            try:
                with patch.dict(
                    os.environ,
                    {"MODEL_STATE_DIR": str(root / "model_state")},
                ):
                    migrated = profile_runtime.initialize_profiles(legacy_path)
                    repeated = profile_runtime.initialize_profiles(legacy_path)
                self.assertIsNotNone(migrated)
                self.assertEqual(migrated["name"], "Configuracion migrada")
                self.assertEqual(
                    profile_runtime.profile_store.snapshot()[
                        "active_profile_id"
                    ],
                    migrated["id"],
                )
                self.assertEqual(repeated["id"], migrated["id"])
                self.assertTrue(legacy_path.exists())
                self.assertEqual(
                    len(profile_runtime.profile_store.list_profiles()),
                    1,
                )
            finally:
                profile_runtime.profile_store = original_store


if __name__ == "__main__":
    unittest.main()
