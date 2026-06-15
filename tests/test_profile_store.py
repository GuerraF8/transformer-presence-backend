from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from inferencia_hub.profile_store import (
    PresenceProfileStore,
    ProfileRevisionError,
    normalize_profile,
    profile_fingerprint,
)


def profile_payload(name: str = "Casa") -> dict:
    return {
        "name": name,
        "source": "manual",
        "rooms": [
            {
                "slug": "cocina",
                "name": "Cocina",
                "area_id": "kitchen",
                "area_name": "Cocina",
            },
            {"slug": "pasillo", "name": "Pasillo"},
        ],
        "areas": [
            {
                "area_id": "kitchen",
                "room_slug": "cocina",
                "name": "Cocina",
            }
        ],
        "assignments": [
            {
                "entity_id": "light.encimera",
                "room_slug": "cocina",
                "enabled": True,
                "sensor_type": "other",
                "area_id": "kitchen",
                "area_name": "Cocina",
            }
        ],
        "edges": [["cocina", "pasillo"]],
    }


class PresenceProfileStoreTest(unittest.TestCase):
    def test_existing_assignments_default_to_signal_role(self) -> None:
        profile = normalize_profile(
            {
                "name": "Casa",
                "rooms": [{"slug": "hall", "name": "Hall"}],
                "assignments": [
                    {
                        "entity_id": "binary_sensor.hall_motion",
                        "room_slug": "hall",
                        "sensor_type": "motion",
                    }
                ],
            }
        )
        self.assertEqual(
            profile["assignments"][0]["training_role"],
            "signal",
        )

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "presence_profiles.json"
        self.store = PresenceProfileStore(self.path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_crud_activation_and_restart(self) -> None:
        created = self.store.create(profile_payload(), activate=True)
        self.assertEqual(self.store.active()["id"], created["id"])
        self.assertTrue(self.path.exists())
        self.assertFalse(self.path.with_suffix(".json.tmp").exists())

        reopened = PresenceProfileStore(self.path)
        snapshot = reopened.load()
        self.assertEqual(snapshot["active_profile_id"], created["id"])
        self.assertEqual(snapshot["profiles"][0]["rooms"][0]["slug"], "cocina")

        updated = reopened.update(
            created["id"],
            {**profile_payload("Casa editada"), "edges": []},
            expected_revision=1,
        )
        self.assertEqual(updated["revision"], 2)
        self.assertEqual(updated["name"], "Casa editada")
        with self.assertRaises(ProfileRevisionError):
            reopened.update(
                created["id"],
                profile_payload("Edición obsoleta"),
                expected_revision=1,
            )

        deleted, was_active = reopened.delete(created["id"])
        self.assertEqual(deleted["id"], created["id"])
        self.assertTrue(was_active)
        self.assertIsNone(reopened.snapshot()["active_profile_id"])

    def test_fingerprint_changes_only_with_structure(self) -> None:
        base = profile_payload()
        renamed = profile_payload("Otro nombre")
        renamed["rooms"][0]["name"] = "Cocina principal"
        changed = profile_payload()
        changed["assignments"][0]["sensor_type"] = "occupancy"
        self.assertEqual(profile_fingerprint(base), profile_fingerprint(renamed))
        self.assertNotEqual(profile_fingerprint(base), profile_fingerprint(changed))

    def test_invalid_edges_and_assignments_are_discarded(self) -> None:
        payload = profile_payload()
        payload["edges"].extend([["cocina", "inexistente"], ["cocina", "cocina"]])
        payload["assignments"].append(
            {
                "entity_id": "sensor.sin_habitacion",
                "room_slug": "inexistente",
            }
        )
        created = self.store.create(payload)
        self.assertEqual(created["edges"], [["cocina", "pasillo"]])
        self.assertEqual(len(created["assignments"]), 1)
        stored = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(stored["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
