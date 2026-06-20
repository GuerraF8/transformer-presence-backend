from __future__ import annotations

import unittest
from datetime import datetime, timezone

import numpy as np

from inferencia_hub.domain import PresenceFilterConfigInput, SensorEventInput
from inferencia_hub.hub_state import InferenceHubState
from inferencia_hub.version import BACKEND_VERSION


class HubStateContractTest(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_exposes_backend_version(self) -> None:
        snapshot = InferenceHubState().snapshot()
        self.assertEqual(snapshot["meta"]["backend_version"], BACKEND_VERSION)

    async def test_profile_supports_arbitrary_rooms_and_clears_state(self) -> None:
        hub = InferenceHubState()
        profile = {
            "id": "profile-1",
            "name": "Departamento",
            "revision": 3,
            "fingerprint": "abc",
            "rooms": [
                {"slug": "estudio", "name": "Estudio"},
                {"slug": "terraza", "name": "Terraza"},
            ],
            "assignments": [
                {
                    "entity_id": "light.escritorio",
                    "room_slug": "estudio",
                    "enabled": True,
                    "sensor_type": "other",
                    "status": "active",
                }
            ],
            "edges": [["estudio", "terraza"]],
        }

        hub.apply_profile(profile)
        snapshot = hub.snapshot()
        self.assertEqual(snapshot["profile"]["active_profile_id"], "profile-1")
        self.assertEqual(
            snapshot["profile"]["room_labels"]["estudio"],
            "Estudio",
        )
        self.assertEqual(
            snapshot["layout_reference"]["edges"],
            [{"a": "estudio", "b": "terraza"}],
        )
        self.assertEqual(
            snapshot["layout_reference"]["room_labels"]["terraza"],
            "Terraza",
        )
        self.assertIn(
            "light.escritorio",
            snapshot["real_sensor_config"]["enabled_entities"],
        )

        hub.clear_active_profile()
        snapshot = hub.snapshot()
        self.assertFalse(snapshot["profile"]["available"])
        self.assertEqual(snapshot["layout_reference"]["rooms"], [])

    async def test_profile_sensor_update_preserves_inferred_layout_state(self) -> None:
        hub = InferenceHubState()
        base_profile = {
            "id": "profile-1",
            "name": "Casa",
            "revision": 1,
            "fingerprint": "layout-1",
            "rooms": [
                {"slug": "cocina", "name": "Cocina"},
                {"slug": "pasillo", "name": "Pasillo"},
            ],
            "assignments": [
                {
                    "entity_id": "binary_sensor.cocina_motion",
                    "room_slug": "cocina",
                    "enabled": True,
                    "sensor_type": "motion",
                    "status": "active",
                }
            ],
            "edges": [["cocina", "pasillo"]],
        }
        hub.apply_profile(base_profile)
        hub.events.append({"entity_id": "binary_sensor.cocina_motion"})
        hub.edge_support[("cocina", "pasillo")] = 7
        hub.latest_touched_edge = ("cocina", "pasillo")
        learned_matrix = np.array(
            [[0.2, 0.8], [0.65, 0.35]],
            dtype=np.float32,
        )
        hub.ai_model.transition_matrix = learned_matrix.copy()
        layout_version = hub.reference_layout_version

        updated_profile = {
            **base_profile,
            "revision": 2,
            "assignments": [
                *base_profile["assignments"],
                {
                    "entity_id": "binary_sensor.pasillo_motion",
                    "room_slug": "pasillo",
                    "enabled": True,
                    "sensor_type": "motion",
                    "status": "active",
                },
            ],
        }
        hub.apply_profile(updated_profile)

        self.assertEqual(hub.reference_layout_version, layout_version)
        self.assertEqual(len(hub.events), 1)
        self.assertEqual(hub.edge_support[("cocina", "pasillo")], 7)
        self.assertEqual(hub.latest_touched_edge, ("cocina", "pasillo"))
        np.testing.assert_allclose(
            hub.ai_model.transition_matrix,
            learned_matrix,
        )
        self.assertIn(
            "binary_sensor.pasillo_motion",
            hub.real_sensor_assignments,
        )

        label_changed_profile = {
            **updated_profile,
            "revision": 3,
            "rooms": [
                {"slug": "cocina", "name": "Cocina principal"},
                {"slug": "pasillo", "name": "Pasillo norte"},
            ],
        }
        hub.apply_profile(label_changed_profile)
        self.assertEqual(hub.reference_layout_version, layout_version + 1)
        self.assertEqual(
            hub.active_profile_room_labels["cocina"],
            "Cocina principal",
        )
        self.assertEqual(len(hub.events), 1)
        self.assertEqual(hub.edge_support[("cocina", "pasillo")], 7)
        self.assertEqual(hub.latest_touched_edge, ("cocina", "pasillo"))
        np.testing.assert_allclose(
            hub.ai_model.transition_matrix,
            learned_matrix,
        )

        layout_changed_profile = {
            **label_changed_profile,
            "revision": 4,
            "edges": [],
        }
        hub.apply_profile(layout_changed_profile)
        self.assertEqual(hub.events, [])
        self.assertEqual(hub.edge_support[("cocina", "pasillo")], 0)
        np.testing.assert_allclose(
            hub.ai_model.transition_matrix,
            np.eye(2, dtype=np.float32),
        )

    async def test_event_processing_snapshot_callbacks_and_reset(self) -> None:
        hub = InferenceHubState()
        await hub.configure_presence_filter(
            PresenceFilterConfigInput(
                enabled=False,
                window_seconds=20,
                min_motion_events=2,
                min_distinct_rooms=1,
            )
        )
        persisted: list[tuple[SensorEventInput, dict, dict]] = []
        published: list[dict] = []

        async def persist(payload: SensorEventInput, event: dict, response: dict) -> None:
            persisted.append((payload, event, response))

        async def publish(payload: dict) -> None:
            published.append(payload)

        hub.event_sink = persist
        hub.snapshot_publisher = publish
        response = await hub.process_event(
            SensorEventInput(
                entity_id="binary_sensor.motion_kitchen",
                state="on",
                sensor_type="motion",
                room="kitchen",
                timestamp=datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc),
                source="sensor_simulator",
            )
        )

        self.assertEqual(response["presencia_inferida"], "Presente")
        self.assertEqual(response["habitacion"], "kitchen")
        self.assertEqual(response["habitacion_inferida_ia"], "kitchen")
        self.assertEqual(response["habitaciones_activas"], ["kitchen"])
        self.assertEqual(len(persisted), 1)
        self.assertEqual(len(published), 1)

        snapshot = hub.snapshot()
        self.assertTrue(snapshot["presence"]["inferred_presence"])
        self.assertEqual(snapshot["presence"]["current_room"], "kitchen")
        self.assertEqual(snapshot["events"][0]["entity_id"], "binary_sensor.motion_kitchen")

        await hub.broadcast_snapshot()
        self.assertEqual(published[-1]["kind"], "snapshot")
        self.assertEqual(published[-1]["sim_data"]["presence"]["current_room"], "kitchen")

        await hub.reset()
        reset_snapshot = hub.snapshot()
        self.assertEqual(reset_snapshot["events"], [])
        self.assertFalse(reset_snapshot["presence"]["inferred_presence"])
        self.assertIsNone(reset_snapshot["presence"]["current_room"])

    async def test_isolated_motion_filter_contract(self) -> None:
        hub = InferenceHubState()
        response = await hub.process_event(
            SensorEventInput(
                entity_id="binary_sensor.motion_living",
                state="on",
                sensor_type="motion",
                room="living",
                timestamp=datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc),
                source="sensor_simulator",
            )
        )

        self.assertEqual(response["presencia_inferida"], "Ausente")
        self.assertEqual(response["relacion_habitaciones"], "filtrado_ventana")
        self.assertFalse(response["filtro_presencia"]["accepted"])


if __name__ == "__main__":
    unittest.main()
