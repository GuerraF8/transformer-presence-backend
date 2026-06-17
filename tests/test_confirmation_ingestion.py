from __future__ import annotations

import unittest

from inferencia_hub.domain import SensorEventInput
from inferencia_hub.runtime import presence


class _ConfirmationStore:
    def __init__(self) -> None:
        self.records: list[dict] = []
        self.signals: list[dict] = []

    def record_confirmation(self, **values):
        self.records.append(values)
        return 17

    def record_signal(self, **values):
        self.signals.append(values)
        return 18


class ConfirmationIngestionTest(unittest.IsolatedAsyncioTestCase):
    async def test_confirmation_is_recorded_without_inference(self) -> None:
        original_store = presence.live_training_store
        original_assignments = presence.hub_state.real_sensor_assignments
        original_profile_id = presence.hub_state.active_profile_id
        original_revision = presence.hub_state.active_profile_revision
        original_fingerprint = presence.hub_state.active_profile_fingerprint
        original_mode = presence.hub_state.input_mode
        original_catalog = presence.ha_entity_catalog.as_dict()
        store = _ConfirmationStore()
        entity_id = "binary_sensor.foyer_person"
        try:
            presence.live_training_store = store
            presence.hub_state.active_profile_id = "profile-1"
            presence.hub_state.active_profile_revision = 3
            presence.hub_state.active_profile_fingerprint = "fingerprint"
            presence.hub_state.input_mode = "listen"
            presence.hub_state.real_sensor_assignments = {
                entity_id: {
                    "room": "foyer",
                    "enabled": True,
                    "sensor_type": "occupancy",
                    "training_role": "person_confirmation",
                }
            }
            presence.ha_entity_catalog.replace(
                {"entities": [{"entity_id": entity_id}]}
            )
            result = await presence.ingest_event(
                SensorEventInput(
                    entity_id=entity_id,
                    state="on",
                    sensor_type="occupancy",
                    room="foyer",
                    source="ha_state_change",
                )
            )
            self.assertEqual(result["status"], "recorded")
            self.assertEqual(result["confirmation_id"], 17)
            self.assertEqual(len(store.records), 1)
            self.assertEqual(
                store.records[0]["training_role"],
                "person_confirmation",
            )
        finally:
            presence.live_training_store = original_store
            presence.hub_state.real_sensor_assignments = original_assignments
            presence.hub_state.active_profile_id = original_profile_id
            presence.hub_state.active_profile_revision = original_revision
            presence.hub_state.active_profile_fingerprint = original_fingerprint
            presence.hub_state.input_mode = original_mode
            presence.ha_entity_catalog.replace(original_catalog)

    async def test_signal_is_saved_as_training_context(self) -> None:
        original_store = presence.live_training_store
        original_assignments = presence.hub_state.real_sensor_assignments
        original_profile_id = presence.hub_state.active_profile_id
        original_revision = presence.hub_state.active_profile_revision
        original_fingerprint = presence.hub_state.active_profile_fingerprint
        original_mode = presence.hub_state.input_mode
        original_catalog = presence.ha_entity_catalog.as_dict()
        original_process_event = presence.hub_state.process_event
        store = _ConfirmationStore()
        entity_id = "binary_sensor.foyer_motion"

        async def process_event(_payload):
            return {"status": "ok", "event": {}}

        try:
            presence.live_training_store = store
            presence.hub_state.active_profile_id = "profile-1"
            presence.hub_state.active_profile_revision = 3
            presence.hub_state.active_profile_fingerprint = "fingerprint"
            presence.hub_state.input_mode = "listen"
            presence.hub_state.real_sensor_assignments = {
                entity_id: {
                    "room": "foyer",
                    "enabled": True,
                    "sensor_type": "motion",
                    "training_role": "signal",
                }
            }
            presence.hub_state.process_event = process_event
            presence.ha_entity_catalog.replace(
                {"entities": [{"entity_id": entity_id}]}
            )
            result = await presence.ingest_event(
                SensorEventInput(
                    entity_id=entity_id,
                    state="on",
                    sensor_type="motion",
                    room="foyer",
                    source="ha_state_change",
                )
            )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(len(store.signals), 1)
            self.assertEqual(store.signals[0]["room"], "foyer")
        finally:
            presence.live_training_store = original_store
            presence.hub_state.real_sensor_assignments = original_assignments
            presence.hub_state.active_profile_id = original_profile_id
            presence.hub_state.active_profile_revision = original_revision
            presence.hub_state.active_profile_fingerprint = original_fingerprint
            presence.hub_state.input_mode = original_mode
            presence.hub_state.process_event = original_process_event
            presence.ha_entity_catalog.replace(original_catalog)

    async def test_people_count_confirmation_corrects_estimate(self) -> None:
        original_store = presence.live_training_store
        original_assignments = presence.hub_state.real_sensor_assignments
        original_profile_id = presence.hub_state.active_profile_id
        original_revision = presence.hub_state.active_profile_revision
        original_fingerprint = presence.hub_state.active_profile_fingerprint
        original_mode = presence.hub_state.input_mode
        original_catalog = presence.ha_entity_catalog.as_dict()
        original_estimate = presence.hub_state.current_people_estimate
        original_count_truth = presence.hub_state.people_count_ground_truth
        original_samples = list(presence.hub_state.ground_truth_samples)
        store = _ConfirmationStore()
        entity_id = "sensor.number_of_people_home"
        try:
            presence.live_training_store = store
            presence.hub_state.active_profile_id = "profile-1"
            presence.hub_state.active_profile_revision = 3
            presence.hub_state.active_profile_fingerprint = "fingerprint"
            presence.hub_state.input_mode = "listen"
            presence.hub_state.current_people_estimate = 1
            presence.hub_state.real_sensor_assignments = {
                entity_id: {
                    "room": "",
                    "enabled": True,
                    "sensor_type": "other",
                    "training_role": "people_count_confirmation",
                }
            }
            presence.ha_entity_catalog.replace(
                {"entities": [{"entity_id": entity_id}]}
            )
            result = await presence.ingest_event(
                SensorEventInput(
                    entity_id=entity_id,
                    state="2",
                    sensor_type="other",
                    source="ha_state_change",
                )
            )
            self.assertEqual(result["status"], "recorded")
            self.assertEqual(result["numeric_value"], 2.0)
            self.assertEqual(store.records[0]["numeric_value"], 2.0)
            self.assertEqual(presence.hub_state.current_people_estimate, 2)
            self.assertEqual(
                presence.hub_state.people_count_ground_truth["count"],
                2,
            )
        finally:
            presence.live_training_store = original_store
            presence.hub_state.real_sensor_assignments = original_assignments
            presence.hub_state.active_profile_id = original_profile_id
            presence.hub_state.active_profile_revision = original_revision
            presence.hub_state.active_profile_fingerprint = original_fingerprint
            presence.hub_state.input_mode = original_mode
            presence.hub_state.current_people_estimate = original_estimate
            presence.hub_state.people_count_ground_truth = original_count_truth
            presence.hub_state.ground_truth_samples.clear()
            presence.hub_state.ground_truth_samples.extend(original_samples)
            presence.ha_entity_catalog.replace(original_catalog)

    async def test_people_count_confirmation_rejects_non_numeric_state(self) -> None:
        original_assignments = presence.hub_state.real_sensor_assignments
        original_profile_id = presence.hub_state.active_profile_id
        original_revision = presence.hub_state.active_profile_revision
        original_fingerprint = presence.hub_state.active_profile_fingerprint
        original_mode = presence.hub_state.input_mode
        original_catalog = presence.ha_entity_catalog.as_dict()
        entity_id = "sensor.number_of_people_home"
        try:
            presence.hub_state.active_profile_id = "profile-1"
            presence.hub_state.active_profile_revision = 3
            presence.hub_state.active_profile_fingerprint = "fingerprint"
            presence.hub_state.input_mode = "listen"
            presence.hub_state.real_sensor_assignments = {
                entity_id: {
                    "room": "",
                    "enabled": True,
                    "sensor_type": "other",
                    "training_role": "people_count_confirmation",
                }
            }
            presence.ha_entity_catalog.replace(
                {"entities": [{"entity_id": entity_id}]}
            )
            result = await presence.ingest_event(
                SensorEventInput(
                    entity_id=entity_id,
                    state="unknown",
                    sensor_type="other",
                    source="ha_state_change",
                )
            )
            self.assertEqual(result["status"], "ignored")
            self.assertEqual(result["reason"], "count_confirmation_state_not_numeric")
        finally:
            presence.hub_state.real_sensor_assignments = original_assignments
            presence.hub_state.active_profile_id = original_profile_id
            presence.hub_state.active_profile_revision = original_revision
            presence.hub_state.active_profile_fingerprint = original_fingerprint
            presence.hub_state.input_mode = original_mode
            presence.ha_entity_catalog.replace(original_catalog)
