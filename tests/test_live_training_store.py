from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from inferencia_hub.live_training_store import LiveTrainingStore


class LiveTrainingStoreTest(unittest.TestCase):
    def test_persists_config_confirmations_and_runs(self) -> None:
        with TemporaryDirectory() as temporary:
            store = LiveTrainingStore(Path(temporary) / "history.sqlite3")
            store.initialize()
            config = store.config()
            self.assertTrue(config["enabled"])
            self.assertEqual(config["minimum_confirmations"], 500)

            updated = store.update_config(
                {
                    **config,
                    "minimum_confirmations": 20,
                    "minimum_person_confirmations": 5,
                    "minimum_pet_confirmations": 5,
                }
            )
            self.assertEqual(updated["minimum_confirmations"], 20)

            person_id = store.record_confirmation(
                timestamp="2026-06-01T10:00:00Z",
                entity_id="binary_sensor.camera_person",
                state="on",
                training_role="person_confirmation",
                room="foyer",
                profile_id="profile-1",
                profile_revision=1,
                profile_fingerprint="fingerprint",
            )
            pet_id = store.record_confirmation(
                timestamp="2026-06-01T10:01:00Z",
                entity_id="binary_sensor.camera_pet",
                state="on",
                training_role="pet_confirmation",
                room="foyer",
                profile_id="profile-1",
                profile_revision=1,
                profile_fingerprint="fingerprint",
            )
            counts = store.confirmation_counts(
                "profile-1",
                "fingerprint",
            )
            self.assertEqual(counts["person"], 1)
            self.assertEqual(counts["pet"], 1)
            self.assertEqual(counts["maximum_id"], pet_id)
            store.record_signal(
                timestamp="2026-06-01T09:59:55Z",
                entity_id="binary_sensor.foyer_motion",
                sensor_type="motion",
                room="foyer",
                state="on",
                profile_id="profile-1",
                profile_revision=1,
                profile_fingerprint="fingerprint",
            )
            signals = store.signal_events(
                "profile-1",
                "fingerprint",
                from_timestamp="2026-06-01T09:00:00Z",
                to_timestamp="2026-06-01T11:00:00Z",
            )
            self.assertEqual(len(signals), 1)
            self.assertEqual(signals[0]["entity_id"], "binary_sensor.foyer_motion")

            run_id = store.begin_run(
                "profile-1",
                "fingerprint",
                "scheduled",
            )
            result = store.finish_run(
                run_id,
                state="activated",
                message="ok",
                confirmation_cutoff_id=pet_id,
                activated_components=["occupancy"],
                metrics={"occupancy": {"candidate": {"f1": 0.9}}},
            )
            self.assertEqual(result["state"], "activated")
            self.assertEqual(result["activated_components"], ["occupancy"])
            self.assertEqual(
                store.confirmation_counts("profile-1", "fingerprint")["total"],
                0,
            )
            self.assertLess(person_id, pet_id)
