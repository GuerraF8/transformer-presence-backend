from __future__ import annotations

from datetime import datetime, timezone
import unittest

from inferencia_hub.hub.state import InferenceHubState


class GroundTruthMetricsTest(unittest.TestCase):
    def test_count_metrics_compare_prediction_to_ground_truth(self) -> None:
        hub = InferenceHubState()
        now = datetime.now(timezone.utc)
        hub.current_people_estimate = 1

        hub._apply_count_ground_truth_locked(
            timestamp=now,
            entity_id="sensor.number_of_people_home",
            room="",
            count=2,
        )
        hub.current_people_estimate = 2
        hub._apply_count_ground_truth_locked(
            timestamp=now,
            entity_id="sensor.number_of_people_home",
            room="",
            count=2,
            predicted_people=2,
        )

        metrics = hub._evaluation_metrics_locked()["ground_truth"]["count"]
        self.assertEqual(metrics["global_samples"], 2)
        self.assertEqual(metrics["count_accuracy"], 0.5)
        self.assertEqual(metrics["count_mae"], 0.5)
        self.assertEqual(hub.current_people_estimate, 2)

    def test_pet_confirmation_reports_false_positive(self) -> None:
        hub = InferenceHubState()
        hub.current_room = "hall"
        hub.current_active_rooms = ["hall"]
        hub.current_people_estimate = 1

        hub._record_confirmation_ground_truth_locked(
            timestamp=datetime.now(timezone.utc),
            entity_id="binary_sensor.hall_pet",
            state="on",
            training_role="pet_confirmation",
            room="hall",
        )

        metrics = hub._evaluation_metrics_locked()["ground_truth"]["presence"]
        self.assertEqual(metrics["pet_samples"], 1)
        self.assertEqual(metrics["pet_false_positive_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
