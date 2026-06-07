from __future__ import annotations

import unittest

from inferencia_hub.presence_contract import build_presence_snapshot


class PresenceSnapshotContractTest(unittest.TestCase):
    def test_empty_snapshot_exposes_automation_fields(self) -> None:
        presence = build_presence_snapshot(
            current_room=None,
            active_rooms=[],
            people_estimate=0,
            latest_event=None,
            occupancy_ground_truth_rooms=[],
            live_sensor_rooms=[],
        )

        self.assertIs(presence["inferred_presence"], False)
        self.assertEqual(presence["people_estimate"], 0)
        self.assertIsNone(presence["confidence"])
        self.assertIsNone(presence["updated_at"])

    def test_snapshot_uses_latest_inference_values(self) -> None:
        presence = build_presence_snapshot(
            current_room="kitchen",
            active_rooms=["kitchen"],
            people_estimate=2,
            latest_event={
                "timestamp": "2026-06-06T12:00:00+00:00",
                "presence_confidence": 0.91,
            },
            occupancy_ground_truth_rooms=["kitchen"],
            live_sensor_rooms=["kitchen"],
        )

        self.assertIs(presence["inferred_presence"], True)
        self.assertEqual(presence["people_estimate"], 2)
        self.assertEqual(presence["confidence"], 0.91)
        self.assertEqual(presence["updated_at"], "2026-06-06T12:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
