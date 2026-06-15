from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from inferencia_hub.ai import AIAdjacencyModel
from inferencia_hub.ai.dependencies import TORCH_AVAILABLE
from inferencia_hub.domain import EventRecord
from inferencia_hub.hub import InferenceHubState


class RelativeOccupancyArtifactTest(unittest.TestCase):
    def test_profile_initializes_neutral_transition_matrix(self) -> None:
        hub = InferenceHubState()
        hub.apply_profile(
            {
                "id": "profile",
                "name": "Casa",
                "revision": 1,
                "fingerprint": "fingerprint",
                "rooms": [
                    {"slug": "atelier", "name": "Atelier"},
                    {"slug": "terraza", "name": "Terraza"},
                ],
                "assignments": [],
                "edges": [["atelier", "terraza"]],
            }
        )
        self.assertEqual(hub.ai_model.transition_matrix.shape, (2, 2))
        self.assertAlmostEqual(
            float(hub.ai_model.transition_matrix.trace()),
            2.0,
        )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch no instalado")
    def test_bundled_model_accepts_arbitrary_room_names_and_layout(self) -> None:
        model = AIAdjacencyModel()
        loaded = model.load_packaged_relative_occupancy()
        self.assertTrue(loaded["loaded"])
        model.rooms = ["atelier", "garage_norte", "terraza"]
        model.adjacency_neighbors = {
            "atelier": ["garage_norte"],
            "garage_norte": ["atelier", "terraza"],
            "terraza": ["garage_norte"],
        }
        now = datetime(2026, 6, 1, tzinfo=timezone.utc)
        events = [
            EventRecord(
                timestamp=now + timedelta(seconds=index * 4),
                entity_id=f"binary_sensor.custom_{index}",
                state="on",
                sensor_type="motion",
                room="atelier",
            )
            for index in range(8)
        ]
        prediction = model.predict_occupancy_state(
            events,
            events[-1].timestamp,
        )
        self.assertIsNotNone(prediction)
        self.assertEqual(
            set(prediction["room_probs"]),
            {"atelier", "garage_norte", "terraza"},
        )
        self.assertEqual(prediction["model_kind"], "relative_transformer")
