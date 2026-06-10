from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from inferencia_hub.ai import AIAdjacencyModel
from inferencia_hub.ai.graph import GraphMixin
from inferencia_hub.ai.occupancy import OccupancyMixin
from inferencia_hub.ai.persistence import PersistenceMixin
from inferencia_hub.ai.simulation import SimulationMixin
from inferencia_hub.ai.training import TrainingMixin
from inferencia_hub.ai.transitions import TransitionsMixin
from inferencia_hub.ai_model import AIAdjacencyModel as ExportedModel
from inferencia_hub.application import app, create_app
from inferencia_hub.hub import InferenceHubState
from inferencia_hub.hub.events import EventsMixin
from inferencia_hub.hub.filtering import FilteringMixin
from inferencia_hub.hub.inference import InferenceMixin
from inferencia_hub.hub.layout import LayoutMixin
from inferencia_hub.hub.metrics import MetricsMixin
from inferencia_hub.hub.sensors import SensorsMixin
from inferencia_hub.hub.snapshot import SnapshotMixin
from inferencia_hub.hub_state import InferenceHubState as ExportedHubState
from inferencia_hub.runtime import HANDLERS


class ComponentStructureTest(unittest.TestCase):
    def test_exported_ai_model_includes_required_capabilities(self) -> None:
        self.assertIs(ExportedModel, AIAdjacencyModel)
        for component in (
            PersistenceMixin,
            TransitionsMixin,
            SimulationMixin,
            OccupancyMixin,
            GraphMixin,
            TrainingMixin,
        ):
            self.assertTrue(issubclass(AIAdjacencyModel, component))

    def test_model_state_round_trip_restores_model_data(self) -> None:
        model = AIAdjacencyModel()
        model.ready = True
        model.rooms = ["living", "kitchen"]
        model.room_to_idx = {"living": 0, "kitchen": 1}
        model.transition_matrix = np.array([[0.25, 0.75], [0.5, 0.5]], dtype=np.float32)
        model.adjacency_neighbors = {
            "living": ["kitchen"],
            "kitchen": ["living"],
        }
        model.adjacency_edges = [{"source": "living", "target": "kitchen"}]

        with tempfile.TemporaryDirectory() as directory:
            saved = model.save_state(directory)
            restored = AIAdjacencyModel()
            loaded = restored.load_state(directory)

            self.assertTrue(Path(saved["core_path"]).exists())
            self.assertTrue(loaded["loaded"])
            self.assertEqual(restored.rooms, model.rooms)
            np.testing.assert_allclose(restored.transition_matrix, model.transition_matrix)
            self.assertTrue(restored.are_adjacent("living", "kitchen"))

    def test_application_uses_registered_runtime_handlers(self) -> None:
        self.assertIs(create_app(), app)
        route_paths = {route.path for route in app.routes}
        self.assertIn("/api/events", route_paths)
        self.assertIn("/api/replay_csv", route_paths)
        self.assertIn("/api/train_model", route_paths)
        self.assertIn("/presencia", route_paths)
        self.assertIn("ingest_event", HANDLERS)
        self.assertIn("replay_csv", HANDLERS)
        self.assertIn("train_model", HANDLERS)

    def test_exported_hub_state_includes_required_operations(self) -> None:
        self.assertIs(ExportedHubState, InferenceHubState)
        for component in (
            LayoutMixin,
            SensorsMixin,
            MetricsMixin,
            FilteringMixin,
            InferenceMixin,
            EventsMixin,
            SnapshotMixin,
        ):
            self.assertTrue(issubclass(InferenceHubState, component))


if __name__ == "__main__":
    unittest.main()
