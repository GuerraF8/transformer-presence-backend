from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from inferencia_hub.ai.model import AIAdjacencyModel
from inferencia_hub.ai import persistence


class DummyNetwork:
    def __init__(self, *args, **kwargs):
        self.loaded = None

    def state_dict(self):
        return {"weight": np.asarray([1.0])}

    def to(self, _device):
        return self

    def load_state_dict(self, state):
        self.loaded = state

    def eval(self):
        return self


def test_save_and_load_optional_transformer_artifacts(tmp_path):
    model = AIAdjacencyModel()
    model.ready = True
    model.rooms = ["kitchen"]
    model.room_to_idx = {"kitchen": 0}
    model.transition_matrix = np.eye(1, dtype=np.float32)
    model.adjacency_neighbors = {"kitchen": []}
    model.transformer_model = DummyNetwork()
    model.occupancy_transformer_model = DummyNetwork()
    model.occupancy_transformer_rooms = ["kitchen"]
    model.occupancy_transformer_count_classes = 2
    model.pet_filter_model = DummyNetwork()
    model.relative_occupancy_model = DummyNetwork()

    def save(payload, path):
        path.write_bytes(b"checkpoint")

    fake_torch = SimpleNamespace(
        save=save,
        load=lambda *_args, **_kwargs: {"state_dict": {"weight": np.asarray([1.0])}},
        device=lambda name: name,
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    with (
        patch.object(persistence, "HF_AVAILABLE", True),
        patch.object(persistence, "TORCH_AVAILABLE", True),
        patch.object(persistence, "torch", fake_torch),
    ):
        saved = model.save_state(tmp_path)
    assert all(saved[key] for key in (
        "transformer_path", "occupancy_transformer_path", "pet_filter_path", "relative_occupancy_path"
    ))

    # Los modelos opcionales pesados se prueban con dobles; el contrato de
    # persistencia (rutas, estado y restauracion) permanece real.
    model.pet_filter_info = {}
    model.relative_occupancy_info = {}
    with (
        patch.object(persistence, "HF_AVAILABLE", True),
        patch.object(persistence, "TORCH_AVAILABLE", False),
        patch.object(persistence, "torch", fake_torch),
        patch.object(persistence, "NextRoomTransformer", DummyNetwork),
        patch.object(persistence, "OccupancyTransformer", DummyNetwork),
    ):
        loaded = model.load_state(tmp_path)
    assert loaded["loaded"] is True
    assert set(loaded["models"]) == {"next_room_transformer", "occupancy_transformer"}
