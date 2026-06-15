"""Inferencia del clasificador supervisado de movimiento."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from .dependencies import EventRecord
from ..supervised.dataset import filter_context_matrix

try:
    import torch
except Exception:  # pragma: no cover - depende de la imagen ML
    torch = None


class SupervisedFilterMixin:
    def load_packaged_pet_filter(
        self,
        artifact_dir: str | None = None,
    ) -> dict[str, Any]:
        from ..supervised.artifact import (
            load_packaged_pet_filter,
        )

        return load_packaged_pet_filter(
            self,
            artifact_dir,
        )

    def predict_human_motion(
        self,
        history_events: list[EventRecord],
        candidate_room: str,
        timestamp: datetime,
        adjacency: dict[str, list[str]],
    ) -> dict[str, Any] | None:
        if (
            torch is None
            or self.pet_filter_model is None
            or self.pet_filter_device is None
            or not self.pet_filter_info.get("enabled")
        ):
            return None
        candidate = EventRecord(
            timestamp=timestamp,
            entity_id="",
            state="on",
            sensor_type="motion",
            room=candidate_room,
        )
        context = list(history_events) + [candidate]
        matrix = filter_context_matrix(
            context,
            candidate_room,
            adjacency,
            self.pet_filter_context_length,
        )
        try:
            self.pet_filter_model.eval()
            with torch.no_grad():
                values = torch.tensor(
                    matrix[np.newaxis, ...],
                    dtype=torch.float32,
                    device=self.pet_filter_device,
                )
                probability = float(
                    torch.sigmoid(self.pet_filter_model(values))[0]
                    .detach()
                    .cpu()
                    .item()
                )
        except Exception:
            return None
        return {
            "human_probability": round(probability, 4),
            "threshold": round(float(self.pet_filter_threshold), 4),
            "suppression_enabled": bool(
                self.pet_filter_info.get("suppression_enabled")
            ),
            "accepted": (
                not self.pet_filter_info.get("suppression_enabled")
                or probability >= self.pet_filter_threshold
            ),
            "strategy": "supervised_transformer",
        }
