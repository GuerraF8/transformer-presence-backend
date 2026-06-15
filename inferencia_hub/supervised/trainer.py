"""Coordinación del entrenamiento supervisado de presencia."""

from __future__ import annotations

from datetime import datetime, timezone
import random
from typing import Any
from uuid import uuid4

import numpy as np

from .dataset import PreparedSupervisedDataset
from .evaluation import utc_iso
from .filter_training import train_pet_filter
from .occupancy_training import fine_tune_occupancy

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - depende de la imagen ML
    torch = None
    TORCH_AVAILABLE = False


class SupervisedPresenceTrainer:
    """Ajusta presencia y filtro usando confirmaciones separadas de las señales."""

    def __init__(
        self,
        *,
        epochs: int = 5,
        seed: int = 42,
        min_human_recall: float = 0.98,
    ) -> None:
        self.epochs = max(1, int(epochs))
        self.seed = int(seed)
        self.min_human_recall = max(
            0.0,
            min(1.0, float(min_human_recall)),
        )

    def train(
        self,
        ai_model: Any,
        dataset: PreparedSupervisedDataset,
        adjacency: dict[str, list[str]],
        *,
        synthetic_dataset: (
            tuple[Any, Any, Any, Any, Any] | None
        ) = None,
    ) -> dict[str, Any]:
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch no está disponible. "
                "Construye la imagen con INSTALL_ML=1."
            )

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        run_id = (
            datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%SZ"
            )
            + "-"
            + uuid4().hex[:8]
        )
        filter_result = train_pet_filter(
            ai_model,
            dataset,
            adjacency,
            epochs=self.epochs,
            min_human_recall=self.min_human_recall,
        )
        occupancy_result = fine_tune_occupancy(
            ai_model,
            dataset,
            epochs=self.epochs,
            synthetic_dataset=synthetic_dataset,
        )
        return {
            "run_id": run_id,
            "status": "ok",
            "created_at": utc_iso(),
            "manifest_id": dataset.manifest_id,
            "dataset_fingerprint": dataset.fingerprint,
            "dataset": dataset.summary(),
            "filter": filter_result,
            "occupancy": occupancy_result,
            "activation": {
                "automatic": True,
                "filter_suppression_enabled": bool(
                    filter_result.get(
                        "suppression_enabled"
                    )
                ),
            },
        }
