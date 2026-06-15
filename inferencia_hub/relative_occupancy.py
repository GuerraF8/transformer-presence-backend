"""Carga, inferencia y metadatos del modelo relativo incluido."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .domain import EventRecord
from .supervised.dataset import FILTER_FEATURE_SIZE, filter_context_matrix

try:
    import torch
except Exception:  # pragma: no cover - depende de la imagen ML
    torch = None


def packaged_relative_occupancy_dir() -> Path:
    return (
        Path(__file__).resolve().parent
        / "defaults"
        / "models"
        / "relative_occupancy"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_packaged_relative_occupancy(
    ai_model: Any,
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    if torch is None:
        ai_model.relative_occupancy_info = {
            "enabled": False,
            "source": "rules",
            "reason": "PyTorch no disponible",
        }
        return {"loaded": False, "reason": "PyTorch no disponible"}

    directory = Path(artifact_dir or packaged_relative_occupancy_dir())
    metadata_path = directory / "metadata.json"
    checkpoint_path = directory / "relative_occupancy_transformer.pt"
    if not metadata_path.exists() or not checkpoint_path.exists():
        ai_model.relative_occupancy_info = {
            "enabled": False,
            "source": "rules",
            "reason": "modelo relativo incluido no disponible",
        }
        return {
            "loaded": False,
            "reason": "modelo relativo incluido no disponible",
        }

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual_hash = sha256_file(checkpoint_path)
    if metadata.get("checkpoint_sha256") not in {"", None, actual_hash}:
        return {"loaded": False, "reason": "hash del checkpoint no coincide"}

    from .models.relative_occupancy import RelativeOccupancyTransformer

    model_config = dict(metadata.get("model") or {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RelativeOccupancyTransformer(
        input_size=int(model_config.get("input_size") or FILTER_FEATURE_SIZE),
        context_length=int(model_config.get("context_length") or 28),
        count_classes=int(model_config.get("count_classes") or 5),
        hidden_size=int(model_config.get("hidden_size") or 48),
    ).to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    ai_model.relative_occupancy_model = model
    ai_model.relative_occupancy_device = device
    ai_model.relative_occupancy_context_length = model.context_length
    ai_model.relative_occupancy_threshold = float(
        metadata.get("threshold") or 0.5
    )
    ai_model.relative_occupancy_info = {
        **metadata,
        "enabled": True,
        "source": "bundled",
        "pretrained": True,
        "checkpoint_sha256": actual_hash,
        "device": str(device),
    }
    return {
        "loaded": True,
        "artifact_id": metadata.get("artifact_id"),
        "checkpoint_path": str(checkpoint_path),
    }


def relative_occupancy_prediction(
    ai_model: Any,
    history_events: list[EventRecord],
    rooms: list[str],
    adjacency: dict[str, list[str]],
) -> dict[str, Any] | None:
    if (
        torch is None
        or ai_model.relative_occupancy_model is None
        or ai_model.relative_occupancy_device is None
        or not ai_model.relative_occupancy_info.get("enabled")
        or not rooms
    ):
        return None
    context_length = int(ai_model.relative_occupancy_context_length or 28)
    if len(history_events) < max(4, context_length // 4):
        return None
    matrices = np.stack(
        [
            filter_context_matrix(
                history_events,
                room,
                adjacency,
                context_length,
            )
            for room in rooms
        ]
    )
    model = ai_model.relative_occupancy_model
    model.eval()
    with torch.no_grad():
        room_logits, count_logits = model(
            torch.tensor(
                matrices,
                dtype=torch.float32,
                device=ai_model.relative_occupancy_device,
            )
        )
        room_probabilities = (
            torch.sigmoid(room_logits).detach().cpu().numpy()
        )
        count_probabilities = (
            torch.softmax(count_logits, dim=1)
            .mean(dim=0)
            .detach()
            .cpu()
            .numpy()
        )
    threshold = float(ai_model.relative_occupancy_threshold or 0.5)
    predicted_count = int(np.argmax(count_probabilities))
    order = list(np.argsort(-room_probabilities))
    selected = [
        rooms[int(index)]
        for index in order[: max(1, predicted_count)]
        if float(room_probabilities[int(index)]) >= threshold
    ]
    if predicted_count == 0:
        selected = [
            rooms[int(index)]
            for index in order
            if float(room_probabilities[int(index)]) >= max(0.6, threshold)
        ]
    return {
        "rooms": selected,
        "people_count": predicted_count,
        "confidence": round(float(np.max(room_probabilities)), 4),
        "room_probs": {
            room: round(float(room_probabilities[index]), 4)
            for index, room in enumerate(rooms)
        },
        "count_probs": {
            str(index): round(float(value), 4)
            for index, value in enumerate(count_probabilities)
        },
        "model_kind": "relative_transformer",
        "threshold": round(threshold, 4),
    }
