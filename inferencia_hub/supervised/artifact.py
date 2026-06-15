"""Carga del clasificador supervisado distribuido con el backend."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - depende de la imagen ML
    torch = None


def packaged_pet_filter_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "defaults"
        / "models"
        / "pet_filter"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def load_packaged_pet_filter(
    ai_model: Any,
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    if torch is None:
        return {
            "loaded": False,
            "reason": "PyTorch no disponible",
        }

    directory = Path(
        artifact_dir or packaged_pet_filter_dir()
    )
    metadata_path = directory / "metadata.json"
    checkpoint_path = directory / "pet_motion_transformer.pt"
    if not metadata_path.exists() or not checkpoint_path.exists():
        return {
            "loaded": False,
            "reason": "artefacto supervisado incluido no disponible",
        }

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )
    expected_hash = str(
        metadata.get("checkpoint_sha256") or ""
    )
    actual_hash = _sha256(checkpoint_path)
    if expected_hash and actual_hash != expected_hash:
        return {
            "loaded": False,
            "reason": "hash del checkpoint incluido no coincide",
        }

    from ..models.pet_filter import PetMotionTransformer

    info = dict(metadata.get("pet_filter_info") or {})
    context_length = int(
        metadata.get("context_length") or 28
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = PetMotionTransformer(
        input_size=int(info.get("input_size") or 12),
        context_length=context_length,
    ).to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    ai_model.pet_filter_model = model
    ai_model.pet_filter_device = device
    ai_model.pet_filter_threshold = float(
        metadata.get("threshold") or 0.0
    )
    ai_model.pet_filter_context_length = context_length
    ai_model.pet_filter_info = {
        **info,
        "enabled": True,
        "suppression_enabled": True,
        "source": "bundled",
        "pretrained": True,
        "artifact_id": metadata.get("artifact_id"),
        "checkpoint_sha256": actual_hash,
        "activation_policy": "operational_preference",
        "device": str(device),
    }
    return {
        "loaded": True,
        "artifact_id": metadata.get("artifact_id"),
        "checkpoint_path": str(checkpoint_path),
        "threshold": ai_model.pet_filter_threshold,
    }
