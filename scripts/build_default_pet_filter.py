"""Genera el filtro supervisado distribuido con el backend."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from inferencia_hub.ai import AIAdjacencyModel
from inferencia_hub.domain import (
    REAL_HOME_LAYOUT_EDGES,
    to_adjacency,
)
from inferencia_hub.supervised.dataset import (
    SupervisedDatasetBuilder,
)
from inferencia_hub.supervised.filter_training import (
    train_pet_filter,
)
from inferencia_hub.supervised.manifest import (
    TrainingManifestStore,
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


def build_artifact(
    data_root: Path,
    output_dir: Path,
    *,
    epochs: int,
    seed: int,
    min_human_recall: float,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    defaults_dir = (
        Path(__file__).resolve().parents[1]
        / "inferencia_hub"
        / "defaults"
        / "training_manifests"
    )
    with tempfile.TemporaryDirectory() as temporary:
        store = TrainingManifestStore(
            manifest_dir=Path(temporary) / "manifests",
            dataset_root=data_root,
            defaults_dir=defaults_dir,
        )
        dataset = SupervisedDatasetBuilder(
            store,
            context_length=28,
        ).build("person_pet_foyer")

    rooms = sorted(
        {
            room
            for edge in REAL_HOME_LAYOUT_EDGES
            for room in edge
        }
    )
    adjacency = to_adjacency(
        rooms,
        REAL_HOME_LAYOUT_EDGES,
    )
    model = AIAdjacencyModel()
    result = train_pet_filter(
        model,
        dataset,
        adjacency,
        epochs=epochs,
        min_human_recall=min_human_recall,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        output_dir / "pet_motion_transformer.pt"
    )
    torch.save(
        {"state_dict": model.pet_filter_model.state_dict()},
        checkpoint_path,
    )
    metadata = {
        "schema_version": 1,
        "artifact_id": (
            f"person_pet_foyer-"
            f"{dataset.fingerprint[:12]}-seed{seed}"
        ),
        "manifest_id": dataset.manifest_id,
        "dataset_fingerprint": dataset.fingerprint,
        "threshold": model.pet_filter_threshold,
        "context_length": model.pet_filter_context_length,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "training": {
            "epochs": epochs,
            "seed": seed,
            "min_human_recall": min_human_recall,
            "torch_version": torch.__version__,
        },
        "pet_filter_info": {
            **model.pet_filter_info,
            "source": "bundled",
            "pretrained": True,
            "suppression_enabled": True,
        },
        "result": result,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path("inferencia_hub")
            / "defaults"
            / "models"
            / "pet_filter"
        ),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-human-recall",
        type=float,
        default=0.98,
    )
    args = parser.parse_args()
    metadata = build_artifact(
        args.data_root.resolve(),
        args.output_dir.resolve(),
        epochs=max(1, args.epochs),
        seed=args.seed,
        min_human_recall=args.min_human_recall,
    )
    print(
        json.dumps(
            {
                "artifact_id": metadata["artifact_id"],
                "threshold": metadata["threshold"],
                "checkpoint_sha256": metadata[
                    "checkpoint_sha256"
                ],
                "test": metadata[
                    "pet_filter_info"
                ]["test"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
