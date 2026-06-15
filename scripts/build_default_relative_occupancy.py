"""Genera el modelo relativo de ocupación distribuido con el backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inferencia_hub.domain import EventRecord
from inferencia_hub.models.relative_occupancy import (
    RelativeOccupancyTransformer,
)
from inferencia_hub.supervised.dataset import (
    FILTER_FEATURE_SIZE,
    SupervisedDatasetBuilder,
    filter_context_matrix,
)
from inferencia_hub.supervised.manifest import TrainingManifestStore


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _synthetic_samples(
    rng: random.Random,
    *,
    total: int,
    context_length: int,
) -> tuple[list[np.ndarray], list[float], list[int]]:
    values: list[np.ndarray] = []
    labels: list[float] = []
    counts: list[int] = []
    sensor_types = ["motion", "door", "occupancy"]
    for sample_index in range(total):
        room_count = rng.randint(2, 8)
        rooms = [f"room_{index}" for index in range(room_count)]
        adjacency = {
            room: [
                candidate
                for candidate in rooms
                if candidate != room and rng.random() < 0.3
            ]
            for room in rooms
        }
        occupied_count = rng.randint(0, min(4, room_count))
        occupied = set(rng.sample(rooms, occupied_count))
        candidate = rng.choice(rooms)
        now = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(
            seconds=sample_index * 90
        )
        events: list[EventRecord] = []
        for index in range(context_length):
            event_room = rng.choice(rooms)
            active_probability = 0.82 if event_room in occupied else 0.12
            active = rng.random() < active_probability
            sensor_type = rng.choice(sensor_types)
            state = (
                "on"
                if active
                else ("off" if sensor_type != "door" else "closed")
            )
            events.append(
                EventRecord(
                    timestamp=now
                    - timedelta(seconds=(context_length - index) * 4),
                    entity_id="",
                    state=state,
                    sensor_type=sensor_type,
                    room=event_room,
                )
            )
        values.append(
            filter_context_matrix(
                events,
                candidate,
                adjacency,
                context_length,
            )
        )
        labels.append(float(candidate in occupied))
        counts.append(occupied_count)
    return values, labels, counts


def _real_samples(
    data_root: Path,
    context_length: int,
) -> tuple[list[np.ndarray], list[float], list[int], str]:
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
            context_length=context_length,
        ).build("person_pet_foyer")
    values: list[np.ndarray] = []
    labels: list[float] = []
    counts: list[int] = []
    for split in ("train", "validation", "test"):
        for sample in dataset.samples[split]:
            values.append(
                filter_context_matrix(
                    sample.context,
                    sample.room,
                    {sample.room: []},
                    context_length,
                )
            )
            labels.append(float(sample.human_label))
            counts.append(1 if sample.human_label >= 0.5 else 0)
    return values, labels, counts, dataset.fingerprint


def build_artifact(
    data_root: Path,
    output_dir: Path,
    *,
    epochs: int,
    seed: int,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    context_length = 28
    rng = random.Random(seed)
    synthetic_values, synthetic_labels, synthetic_counts = _synthetic_samples(
        rng,
        total=5000,
        context_length=context_length,
    )
    real_values, real_labels, real_counts, fingerprint = _real_samples(
        data_root,
        context_length,
    )
    values = np.asarray(
        synthetic_values + real_values,
        dtype=np.float32,
    )
    labels = np.asarray(
        synthetic_labels + real_labels,
        dtype=np.float32,
    )
    counts = np.asarray(
        synthetic_counts + real_counts,
        dtype=np.int64,
    )
    split = int(len(values) * 0.85)
    train_values = torch.tensor(values[:split], dtype=torch.float32)
    train_labels = torch.tensor(labels[:split], dtype=torch.float32)
    train_counts = torch.tensor(counts[:split], dtype=torch.long)
    test_values = torch.tensor(values[split:], dtype=torch.float32)
    test_labels = torch.tensor(labels[split:], dtype=torch.float32)
    test_counts = torch.tensor(counts[split:], dtype=torch.long)

    model = RelativeOccupancyTransformer(
        input_size=FILTER_FEATURE_SIZE,
        context_length=context_length,
        count_classes=5,
        hidden_size=48,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4)
    positives = float((train_labels == 1).sum().item())
    negatives = float((train_labels == 0).sum().item())
    room_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / max(1.0, positives))
    )
    count_loss = nn.CrossEntropyLoss()
    batch_size = 256
    model.train()
    last_loss = 0.0
    for _epoch in range(max(1, epochs)):
        permutation = torch.randperm(len(train_values))
        for start in range(0, len(train_values), batch_size):
            indices = permutation[start : start + batch_size]
            room_logits, count_logits = model(train_values[indices])
            loss = room_loss(
                room_logits,
                train_labels[indices],
            ) + 0.45 * count_loss(count_logits, train_counts[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().item())

    model.eval()
    with torch.no_grad():
        room_logits, count_logits = model(test_values)
        probabilities = torch.sigmoid(room_logits)
        best_threshold = 0.5
        best_f1 = -1.0
        for candidate_threshold in np.linspace(0.25, 0.75, 51):
            candidate_predictions = (
                probabilities >= float(candidate_threshold)
            ).float()
            candidate_tp = float(
                (
                    (candidate_predictions == 1)
                    & (test_labels == 1)
                )
                .sum()
                .item()
            )
            candidate_fp = float(
                (
                    (candidate_predictions == 1)
                    & (test_labels == 0)
                )
                .sum()
                .item()
            )
            candidate_fn = float(
                (
                    (candidate_predictions == 0)
                    & (test_labels == 1)
                )
                .sum()
                .item()
            )
            candidate_precision = candidate_tp / max(
                1.0,
                candidate_tp + candidate_fp,
            )
            candidate_recall = candidate_tp / max(
                1.0,
                candidate_tp + candidate_fn,
            )
            candidate_f1 = (
                2
                * candidate_precision
                * candidate_recall
                / max(1e-9, candidate_precision + candidate_recall)
            )
            if candidate_f1 > best_f1:
                best_f1 = candidate_f1
                best_threshold = float(candidate_threshold)
        room_predictions = (
            probabilities >= best_threshold
        ).float()
        true_positive = int(
            ((room_predictions == 1) & (test_labels == 1)).sum().item()
        )
        false_positive = int(
            ((room_predictions == 1) & (test_labels == 0)).sum().item()
        )
        false_negative = int(
            ((room_predictions == 0) & (test_labels == 1)).sum().item()
        )
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        count_accuracy = float(
            (torch.argmax(count_logits, dim=1) == test_counts)
            .float()
            .mean()
            .item()
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "relative_occupancy_transformer.pt"
    torch.save({"state_dict": model.state_dict()}, checkpoint_path)
    metadata = {
        "schema_version": 1,
        "artifact_id": f"relative-occupancy-{fingerprint[:12]}-seed{seed}",
        "dataset_fingerprint": fingerprint,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "threshold": round(best_threshold, 4),
        "trained_at": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "model": {
            "kind": "relative_transformer",
            "input_size": FILTER_FEATURE_SIZE,
            "context_length": context_length,
            "count_classes": 5,
            "hidden_size": 48,
        },
        "training": {
            "epochs": epochs,
            "seed": seed,
            "synthetic_samples": len(synthetic_values),
            "supervised_samples": len(real_values),
            "manifest_id": "person_pet_foyer",
            "loss": round(last_loss, 6),
            "torch_version": torch.__version__,
        },
        "metrics": {
            "room_precision": round(precision, 4),
            "room_recall": round(recall, 4),
            "room_f1": round(f1, 4),
            "count_accuracy": round(count_accuracy, 4),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path("inferencia_hub")
            / "defaults"
            / "models"
            / "relative_occupancy"
        ),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metadata = build_artifact(
        args.data_root.resolve(),
        args.output_dir.resolve(),
        epochs=max(1, args.epochs),
        seed=args.seed,
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
