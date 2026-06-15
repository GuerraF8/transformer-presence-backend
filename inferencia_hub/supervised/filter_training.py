"""Entrenamiento del clasificador de origen del movimiento."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..domain import is_activation
from .dataset import (
    FILTER_FEATURE_SIZE,
    PreparedSupervisedDataset,
    filter_context_matrix,
)
from .evaluation import (
    binary_metrics,
    filter_metrics_by_period,
    pet_suppression_rate,
    temporal_baseline,
    utc_iso,
)

try:
    import torch
    import torch.nn as nn

    from ..models.pet_filter import PetMotionTransformer
except Exception:  # pragma: no cover - depende de la imagen ML
    torch = None
    nn = None
    PetMotionTransformer = None


def _filter_arrays(
    dataset: PreparedSupervisedDataset,
    split: str,
    adjacency: dict[str, list[str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Any]]:
    samples = [
        sample
        for sample in dataset.samples[split]
        if sample.context
        and sample.context[-1].sensor_type == "motion"
        and is_activation(
            sample.context[-1].sensor_type,
            sample.context[-1].state,
        )
    ]
    if not samples:
        return (
            np.zeros(
                (
                    0,
                    dataset.context_length,
                    FILTER_FEATURE_SIZE,
                ),
                dtype=np.float32,
            ),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            [],
        )
    values = np.stack(
        [
            filter_context_matrix(
                sample.context,
                sample.room,
                adjacency,
                dataset.context_length,
            )
            for sample in samples
        ]
    )
    labels = np.asarray(
        [sample.human_label for sample in samples],
        dtype=np.float32,
    )
    weights = np.asarray(
        [sample.sample_weight for sample in samples],
        dtype=np.float32,
    )
    return values, labels, weights, samples


def _probabilities(
    model: Any,
    values: np.ndarray,
    device: Any,
) -> np.ndarray:
    if len(values) == 0:
        return np.zeros((0,), dtype=np.float32)
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), 512):
            batch = torch.tensor(
                values[start : start + 512],
                dtype=torch.float32,
                device=device,
            )
            output.append(
                torch.sigmoid(model(batch))
                .detach()
                .cpu()
                .numpy()
            )
    return np.concatenate(output).astype(np.float32)


def _select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    min_human_recall: float,
) -> tuple[float, bool, dict[str, Any]]:
    candidates: list[
        tuple[float, float, dict[str, Any]]
    ] = []
    for threshold in np.linspace(0.01, 0.99, 99):
        predictions = (
            probabilities >= threshold
        ).astype(np.float32)
        metrics = binary_metrics(labels, predictions)
        if float(metrics["recall"]) < min_human_recall:
            continue
        negative_mask = labels == 0
        suppression = (
            float(np.mean(predictions[negative_mask] == 0))
            if np.any(negative_mask)
            else 0.0
        )
        candidates.append(
            (suppression, float(threshold), metrics)
        )
    if not candidates:
        predictions = np.ones_like(labels)
        return 0.0, False, {
            **binary_metrics(labels, predictions),
            "reason": (
                "ningún umbral conserva el recall humano mínimo"
            ),
        }
    suppression, threshold, metrics = max(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    return threshold, True, {
        **metrics,
        "pet_suppression_rate": round(suppression, 4),
    }


def train_pet_filter(
    ai_model: Any,
    dataset: PreparedSupervisedDataset,
    adjacency: dict[str, list[str]],
    *,
    epochs: int,
    min_human_recall: float,
) -> dict[str, Any]:
    train_values, train_labels, train_weights, _ = (
        _filter_arrays(dataset, "train", adjacency)
    )
    validation_values, validation_labels, _, _ = (
        _filter_arrays(dataset, "validation", adjacency)
    )
    test_values, test_labels, _, test_samples = (
        _filter_arrays(dataset, "test", adjacency)
    )
    if len(train_values) < 50 or len(validation_values) < 10:
        raise ValueError(
            "El manifiesto no genera suficientes activaciones "
            "de movimiento etiquetadas"
        )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = PetMotionTransformer(
        input_size=FILTER_FEATURE_SIZE,
        context_length=dataset.context_length,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    values_t = torch.tensor(
        train_values,
        dtype=torch.float32,
        device=device,
    )
    labels_t = torch.tensor(
        train_labels,
        dtype=torch.float32,
        device=device,
    )
    weights_t = torch.tensor(
        train_weights,
        dtype=torch.float32,
        device=device,
    )
    batch_size = 256
    last_loss = 0.0
    model.train()
    for _epoch in range(epochs):
        permutation = torch.randperm(
            len(values_t),
            device=device,
        )
        for start in range(0, len(values_t), batch_size):
            indices = permutation[start : start + batch_size]
            logits = model(values_t[indices])
            losses = criterion(logits, labels_t[indices])
            loss = (
                losses * weights_t[indices]
            ).sum() / weights_t[indices].sum().clamp_min(1e-6)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(
                loss.detach().cpu().item()
            )

    validation_probabilities = _probabilities(
        model,
        validation_values,
        device,
    )
    (
        threshold,
        suppression_enabled,
        validation_metrics,
    ) = _select_threshold(
        validation_labels,
        validation_probabilities,
        min_human_recall,
    )
    test_probabilities = _probabilities(
        model,
        test_values,
        device,
    )
    test_predictions = (
        test_probabilities >= threshold
    ).astype(np.float32)
    test_metrics = binary_metrics(
        test_labels,
        test_predictions,
    )
    test_metrics["pet_suppression_rate"] = (
        pet_suppression_rate(
            test_samples,
            test_predictions,
        )
    )
    test_metrics["human_false_suppression_rate"] = round(
        1.0 - float(test_metrics["recall"]),
        4,
    )
    if float(test_metrics["recall"]) < min_human_recall:
        test_metrics["recall_below_target"] = True
    test_metrics["by_period"] = filter_metrics_by_period(
        test_samples,
        test_labels,
        test_predictions,
    )
    baseline = temporal_baseline(test_samples, adjacency)
    baseline["by_period"] = {
        period_id: temporal_baseline(
            [
                sample
                for sample in test_samples
                if sample.period_id == period_id
            ],
            adjacency,
        )
        for period_id in sorted(
            {sample.period_id for sample in test_samples}
        )
    }
    ai_model.pet_filter_model = model
    ai_model.pet_filter_device = device
    ai_model.pet_filter_threshold = float(threshold)
    ai_model.pet_filter_context_length = (
        dataset.context_length
    )
    ai_model.pet_filter_info = {
        "enabled": True,
        "suppression_enabled": suppression_enabled,
        "threshold": round(float(threshold), 4),
        "manifest_id": dataset.manifest_id,
        "dataset_fingerprint": dataset.fingerprint,
        "input_size": FILTER_FEATURE_SIZE,
        "context_length": dataset.context_length,
        "device": str(device),
        "min_human_recall": min_human_recall,
        "activation_policy": "operational_preference",
        "validation": validation_metrics,
        "test": test_metrics,
        "baseline_temporal": baseline,
        "trained_at": utc_iso(),
    }
    return {
        **ai_model.pet_filter_info,
        "samples": {
            "train": len(train_values),
            "validation": len(validation_values),
            "test": len(test_values),
        },
        "loss": round(last_loss, 6),
    }
