"""Métricas para presencia humana, mascotas y reglas temporales."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def binary_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | int]:
    labels_i = labels.astype(np.int64)
    predictions_i = predictions.astype(np.int64)
    tp = int(np.sum((labels_i == 1) & (predictions_i == 1)))
    fp = int(np.sum((labels_i == 0) & (predictions_i == 1)))
    fn = int(np.sum((labels_i == 1) & (predictions_i == 0)))
    tn = int(np.sum((labels_i == 0) & (predictions_i == 0)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(
        1e-9,
        precision + recall,
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(
            (tp + tn) / max(1, len(labels_i)),
            4,
        ),
    }


def pet_suppression_rate(
    samples: list[Any],
    predictions: np.ndarray,
) -> float:
    pet_indices = [
        index
        for index, sample in enumerate(samples)
        if sample.pet_label > 0 and sample.human_label == 0
    ]
    if not pet_indices:
        return 0.0
    return round(
        float(
            np.mean(
                [
                    predictions[index] == 0
                    for index in pet_indices
                ]
            )
        ),
        4,
    )


def filter_metrics_by_period(
    samples: list[Any],
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for period_id in sorted(
        {sample.period_id for sample in samples}
    ):
        indices = [
            index
            for index, sample in enumerate(samples)
            if sample.period_id == period_id
        ]
        period_predictions = predictions[indices]
        metrics = binary_metrics(
            labels[indices],
            period_predictions,
        )
        metrics["pet_suppression_rate"] = pet_suppression_rate(
            [samples[index] for index in indices],
            period_predictions,
        )
        metrics["human_false_suppression_rate"] = round(
            1.0 - float(metrics["recall"]),
            4,
        )
        output[period_id] = metrics
    return output


def temporal_baseline(
    samples: list[Any],
    adjacency: dict[str, list[str]],
) -> dict[str, Any]:
    recent: list[tuple[datetime, str]] = []
    predictions: list[float] = []
    labels: list[float] = []
    for sample in sorted(
        samples,
        key=lambda item: item.timestamp,
    ):
        cutoff = sample.timestamp.timestamp() - 20.0
        recent = [
            item
            for item in recent
            if item[0].timestamp() >= cutoff
        ]
        recent.append((sample.timestamp, sample.room))
        related = [
            room
            for _timestamp, room in recent
            if room == sample.room
            or room in set(adjacency.get(sample.room, []))
        ]
        predictions.append(
            1.0 if len(related) >= 2 else 0.0
        )
        labels.append(sample.human_label)
    if not labels:
        return {}
    return binary_metrics(
        np.asarray(labels, dtype=np.float32),
        np.asarray(predictions, dtype=np.float32),
    )
