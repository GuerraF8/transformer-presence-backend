"""Ajuste supervisado del modelo de ocupación por habitación."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..domain import time_features_from_dt
from .dataset import PreparedSupervisedDataset
from .evaluation import binary_metrics, utc_iso

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - depende de la imagen ML
    torch = None
    nn = None


def _occupancy_arrays(
    ai_model: Any,
    samples: list[Any],
    rooms: list[str],
    room_to_idx: dict[str, int],
) -> tuple[np.ndarray, ...]:
    x_values: list[np.ndarray] = []
    x_time: list[np.ndarray] = []
    x_future: list[np.ndarray] = []
    y_rooms: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for sample in samples:
        previous = None
        values = []
        times = []
        for event in sample.context:
            values.append(
                ai_model._event_feature_vector(
                    event,
                    rooms,
                    room_to_idx,
                    previous,
                )
            )
            times.append(
                time_features_from_dt(event.timestamp)
            )
            previous = event.timestamp
        target = np.zeros(
            (len(rooms),),
            dtype=np.float32,
        )
        mask = np.zeros(
            (len(rooms),),
            dtype=np.float32,
        )
        sample_weights = np.zeros(
            (len(rooms),),
            dtype=np.float32,
        )
        index = room_to_idx[sample.room]
        target[index] = sample.human_label
        mask[index] = 1.0
        sample_weights[index] = sample.sample_weight
        x_values.append(np.stack(values))
        x_time.append(np.stack(times))
        x_future.append(times[-1])
        y_rooms.append(target)
        masks.append(mask)
        weights.append(sample_weights)
    return tuple(
        np.asarray(value, dtype=np.float32)
        for value in (
            x_values,
            x_time,
            x_future,
            y_rooms,
            masks,
            weights,
        )
    )


def _synthetic_regularization_loss(
    model: Any,
    dataset: tuple[Any, Any, Any, Any, Any],
    device: Any,
    start: int,
    batch_size: int,
) -> Any:
    x_values, x_time, x_future, y_rooms, _y_count = (
        dataset
    )
    if len(x_values) == 0:
        return torch.tensor(0.0, device=device)
    indices = (
        np.arange(start, start + batch_size)
        % len(x_values)
    )
    values_t = torch.tensor(
        x_values[indices],
        dtype=torch.float32,
        device=device,
    )
    time_t = torch.tensor(
        x_time[indices],
        dtype=torch.float32,
        device=device,
    )
    future_t = torch.tensor(
        x_future[indices],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)
    rooms_t = torch.tensor(
        y_rooms[indices],
        dtype=torch.float32,
        device=device,
    )
    room_logits, _ = model(
        values_t,
        time_t,
        future_t,
    )
    return nn.functional.binary_cross_entropy_with_logits(
        room_logits,
        rooms_t,
    )


def _occupancy_metrics(
    model: Any,
    arrays: tuple[np.ndarray, ...],
    device: Any,
) -> dict[str, Any]:
    if not arrays or len(arrays[0]) == 0:
        return {}
    (
        x_values,
        x_time,
        x_future,
        y_rooms,
        masks,
        _weights,
    ) = arrays
    model.eval()
    with torch.no_grad():
        room_logits, _ = model(
            torch.tensor(
                x_values,
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                x_time,
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                x_future,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(1),
        )
        predictions = (
            torch.sigmoid(room_logits)
            .detach()
            .cpu()
            .numpy()
            >= 0.5
        ).astype(np.float32)
    selected = masks.astype(bool)
    return binary_metrics(
        y_rooms[selected],
        predictions[selected],
    )


def fine_tune_occupancy(
    ai_model: Any,
    dataset: PreparedSupervisedDataset,
    *,
    epochs: int,
    synthetic_dataset: (
        tuple[Any, Any, Any, Any, Any] | None
    ),
) -> dict[str, Any]:
    model = getattr(
        ai_model,
        "occupancy_transformer_model",
        None,
    )
    device = getattr(
        ai_model,
        "occupancy_transformer_device",
        None,
    )
    rooms = list(
        getattr(
            ai_model,
            "occupancy_transformer_rooms",
            [],
        )
    )
    if model is None or device is None or not rooms:
        return {
            "enabled": False,
            "reason": "modelo de ocupación base no disponible",
        }
    room_to_idx = {
        room: index
        for index, room in enumerate(rooms)
    }
    supervised = [
        sample
        for sample in dataset.samples["train"]
        if sample.room in room_to_idx
    ]
    validation = [
        sample
        for sample in dataset.samples["validation"]
        if sample.room in room_to_idx
    ]
    test = [
        sample
        for sample in dataset.samples["test"]
        if sample.room in room_to_idx
    ]
    if len(supervised) < 50:
        return {
            "enabled": False,
            "reason": (
                "muestras supervisadas insuficientes "
                "para ocupación"
            ),
        }

    train_arrays = _occupancy_arrays(
        ai_model,
        supervised,
        rooms,
        room_to_idx,
    )
    validation_arrays = _occupancy_arrays(
        ai_model,
        validation,
        rooms,
        room_to_idx,
    )
    test_arrays = _occupancy_arrays(
        ai_model,
        test,
        rooms,
        room_to_idx,
    )
    tensors = [
        torch.tensor(
            value,
            dtype=torch.float32,
            device=device,
        )
        for value in train_arrays
    ]
    (
        x_values_t,
        x_time_t,
        x_future_t,
        y_rooms_t,
        masks_t,
        weights_t,
    ) = tensors
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=2e-4,
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    batch_size = 128
    last_loss = 0.0
    model.train()
    for _epoch in range(max(1, min(3, epochs))):
        permutation = torch.randperm(
            len(x_values_t),
            device=device,
        )
        for start in range(
            0,
            len(x_values_t),
            batch_size,
        ):
            indices = permutation[start : start + batch_size]
            room_logits, _count_logits = model(
                x_values_t[indices],
                x_time_t[indices],
                x_future_t[indices].unsqueeze(1),
            )
            losses = (
                criterion(
                    room_logits,
                    y_rooms_t[indices],
                )
                * masks_t[indices]
                * weights_t[indices]
            )
            denominator = (
                masks_t[indices] * weights_t[indices]
            ).sum().clamp_min(1e-6)
            loss = losses.sum() / denominator
            if synthetic_dataset is not None:
                loss = loss + (
                    0.15
                    * _synthetic_regularization_loss(
                        model,
                        synthetic_dataset,
                        device,
                        start,
                        batch_size,
                    )
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(
                loss.detach().cpu().item()
            )

    validation_metrics = _occupancy_metrics(
        model,
        validation_arrays,
        device,
    )
    test_metrics = _occupancy_metrics(
        model,
        test_arrays,
        device,
    )
    validation_metrics["by_period"] = {
        period_id: _occupancy_metrics(
            model,
            _occupancy_arrays(
                ai_model,
                [
                    sample
                    for sample in validation
                    if sample.period_id == period_id
                ],
                rooms,
                room_to_idx,
            ),
            device,
        )
        for period_id in sorted(
            {sample.period_id for sample in validation}
        )
    }
    test_metrics["by_period"] = {
        period_id: _occupancy_metrics(
            model,
            _occupancy_arrays(
                ai_model,
                [
                    sample
                    for sample in test
                    if sample.period_id == period_id
                ],
                rooms,
                room_to_idx,
            ),
            device,
        )
        for period_id in sorted(
            {sample.period_id for sample in test}
        )
    }
    ai_model.occupancy_transformer_info.update(
        {
            "supervised": {
                "enabled": True,
                "manifest_id": dataset.manifest_id,
                "dataset_fingerprint": dataset.fingerprint,
                "samples": {
                    "train": len(supervised),
                    "validation": len(validation),
                    "test": len(test),
                },
                "validation": validation_metrics,
                "test": test_metrics,
                "loss": round(last_loss, 6),
                "count_head_updated": False,
                "trained_at": utc_iso(),
            }
        }
    )
    return ai_model.occupancy_transformer_info[
        "supervised"
    ]
