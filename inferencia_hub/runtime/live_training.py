"""Adaptación automática con confirmaciones de persona y mascota."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from .shared import *  # noqa: F401,F403
from .lifecycle import persist_model_state_atomic
from ..domain import EventRecord
from ..supervised.dataset import filter_context_matrix

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - depende de la imagen ML
    torch = None
    nn = None


_run_lock = asyncio.Lock()
_scheduler_task: asyncio.Task | None = None


def _active_identity() -> tuple[str, str]:
    return (
        str(hub_state.active_profile_id or ""),
        str(hub_state.active_profile_fingerprint or ""),
    )


def _last_activation_time(run: dict[str, Any] | None) -> datetime | None:
    if not run or run.get("state") != "activated":
        return None
    value = run.get("finished_at")
    return parse_iso_datetime(value) if value else None


def live_training_status() -> dict[str, Any]:
    profile_id, fingerprint = _active_identity()
    config = live_training_store.config()
    if not profile_id:
        return {
            "enabled": config["enabled"],
            "available": False,
            "reason": "no_active_profile",
            "config": config,
        }
    counts = live_training_store.confirmation_counts(
        profile_id,
        fingerprint,
        only_new=True,
    )
    latest = live_training_store.latest_run(profile_id, fingerprint)
    activated_run = live_training_store.last_activated_run(
        profile_id,
        fingerprint,
    )
    last_activation = _last_activation_time(activated_run)
    next_activation_at = (
        last_activation
        + timedelta(days=config["minimum_days_between_activations"])
        if last_activation
        else None
    )
    thresholds_met = (
        counts["total"] >= config["minimum_confirmations"]
        and counts["person"] >= config["minimum_person_confirmations"]
        and counts["pet"] >= config["minimum_pet_confirmations"]
    )
    cooldown_met = (
        next_activation_at is None
        or datetime.now(timezone.utc) >= next_activation_at
    )
    return {
        "enabled": config["enabled"],
        "available": torch is not None,
        "profile_id": profile_id,
        "profile_fingerprint": fingerprint,
        "confirmations": counts,
        "thresholds_met": thresholds_met,
        "cooldown_met": cooldown_met,
        "eligible": bool(
            config["enabled"]
            and torch is not None
            and thresholds_met
            and cooldown_met
        ),
        "last_run": latest,
        "next_activation_at": (
            to_utc_iso(next_activation_at) if next_activation_at else None
        ),
        "next_evaluation_at": to_utc_iso(
            datetime.now(timezone.utc) + timedelta(days=1)
        ),
        "config": config,
    }


def get_live_training_config() -> dict[str, Any]:
    return live_training_store.config()


async def update_live_training_config(
    req: LiveTrainingConfigInput,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        live_training_store.update_config,
        req.model_dump(),
    )


def _binary_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    true_positive = float(((labels == 1) & (predictions == 1)).sum())
    false_positive = float(((labels == 0) & (predictions == 1)).sum())
    false_negative = float(((labels == 1) & (predictions == 0)).sum())
    true_negative = float(((labels == 0) & (predictions == 0)).sum())
    precision = true_positive / max(1.0, true_positive + false_positive)
    recall = true_positive / max(1.0, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    specificity = true_negative / max(1.0, true_negative + false_positive)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "pet_suppression_rate": round(specificity, 4),
    }


def _build_live_arrays() -> tuple[np.ndarray, np.ndarray]:
    profile_id, fingerprint = _active_identity()
    confirmations = live_training_store.confirmations(
        profile_id,
        fingerprint,
    )
    active_confirmations = [
        item
        for item in confirmations
        if str(item.get("state") or "").lower() == "on"
    ]
    if not active_confirmations:
        return (
            np.zeros((0, 28, 12), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    person_times_by_room: dict[str, list[datetime]] = {}
    for item in active_confirmations:
        if item["training_role"] != "person_confirmation":
            continue
        person_times_by_room.setdefault(item["room"], []).append(
            parse_iso_datetime(item["event_timestamp"])
        )
    start = parse_iso_datetime(active_confirmations[0]["event_timestamp"])
    end = parse_iso_datetime(active_confirmations[-1]["event_timestamp"])
    rows = live_training_store.signal_events(
        profile_id,
        fingerprint,
        from_timestamp=to_utc_iso(start - timedelta(minutes=15)),
        to_timestamp=to_utc_iso(end + timedelta(seconds=20)),
    )
    events = [
        EventRecord(
            timestamp=parse_iso_datetime(row["event_timestamp"]),
            entity_id=row["entity_id"],
            state=row["state"],
            sensor_type=row["sensor_type"],
            room=row["room"],
        )
        for row in rows
    ]
    matrices: list[np.ndarray] = []
    labels: list[float] = []
    adjacency = dict(hub_state.reference_layout)
    for confirmation in active_confirmations:
        timestamp = parse_iso_datetime(confirmation["event_timestamp"])
        room = str(confirmation["room"])
        if confirmation["training_role"] == "person_confirmation":
            label = 1.0
        else:
            has_person = any(
                timestamp - timedelta(seconds=10)
                <= person_time
                <= timestamp + timedelta(seconds=20)
                for person_time in person_times_by_room.get(room, [])
            )
            label = 1.0 if has_person else 0.0
        context_events = [
            event
            for event in events
            if event.timestamp <= timestamp
        ][-28:]
        if len(context_events) < 4:
            continue
        matrices.append(
            filter_context_matrix(
                context_events,
                room,
                adjacency,
                28,
            )
        )
        labels.append(label)
    return (
        np.asarray(matrices, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
    )


def _predict_room_model(
    model: Any,
    values: Any,
    device: Any,
    threshold: float,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits, _counts = model(
            torch.tensor(values, dtype=torch.float32, device=device)
        )
        return (
            (torch.sigmoid(logits) >= threshold)
            .float()
            .detach()
            .cpu()
            .numpy()
        )


def _predict_pet_model(model: Any, values: Any, device: Any, threshold: float) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(
            model(torch.tensor(values, dtype=torch.float32, device=device))
        )
        return (probabilities >= threshold).float().cpu().numpy()


def _train_candidate_sync() -> dict[str, Any]:
    values, labels = _build_live_arrays()
    if len(values) < 20 or len(np.unique(labels)) < 2:
        raise ValueError("Confirmaciones útiles insuficientes para validar")
    split = max(1, int(len(values) * 0.8))
    train_values, validation_values = values[:split], values[split:]
    train_labels, validation_labels = labels[:split], labels[split:]
    if len(validation_values) < 4 or len(np.unique(validation_labels)) < 2:
        raise ValueError("La reserva cronológica no contiene ambas clases")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: dict[str, Any] = {}

    relative = hub_state.ai_model.relative_occupancy_model
    if relative is not None:
        candidate = deepcopy(relative).to(device)
        for parameter in candidate.parameters():
            parameter.requires_grad = False
        for parameter in candidate.room_head.parameters():
            parameter.requires_grad = True
        baseline_predictions = _predict_room_model(
            relative,
            validation_values,
            hub_state.ai_model.relative_occupancy_device,
            float(hub_state.ai_model.relative_occupancy_threshold),
        )
        optimizer = torch.optim.Adam(candidate.room_head.parameters(), lr=2e-4)
        criterion = nn.BCEWithLogitsLoss()
        values_tensor = torch.tensor(
            train_values,
            dtype=torch.float32,
            device=device,
        )
        labels_tensor = torch.tensor(
            train_labels,
            dtype=torch.float32,
            device=device,
        )
        candidate.train()
        for _epoch in range(3):
            logits, _counts = candidate(values_tensor)
            loss = criterion(logits, labels_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        candidate_predictions = _predict_room_model(
            candidate,
            validation_values,
            device,
            float(hub_state.ai_model.relative_occupancy_threshold),
        )
        baseline = _binary_metrics(validation_labels, baseline_predictions)
        metrics = _binary_metrics(validation_labels, candidate_predictions)
        accepted = (
            metrics["f1"] >= baseline["f1"] + 0.01
            and metrics["recall"] >= baseline["recall"] - 0.01
        )
        results["occupancy"] = {
            "accepted": accepted,
            "baseline": baseline,
            "candidate": metrics,
            "model": candidate,
            "device": device,
        }

    pet_model = hub_state.ai_model.pet_filter_model
    if pet_model is not None:
        candidate = deepcopy(pet_model).to(device)
        threshold = float(hub_state.ai_model.pet_filter_threshold)
        baseline_predictions = _predict_pet_model(
            pet_model,
            validation_values,
            hub_state.ai_model.pet_filter_device,
            threshold,
        )
        optimizer = torch.optim.Adam(candidate.parameters(), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        values_tensor = torch.tensor(
            train_values,
            dtype=torch.float32,
            device=device,
        )
        labels_tensor = torch.tensor(
            train_labels,
            dtype=torch.float32,
            device=device,
        )
        candidate.train()
        for _epoch in range(2):
            logits = candidate(values_tensor)
            loss = criterion(logits, labels_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        candidate_predictions = _predict_pet_model(
            candidate,
            validation_values,
            device,
            threshold,
        )
        baseline = _binary_metrics(validation_labels, baseline_predictions)
        metrics = _binary_metrics(validation_labels, candidate_predictions)
        accepted = (
            metrics["recall"] >= 0.95
            and metrics["recall"] >= baseline["recall"] - 0.01
            and metrics["pet_suppression_rate"]
            >= baseline["pet_suppression_rate"] + 0.02
        )
        results["pet_filter"] = {
            "accepted": accepted,
            "baseline": baseline,
            "candidate": metrics,
            "model": candidate,
            "device": device,
        }
    results["samples"] = {
        "total": len(values),
        "train": len(train_values),
        "validation": len(validation_values),
    }
    return results


async def run_live_training(
    *,
    force: bool = False,
    trigger: str = "manual",
) -> dict[str, Any]:
    if _run_lock.locked():
        raise HTTPException(status_code=409, detail="Ya existe un entrenamiento activo")
    status = live_training_status()
    if not status.get("available"):
        raise HTTPException(status_code=409, detail="PyTorch no disponible")
    if not force and not status.get("eligible"):
        return {"status": "deferred", **status}
    profile_id, fingerprint = _active_identity()
    counts = status["confirmations"]
    run_id = await asyncio.to_thread(
        live_training_store.begin_run,
        profile_id,
        fingerprint,
        trigger,
    )
    async with _run_lock:
        original_models = {
            "relative": hub_state.ai_model.relative_occupancy_model,
            "relative_device": hub_state.ai_model.relative_occupancy_device,
            "relative_info": deepcopy(
                hub_state.ai_model.relative_occupancy_info
            ),
            "pet": hub_state.ai_model.pet_filter_model,
            "pet_device": hub_state.ai_model.pet_filter_device,
            "pet_info": deepcopy(hub_state.ai_model.pet_filter_info),
        }
        try:
            result = await asyncio.to_thread(_train_candidate_sync)
            activated: list[str] = []
            occupancy = result.get("occupancy") or {}
            pet_filter = result.get("pet_filter") or {}
            if occupancy.get("accepted"):
                hub_state.ai_model.relative_occupancy_model = occupancy["model"]
                hub_state.ai_model.relative_occupancy_device = occupancy["device"]
                hub_state.ai_model.relative_occupancy_info.update(
                    {
                        "source": "live_adapted",
                        "pretrained": False,
                        "live_metrics": occupancy["candidate"],
                        "live_run_id": run_id,
                    }
                )
                activated.append("occupancy")
            if pet_filter.get("accepted"):
                hub_state.ai_model.pet_filter_model = pet_filter["model"]
                hub_state.ai_model.pet_filter_device = pet_filter["device"]
                hub_state.ai_model.pet_filter_info.update(
                    {
                        "source": "live_adapted",
                        "pretrained": False,
                        "live_metrics": pet_filter["candidate"],
                        "live_run_id": run_id,
                    }
                )
                activated.append("pet_filter")
            public_metrics = {
                key: {
                    nested_key: nested_value
                    for nested_key, nested_value in value.items()
                    if nested_key not in {"model", "device"}
                }
                if isinstance(value, dict)
                else value
                for key, value in result.items()
            }
            state = "activated" if activated else "rejected"
            message = (
                "Componentes validados y activados"
                if activated
                else "El candidato no superó las métricas del modelo activo"
            )
            if activated:
                saved = await asyncio.to_thread(persist_model_state_atomic)
                if not saved:
                    raise RuntimeError("No se pudo persistir el modelo validado")
            run = await asyncio.to_thread(
                live_training_store.finish_run,
                run_id,
                state=state,
                message=message,
                confirmation_cutoff_id=counts["maximum_id"],
                activated_components=activated,
                metrics=public_metrics,
            )
            await hub_state.broadcast_snapshot()
            return run
        except Exception as exc:
            hub_state.ai_model.relative_occupancy_model = original_models[
                "relative"
            ]
            hub_state.ai_model.relative_occupancy_device = original_models[
                "relative_device"
            ]
            hub_state.ai_model.relative_occupancy_info = original_models[
                "relative_info"
            ]
            hub_state.ai_model.pet_filter_model = original_models["pet"]
            hub_state.ai_model.pet_filter_device = original_models[
                "pet_device"
            ]
            hub_state.ai_model.pet_filter_info = original_models["pet_info"]
            await asyncio.to_thread(
                live_training_store.finish_run,
                run_id,
                state="error",
                message=str(exc),
                confirmation_cutoff_id=counts["maximum_id"],
                activated_components=[],
                metrics={},
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _scheduler_loop() -> None:
    first_run = True
    while True:
        try:
            await asyncio.sleep(60 if first_run else 24 * 60 * 60)
            first_run = False
            await run_live_training(force=False, trigger="scheduled")
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("Falló la evaluación diaria del aprendizaje en vivo")


def start_live_training_scheduler() -> None:
    global _scheduler_task
    if _scheduler_task is None or _scheduler_task.done():
        _scheduler_task = asyncio.create_task(
            _scheduler_loop(),
            name="live_training_scheduler",
        )


async def stop_live_training_scheduler() -> None:
    global _scheduler_task
    if _scheduler_task is None:
        return
    _scheduler_task.cancel()
    try:
        await _scheduler_task
    except asyncio.CancelledError:
        pass
    _scheduler_task = None
