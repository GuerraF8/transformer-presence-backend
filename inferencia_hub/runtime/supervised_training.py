"""Entrenamiento supervisado con confirmaciones externas."""

from __future__ import annotations

from .shared import *  # noqa: F401,F403
from .lifecycle import (
    mark_training_status,
    persist_model_state_atomic,
    rollback_model_state,
)
from ..supervised.dataset import SupervisedDatasetBuilder
from ..supervised.trainer import SupervisedPresenceTrainer


def list_training_manifests() -> dict[str, Any]:
    return {"items": training_manifests.list()}


async def validate_training_manifest(
    req: TrainingManifestInput,
) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(
            training_manifests.validate,
            req.manifest_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def get_training_report(run_id: str) -> dict[str, Any]:
    try:
        return training_manifests.load_report(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _synthetic_request(
    req: TrainPresenceSupervisedRequest,
    rooms: list[str],
    adjacency: dict[str, list[str]],
) -> TrainSimulatorPresenceRequest:
    return TrainSimulatorPresenceRequest(
        rooms=rooms,
        template="personalizado",
        layout_edges=[
            [edge["a"], edge["b"]]
            for edge in edge_list_from_adjacency(adjacency)
        ],
        scenarios=req.synthetic_scenarios,
        steps_per_scenario=req.synthetic_steps,
        max_people=max(1, min(4, len(rooms))),
        epochs=req.epochs,
        max_samples=req.max_samples,
        seed=req.seed,
        use_real_profile=False,
    )


def _train_supervised_sync(
    req: TrainPresenceSupervisedRequest,
    reference_layout: dict[str, list[str]],
) -> dict[str, Any]:
    builder = SupervisedDatasetBuilder(
        training_manifests,
        context_length=hub_state.ai_model.transformer_context_length,
    )
    dataset = builder.build(req.manifest_id)
    rooms = sorted(reference_layout)
    if not rooms:
        rooms = dataset.rooms
    synthetic_request = _synthetic_request(
        req,
        rooms,
        reference_layout,
    )
    base_result = hub_state.ai_model.train_occupancy_from_simulator(
        synthetic_request,
        reference_layout,
    )
    if base_result.get("status") != "ok":
        reason = base_result.get("training_info", {}).get(
            "reason",
            "dependencias ML no disponibles",
        )
        raise RuntimeError(f"No se pudo preparar el modelo de ocupación: {reason}")

    labeled_events, synthetic_rooms, _layout = (
        hub_state.ai_model._generate_simulated_presence_events(
            synthetic_request,
            reference_layout,
        )
    )
    synthetic_dataset = (
        hub_state.ai_model._prepare_occupancy_transformer_dataset(
            labeled_events,
            synthetic_rooms,
        )
    )
    trainer = SupervisedPresenceTrainer(
        epochs=req.epochs,
        seed=req.seed,
        min_human_recall=req.min_human_recall,
    )
    report = trainer.train(
        hub_state.ai_model,
        dataset,
        reference_layout,
        synthetic_dataset=synthetic_dataset,
    )
    report["request"] = req.model_dump()
    report["synthetic_base"] = base_result.get("training_info", {})
    hub_state.ai_model.training_info["supervised"] = {
        "enabled": True,
        "run_id": report["run_id"],
        "manifest_id": report["manifest_id"],
        "dataset_fingerprint": report["dataset_fingerprint"],
        "filter": report["filter"],
        "occupancy": report["occupancy"],
        "trained_at": report["created_at"],
    }
    return report


async def train_presence_supervised(
    req: TrainPresenceSupervisedRequest,
) -> dict[str, Any]:
    if not hub_state.active_profile_id:
        raise HTTPException(
            status_code=409,
            detail="Debe activar un perfil antes de entrenar",
        )
    mark_training_status(
        "supervised",
        "running",
        "preparando historiales y entrenando presencia supervisada",
        request=req.model_dump(),
    )
    try:
        async with hub_state.lock:
            reference_layout = dict(hub_state.reference_layout)
        report = await asyncio.to_thread(
            _train_supervised_sync,
            req,
            reference_layout,
        )
        report_path = await asyncio.to_thread(
            training_manifests.save_report,
            report["run_id"],
            report,
        )
        report["report_path"] = str(report_path)
        model_state = await asyncio.to_thread(persist_model_state_atomic)
        if not model_state:
            raise RuntimeError("No se pudo activar el artefacto entrenado")
        report["model_state"] = model_state
    except Exception as exc:
        mark_training_status(
            "supervised",
            "error",
            "falló el entrenamiento supervisado",
            error=str(exc),
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    mark_training_status(
        "supervised",
        "completed",
        "entrenamiento supervisado activado",
        result=report,
    )
    await hub_state.broadcast_snapshot()
    return report


async def rollback_model() -> dict[str, Any]:
    try:
        result = await asyncio.to_thread(rollback_model_state)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    await hub_state.broadcast_snapshot()
    return result
