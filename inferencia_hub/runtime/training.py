"""Operaciones de entrenamiento y generación de artefactos."""

from .shared import *  # noqa: F401,F403
from .lifecycle import (
    export_simulated_sensor_csv,
    mark_training_status,
    persist_model_state,
    resolve_training_csv,
    store_training_artifact,
)


def download_training_export(filename: str) -> FileResponse:
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.endswith(".csv"):
        raise HTTPException(status_code=404, detail="Export no encontrado")
    export_dir = Path(os.getenv("TRAINING_EXPORT_DIR", "/app/data/training_exports")).resolve()
    export_path = (export_dir / safe_name).resolve()
    if export_dir not in export_path.parents or not export_path.exists():
        raise HTTPException(status_code=404, detail="Export no encontrado")
    return FileResponse(
        path=str(export_path),
        media_type="text/csv",
        filename=safe_name,
    )


async def train_model(req: TrainModelRequest) -> dict[str, Any]:
    mark_training_status(
        "historical",
        "running",
        "entrenando modelo desde CSV",
        request=req.model_dump(),
    )
    try:
        async with hub_state.lock:
            reference_layout = dict(hub_state.reference_layout)
        result = await asyncio.to_thread(
            hub_state.ai_model.train_from_csv_with_reference,
            req,
            reference_layout,
        )
        result["training_info"]["csv_path"] = req.csv_path
    except Exception as exc:
        mark_training_status("historical", "error", "fallo el entrenamiento desde CSV", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async with hub_state.lock:
        hub_state.rooms.update(result.get("rooms", []))
        hub_state._ensure_reference_layout_locked()
        map_validation = hub_state.training_map_validation_locked()

    artifact_payload = {
        "timestamp": to_utc_iso(datetime.now(timezone.utc)),
        "request": req.model_dump(),
        "training_result": result,
        "map_validation": map_validation,
    }
    artifact_path = await asyncio.to_thread(store_training_artifact, artifact_payload)

    result["map_validation"] = map_validation
    result["artifact_path"] = artifact_path
    model_state = await asyncio.to_thread(persist_model_state)
    if model_state:
        result["model_state"] = model_state
    mark_training_status("historical", "completed", "entrenamiento desde CSV completado", result=result)

    await hub_state.reset()
    return result


async def train_model_full(req: TrainModelFullRequest) -> dict[str, Any]:
    mark_training_status(
        "historical",
        "running",
        "entrenando modelo completo desde CSV",
        request=req.model_dump(),
    )
    """Entrenamiento completo optimizado para mÃ¡xima captura del historial."""
    # Utiliza el contrato común de entrenamiento con los parámetros del modo completo.
    full_req = TrainModelRequest(
        csv_path=req.csv_path,
        debounce_seconds=req.debounce_seconds,
        include_all_state_transitions=req.include_all_state_transitions,
        min_gap_seconds=req.min_gap_seconds,
        max_gap_seconds=req.max_gap_seconds,
        epochs=req.epochs,
        max_samples=req.max_samples,
        degree_limit=req.degree_limit,
        use_ollama_validation=req.use_ollama_validation,
    )
    try:
        async with hub_state.lock:
            reference_layout = dict(hub_state.reference_layout)
        result = await asyncio.to_thread(
            hub_state.ai_model.train_from_csv_with_reference,
            full_req,
            reference_layout,
        )
        result["training_info"]["csv_path"] = req.csv_path
        result["training_type"] = "full_historical"
        result["training_info"]["note"] = (
            f"Entrenamiento completo con debounce={req.debounce_seconds}s, "
            f"min_gap={req.min_gap_seconds}s, max_gap={req.max_gap_seconds}s"
        )
    except Exception as exc:
        mark_training_status("historical", "error", "fallo el entrenamiento completo desde CSV", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async with hub_state.lock:
        hub_state.rooms.update(result.get("rooms", []))
        hub_state._ensure_reference_layout_locked()
        map_validation = hub_state.training_map_validation_locked()

    artifact_payload = {
        "timestamp": to_utc_iso(datetime.now(timezone.utc)),
        "request": req.model_dump(),
        "training_result": result,
        "map_validation": map_validation,
    }
    artifact_path = await asyncio.to_thread(store_training_artifact, artifact_payload)

    result["map_validation"] = map_validation
    result["artifact_path"] = artifact_path
    model_state = await asyncio.to_thread(persist_model_state)
    if model_state:
        result["model_state"] = model_state
    mark_training_status("historical", "completed", "entrenamiento completo desde CSV finalizado", result=result)

    await hub_state.reset()
    return result


async def train_presence_simulator(req: TrainSimulatorPresenceRequest) -> dict[str, Any]:
    if req.use_real_profile and not req.real_profile_csv_path:
        resolved_csv = resolve_training_csv()
        if resolved_csv:
            req = req.model_copy(update={"real_profile_csv_path": resolved_csv})
    mark_training_status(
        "presence",
        "running",
        "entrenando ocupacion desde simulador",
        request=req.model_dump(),
    )
    try:
        async with hub_state.lock:
            reference_layout = dict(hub_state.reference_layout)
        result = await asyncio.to_thread(
            hub_state.ai_model.train_occupancy_from_simulator,
            req,
            reference_layout,
        )
    except Exception as exc:
        mark_training_status("presence", "error", "fallo el entrenamiento de presencia", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async with hub_state.lock:
        hub_state.rooms.update(result.get("rooms", []))
        hub_state._ensure_reference_layout_locked()

    artifact_payload = {
        "timestamp": to_utc_iso(datetime.now(timezone.utc)),
        "request": req.model_dump(),
        "training_result": result,
    }
    artifact_path = await asyncio.to_thread(store_training_artifact, artifact_payload)
    result["artifact_path"] = artifact_path
    export_info = await asyncio.to_thread(export_simulated_sensor_csv, req, reference_layout)
    if export_info:
        result["simulated_sensor_csv"] = export_info
    model_state = await asyncio.to_thread(persist_model_state)
    if model_state:
        result["model_state"] = model_state
    mark_training_status("presence", "completed", "entrenamiento de presencia finalizado", result=result)

    await hub_state.broadcast_snapshot()
    return result
