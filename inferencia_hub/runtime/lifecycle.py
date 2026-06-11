"""Inicio, cierre y persistencia del servicio."""

from .shared import *  # noqa: F401,F403


def data_dir() -> Path:
    return Path(os.getenv("INFERENCIA_DATA_DIR", "/app/data"))


def model_state_dir() -> Path:
    return Path(os.getenv("MODEL_STATE_DIR", str(data_dir() / "model_state")))


def training_status_path() -> Path:
    return Path(os.getenv("TRAINING_STATUS_PATH", str(data_dir() / "training_status.json")))


def real_sensor_config_path() -> Path:
    return Path(os.getenv("REAL_SENSOR_CONFIG_PATH", str(data_dir() / "real_sensor_config.json")))


def history_sensor_name(entity_id: str) -> str:
    return ha_entity_catalog.sensor_name(entity_id)


async def persist_history_event(
    payload: SensorEventInput,
    event: dict[str, Any],
    response: dict[str, Any],
) -> None:
    input_mode = str(event.get("input_mode") or "listen")
    record = {
        "event_timestamp": event["timestamp"],
        "entity_id": event["entity_id"],
        "sensor_name": history_sensor_name(event["entity_id"]),
        "sensor_type": event["sensor_type"],
        "room": event["room"],
        "state": event["state"],
        "source": event.get("source") or payload.source,
        "input_mode": input_mode,
        "inferred_presence": str(event.get("inferred_presence") or "").lower()
        == "presente",
        "inferred_room": event.get("presence_room") or "",
        "confidence": event.get("presence_confidence"),
        "estimated_people": event.get("estimated_people") or 0,
        "active_rooms": event.get("active_rooms") or [],
        "layout_alert": event.get("layout_alert"),
        "raw_payload": {
            "entity_id": payload.entity_id,
            "state": payload.state,
            "sensor_type": payload.sensor_type,
            "room": payload.room,
            "timestamp": event["timestamp"],
            "source": payload.source,
        },
        "inference_payload": response,
    }
    stored = await history_store.enqueue(record, wait=input_mode == "listen")
    if input_mode == "listen" and not stored and history_store.should_persist(input_mode):
        LOGGER.error("No se pudo confirmar la escritura del evento %s", event["entity_id"])


hub_state.event_sink = persist_history_event


def resolve_training_csv() -> str | None:
    candidates = [
        os.getenv("TRAINING_CSV_PATH", ""),
        "/data/history-1mes_sorted.csv",
        "/data/history-1mes.csv",
        "/data/history-1mes_sorted.csv",
        "/data/history-1mes.csv",
        "history-1mes_sorted.csv",
        "history-1mes.csv",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if Path(candidate).exists():
            return str(candidate)
    return None


def store_training_artifact(payload: dict[str, Any]) -> str | None:
    artifact_dir = Path(os.getenv("TRAINING_ARTIFACT_DIR", "/app/data/training_artifacts"))
    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        artifact_path = artifact_dir / f"training-{stamp}.json"
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        artifact_path.write_text(serialized, encoding="utf-8")
        (artifact_dir / "latest_training.json").write_text(serialized, encoding="utf-8")
        return str(artifact_path)
    except Exception:
        return None


def persist_real_sensor_config() -> None:
    path = real_sensor_config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(hub_state.real_sensor_config(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        LOGGER.exception("No se pudo persistir real_sensor_config")


def load_real_sensor_config() -> None:
    path = real_sensor_config_path()
    if not path.exists():
        return
    try:
        restored = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(restored, dict):
            hub_state.load_real_sensor_config(restored)
    except Exception:
        LOGGER.exception("No se pudo cargar real_sensor_config persistido")


async def activate_listen_mode() -> None:
    hub_state.input_mode = "listen"
    hub_state.replay_stop_requested = True
    hub_state.replay_step_budget = 0
    if hub_state.replay_task and not hub_state.replay_task.done():
        hub_state.replay_task.cancel()
    hub_state.replay_paused = False
    await hub_state.broadcast_snapshot()


def persist_training_status() -> None:
    path = training_status_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(training_status, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        LOGGER.exception("No se pudo persistir training_status")


def load_training_status() -> None:
    path = training_status_path()
    if not path.exists():
        return
    try:
        restored = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(restored, dict):
            for key, value in restored.items():
                if isinstance(value, dict):
                    training_status[key] = value
            for value in training_status.values():
                if isinstance(value, dict) and value.get("state") == "running":
                    value["state"] = "error"
                    value["message"] = "entrenamiento interrumpido por reinicio del backend"
                    value["finished_at"] = to_utc_iso(datetime.now(timezone.utc))
    except Exception:
        LOGGER.exception("No se pudo cargar training_status persistido")


def persist_model_state() -> dict[str, Any] | None:
    try:
        return hub_state.ai_model.save_state(model_state_dir())
    except Exception:
        LOGGER.exception("No se pudo persistir el estado del modelo")
        return None


def export_simulated_sensor_csv(
    req: TrainSimulatorPresenceRequest,
    reference_layout: dict[str, list[str]] | None,
) -> dict[str, Any] | None:
    export_dir = Path(os.getenv("TRAINING_EXPORT_DIR", "/app/data/training_exports"))
    try:
        export_dir.mkdir(parents=True, exist_ok=True)
        real_profile = (
            hub_state.ai_model.real_profile_info
            if hub_state.ai_model.real_profile_info.get("enabled")
            else None
        )
        labeled_events, rooms, _layout = hub_state.ai_model._generate_simulated_presence_events(req, reference_layout, real_profile)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        filename = f"simulated-sensors-{stamp}.csv"
        export_path = export_dir / filename
        fieldnames = [
            "last_changed",
            "entity_id",
            "state",
            "sensor_type",
            "room",
            "occupied_rooms",
            "occupied_count",
            "rooms_total",
            "source",
        ]
        with export_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for event, occupied_rooms in labeled_events:
                writer.writerow(
                    {
                        "last_changed": to_utc_iso(event.timestamp),
                        "entity_id": event.entity_id,
                        "state": event.state,
                        "sensor_type": event.sensor_type,
                        "room": event.room,
                        "occupied_rooms": "|".join(sorted(occupied_rooms)),
                        "occupied_count": len(occupied_rooms),
                        "rooms_total": len(rooms),
                        "source": "simulator_training",
                    }
                )
        return {
            "filename": filename,
            "path": str(export_path),
            "url": f"/api/training_exports/{filename}",
            "rows": len(labeled_events),
        }
    except Exception:
        return None


def mark_training_status(
    kind: str,
    state: str,
    message: str,
    *,
    request: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    now = to_utc_iso(datetime.now(timezone.utc))
    current = training_status.setdefault(kind, {})
    current["state"] = state
    current["message"] = message
    if state == "running":
        current["started_at"] = now
        current["finished_at"] = None
        current["request"] = request or {}
        current.pop("result_summary", None)
        current.pop("error", None)
        persist_training_status()
        return

    current["finished_at"] = now
    if result is not None:
        info = result.get("training_info", {}) if isinstance(result, dict) else {}
        current["result_summary"] = {
            "status": result.get("status") if isinstance(result, dict) else None,
            "samples": info.get("samples"),
            "events_total": info.get("events_total"),
            "synthetic_events": info.get("synthetic_events"),
            "count_accuracy": info.get("count_accuracy"),
            "room_exact_match_rate": info.get("room_exact_match_rate"),
            "transformer_enabled": info.get("transformer", {}).get("enabled")
            if isinstance(info.get("transformer"), dict)
            else None,
            "simulated_csv_url": result.get("simulated_sensor_csv", {}).get("url")
            if isinstance(result.get("simulated_sensor_csv"), dict)
            else None,
            "simulated_csv_rows": result.get("simulated_sensor_csv", {}).get("rows")
            if isinstance(result.get("simulated_sensor_csv"), dict)
            else None,
            "model_state_saved": bool(result.get("model_state")),
        }
    if error:
        current["error"] = error
    persist_training_status()


async def startup_train_model() -> None:
    await history_store.start()
    load_training_status()
    load_real_sensor_config()
    model_load = await asyncio.to_thread(hub_state.ai_model.load_state, model_state_dir())
    if model_load.get("loaded"):
        async with hub_state.lock:
            hub_state.rooms.update(hub_state.ai_model.rooms)
            hub_state._ensure_reference_layout_locked()
            n_rooms = len(hub_state.ai_model.rooms)
            if n_rooms > 0:
                import numpy as np

                hub_state.presence_belief = np.full((n_rooms,), 1.0 / n_rooms, dtype=np.float32)
        training_status["model_state"] = {
            "state": "loaded",
            "label": "Estado persistido",
            "message": "modelo cargado desde disco",
            "loaded_at": to_utc_iso(datetime.now(timezone.utc)),
            "result_summary": model_load,
        }
        persist_training_status()
        if os.getenv("FORCE_AUTO_TRAIN_ON_START", "0") != "1":
            return

    if os.getenv("AUTO_TRAIN_ON_START", "1") == "0":
        return
    csv_path = resolve_training_csv()
    if not csv_path:
        return

    async def _run() -> None:
        req = TrainModelRequest(csv_path=csv_path)
        try:
            async with hub_state.lock:
                reference_layout = dict(hub_state.reference_layout)
            await asyncio.to_thread(
                hub_state.ai_model.train_from_csv_with_reference,
                req,
                reference_layout,
            )
            async with hub_state.lock:
                hub_state.rooms.update(hub_state.ai_model.rooms)
                hub_state._ensure_reference_layout_locked()
            # No se reinicia el estado porque el entrenamiento automático se ejecuta
            # en segundo plano y puede finalizar con eventos ya procesados.
        except Exception:
            # Un fallo del entrenamiento automático no detiene el servicio.
            return

    asyncio.create_task(_run())


async def shutdown_history_store() -> None:
    await history_store.stop()
