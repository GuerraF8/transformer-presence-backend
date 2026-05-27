from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from .domain import (
        CsvReplayRequest,
        HAActionRequestInput,
        HAEntityCatalogInput,
        LayoutReferenceInput,
        PresenceFilterConfigInput,
        RealSensorConfigInput,
        ReplayControlInput,
        RuntimeModeInput,
        SensorEventInput,
        TrainModelFullRequest,
        TrainModelRequest,
        TrainSimulatorPresenceRequest,
        build_layout_for_request,
        build_scenario_templates,
        classify_sensor_type,
        edge_list_from_adjacency,
        infer_room_from_entity,
        is_activation,
        normalize_room_name,
        parse_iso_datetime,
        shortest_path_rooms,
        to_utc_iso,
    )
    from .hub_state import InferenceHubState
except ImportError:  # pragma: no cover - supports `uvicorn server:app` in Docker
    from domain import (
        CsvReplayRequest,
        HAActionRequestInput,
        HAEntityCatalogInput,
        LayoutReferenceInput,
        PresenceFilterConfigInput,
        RealSensorConfigInput,
        ReplayControlInput,
        RuntimeModeInput,
        SensorEventInput,
        TrainModelFullRequest,
        TrainModelRequest,
        TrainSimulatorPresenceRequest,
        build_layout_for_request,
        build_scenario_templates,
        classify_sensor_type,
        edge_list_from_adjacency,
        infer_room_from_entity,
        is_activation,
        normalize_room_name,
        parse_iso_datetime,
        shortest_path_rooms,
        to_utc_iso,
    )
    from hub_state import InferenceHubState

TAG_STATUS = "01 Estado"
TAG_INGESTION = "02 Ingesta"
TAG_PRESENCE = "03 Presencia"
TAG_MODEL = "04 Modelo"
TAG_TRAINING = "05 Entrenamiento"
TAG_REPLAY = "06 Replay"
TAG_LAYOUT = "07 Layout y metricas"
TAG_SCENARIOS = "08 Escenarios"
TAG_DOWNLOADS = "09 Descargas"
TAG_SYSTEM = "10 Sistema"

OPENAPI_TAGS = [
    {"name": TAG_STATUS, "description": "Salud del backend y estado operativo general."},
    {"name": TAG_INGESTION, "description": "Recepcion de eventos normalizados desde Home Assistant o simulador."},
    {"name": TAG_PRESENCE, "description": "Snapshot de presencia, filtros temporales y estado inferido."},
    {"name": TAG_MODEL, "description": "Metadata del modelo de inferencia y transformadores entrenados."},
    {"name": TAG_TRAINING, "description": "Entrenamiento desde historico CSV o datos sinteticos del simulador."},
    {"name": TAG_REPLAY, "description": "Replay de historicos CSV y control de ejecucion paso a paso."},
    {"name": TAG_LAYOUT, "description": "Mapa de referencia, adyacencia y metricas de evaluacion."},
    {"name": TAG_SCENARIOS, "description": "Plantillas de layouts para simulacion y entrenamiento."},
    {"name": TAG_DOWNLOADS, "description": "Descarga de artefactos generados por entrenamiento."},
    {"name": TAG_SYSTEM, "description": "Operaciones administrativas del runtime."},
]

app = FastAPI(
    title="Inferencia Presencia Hub",
    version="0.2.0",
    openapi_tags=OPENAPI_TAGS,
)
hub_state = InferenceHubState()
WEB_DIR = Path(os.getenv("WEB_DIR", "/app/web")).resolve()
LOGGER = logging.getLogger("inferencia_hub")
ha_entity_catalog: dict[str, Any] = {
    "source": None,
    "entry_id": None,
    "scanned_at": None,
    "received_at": None,
    "auto_discovery": True,
    "tracked_entities": [],
    "entities": [],
    "entities_total": 0,
    "supported_total": 0,
}
ha_action_queue: deque[dict[str, Any]] = deque(maxlen=100)
ha_action_results: deque[dict[str, Any]] = deque(maxlen=50)
ha_action_sequence = 0
ha_integration_status: dict[str, Any] = {
    "last_seen_at": None,
    "entries": {},
}
training_status: dict[str, dict[str, Any]] = {
    "historical": {
        "state": "idle",
        "label": "Historico CSV",
        "started_at": None,
        "finished_at": None,
        "message": "sin entrenamiento iniciado desde UI",
    },
    "presence": {
        "state": "idle",
        "label": "Presencia simulador",
        "started_at": None,
        "finished_at": None,
        "message": "sin entrenamiento iniciado desde UI",
    },
}


def data_dir() -> Path:
    return Path(os.getenv("INFERENCIA_DATA_DIR", "/app/data"))


def model_state_dir() -> Path:
    return Path(os.getenv("MODEL_STATE_DIR", str(data_dir() / "model_state")))


def training_status_path() -> Path:
    return Path(os.getenv("TRAINING_STATUS_PATH", str(data_dir() / "training_status.json")))


def real_sensor_config_path() -> Path:
    return Path(os.getenv("REAL_SENSOR_CONFIG_PATH", str(data_dir() / "real_sensor_config.json")))

cors_origins_raw = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
cors_origins = ["*"] if cors_origins_raw == "*" else [
    origin.strip()
    for origin in cors_origins_raw.split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.on_event("startup")
async def startup_train_model() -> None:
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
            # No hacer reset aqui: el auto-train corre en background y puede terminar
            # despues de que ya existan eventos/replay en curso.
        except Exception:
            # No se detiene el backend si falla el entrenamiento automatico.
            return

    asyncio.create_task(_run())


@app.get("/api/health", tags=[TAG_STATUS], summary="Estado del backend")
def health() -> dict[str, Any]:
    running = bool(hub_state.replay_task and not hub_state.replay_task.done())
    return {
        "status": "ok",
        "events": len(hub_state.events),
        "rooms": len(hub_state.rooms),
        "model_ready": hub_state.ai_model.ready,
        "transformer_enabled": bool(
            hub_state.ai_model.training_info.get("transformer", {}).get("enabled")
        ),
        "presence_transformer_enabled": bool(
            hub_state.ai_model.occupancy_transformer_info.get("enabled")
        ),
        "input_mode": hub_state.input_mode,
        "replay_running": running,
        "replay_paused": hub_state.replay_paused,
        "presence_filter": hub_state.presence_filter_config(),
        "real_sensor_config": {
            "assigned_total": len(hub_state.real_sensor_config().get("assignments", [])),
            "enabled_total": len(hub_state.real_sensor_config().get("enabled_entities", [])),
            "rejected_events": hub_state.real_sensor_config().get("rejected_events", 0),
        },
        "ha_entity_catalog": {
            "scanned_at": ha_entity_catalog.get("scanned_at"),
            "received_at": ha_entity_catalog.get("received_at"),
            "entities_total": ha_entity_catalog.get("entities_total", 0),
            "supported_total": ha_entity_catalog.get("supported_total", 0),
        },
    }


@app.post("/api/events", tags=[TAG_INGESTION], summary="Ingestar evento de sensor")
async def ingest_event(payload: SensorEventInput) -> dict[str, Any]:
    source = str(payload.source or "").lower()
    is_csv = source.startswith("csv_")
    is_simulator = "simulator" in source or source in {"manual_send", "sensor_simulator"}
    is_real_ha = source.startswith("ha") or source.startswith("home_assistant") or source.startswith("hass")
    if is_csv and hub_state.input_mode != "replay":
        return {
            "status": "ignored",
            "reason": "replay_not_active",
            "input_mode": hub_state.input_mode,
        }
    if is_simulator and hub_state.input_mode != "simulator":
        return {
            "status": "ignored",
            "reason": "simulator_not_active",
            "input_mode": hub_state.input_mode,
        }
    if is_real_ha and hub_state.input_mode != "listen":
        return {
            "status": "ignored",
            "reason": "real_sensors_not_active",
            "input_mode": hub_state.input_mode,
        }
    if (not is_csv and not is_simulator and not is_real_ha) and hub_state.input_mode != "listen":
        return {
            "status": "ignored",
            "reason": "input_mode_not_listen",
            "input_mode": hub_state.input_mode,
        }
    return await hub_state.process_event(payload)


@app.get("/api/sim_data", tags=[TAG_PRESENCE], summary="Snapshot completo para la UI")
def get_sim_data() -> dict[str, Any]:
    snapshot = hub_state.snapshot()
    snapshot["ha_entity_catalog"] = ha_entity_catalog
    return snapshot


@app.get("/api/ha_entities", tags=[TAG_SYSTEM], summary="Listar entidades Home Assistant")
def get_ha_entities() -> dict[str, Any]:
    return ha_entity_catalog


@app.post("/api/ha_entities", tags=[TAG_SYSTEM], summary="Actualizar entidades Home Assistant")
async def update_ha_entities(payload: HAEntityCatalogInput) -> dict[str, Any]:
    global ha_entity_catalog
    entities = [entity.model_dump() for entity in payload.entities]
    ha_entity_catalog = {
        "source": payload.source,
        "entry_id": payload.entry_id,
        "scanned_at": to_utc_iso(payload.scanned_at or datetime.now(timezone.utc)),
        "received_at": to_utc_iso(datetime.now(timezone.utc)),
        "auto_discovery": payload.auto_discovery,
        "tracked_entities": sorted(set(payload.tracked_entities)),
        "entities": entities,
        "entities_total": len(entities),
        "supported_total": len([entity for entity in entities if entity.get("supported")]),
    }
    await hub_state.broadcast_snapshot()
    return {"status": "ok", **ha_entity_catalog}


def _real_sensor_payload() -> dict[str, Any]:
    config = hub_state.real_sensor_config()
    entities = ha_entity_catalog.get("entities") if isinstance(ha_entity_catalog, dict) else []
    entities = entities if isinstance(entities, list) else []
    assignments = {
        str(item.get("entity_id") or ""): item
        for item in config.get("assignments", [])
        if isinstance(item, dict)
    }
    rooms = list(config.get("rooms", []))
    room_sensor_counts: dict[str, int] = {room: 0 for room in rooms}
    for item in assignments.values():
        if item.get("enabled"):
            room = str(item.get("room") or "")
            room_sensor_counts[room] = room_sensor_counts.get(room, 0) + 1

    return {
        "status": "ok",
        "config": config,
        **config,
        "catalog": {
            "received_at": ha_entity_catalog.get("received_at"),
            "scanned_at": ha_entity_catalog.get("scanned_at"),
            "entities_total": ha_entity_catalog.get("entities_total", 0),
            "supported_total": ha_entity_catalog.get("supported_total", 0),
            "entities": entities,
        },
        "room_sensor_counts": room_sensor_counts,
    }


@app.get("/api/real_sensor_config", tags=[TAG_SYSTEM], summary="Configuracion de sensores reales")
def get_real_sensor_config() -> dict[str, Any]:
    return _real_sensor_payload()


@app.post("/api/real_sensor_config", tags=[TAG_SYSTEM], summary="Actualizar sensores reales")
async def set_real_sensor_config(config: RealSensorConfigInput) -> dict[str, Any]:
    await hub_state.configure_real_sensors(config)
    persist_real_sensor_config()
    await hub_state.broadcast_snapshot()
    return _real_sensor_payload()


@app.get("/api/ha_actions", tags=[TAG_SYSTEM], summary="Estado de acciones Home Assistant")
def list_ha_actions() -> dict[str, Any]:
    return {
        "pending": list(ha_action_queue),
        "recent_results": list(ha_action_results),
        "integration_status": ha_integration_status,
    }


@app.post("/api/ha_integration_status", tags=[TAG_SYSTEM], summary="Actualizar estado integracion Home Assistant")
def update_ha_integration_status(payload: dict[str, Any]) -> dict[str, Any]:
    entry_id = str(payload.get("entry_id") or "default")
    now = to_utc_iso(datetime.now(timezone.utc))
    entry_status = {
        **payload,
        "entry_id": entry_id,
        "last_seen_at": now,
    }
    ha_integration_status["last_seen_at"] = now
    ha_integration_status.setdefault("entries", {})[entry_id] = entry_status
    return {"status": "ok", "integration_status": ha_integration_status}


@app.post("/api/ha_actions", tags=[TAG_SYSTEM], summary="Solicitar accion Home Assistant")
def request_ha_action(req: HAActionRequestInput) -> dict[str, Any]:
    global ha_action_sequence
    ha_action_sequence += 1
    now = to_utc_iso(datetime.now(timezone.utc))
    request_id = f"ha-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{ha_action_sequence}"
    item = {
        "request_id": request_id,
        "status": "pending",
        "action": req.action,
        "entry_id": req.entry_id or "",
        "requested_at": now,
        "payload": {
            "rooms": req.rooms,
            "include_occupancy": req.include_occupancy,
            "initial_state": req.initial_state,
        },
    }
    ha_action_queue.append(item)
    return {"status": "queued", **item, "pending_total": len(ha_action_queue)}


@app.get("/api/ha_actions/pending", tags=[TAG_SYSTEM], summary="Tomar accion pendiente Home Assistant")
def claim_ha_action(entry_id: str = "") -> dict[str, Any]:
    selected: dict[str, Any] | None = None
    for item in list(ha_action_queue):
        target_entry = str(item.get("entry_id") or "")
        if target_entry and entry_id and target_entry != entry_id:
            continue
        selected = item
        break

    if selected is None:
        return {"status": "empty", "action": None}

    ha_action_queue.remove(selected)
    claimed = {
        **selected,
        "status": "claimed",
        "claimed_at": to_utc_iso(datetime.now(timezone.utc)),
        "claimed_by": entry_id,
    }
    return claimed


@app.post("/api/ha_actions/{request_id}/result", tags=[TAG_SYSTEM], summary="Registrar resultado Home Assistant")
def complete_ha_action(request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    result = {
        "request_id": request_id,
        "received_at": to_utc_iso(datetime.now(timezone.utc)),
        **payload,
    }
    ha_action_results.appendleft(result)
    return {"status": "ok", "result": result}


@app.get("/api/presence_filter", tags=[TAG_PRESENCE], summary="Obtener filtro de presencia")
def get_presence_filter() -> dict[str, Any]:
    return hub_state.presence_filter_config()


@app.post("/api/presence_filter", tags=[TAG_PRESENCE], summary="Actualizar filtro de presencia")
async def set_presence_filter(req: PresenceFilterConfigInput) -> dict[str, Any]:
    config = await hub_state.configure_presence_filter(req)
    await hub_state.broadcast_snapshot()
    return config


@app.get("/api/model_info", tags=[TAG_MODEL], summary="Informacion del modelo")
def model_info() -> dict[str, Any]:
    return {
        "ready": hub_state.ai_model.ready,
        "rooms": hub_state.ai_model.rooms,
        "edges": hub_state.ai_model.adjacency_edges,
        "training_info": hub_state.ai_model.training_info,
        "presence_rooms": hub_state.ai_model.occupancy_transformer_rooms,
        "presence_training_info": hub_state.ai_model.occupancy_transformer_info,
        "training_status": training_status,
    }


@app.get("/api/training_exports/{filename}", tags=[TAG_DOWNLOADS], summary="Descargar CSV sintetico")
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


@app.get("/api/input_mode", tags=[TAG_SYSTEM], summary="Obtener modo de entrada")
def get_input_mode() -> dict[str, Any]:
    return {
        "mode": hub_state.input_mode,
        "replay_running": bool(hub_state.replay_task and not hub_state.replay_task.done()),
    }


@app.post("/api/input_mode", tags=[TAG_SYSTEM], summary="Cambiar modo de entrada")
async def set_input_mode(req: RuntimeModeInput) -> dict[str, Any]:
    hub_state.input_mode = req.mode
    if req.mode == "listen":
        hub_state.replay_stop_requested = True
        hub_state.replay_step_budget = 0
        if hub_state.replay_task and not hub_state.replay_task.done():
            hub_state.replay_task.cancel()
        hub_state.replay_paused = False
        await hub_state.broadcast_snapshot()
    return get_input_mode()


@app.get("/api/scenario_templates", tags=[TAG_SCENARIOS], summary="Listar plantillas de escenario")
def scenario_templates() -> dict[str, Any]:
    rooms_set = set(hub_state.ai_model.rooms) | set(hub_state.rooms) | set(hub_state.reference_layout.keys())
    for neighbors in hub_state.reference_layout.values():
        rooms_set.update(neighbors)
    rooms = sorted(rooms_set)
    base_templates = build_scenario_templates(rooms)
    descriptions = {
        "real_home": "Layout base inferido para el hogar real",
        "lineal": "Habitaciones conectadas en cadena",
        "anillo": "Cadena cerrada con retorno",
        "estrella": "Un nodo central conecta todas las salas",
    }

    templates = {
        name: {
            "description": descriptions.get(name, "Template de escenario"),
            "adjacency": adjacency,
            "edges": edge_list_from_adjacency(adjacency),
        }
        for name, adjacency in base_templates.items()
    }
    return {
        "templates": templates,
        "rooms": rooms,
    }


@app.get("/api/layout_reference", tags=[TAG_LAYOUT], summary="Obtener layout de referencia")
async def get_layout_reference() -> dict[str, Any]:
    async with hub_state.lock:
        return {
            "layout_reference": hub_state._layout_payload_locked(),
            "metrics": hub_state._evaluation_metrics_locked(),
        }


@app.post("/api/layout_reference", tags=[TAG_LAYOUT], summary="Actualizar layout de referencia")
async def set_layout_reference(config: LayoutReferenceInput) -> dict[str, Any]:
    try:
        result = await hub_state.configure_reference_layout(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await hub_state.broadcast_snapshot()
    return result


@app.get("/api/evaluation_metrics", tags=[TAG_LAYOUT], summary="Obtener metricas de evaluacion")
async def evaluation_metrics(limit: int = 100) -> dict[str, Any]:
    limit = max(1, min(limit, 500))
    async with hub_state.lock:
        metrics = hub_state._evaluation_metrics_locked()
        metrics["non_adjacent"]["recent"] = metrics["non_adjacent"]["recent"][-limit:]
        return {
            "metrics": metrics,
            "layout_reference": hub_state._layout_payload_locked(),
        }


@app.post("/api/train_model", tags=[TAG_TRAINING], summary="Entrenar modelo desde CSV")
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


@app.post("/api/train_model_full", tags=[TAG_TRAINING], summary="Entrenar modelo completo desde CSV")
async def train_model_full(req: TrainModelFullRequest) -> dict[str, Any]:
    mark_training_status(
        "historical",
        "running",
        "entrenando modelo completo desde CSV",
        request=req.model_dump(),
    )
    """Entrenamiento completo optimizado para máxima captura del historial."""
    # Convertir a TrainModelRequest con los parámetros mejorados
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


@app.post("/api/train_presence_simulator", tags=[TAG_TRAINING], summary="Entrenar presencia desde simulador")
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


@app.post("/api/reset", tags=[TAG_SYSTEM], summary="Reiniciar estado runtime")
async def reset_state() -> dict[str, str]:
    hub_state.replay_stop_requested = True
    if hub_state.replay_task and not hub_state.replay_task.done():
        hub_state.replay_task.cancel()
    await hub_state.reset()
    await hub_state.broadcast_snapshot()
    return {"status": "ok"}


async def _load_csv_events(
    csv_path: str,
    debounce_seconds: int,
    include_all_state_transitions: bool,
) -> list[SensorEventInput]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe CSV: {csv_path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    parsed_rows: list[tuple[datetime, SensorEventInput]] = []
    for row in rows:
        entity_id = str(row.get("entity_id", "")).strip()
        state = str(row.get("state", "")).strip().lower()
        ts_raw = str(row.get("last_changed", "")).strip()

        if not entity_id or not ts_raw:
            continue

        try:
            timestamp = parse_iso_datetime(ts_raw)
        except Exception:
            continue

        sensor_type = classify_sensor_type(entity_id)
        room = normalize_room_name(infer_room_from_entity(entity_id))

        parsed_rows.append(
            (
                timestamp,
                SensorEventInput(
                    entity_id=entity_id,
                    state=state,
                    sensor_type=sensor_type,
                    room=room,
                    timestamp=timestamp,
                    source="csv_replay",
                ),
            )
        )

    parsed_rows.sort(key=lambda item: item[0])

    if include_all_state_transitions:
        return [event for _, event in parsed_rows]

    last_by_entity: dict[str, datetime] = {}
    out: list[SensorEventInput] = []
    for ts, event in parsed_rows:
        if not is_activation(event.sensor_type or "other", event.state):
            continue
        prev = last_by_entity.get(event.entity_id)
        if prev is not None and (ts - prev).total_seconds() <= debounce_seconds:
            continue
        last_by_entity[event.entity_id] = ts
        out.append(event)

    return out


def _normalize_room_mapping(mapping: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for source, target in mapping.items():
        source_room = normalize_room_name(source)
        target_room = normalize_room_name(target)
        if source_room and target_room:
            out[source_room] = target_room
    return out


def _build_scenario_events(
    events: list[SensorEventInput],
    req: CsvReplayRequest,
) -> tuple[list[SensorEventInput], dict[str, list[str]]]:
    if not events:
        return events, {}

    room_mapping = _normalize_room_mapping(req.room_mapping)
    base_rooms = sorted(
        {
            normalize_room_name(event.room)
            for event in events
            if normalize_room_name(event.room)
        }
    )
    mapped_rooms = sorted({room_mapping.get(room, room) for room in base_rooms})
    layout = build_layout_for_request(mapped_rooms, req.template, req.layout_edges)

    scenario_events: list[SensorEventInput] = []
    current_room = ""
    cursor_time = events[0].timestamp or datetime.now(timezone.utc)
    if cursor_time.tzinfo is None:
        cursor_time = cursor_time.replace(tzinfo=timezone.utc)

    for event in events:
        original_room = normalize_room_name(event.room)
        mapped_room = room_mapping.get(original_room, original_room)

        event_time = event.timestamp or cursor_time
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
        else:
            event_time = event_time.astimezone(timezone.utc)
        if event_time <= cursor_time:
            event_time = cursor_time + timedelta(seconds=1)

        if current_room and mapped_room and mapped_room != current_room:
            path = shortest_path_rooms(layout, current_room, mapped_room)
            if path and len(path) > 2:
                for intermediate in path[1:-1]:
                    event_time = event_time + timedelta(seconds=req.step_seconds)
                    scenario_events.append(
                        SensorEventInput(
                            entity_id=f"simulated_sensor.{intermediate}_scenario",
                            state="on",
                            sensor_type="motion",
                            room=intermediate,
                            timestamp=event_time,
                            source=f"csv_scenario_intermediate:{req.template}",
                        )
                    )

        scenario_events.append(
            SensorEventInput(
                entity_id=event.entity_id,
                state=event.state,
                sensor_type=event.sensor_type,
                room=mapped_room,
                timestamp=event_time,
                source=f"csv_scenario:{req.template}",
            )
        )
        cursor_time = event_time

        if is_activation(event.sensor_type or "other", event.state):
            current_room = mapped_room

    return scenario_events, layout


async def _run_csv_replay(req: CsvReplayRequest) -> None:
    await hub_state.reset()
    hub_state.input_mode = "replay"
    hub_state.replay_paused = False
    hub_state.replay_stop_requested = False
    hub_state.replay_step_budget = 0
    hub_state.replay_last_error = None

    if not hub_state.ai_model.ready and os.getenv("AUTO_TRAIN_ON_REPLAY", "1") != "0":
        train_req = TrainModelRequest(
            csv_path=req.csv_path,
            debounce_seconds=req.debounce_seconds,
            include_all_state_transitions=req.include_all_state_transitions,
        )
        try:
            await asyncio.to_thread(hub_state.ai_model.train_from_csv, train_req)
            await hub_state.reset()
        except Exception:
            # Si falla el entrenamiento, replay sigue con modo fallback.
            pass

    hub_state.input_mode = "replay"

    try:
        events = await _load_csv_events(
            req.csv_path,
            req.debounce_seconds,
            req.include_all_state_transitions,
        )
    except Exception:
        hub_state.replay_task = None
        raise

    if req.max_events > 0:
        events = events[: req.max_events]

    scenario_layout: dict[str, list[str]] = {}
    if req.use_scenario_layout:
        events, scenario_layout = _build_scenario_events(events, req)

    hub_state.replay_total_events = len(events)
    hub_state.replay_processed_events = 0

    hub_state.last_replay_config = {
        "csv_path": req.csv_path,
        "speed_events_per_second": req.speed_events_per_second,
        "debounce_seconds": req.debounce_seconds,
        "include_all_state_transitions": req.include_all_state_transitions,
        "max_events": req.max_events,
        "use_scenario_layout": req.use_scenario_layout,
        "template": req.template,
        "layout_edges": edge_list_from_adjacency(scenario_layout),
        "room_mapping": req.room_mapping,
        "step_seconds": req.step_seconds,
        "events_loaded": len(events),
    }

    delay = 1.0 / req.speed_events_per_second
    for event in events:
        if hub_state.replay_stop_requested:
            break
        while hub_state.replay_paused and not hub_state.replay_stop_requested:
            if hub_state.replay_step_budget > 0:
                hub_state.replay_step_budget -= 1
                break
            await asyncio.sleep(0.2)
        if hub_state.replay_stop_requested:
            break

        await hub_state.process_event(event)
        hub_state.replay_processed_events += 1
        await asyncio.sleep(delay)

    hub_state.replay_paused = False
    hub_state.replay_stop_requested = False
    hub_state.replay_step_budget = 0
    hub_state.replay_task = None
    await hub_state.broadcast_snapshot()


@app.post("/api/replay_csv", tags=[TAG_REPLAY], summary="Iniciar replay de CSV")
async def replay_csv(req: CsvReplayRequest) -> dict[str, Any]:
    if hub_state.replay_task and not hub_state.replay_task.done():
        raise HTTPException(status_code=409, detail="Ya hay una simulacion en ejecucion")

    async def _runner() -> None:
        try:
            await _run_csv_replay(req)
        except asyncio.CancelledError:
            hub_state.replay_task = None
            hub_state.replay_paused = False
            hub_state.replay_stop_requested = False
            raise
        except Exception as exc:
            hub_state.replay_last_error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("CSV replay failed after %s/%s events", hub_state.replay_processed_events, hub_state.replay_total_events)
            hub_state.replay_task = None
            hub_state.replay_paused = False
            hub_state.replay_stop_requested = False
            await hub_state.broadcast_snapshot()

    hub_state.input_mode = "replay"
    hub_state.replay_task = asyncio.create_task(_runner())

    return {
        "status": "started",
        "csv_path": req.csv_path,
        "speed_events_per_second": req.speed_events_per_second,
        "debounce_seconds": req.debounce_seconds,
        "include_all_state_transitions": req.include_all_state_transitions,
        "max_events": req.max_events,
        "use_scenario_layout": req.use_scenario_layout,
        "template": req.template,
        "step_seconds": req.step_seconds,
        "input_mode": hub_state.input_mode,
    }


@app.post("/api/replay_control", tags=[TAG_REPLAY], summary="Controlar replay")
async def replay_control(req: ReplayControlInput) -> dict[str, Any]:
    running = bool(hub_state.replay_task and not hub_state.replay_task.done())

    if req.action == "pause":
        if not running:
            raise HTTPException(status_code=409, detail="No hay replay activo para pausar")
        hub_state.replay_paused = True

    elif req.action == "start":
        if not running:
            raise HTTPException(status_code=409, detail="No hay replay activo. Inicia uno nuevo con /api/replay_csv")
        hub_state.replay_paused = False
        hub_state.replay_step_budget = 0

    elif req.action == "step":
        if not running:
            raise HTTPException(status_code=409, detail="No hay replay activo para avanzar paso a paso")
        if not hub_state.replay_paused:
            raise HTTPException(status_code=409, detail="Step solo disponible cuando el replay esta pausado")
        hub_state.replay_step_budget += 1

    elif req.action == "reset":
        hub_state.replay_stop_requested = True
        hub_state.replay_step_budget = 0
        if hub_state.replay_task and not hub_state.replay_task.done():
            hub_state.replay_task.cancel()
        await hub_state.reset()
        await hub_state.broadcast_snapshot()

    return replay_status()


@app.get("/api/replay_status", tags=[TAG_REPLAY], summary="Estado del replay")
def replay_status() -> dict[str, Any]:
    running = bool(hub_state.replay_task and not hub_state.replay_task.done())
    return {
        "running": running,
        "mode": hub_state.input_mode,
        "paused": hub_state.replay_paused,
        "step_budget": hub_state.replay_step_budget,
        "events": len(hub_state.events),
        "rooms": len(hub_state.rooms),
        "model_ready": hub_state.ai_model.ready,
        "processed_events": hub_state.replay_processed_events,
        "total_events": hub_state.replay_total_events,
        "last_error": hub_state.replay_last_error,
        "progress": (
            round(hub_state.replay_processed_events / hub_state.replay_total_events, 4)
            if hub_state.replay_total_events > 0
            else 0.0
        ),
        "last_replay_config": hub_state.last_replay_config,
    }


@app.websocket("/presencia")
async def presencia_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    hub_state.websockets.add(websocket)
    try:
        await websocket.send_json({"kind": "snapshot", "sim_data": hub_state.snapshot()})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        hub_state.websockets.discard(websocket)
    except Exception:
        hub_state.websockets.discard(websocket)


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
else:

    @app.get("/", tags=[TAG_SYSTEM], summary="Fallback de interfaz web")
    def root_fallback() -> dict[str, str]:
        return {
            "message": "inferencia_hub operativo, pero WEB_DIR no existe",
            "web_dir": str(WEB_DIR),
        }
