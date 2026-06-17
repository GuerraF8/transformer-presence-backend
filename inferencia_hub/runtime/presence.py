"""Operaciones de ingesta, consulta y publicación de presencia."""

from .shared import *  # noqa: F401,F403
from .lifecycle import activate_listen_mode
from .live_training import live_training_status


def _parse_people_count_state(value: Any) -> int | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        number = float(raw)
    except ValueError:
        return None
    if not (0 <= number < 1000000):
        return None
    count = int(number)
    return count if number == float(count) else None


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
            or hub_state.ai_model.relative_occupancy_info.get("enabled")
        ),
        "input_mode": hub_state.input_mode,
        "profile": hub_state._profile_payload_locked(),
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
        "history": history_store.status(),
    }


def catalog_has_entity(entity_id: str) -> bool:
    return ha_entity_catalog.has_entity(entity_id)


async def ingest_event(payload: SensorEventInput) -> dict[str, Any]:
    if not hub_state.active_profile_id:
        return {
            "status": "ignored",
            "reason": "no_active_profile",
            "input_mode": hub_state.input_mode,
        }
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
    if is_real_ha and not catalog_has_entity(payload.entity_id):
        return {
            "status": "ignored",
            "reason": "sensor_not_in_ha_catalog",
            "input_mode": hub_state.input_mode,
        }
    if not is_csv and not is_simulator and not is_real_ha:
        return {
            "status": "ignored",
            "reason": "unknown_event_source",
            "input_mode": hub_state.input_mode,
        }
    assignment = hub_state.real_sensor_assignments.get(
        str(payload.entity_id or "").strip().lower()
    )
    training_role = str(
        (assignment or {}).get("training_role") or "signal"
    )
    if training_role in {
        "person_confirmation",
        "pet_confirmation",
        "people_count_confirmation",
    }:
        if not is_real_ha or hub_state.input_mode != "listen":
            return {
                "status": "ignored",
                "reason": "confirmation_requires_listen_mode",
                "input_mode": hub_state.input_mode,
            }
        if not assignment or not assignment.get("enabled", True):
            return {
                "status": "ignored",
                "reason": "confirmation_not_enabled",
                "input_mode": hub_state.input_mode,
            }
        timestamp = to_utc_iso(
            payload.timestamp or datetime.now(timezone.utc)
        )
        numeric_value = None
        count_value = None
        if training_role == "people_count_confirmation":
            count_value = _parse_people_count_state(payload.state)
            if count_value is None:
                return {
                    "status": "ignored",
                    "reason": "count_confirmation_state_not_numeric",
                    "entity_id": str(payload.entity_id or "").strip().lower(),
                    "state": str(payload.state or ""),
                    "input_mode": hub_state.input_mode,
                }
            numeric_value = float(count_value)
        confirmation_id = await asyncio.to_thread(
            live_training_store.record_confirmation,
            timestamp=timestamp,
            entity_id=str(payload.entity_id).strip().lower(),
            state=str(payload.state or "").strip().lower(),
            training_role=training_role,
            room=str(assignment.get("room") or ""),
            profile_id=str(hub_state.active_profile_id or ""),
            profile_revision=int(hub_state.active_profile_revision or 1),
            profile_fingerprint=str(
                hub_state.active_profile_fingerprint or ""
            ),
            numeric_value=numeric_value,
        )
        event_timestamp = payload.timestamp or datetime.now(timezone.utc)
        async with hub_state.lock:
            if training_role == "people_count_confirmation":
                hub_state._apply_count_ground_truth_locked(
                    timestamp=event_timestamp,
                    entity_id=str(payload.entity_id).strip().lower(),
                    room=str(assignment.get("room") or ""),
                    count=int(count_value or 0),
                )
            else:
                hub_state._record_confirmation_ground_truth_locked(
                    timestamp=event_timestamp,
                    entity_id=str(payload.entity_id).strip().lower(),
                    state=str(payload.state or "").strip().lower(),
                    training_role=training_role,
                    room=str(assignment.get("room") or ""),
                )
        await hub_state.broadcast_snapshot()
        return {
            "status": "recorded",
            "reason": "training_confirmation",
            "confirmation_id": confirmation_id,
            "training_role": training_role,
            "room": assignment.get("room"),
            "numeric_value": numeric_value,
            "input_mode": hub_state.input_mode,
        }
    if (
        is_real_ha
        and hub_state.input_mode == "listen"
        and assignment
        and assignment.get("enabled", True)
        and training_role == "signal"
    ):
        assigned_type = str(
            assignment.get("sensor_type") or payload.sensor_type or "other"
        )
        if assigned_type == "auto":
            assigned_type = str(payload.sensor_type or "other")
        await asyncio.to_thread(
            live_training_store.record_signal,
            timestamp=to_utc_iso(
                payload.timestamp or datetime.now(timezone.utc)
            ),
            entity_id=str(payload.entity_id).strip().lower(),
            sensor_type=assigned_type,
            room=str(assignment.get("room") or payload.room or ""),
            state=str(payload.state or "").strip().lower(),
            profile_id=str(hub_state.active_profile_id or ""),
            profile_revision=int(hub_state.active_profile_revision or 1),
            profile_fingerprint=str(
                hub_state.active_profile_fingerprint or ""
            ),
        )
    return await hub_state.process_event(payload)


def get_sim_data() -> dict[str, Any]:
    snapshot = hub_state.snapshot()
    snapshot["ha_entity_catalog"] = ha_entity_catalog.as_dict()
    snapshot["live_training"] = live_training_status()
    return snapshot


def get_presence_filter() -> dict[str, Any]:
    return hub_state.presence_filter_config()


async def set_presence_filter(req: PresenceFilterConfigInput) -> dict[str, Any]:
    config = await hub_state.configure_presence_filter(req)
    await hub_state.broadcast_snapshot()
    return config


def model_info() -> dict[str, Any]:
    return {
        "profile": hub_state._profile_payload_locked(),
        "ready": hub_state.ai_model.ready,
        "rooms": hub_state.ai_model.rooms,
        "edges": hub_state.ai_model.adjacency_edges,
        "training_info": hub_state.ai_model.training_info,
        "presence_rooms": hub_state.ai_model.occupancy_transformer_rooms,
        "presence_training_info": hub_state.ai_model.occupancy_transformer_info,
        "pet_filter": hub_state.ai_model.pet_filter_info,
        "relative_occupancy": hub_state.ai_model.relative_occupancy_info,
        "live_training": live_training_status(),
        "training_status": training_status,
    }


def get_input_mode() -> dict[str, Any]:
    return {
        "mode": hub_state.input_mode,
        "replay_running": bool(hub_state.replay_task and not hub_state.replay_task.done()),
    }


async def set_input_mode(req: RuntimeModeInput) -> dict[str, Any]:
    if req.mode == "listen":
        await activate_listen_mode()
    else:
        hub_state.input_mode = req.mode
    return get_input_mode()


async def reset_state() -> dict[str, str]:
    hub_state.replay_stop_requested = True
    if hub_state.replay_task and not hub_state.replay_task.done():
        hub_state.replay_task.cancel()
    await hub_state.reset()
    await hub_state.broadcast_snapshot()
    return {"status": "ok"}


async def presencia_socket(websocket: WebSocket) -> None:
    await context.websocket.connect(websocket)
    try:
        await websocket.send_json({"kind": "snapshot", "sim_data": hub_state.snapshot()})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        context.websocket.disconnect(websocket)
    except Exception:
        context.websocket.disconnect(websocket)
