"""Operaciones del catálogo, sensores y acciones de Home Assistant."""

from .shared import *  # noqa: F401,F403
from .lifecycle import activate_listen_mode
from .profiles import (
    _apply_profile,
    _catalog_entities,
    reconcile_profiles_with_catalog,
)


def get_ha_entities() -> dict[str, Any]:
    return ha_entity_catalog.as_dict()


async def update_ha_entities(payload: HAEntityCatalogInput) -> dict[str, Any]:
    entities = [entity.model_dump() for entity in payload.entities]
    catalog_payload = {
        "source": payload.source,
        "entry_id": payload.entry_id,
        "scanned_at": to_utc_iso(payload.scanned_at or datetime.now(timezone.utc)),
        "received_at": to_utc_iso(datetime.now(timezone.utc)),
        "auto_discovery": payload.auto_discovery,
        "tracked_entities": sorted(set(payload.tracked_entities)),
        "areas": [area.model_dump() for area in payload.areas],
        "entities": entities,
        "entities_total": len(entities),
        "supported_total": len([entity for entity in entities if entity.get("supported")]),
    }
    ha_entity_catalog.replace(catalog_payload)
    reconciliation = await reconcile_profiles_with_catalog()
    await hub_state.broadcast_snapshot()
    return {"status": "ok", "reconciliation": reconciliation, **catalog_payload}


def _real_sensor_payload() -> dict[str, Any]:
    config = hub_state.real_sensor_config()
    entities = ha_entity_catalog.get("entities")
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
        "profile": hub_state._profile_payload_locked(),
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


def get_real_sensor_config() -> dict[str, Any]:
    return _real_sensor_payload()


async def set_real_sensor_config(config: RealSensorConfigInput) -> dict[str, Any]:
    profile = profile_store.active()
    if profile is None:
        raise HTTPException(
            status_code=409,
            detail="Debe crear y activar un perfil antes de seleccionar sensores",
        )
    room_by_slug = {
        str(room.get("slug") or ""): dict(room)
        for room in profile.get("rooms", [])
        if isinstance(room, dict)
    }
    for room_name in config.rooms:
        slug = normalize_room_name(room_name)
        if slug and slug not in room_by_slug:
            room_by_slug[slug] = {
                "slug": slug,
                "name": str(room_name).replace("_", " "),
                "area_id": "",
                "area_name": "",
            }
    catalog_by_entity = {
        str(item.get("entity_id") or "").strip().lower(): item
        for item in _catalog_entities()
        if isinstance(item, dict)
    }
    assignments = []
    for assignment in config.assignments:
        entity_id = str(assignment.entity_id or "").strip().lower()
        room_slug = normalize_room_name(assignment.room)
        if not entity_id or not room_slug:
            continue
        entity = catalog_by_entity.get(entity_id, {})
        sensor_type = str(assignment.sensor_type or "auto")
        if sensor_type == "auto":
            sensor_type = str(entity.get("sensor_type") or "other")
        if sensor_type not in {"motion", "door", "occupancy", "other"}:
            sensor_type = "other"
        assignments.append(
            {
                "entity_id": entity_id,
                "room_slug": room_slug,
                "enabled": bool(assignment.enabled),
                "sensor_type": sensor_type,
                "area_id": str(entity.get("area_id") or ""),
                "area_name": str(entity.get("area_name") or ""),
                "status": "active",
                "warning": "",
                "unique_id": str(entity.get("unique_id") or ""),
                "platform": str(entity.get("platform") or ""),
            }
        )
    updated = await asyncio.to_thread(
        profile_store.update,
        profile["id"],
        {
            **profile,
            "rooms": list(room_by_slug.values()),
            "assignments": assignments,
        },
        expected_revision=int(profile["revision"]),
    )
    await _apply_profile(updated)
    if hub_state.real_sensor_config().get("enabled_entities"):
        await activate_listen_mode()
    return _real_sensor_payload()


def list_ha_actions() -> dict[str, Any]:
    return {
        "pending": list(context.actions.pending),
        "recent_results": list(context.actions.recent_results),
        "integration_status": context.actions.integration_status,
    }


def update_ha_integration_status(payload: dict[str, Any]) -> dict[str, Any]:
    entry_id = str(payload.get("entry_id") or "default")
    now = to_utc_iso(datetime.now(timezone.utc))
    entry_status = {
        **payload,
        "entry_id": entry_id,
        "last_seen_at": now,
    }
    context.actions.integration_status["last_seen_at"] = now
    context.actions.integration_status.setdefault("entries", {})[entry_id] = entry_status
    return {"status": "ok", "integration_status": context.actions.integration_status}


def request_ha_action(req: HAActionRequestInput) -> dict[str, Any]:
    return context.actions.request(
        req.action,
        req.entry_id or "",
        {
            "rooms": req.rooms,
            "include_occupancy": req.include_occupancy,
            "initial_state": req.initial_state,
        },
    )


def claim_ha_action(entry_id: str = "") -> dict[str, Any]:
    return context.actions.claim(entry_id)


def complete_ha_action(request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    result = context.actions.complete(request_id, payload)
    return {"status": "ok", "result": result}
