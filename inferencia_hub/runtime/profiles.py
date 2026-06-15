"""Operaciones de perfiles persistentes de presencia."""

from __future__ import annotations

import shutil

import numpy as np

from .shared import *  # noqa: F401,F403
from ..profile_store import (
    ProfileNotFoundError,
    ProfileRevisionError,
    normalize_profile,
)


def _room_slug(value: str) -> str:
    return normalize_room_name(value) or "habitacion"


def _catalog_entities() -> list[dict[str, Any]]:
    entities = ha_entity_catalog.get("entities", [])
    return entities if isinstance(entities, list) else []


def _catalog_areas() -> list[dict[str, Any]]:
    areas = ha_entity_catalog.get("areas", [])
    return areas if isinstance(areas, list) else []


def _profile_summary(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        **profile,
        "active": profile.get("id") == profile_store.snapshot().get(
            "active_profile_id"
        ),
        "model": _profile_model_status(profile),
    }


def _model_root() -> Path:
    return Path(
        os.getenv(
            "MODEL_STATE_DIR",
            str(Path(os.getenv("INFERENCIA_DATA_DIR", "/app/data")) / "model_state"),
        )
    )


def _profile_model_dir(profile_id: str) -> Path:
    return _model_root() / "profiles" / profile_id


def _profile_model_status(profile: dict[str, Any]) -> dict[str, Any]:
    state_path = _profile_model_dir(str(profile.get("id") or "")) / "model_state.json"
    if not state_path.exists():
        return {"available": False, "compatible": False}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"available": True, "compatible": False}
    stored = str(
        dict(payload.get("training_info") or {}).get("profile_fingerprint") or ""
    )
    return {
        "available": True,
        "compatible": bool(stored and stored == profile.get("fingerprint")),
        "fingerprint": stored or None,
    }


def _real_home_profile(name: str) -> dict[str, Any]:
    rooms = sorted({room for edge in REAL_HOME_LAYOUT_EDGES for room in edge})
    return {
        "name": name,
        "source": "real_home",
        "rooms": [
            {
                "slug": room,
                "name": room.replace("_", " "),
                "area_id": "",
                "area_name": "",
            }
            for room in rooms
        ],
        "areas": [],
        "assignments": [],
        "edges": [list(edge) for edge in REAL_HOME_LAYOUT_EDGES],
    }


def _detected_profile(req: ProfileCreateInput) -> dict[str, Any]:
    area_by_id = {
        str(area.get("area_id") or ""): area
        for area in _catalog_areas()
        if isinstance(area, dict) and str(area.get("area_id") or "")
    }
    selected_area_ids = req.area_ids or sorted(area_by_id)
    rooms: list[dict[str, Any]] = []
    areas: list[dict[str, Any]] = []
    room_by_area: dict[str, str] = {}
    used_slugs: set[str] = set()
    for area_id in selected_area_ids:
        area = area_by_id.get(area_id)
        if area is None:
            continue
        base = _room_slug(str(area.get("name") or area_id))
        slug = base
        suffix = 2
        while slug in used_slugs:
            slug = f"{base}_{suffix}"
            suffix += 1
        used_slugs.add(slug)
        room_by_area[area_id] = slug
        rooms.append(
            {
                "slug": slug,
                "name": str(area.get("name") or slug),
                "area_id": area_id,
                "area_name": str(area.get("name") or ""),
            }
        )
        areas.append(
            {
                "area_id": area_id,
                "room_slug": slug,
                "name": str(area.get("name") or ""),
            }
        )

    selected_entities = set(req.entity_ids)
    if not selected_entities and selected_area_ids:
        selected_entities = {
            str(entity.get("entity_id") or "")
            for entity in _catalog_entities()
            if str(entity.get("area_id") or "") in room_by_area
        }
    assignments: list[dict[str, Any]] = []
    for entity in _catalog_entities():
        entity_id = str(entity.get("entity_id") or "")
        if entity_id not in selected_entities:
            continue
        area_id = str(entity.get("area_id") or "")
        room_slug = room_by_area.get(area_id)
        if not room_slug:
            room_slug = _room_slug(
                str(entity.get("room") or entity_id.split(".", 1)[-1])
            )
            if room_slug not in used_slugs:
                used_slugs.add(room_slug)
                rooms.append(
                    {
                        "slug": room_slug,
                        "name": room_slug.replace("_", " "),
                        "area_id": "",
                        "area_name": "",
                    }
                )
        sensor_type = str(entity.get("sensor_type") or "other")
        if sensor_type not in {"motion", "door", "occupancy", "other"}:
            sensor_type = "other"
        assignments.append(
            {
                "entity_id": entity_id,
                "room_slug": room_slug,
                "enabled": True,
                "sensor_type": sensor_type,
                "training_role": "signal",
                "area_id": area_id,
                "area_name": str(entity.get("area_name") or ""),
                "status": "active",
                "warning": "",
                "unique_id": str(entity.get("unique_id") or ""),
                "platform": str(entity.get("platform") or ""),
            }
        )
    return {
        "name": req.name,
        "source": "detected",
        "rooms": rooms,
        "areas": areas,
        "assignments": assignments,
        "edges": [],
    }


def _build_profile(req: ProfileCreateInput) -> dict[str, Any]:
    if req.source == "real_home":
        return _real_home_profile(req.name)
    if req.source == "detected":
        return _detected_profile(req)
    return {
        "name": req.name,
        "source": "manual",
        "rooms": [],
        "areas": [],
        "assignments": [],
        "edges": [],
    }


async def _apply_profile(profile: dict[str, Any]) -> None:
    persist_packaged_model = False
    async with hub_state.lock:
        hub_state.apply_profile(profile)
        candidate = hub_state.ai_model
        model_status = _profile_model_status(profile)
        if model_status.get("compatible"):
            candidate = AIAdjacencyModel()
            loaded = await asyncio.to_thread(
                candidate.load_state,
                _profile_model_dir(profile["id"]),
            )
            if loaded.get("loaded"):
                hub_state.rooms.update(candidate.rooms)
                hub_state.active_profile_model_compatible = True
                count = len(candidate.rooms)
                hub_state.presence_belief = (
                    np.full((count,), 1.0 / count, dtype=np.float32)
                    if count
                    else np.zeros((0,), dtype=np.float32)
                )
        if (
            getattr(candidate, "pet_filter_model", None)
            is None
            and hasattr(
                candidate,
                "load_packaged_pet_filter",
            )
        ):
            await asyncio.to_thread(
                candidate.load_packaged_pet_filter,
            )
        if (
            getattr(candidate, "relative_occupancy_model", None)
            is None
            and getattr(candidate, "occupancy_transformer_model", None)
            is None
            and hasattr(candidate, "load_packaged_relative_occupancy")
        ):
            loaded_relative = await asyncio.to_thread(
                candidate.load_packaged_relative_occupancy,
            )
            persist_packaged_model = bool(loaded_relative.get("loaded"))
        candidate.rooms = sorted(hub_state.reference_layout)
        candidate.room_to_idx = {
            room: index for index, room in enumerate(candidate.rooms)
        }
        transition_matrix = getattr(
            candidate,
            "transition_matrix",
            np.zeros((0, 0), dtype=np.float32),
        )
        if transition_matrix.shape != (
            len(candidate.rooms),
            len(candidate.rooms),
        ):
            candidate.transition_matrix = np.eye(
                len(candidate.rooms),
                dtype=np.float32,
            )
        candidate.adjacency_neighbors = {
            room: list(neighbors)
            for room, neighbors in hub_state.reference_layout.items()
        }
        candidate.ready = bool(candidate.rooms)
        hub_state.ai_model = candidate
        if persist_packaged_model:
            candidate.training_info["profile_fingerprint"] = (
                hub_state.active_profile_fingerprint
            )
            hub_state.active_profile_model_compatible = True
    if persist_packaged_model:
        await asyncio.to_thread(
            candidate.save_state,
            _profile_model_dir(profile["id"]),
        )
    await hub_state.broadcast_snapshot()


def list_profiles() -> dict[str, Any]:
    snapshot = profile_store.snapshot()
    return {
        "profiles": [
            _profile_summary(profile)
            for profile in snapshot.get("profiles", [])
        ],
        "active_profile_id": snapshot.get("active_profile_id"),
        "active_profile": (
            _profile_summary(profile_store.active())
            if profile_store.active()
            else None
        ),
    }


def get_profile(profile_id: str) -> dict[str, Any]:
    try:
        return _profile_summary(profile_store.get(profile_id))
    except ProfileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Perfil no encontrado") from exc


async def create_profile(req: ProfileCreateInput) -> dict[str, Any]:
    profile = await asyncio.to_thread(profile_store.create, _build_profile(req))
    return _profile_summary(profile)


async def update_profile(
    profile_id: str,
    req: ProfileUpdateInput,
) -> dict[str, Any]:
    try:
        updated = await asyncio.to_thread(
            profile_store.update,
            profile_id,
            {
                "name": req.name,
                "rooms": [room.model_dump() for room in req.rooms],
                "areas": [area.model_dump() for area in req.areas],
                "assignments": [
                    assignment.model_dump()
                    for assignment in req.assignments
                ],
                "edges": req.edges,
            },
            expected_revision=req.revision,
        )
    except ProfileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Perfil no encontrado") from exc
    except ProfileRevisionError as exc:
        raise HTTPException(
            status_code=409,
            detail="El perfil fue modificado; vuelve a cargarlo",
        ) from exc
    if profile_store.snapshot().get("active_profile_id") == profile_id:
        await _apply_profile(updated)
    return _profile_summary(updated)


async def activate_profile(profile_id: str) -> dict[str, Any]:
    try:
        profile = await asyncio.to_thread(profile_store.activate, profile_id)
    except ProfileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Perfil no encontrado") from exc
    await _apply_profile(profile)
    return _profile_summary(profile)


async def delete_profile(profile_id: str) -> dict[str, Any]:
    try:
        profile, was_active = await asyncio.to_thread(
            profile_store.delete,
            profile_id,
        )
    except ProfileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Perfil no encontrado") from exc
    model_dir = _profile_model_dir(profile_id)
    if model_dir.exists():
        await asyncio.to_thread(shutil.rmtree, model_dir)
    if was_active:
        async with hub_state.lock:
            hub_state.clear_active_profile()
        await hub_state.broadcast_snapshot()
    return {"status": "ok", "deleted": profile, "was_active": was_active}


async def infer_profile_layout(
    profile_id: str,
    req: ProfileInferLayoutInput,
) -> dict[str, Any]:
    profile = get_profile(profile_id)
    enabled = [
        str(item.get("entity_id") or "")
        for item in profile.get("assignments", [])
        if item.get("enabled") and item.get("status") == "active"
    ]
    supports = await asyncio.to_thread(
        history_store.transition_support,
        enabled,
        max_gap_seconds=req.max_gap_seconds,
    )
    room_slugs = {
        str(room.get("slug") or "")
        for room in profile.get("rooms", [])
    }
    proposals = [
        item
        for item in supports
        if item["support"] >= req.min_support
        and item["a"] in room_slugs
        and item["b"] in room_slugs
    ]
    return {
        "profile_id": profile_id,
        "min_support": req.min_support,
        "max_gap_seconds": req.max_gap_seconds,
        "proposals": proposals,
        "source_events": len(supports),
    }


async def reconcile_profiles_with_catalog() -> dict[str, int]:
    entities = {
        str(item.get("entity_id") or "").strip().lower(): item
        for item in _catalog_entities()
        if isinstance(item, dict)
        and str(item.get("entity_id") or "").strip()
    }
    entities_by_registry_key = {
        (
            str(item.get("platform") or ""),
            str(item.get("unique_id") or ""),
        ): item
        for item in _catalog_entities()
        if isinstance(item, dict)
        and str(item.get("platform") or "")
        and str(item.get("unique_id") or "")
    }
    areas = {
        str(item.get("area_id") or ""): item
        for item in _catalog_areas()
        if isinstance(item, dict) and str(item.get("area_id") or "")
    }
    changed = 0
    disabled = 0
    active_updated: dict[str, Any] | None = None
    for profile in profile_store.list_profiles():
        next_profile = dict(profile)
        next_profile["rooms"] = [dict(room) for room in profile.get("rooms", [])]
        next_profile["areas"] = [dict(area) for area in profile.get("areas", [])]
        next_profile["assignments"] = [
                    dict(item) for item in profile.get("assignments", [])
        ]
        profile_changed = False
        for area in next_profile["areas"]:
            catalog_area = areas.get(str(area.get("area_id") or ""))
            if catalog_area and area.get("name") != catalog_area.get("name"):
                area["name"] = str(catalog_area.get("name") or "")
                profile_changed = True
        for room in next_profile["rooms"]:
            room_area_id = str(room.get("area_id") or "")
            catalog_area = areas.get(room_area_id)
            if catalog_area:
                new_name = str(catalog_area.get("name") or room.get("name") or "")
                if (
                    room.get("name") != new_name
                    or room.get("area_name") != new_name
                    or room.get("status") != "active"
                    or room.get("warning")
                ):
                    room["name"] = new_name
                    room["area_name"] = new_name
                    room["status"] = "active"
                    room["warning"] = ""
                    profile_changed = True
            elif room_area_id and (
                room.get("status") != "missing" or not room.get("warning")
            ):
                room["status"] = "missing"
                room["warning"] = "El área ya no existe en Home Assistant"
                profile_changed = True
        for assignment in next_profile["assignments"]:
            entity = entities.get(
                str(assignment.get("entity_id") or "").strip().lower()
            )
            registry_key = (
                str(assignment.get("platform") or ""),
                str(assignment.get("unique_id") or ""),
            )
            if entity is None and all(registry_key):
                entity = entities_by_registry_key.get(registry_key)
                if entity is not None:
                    assignment["entity_id"] = str(
                        entity.get("entity_id") or ""
                    ).strip().lower()
                    profile_changed = True
            expected_area = str(assignment.get("area_id") or "")
            if entity is None:
                if (
                    assignment.get("status") != "missing"
                    or assignment.get("enabled")
                ):
                    assignment["status"] = "missing"
                    assignment["enabled"] = False
                    assignment["warning"] = "Entidad no disponible en Home Assistant"
                    disabled += 1
                    profile_changed = True
                continue
            current_area = str(entity.get("area_id") or "")
            has_registry_identity = bool(
                assignment.get("platform") and assignment.get("unique_id")
            )
            if not has_registry_identity:
                assignment["area_id"] = current_area
                expected_area = current_area
                profile_changed = True
            elif current_area != expected_area:
                if (
                    assignment.get("status") != "moved"
                    or assignment.get("enabled")
                ):
                    assignment["status"] = "moved"
                    assignment["enabled"] = False
                    assignment["warning"] = (
                        "La entidad cambio de area en Home Assistant"
                    )
                    disabled += 1
                    profile_changed = True
                continue
            for key in ("platform", "unique_id"):
                value = str(entity.get(key) or "")
                if assignment.get(key) != value:
                    assignment[key] = value
                    profile_changed = True
            area_name = str(entity.get("area_name") or "")
            if assignment.get("area_name") != area_name:
                assignment["area_name"] = area_name
                profile_changed = True
            if assignment.get("status") != "active":
                assignment["status"] = "active"
                assignment["warning"] = ""
                profile_changed = True
        if not profile_changed:
            continue
        updated = profile_store.update(
            profile["id"],
            next_profile,
            expected_revision=int(profile["revision"]),
        )
        changed += 1
        if profile_store.snapshot().get("active_profile_id") == profile["id"]:
            active_updated = updated
    if active_updated is not None:
        await _apply_profile(active_updated)
    return {"profiles_changed": changed, "assignments_disabled": disabled}


def _legacy_profile_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rooms = sorted(
        {
            normalize_room_name(str(room))
            for room in payload.get("rooms", [])
            if normalize_room_name(str(room))
        }
    )
    assignments = []
    for item in payload.get("assignments", []):
        if not isinstance(item, dict):
            continue
        room = normalize_room_name(str(item.get("room") or ""))
        entity_id = str(item.get("entity_id") or "").strip().lower()
        if not room or not entity_id:
            continue
        sensor_type = str(item.get("sensor_type") or "other")
        assignments.append(
            {
                "entity_id": entity_id,
                "room_slug": room,
                "enabled": bool(item.get("enabled", True)),
                "sensor_type": (
                    sensor_type
                    if sensor_type in {"motion", "door", "occupancy", "other"}
                    else "other"
                ),
                "training_role": (
                    str(item.get("training_role") or "signal")
                    if str(item.get("training_role") or "signal")
                    in {"signal", "person_confirmation", "pet_confirmation"}
                    else "signal"
                ),
                "status": "active",
                "unique_id": "",
                "platform": "",
            }
        )
    edges = [
        list(edge)
        for edge in REAL_HOME_LAYOUT_EDGES
        if edge[0] in rooms and edge[1] in rooms
    ]
    return normalize_profile(
        {
            "name": "Configuracion migrada",
            "source": "migrated",
            "rooms": [
                {"slug": room, "name": room.replace("_", " ")}
                for room in rooms
            ],
            "areas": [],
            "assignments": assignments,
            "edges": edges,
        }
    )


def initialize_profiles(legacy_config_path: Path) -> dict[str, Any] | None:
    profile_store.load()
    if not profile_store.list_profiles() and legacy_config_path.exists():
        try:
            legacy = json.loads(legacy_config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            legacy = None
        if isinstance(legacy, dict):
            migrated = profile_store.create(
                _legacy_profile_payload(legacy),
                activate=True,
            )
            old_model_dir = _model_root()
            new_model_dir = _profile_model_dir(migrated["id"])
            old_state = old_model_dir / "model_state.json"
            if old_state.exists() and not new_model_dir.exists():
                new_model_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(
                    old_model_dir,
                    new_model_dir,
                    ignore=shutil.ignore_patterns("profiles"),
                )
                copied_state = new_model_dir / "model_state.json"
                try:
                    model_payload = json.loads(
                        copied_state.read_text(encoding="utf-8")
                    )
                    profile_rooms = {
                        room["slug"] for room in migrated.get("rooms", [])
                    }
                    model_rooms = {
                        str(room) for room in model_payload.get("rooms", [])
                    }
                    if model_rooms == profile_rooms:
                        training_info = dict(
                            model_payload.get("training_info") or {}
                        )
                        training_info["profile_fingerprint"] = migrated[
                            "fingerprint"
                        ]
                        model_payload["training_info"] = training_info
                        copied_state.write_text(
                            json.dumps(
                                model_payload,
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                except (OSError, ValueError):
                    pass
            return migrated
    return profile_store.active()
