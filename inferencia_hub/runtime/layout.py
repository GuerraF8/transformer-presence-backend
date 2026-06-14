"""Operaciones de escenarios, mapas de referencia y métricas."""

from .shared import *  # noqa: F401,F403
from .profiles import _apply_profile


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


async def get_layout_reference() -> dict[str, Any]:
    async with hub_state.lock:
        return {
            "layout_reference": hub_state._layout_payload_locked(),
            "metrics": hub_state._evaluation_metrics_locked(),
        }


async def set_layout_reference(config: LayoutReferenceInput) -> dict[str, Any]:
    profile = profile_store.active()
    if profile is None:
        raise HTTPException(
            status_code=409,
            detail="Debe crear y activar un perfil antes de editar el mapa",
        )
    try:
        result = await hub_state.configure_reference_layout(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    layout = result["layout_reference"]
    existing_rooms = {
        str(room.get("slug") or ""): dict(room)
        for room in profile.get("rooms", [])
        if isinstance(room, dict)
    }
    for slug in layout.get("rooms", []):
        existing_rooms.setdefault(
            slug,
            {
                "slug": slug,
                "name": str(slug).replace("_", " "),
                "area_id": "",
                "area_name": "",
            },
        )
    updated = await asyncio.to_thread(
        profile_store.update,
        profile["id"],
        {
            **profile,
            "rooms": list(existing_rooms.values()),
            "edges": [
                [edge["a"], edge["b"]]
                for edge in layout.get("edges", [])
            ],
        },
        expected_revision=int(profile["revision"]),
    )
    await _apply_profile(updated)
    return {
        **result,
        "profile": updated,
    }


async def evaluation_metrics(limit: int = 100) -> dict[str, Any]:
    limit = max(1, min(limit, 500))
    async with hub_state.lock:
        metrics = hub_state._evaluation_metrics_locked()
        metrics["non_adjacent"]["recent"] = metrics["non_adjacent"]["recent"][-limit:]
        return {
            "metrics": metrics,
            "layout_reference": hub_state._layout_payload_locked(),
        }
