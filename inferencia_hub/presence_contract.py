from __future__ import annotations

from typing import Any


def build_presence_snapshot(
    *,
    current_room: str | None,
    active_rooms: list[str],
    people_estimate: int,
    latest_event: dict[str, Any] | None,
    occupancy_ground_truth_rooms: list[str],
    live_sensor_rooms: list[str],
) -> dict[str, Any]:
    return {
        "current_room": current_room,
        "active_rooms": active_rooms,
        "inferred_presence": bool(active_rooms),
        "people_estimate": people_estimate,
        "confidence": latest_event.get("presence_confidence") if latest_event else None,
        "updated_at": latest_event.get("timestamp") if latest_event else None,
        "occupancy_ground_truth_rooms": occupancy_ground_truth_rooms,
        "live_sensor_rooms": live_sensor_rooms,
    }
