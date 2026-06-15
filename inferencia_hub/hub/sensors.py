"""Configuración y asignación de sensores reales."""

from .dependencies import *  # noqa: F401,F403


class SensorsMixin:
    def _real_sensor_config_locked(self) -> dict[str, Any]:
        assignments = sorted(
            [
                {
                    "entity_id": entity_id,
                    "room": str(config.get("room") or ""),
                    "enabled": bool(config.get("enabled", True)),
                    "sensor_type": str(config.get("sensor_type") or "auto"),
                    "training_role": str(
                        config.get("training_role") or "signal"
                    ),
                }
                for entity_id, config in self.real_sensor_assignments.items()
            ],
            key=lambda item: item["entity_id"],
        )
        enabled_assignments = [item for item in assignments if item["enabled"]]
        return {
            "rooms": sorted(self.real_sensor_rooms),
            "assignments": assignments,
            "enabled_entities": [item["entity_id"] for item in enabled_assignments],
            "require_explicit_selection": self.real_sensor_require_explicit_selection,
            "rejected_events": self.real_sensor_rejected_events,
            "last_rejected": self.real_sensor_last_rejected,
        }

    def real_sensor_config(self) -> dict[str, Any]:
        return self._real_sensor_config_locked()

    async def configure_real_sensors(self, config: RealSensorConfigInput) -> dict[str, Any]:
        async with self.lock:
            rooms = {
                normalize_room_name(room)
                for room in config.rooms
                if normalize_room_name(room)
            }
            for assignment in config.assignments:
                room = normalize_room_name(assignment.room)
                if room:
                    rooms.add(room)
            if not rooms:
                rooms = set(self._real_map_rooms_locked())

            assignments: dict[str, dict[str, Any]] = {}
            for assignment in config.assignments:
                entity_id = str(assignment.entity_id or "").strip().lower()
                room = normalize_room_name(assignment.room)
                if not entity_id or not room:
                    continue
                sensor_type = str(assignment.sensor_type or "auto").strip().lower()
                assignments[entity_id] = {
                    "room": room,
                    "enabled": bool(assignment.enabled),
                    "sensor_type": sensor_type if sensor_type in {"auto", "motion", "door", "occupancy", "other"} else "auto",
                    "training_role": "signal",
                }

            self.real_sensor_rooms = rooms
            self.real_sensor_assignments = assignments
            self.real_sensor_require_explicit_selection = bool(config.require_explicit_selection)
            self.reference_layout = normalize_adjacency_map(self.reference_layout, sorted(self.real_sensor_rooms))
            self.reference_layout_version += 1
            return self._real_sensor_config_locked()

    def load_real_sensor_config(self, payload: dict[str, Any]) -> None:
        try:
            config = RealSensorConfigInput(**payload)
        except Exception:
            return
        rooms = {
            normalize_room_name(room)
            for room in config.rooms
            if normalize_room_name(room)
        }
        assignments: dict[str, dict[str, Any]] = {}
        for assignment in config.assignments:
            entity_id = str(assignment.entity_id or "").strip().lower()
            room = normalize_room_name(assignment.room)
            if not entity_id or not room:
                continue
            rooms.add(room)
            assignments[entity_id] = {
                "room": room,
                "enabled": bool(assignment.enabled),
                "sensor_type": str(assignment.sensor_type or "auto"),
                "training_role": "signal",
            }
        if rooms:
            self.real_sensor_rooms = rooms
        self.real_sensor_assignments = assignments
        self.real_sensor_require_explicit_selection = bool(config.require_explicit_selection)
