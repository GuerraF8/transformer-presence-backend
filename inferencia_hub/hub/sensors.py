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
