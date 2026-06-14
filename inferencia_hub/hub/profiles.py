"""Aplicacion de perfiles sobre el estado de inferencia."""

from .dependencies import *  # noqa: F401,F403


class ProfilesMixin:
    def _profile_payload_locked(self) -> dict[str, Any]:
        return {
            "active_profile_id": self.active_profile_id,
            "name": self.active_profile_name,
            "revision": self.active_profile_revision,
            "fingerprint": self.active_profile_fingerprint,
            "available": bool(self.active_profile_id),
            "model_compatible": self.active_profile_model_compatible,
            "room_labels": dict(self.active_profile_room_labels),
        }

    def apply_profile(self, profile: dict[str, Any]) -> None:
        rooms = {
            normalize_room_name(str(room.get("slug") or ""))
            for room in profile.get("rooms", [])
            if isinstance(room, dict)
            and normalize_room_name(str(room.get("slug") or ""))
        }
        assignments: dict[str, dict[str, Any]] = {}
        for item in profile.get("assignments", []):
            if not isinstance(item, dict):
                continue
            entity_id = str(item.get("entity_id") or "").strip().lower()
            room = normalize_room_name(str(item.get("room_slug") or ""))
            if not entity_id or not room:
                continue
            rooms.add(room)
            assignments[entity_id] = {
                "room": room,
                "enabled": bool(item.get("enabled", True))
                and str(item.get("status") or "active") == "active",
                "sensor_type": str(item.get("sensor_type") or "other"),
                "area_id": str(item.get("area_id") or ""),
                "status": str(item.get("status") or "active"),
                "warning": str(item.get("warning") or ""),
            }

        edges = [
            (str(edge[0]), str(edge[1]))
            for edge in profile.get("edges", [])
            if isinstance(edge, list) and len(edge) == 2
        ]
        self._reset_transient_locked()
        self.ai_model = AIAdjacencyModel()
        self.real_sensor_rooms = rooms
        self.real_sensor_assignments = assignments
        self.real_sensor_require_explicit_selection = True
        self.reference_layout = to_adjacency(sorted(rooms), edges)
        self.reference_layout_source = "profile"
        self.reference_layout_version += 1
        self.active_profile_id = str(profile.get("id") or "")
        self.active_profile_name = str(profile.get("name") or "")
        self.active_profile_revision = int(profile.get("revision") or 1)
        self.active_profile_fingerprint = str(profile.get("fingerprint") or "")
        self.active_profile_model_compatible = False
        self.active_profile_room_labels = {
            normalize_room_name(str(room.get("slug") or "")): str(
                room.get("name") or room.get("slug") or ""
            )
            for room in profile.get("rooms", [])
            if isinstance(room, dict)
            and normalize_room_name(str(room.get("slug") or ""))
        }

    def clear_active_profile(self) -> None:
        self._reset_transient_locked()
        self.ai_model = AIAdjacencyModel()
        self.real_sensor_rooms = set()
        self.real_sensor_assignments = {}
        self.reference_layout = {}
        self.reference_layout_source = "none"
        self.reference_layout_version += 1
        self.active_profile_id = None
        self.active_profile_name = None
        self.active_profile_revision = None
        self.active_profile_fingerprint = None
        self.active_profile_model_compatible = False
        self.active_profile_room_labels = {}
