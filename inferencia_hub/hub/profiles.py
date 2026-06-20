"""Aplicacion de perfiles sobre el estado de inferencia."""

from .dependencies import *  # noqa: F401,F403


class ProfilesMixin:
    @staticmethod
    def _layout_signature(
        rooms: set[str],
        layout: dict[str, list[str]],
    ) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
        return (
            tuple(sorted(room for room in rooms if room)),
            tuple(sorted(adjacency_edge_set(layout))),
        )

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
        next_room_labels = {
            normalize_room_name(str(room.get("slug") or "")): str(
                room.get("name") or room.get("slug") or ""
            ).strip()
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
            training_role = str(
                item.get("training_role") or "signal"
            )
            if not entity_id or (not room and training_role != "people_count_confirmation"):
                continue
            if room:
                rooms.add(room)
            assignments[entity_id] = {
                "room": room,
                "enabled": bool(item.get("enabled", True))
                and str(item.get("status") or "active") == "active",
                "sensor_type": str(item.get("sensor_type") or "other"),
                "training_role": training_role,
                "area_id": str(item.get("area_id") or ""),
                "status": str(item.get("status") or "active"),
                "warning": str(item.get("warning") or ""),
            }

        edges = [
            (str(edge[0]), str(edge[1]))
            for edge in profile.get("edges", [])
            if isinstance(edge, list) and len(edge) == 2
        ]
        next_reference_layout = to_adjacency(sorted(rooms), edges)
        previous_signature = self._layout_signature(
            set(self.real_sensor_rooms),
            self.reference_layout,
        )
        next_signature = self._layout_signature(
            rooms,
            next_reference_layout,
        )
        layout_changed = (
            not self.active_profile_id
            or previous_signature != next_signature
        )
        labels_changed = self.active_profile_room_labels != next_room_labels

        if layout_changed:
            self._reset_transient_locked()
            self.ai_model = AIAdjacencyModel()

        self.real_sensor_rooms = rooms
        self.real_sensor_assignments = assignments
        self.real_sensor_require_explicit_selection = True
        self.reference_layout = next_reference_layout
        self.reference_layout_source = "profile"
        if layout_changed or labels_changed:
            self.reference_layout_version += 1
        self.active_profile_id = str(profile.get("id") or "")
        self.active_profile_name = str(profile.get("name") or "")
        self.active_profile_revision = int(profile.get("revision") or 1)
        self.active_profile_fingerprint = str(profile.get("fingerprint") or "")
        self.active_profile_model_compatible = False
        self.active_profile_room_labels = next_room_labels
        self.ai_model.rooms = sorted(rooms)
        self.ai_model.room_to_idx = {
            room: index for index, room in enumerate(self.ai_model.rooms)
        }
        if (
            layout_changed
            or self.ai_model.transition_matrix.shape
            != (len(self.ai_model.rooms), len(self.ai_model.rooms))
        ):
            self.ai_model.transition_matrix = np.eye(
                len(self.ai_model.rooms),
                dtype=np.float32,
            )
        self.ai_model.adjacency_neighbors = {
            room: list(neighbors)
            for room, neighbors in self.reference_layout.items()
        }
        self.ai_model.ready = bool(rooms)

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
