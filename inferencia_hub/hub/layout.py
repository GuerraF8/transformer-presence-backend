"""Administración de habitaciones, mapas y relaciones de adyacencia."""

from .dependencies import *  # noqa: F401,F403


class LayoutMixin:
    @staticmethod
    def _edge_quality(
        inferred_edges: set[tuple[str, str]],
        reference_edges: set[tuple[str, str]],
    ) -> dict[str, Any]:
        tp = len(inferred_edges & reference_edges)
        fp = len(inferred_edges - reference_edges)
        fn = len(reference_edges - inferred_edges)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = ((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
        }

    def _known_rooms_locked(self) -> list[str]:
        rooms = set(self.rooms)
        rooms.update(self.real_sensor_rooms)
        rooms.update(self.ai_model.rooms)
        for a, b in self.edge_support.keys():
            if a:
                rooms.add(a)
            if b:
                rooms.add(b)
        for room, neighbors in self.reference_layout.items():
            if room:
                rooms.add(room)
            for nb in neighbors:
                if nb:
                    rooms.add(nb)
        return sorted([room for room in rooms if room])

    def _real_map_rooms_locked(self) -> list[str]:
        rooms = set(self.real_sensor_rooms)
        for room, neighbors in self.reference_layout.items():
            if room:
                rooms.add(room)
            for nb in neighbors:
                if nb:
                    rooms.add(nb)
        return sorted(room for room in rooms if room)

    def _default_reference_layout_locked(self, rooms: list[str]) -> dict[str, list[str]]:
        templates = build_scenario_templates(rooms)
        base = templates.get("real_home") or templates.get("lineal") or {}
        if base and any(base.values()):
            return normalize_adjacency_map(base, rooms)
        return {room: [] for room in sorted(rooms)}

    def _ensure_reference_layout_locked(self) -> None:
        rooms = self._real_map_rooms_locked()
        if not rooms:
            return

        if not self.reference_layout:
            self.reference_layout = self._default_reference_layout_locked(rooms)
            self.reference_layout_source = "auto"
            self.reference_layout_version += 1
            return

        self.reference_layout = normalize_adjacency_map(self.reference_layout, rooms)

    def _layout_payload_locked(self) -> dict[str, Any]:
        self._ensure_reference_layout_locked()
        return {
            "version": self.reference_layout_version,
            "source": self.reference_layout_source,
            "rooms": sorted(self.reference_layout.keys()),
            "room_labels": dict(self.active_profile_room_labels),
            "adjacency": self.reference_layout,
            "adjacency_text": adjacency_to_text(self.reference_layout),
            "edges": edge_list_from_adjacency(self.reference_layout),
        }

    async def configure_reference_layout(self, config: LayoutReferenceInput) -> dict[str, Any]:
        async with self.lock:
            incoming_rooms = {
                normalize_room_name(room)
                for room in config.rooms
                if normalize_room_name(room)
            }
            known_rooms = set(self._real_map_rooms_locked())

            adjacency: dict[str, list[str]] = {}
            if config.adjacency:
                adjacency = {
                    normalize_room_name(room): [normalize_room_name(nb) for nb in neighbors]
                    for room, neighbors in config.adjacency.items()
                    if normalize_room_name(room)
                }

            if config.adjacency_text.strip():
                parsed_from_text = parse_adjacency_text(config.adjacency_text)
                if parsed_from_text:
                    adjacency = parsed_from_text

            if not adjacency and not (incoming_rooms or known_rooms):
                raise ValueError("No hay habitaciones disponibles para construir layout")

            all_rooms = sorted(known_rooms | incoming_rooms)
            if adjacency:
                self.reference_layout = normalize_adjacency_map(adjacency, all_rooms)
            else:
                self.reference_layout = self._default_reference_layout_locked(all_rooms)

            self.reference_layout_source = "manual"
            self.reference_layout_version += 1
            metrics = self._evaluation_metrics_locked()
            layout_payload = self._layout_payload_locked()

        return {
            "status": "ok",
            "layout_reference": layout_payload,
            "metrics": metrics,
        }

    def _active_layout_graph_locked(self) -> dict[str, list[str]]:
        if self.reference_layout_source == "manual" and self.reference_layout:
            return self.reference_layout
        if self.ai_model.ready and self.ai_model.adjacency_neighbors:
            return normalize_adjacency_map(self.ai_model.adjacency_neighbors, self.ai_model.rooms)
        if self.reference_layout:
            return self.reference_layout
        rooms = self._known_rooms_locked()
        return to_adjacency(rooms, list(self.edge_support.keys()))

    def _movement_adjacent_locked(self, a: str, b: str) -> bool:
        a_n = normalize_room_name(a)
        b_n = normalize_room_name(b)
        if not a_n or not b_n or a_n == b_n:
            return True
        graph = self._active_layout_graph_locked()
        if b_n in graph.get(a_n, []):
            return True
        if self.reference_layout_source == "manual":
            return False

        # Al iniciar desde el simulador, el backend puede conocer solo los cuartos
        # que ya recibieron eventos. Para el hogar real, usamos la plantilla base
        # como respaldo antes de bloquear una transición observada por sensores.
        rooms = sorted(set(self._known_rooms_locked()) | {a_n, b_n})
        real_home = build_scenario_templates(rooms).get("real_home", {})
        return b_n in real_home.get(a_n, [])

    def _reference_adjacent_locked(self, a: str, b: str) -> bool:
        a_n = normalize_room_name(a)
        b_n = normalize_room_name(b)
        if not a_n or not b_n or a_n == b_n:
            return True
        self._ensure_reference_layout_locked()
        return b_n in self.reference_layout.get(a_n, [])
