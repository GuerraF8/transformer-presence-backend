from __future__ import annotations

import asyncio
import os
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from fastapi import WebSocket

try:
    from .ai_model import AIAdjacencyModel
    from .domain import (
        EventRecord,
        LastActivation,
        LayoutReferenceInput,
        PresenceFilterConfigInput,
        RealSensorConfigInput,
        SENSOR_RELIABILITY,
        SensorEventInput,
        adjacency_edge_set,
        adjacency_to_text,
        build_scenario_templates,
        classify_sensor_type,
        edge_key,
        edge_list_from_adjacency,
        infer_room_from_entity,
        is_activation,
        normalize_adjacency_map,
        normalize_room_name,
        parse_adjacency_text,
        to_adjacency,
        to_utc_iso,
    )
except ImportError:  # pragma: no cover - supports `uvicorn server:app` in Docker
    from ai_model import AIAdjacencyModel
    from domain import (
        EventRecord,
        LastActivation,
        LayoutReferenceInput,
        PresenceFilterConfigInput,
        RealSensorConfigInput,
        SENSOR_RELIABILITY,
        SensorEventInput,
        adjacency_edge_set,
        adjacency_to_text,
        build_scenario_templates,
        classify_sensor_type,
        edge_key,
        edge_list_from_adjacency,
        infer_room_from_entity,
        is_activation,
        normalize_adjacency_map,
        normalize_room_name,
        parse_adjacency_text,
        to_adjacency,
        to_utc_iso,
    )

class InferenceHubState:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.events: list[dict[str, Any]] = []
        self.rooms: set[str] = set()
        self.edge_support: Counter[tuple[str, str]] = Counter()
        self.last_active_by_room: dict[str, datetime] = {}
        self.occupancy_confirmed_by_room: dict[str, datetime] = {}
        self.active_sensor_types_by_room: dict[str, set[str]] = {}
        self.last_activation: LastActivation | None = None
        self.current_room: str | None = None
        self.current_active_rooms: list[str] = []
        self.latest_touched_edge: tuple[str, str] | None = None
        self.websockets: set[WebSocket] = set()
        self.rejected_transitions = 0

        self.reference_layout: dict[str, list[str]] = {}
        self.reference_layout_source = "auto"
        self.reference_layout_version = 0

        self.non_adjacent_records: list[dict[str, Any]] = []
        self.max_non_adjacent_records = 2000
        self.non_adjacent_total = 0
        self.non_adjacent_multi_person = 0
        self.non_adjacent_pet_or_noise = 0
        self.non_adjacent_sensor_error = 0

        self.current_people_estimate = 0
        self.max_people_estimate = 0

        self.ingestion_latency_ms: deque[float] = deque(maxlen=5000)
        self.processing_latency_ms: deque[float] = deque(maxlen=5000)

        self.presence_hold_seconds = int(os.getenv("PRESENCE_HOLD_SECONDS", "180"))
        self.min_gap_seconds = int(os.getenv("MIN_GAP_SECONDS", "2"))
        self.max_gap_seconds = int(os.getenv("MAX_GAP_SECONDS", "600"))
        self.confirmed_edge_support = int(os.getenv("CONFIRMED_EDGE_SUPPORT", "2"))
        self.max_events_buffer = int(os.getenv("MAX_EVENTS_BUFFER", "30000"))
        self.include_all_state_transitions = os.getenv("INCLUDE_ALL_STATE_TRANSITIONS", "1") != "0"
        self.presence_filter_enabled = os.getenv("PET_FILTER_ENABLED", "1") != "0"
        self.presence_filter_window_seconds = max(1, min(600, int(os.getenv("PET_FILTER_WINDOW_SECONDS", "20"))))
        self.presence_filter_min_motion_events = max(1, min(20, int(os.getenv("PET_FILTER_MIN_EVENTS", "2"))))
        self.presence_filter_min_distinct_rooms = max(1, min(20, int(os.getenv("PET_FILTER_MIN_DISTINCT_ROOMS", "1"))))
        self.presence_filter_events: deque[dict[str, Any]] = deque(maxlen=512)
        self.presence_filter_suppressed_total = 0

        self.input_mode = "listen"
        self.replay_task: asyncio.Task | None = None
        self.replay_paused = False
        self.replay_stop_requested = False
        self.replay_step_budget = 0
        self.replay_total_events = 0
        self.replay_processed_events = 0
        self.replay_last_error: str | None = None
        self.last_replay_config: dict[str, Any] = {}

        self.ai_model = AIAdjacencyModel()
        self.presence_belief = np.zeros((0,), dtype=np.float32)
        self.sequence_history: deque[EventRecord] = deque(maxlen=512)
        self.real_sensor_rooms: set[str] = {
            "bedroom",
            "entertainment_room",
            "foyer",
            "kitchen",
            "living",
            "sittingroom",
        }
        self.real_sensor_assignments: dict[str, dict[str, Any]] = {}
        self.real_sensor_require_explicit_selection = True
        self.real_sensor_rejected_events = 0
        self.real_sensor_last_rejected: dict[str, Any] | None = None

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

    def _real_sensor_config_locked(self) -> dict[str, Any]:
        assignments = sorted(
            [
                {
                    "entity_id": entity_id,
                    "room": str(config.get("room") or ""),
                    "enabled": bool(config.get("enabled", True)),
                    "sensor_type": str(config.get("sensor_type") or "auto"),
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
            }
        if rooms:
            self.real_sensor_rooms = rooms
        self.real_sensor_assignments = assignments
        self.real_sensor_require_explicit_selection = bool(config.require_explicit_selection)

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
        # como fallback antes de bloquear una transicion observada por sensores.
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

    def _estimate_people_locked(self, active_rooms: list[str]) -> int:
        active = [normalize_room_name(room) for room in active_rooms if normalize_room_name(room)]
        active_unique = sorted(set(active))
        if not active_unique:
            return 0

        graph = self._active_layout_graph_locked()
        active_set = set(active_unique)
        visited: set[str] = set()
        components = 0

        for room in active_unique:
            if room in visited:
                continue
            components += 1
            queue = deque([room])
            visited.add(room)
            while queue:
                current = queue.popleft()
                for nb in graph.get(current, []):
                    if nb in active_set and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)

        estimate = max(1, components)
        if len(active_unique) >= 3 and components >= 2:
            estimate = min(len(active_unique), estimate + 1)
        occupancy_rooms = {
            normalize_room_name(room)
            for room in self.occupancy_confirmed_by_room.keys()
            if normalize_room_name(room)
        }
        live_sensor_rooms = {
            normalize_room_name(room)
            for room, sensor_types in self.active_sensor_types_by_room.items()
            if normalize_room_name(room) and sensor_types
        }
        if occupancy_rooms:
            # Occupancy es ground truth: una ocupacion confirmada mas movimiento
            # simultaneo en otra habitacion implica multiples personas.
            estimate = max(estimate, len(occupancy_rooms) + len(live_sensor_rooms - occupancy_rooms))
        return estimate

    def _record_non_adjacent_locked(
        self,
        *,
        timestamp: datetime,
        transition: dict[str, Any],
        sensor_type: str,
        estimated_people: int,
        active_rooms: list[str],
    ) -> dict[str, Any]:
        gap_seconds = float(transition.get("gap_seconds", 0.0))
        if estimated_people >= 2 or len(active_rooms) >= 2:
            cause = "multiples_personas_probable"
            self.non_adjacent_multi_person += 1
        elif sensor_type in {"motion", "occupancy"} and gap_seconds <= 12.0:
            cause = "mascota_o_ruido"
            self.non_adjacent_pet_or_noise += 1
        else:
            cause = "error_sensor_o_datos"
            self.non_adjacent_sensor_error += 1

        self.non_adjacent_total += 1
        record = {
            "timestamp": to_utc_iso(timestamp),
            "from": transition.get("from"),
            "to": transition.get("to"),
            "gap_seconds": round(gap_seconds, 3),
            "sensor_type": sensor_type,
            "estimated_people": estimated_people,
            "active_rooms": active_rooms,
            "cause": cause,
        }
        self.non_adjacent_records.append(record)
        if len(self.non_adjacent_records) > self.max_non_adjacent_records:
            self.non_adjacent_records = self.non_adjacent_records[-self.max_non_adjacent_records :]
        return record

    def _inference_quality_metrics_locked(self) -> dict[str, Any]:
        activation_events = [
            event
            for event in self.events
            if is_activation(str(event.get("sensor_type") or "other"), str(event.get("state") or ""))
        ]
        ai_events = [
            event
            for event in activation_events
            if str(event.get("ai_mode") or "") in {"hf_transformer_markov", "markov_ai"}
        ]
        if not ai_events:
            return {
                "activation_events": len(activation_events),
                "ai_events": 0,
                "transformer_guided_events": 0,
                "transformer_usage_rate": None,
                "avg_presence_confidence": None,
                "observed_room_match_rate": None,
                "resolved_room_match_rate": None,
                "transition_acceptance_rate": None,
                "occupancy_anchor_events": 0,
            }

        transformer_events = [
            event
            for event in ai_events
            if bool(((event.get("inference_debug") or {}).get("transformer_used")))
        ]
        observed_room_matches = 0
        resolved_room_matches = 0
        accepted_transitions = 0
        total_transitions = 0
        occupancy_anchor_events = 0
        confidence_values: list[float] = []

        for event in ai_events:
            debug = event.get("inference_debug") or {}
            observed_room = normalize_room_name(str(event.get("room") or ""))
            presence_room = normalize_room_name(str(event.get("presence_room") or ""))
            hybrid_top_room = normalize_room_name(str(debug.get("hybrid_top_room") or ""))
            if hybrid_top_room and hybrid_top_room == observed_room:
                observed_room_matches += 1
            if hybrid_top_room and hybrid_top_room == presence_room:
                resolved_room_matches += 1
            confidence = event.get("presence_confidence")
            if isinstance(confidence, (int, float)):
                confidence_values.append(float(confidence))
            if str(event.get("sensor_type") or "") == "occupancy":
                occupancy_anchor_events += 1
            transition = event.get("transition")
            if isinstance(transition, dict) and not transition.get("same_room"):
                total_transitions += 1
                if not transition.get("rejected_by_ai"):
                    accepted_transitions += 1

        transformer_observed_matches = 0
        transformer_resolved_matches = 0
        transformer_observed_probs: list[float] = []
        for event in transformer_events:
            debug = event.get("inference_debug") or {}
            observed_room = normalize_room_name(str(event.get("room") or ""))
            presence_room = normalize_room_name(str(event.get("presence_room") or ""))
            transformer_top_room = normalize_room_name(str(debug.get("transformer_top_room") or ""))
            if transformer_top_room and transformer_top_room == observed_room:
                transformer_observed_matches += 1
            if transformer_top_room and transformer_top_room == presence_room:
                transformer_resolved_matches += 1
            observed_prob = debug.get("transformer_observed_room_prob")
            if isinstance(observed_prob, (int, float)):
                transformer_observed_probs.append(float(observed_prob))

        return {
            "activation_events": len(activation_events),
            "ai_events": len(ai_events),
            "transformer_guided_events": len(transformer_events),
            "transformer_usage_rate": round(len(transformer_events) / len(ai_events), 4),
            "avg_presence_confidence": round(float(np.mean(confidence_values)), 4) if confidence_values else None,
            "observed_room_match_rate": round(observed_room_matches / len(ai_events), 4),
            "resolved_room_match_rate": round(resolved_room_matches / len(ai_events), 4),
            "transition_acceptance_rate": (
                round(accepted_transitions / total_transitions, 4) if total_transitions > 0 else None
            ),
            "occupancy_anchor_events": occupancy_anchor_events,
            "transformer_observed_room_match_rate": (
                round(transformer_observed_matches / len(transformer_events), 4)
                if transformer_events
                else None
            ),
            "transformer_resolved_room_match_rate": (
                round(transformer_resolved_matches / len(transformer_events), 4)
                if transformer_events
                else None
            ),
            "transformer_avg_observed_room_prob": (
                round(float(np.mean(transformer_observed_probs)), 4)
                if transformer_observed_probs
                else None
            ),
        }

    def _evaluation_metrics_locked(self) -> dict[str, Any]:
        self._ensure_reference_layout_locked()
        reference_edges = adjacency_edge_set(self.reference_layout)
        visible_room_set = set(self._real_map_rooms_locked())

        live_edges_all = {
            edge_key(a, b)
            for (a, b), support in self.edge_support.items()
            if support > 0 and a in visible_room_set and b in visible_room_set
        }
        live_edges_confirmed = {
            edge_key(a, b)
            for (a, b), support in self.edge_support.items()
            if support >= self.confirmed_edge_support and a in visible_room_set and b in visible_room_set
        }
        model_edges = {
            edge_key(str(edge.get("a", "")), str(edge.get("b", "")))
            for edge in self.ai_model.adjacency_edges
            if edge.get("a") and edge.get("b") and edge.get("a") in visible_room_set and edge.get("b") in visible_room_set
        }

        return {
            "map": {
                "reference_edges": len(reference_edges),
                "live_edges_total": len(live_edges_all),
                "live_edges_confirmed": len(live_edges_confirmed),
                "model_edges": len(model_edges),
                "live_confirmed_quality": self._edge_quality(live_edges_confirmed, reference_edges),
                "model_quality": self._edge_quality(model_edges, reference_edges),
            },
            "people": {
                "current_estimate": self.current_people_estimate,
                "max_observed": self.max_people_estimate,
                "occupancy_ground_truth_rooms": sorted(self.occupancy_confirmed_by_room.keys()),
                "live_sensor_rooms": sorted(self.active_sensor_types_by_room.keys()),
            },
            "real_sensors": {
                "rooms_total": len(self.real_sensor_rooms),
                "assigned_total": len(self.real_sensor_assignments),
                "enabled_total": len([item for item in self.real_sensor_assignments.values() if item.get("enabled", True)]),
                "rejected_events": self.real_sensor_rejected_events,
                "last_rejected": self.real_sensor_last_rejected,
            },
            "inference": self._inference_quality_metrics_locked(),
            "non_adjacent": {
                "total": self.non_adjacent_total,
                "multi_person_probable": self.non_adjacent_multi_person,
                "pet_or_noise": self.non_adjacent_pet_or_noise,
                "sensor_or_data_error": self.non_adjacent_sensor_error,
                "recent": self.non_adjacent_records[-25:],
            },
            "latency": {
                "ingestion": self._summarize_latency(self.ingestion_latency_ms),
                "processing": self._summarize_latency(self.processing_latency_ms),
            },
        }

    def evaluation_metrics(self) -> dict[str, Any]:
        return self._evaluation_metrics_locked()

    def training_map_validation_locked(self) -> dict[str, Any]:
        self._ensure_reference_layout_locked()
        reference_edges = adjacency_edge_set(self.reference_layout)
        model_edges = {
            edge_key(str(edge.get("a", "")), str(edge.get("b", "")))
            for edge in self.ai_model.adjacency_edges
            if edge.get("a") and edge.get("b")
        }
        return {
            "reference_edges": len(reference_edges),
            "model_edges": len(model_edges),
            **self._edge_quality(model_edges, reference_edges),
        }

    @staticmethod
    def _summarize_latency(values: deque[float]) -> dict[str, Any]:
        if not values:
            return {
                "count": 0,
                "avg_ms": None,
                "p50_ms": None,
                "p95_ms": None,
                "max_ms": None,
            }
        arr = np.asarray(list(values), dtype=np.float32)
        return {
            "count": int(arr.size),
            "avg_ms": round(float(arr.mean()), 3),
            "p50_ms": round(float(np.quantile(arr, 0.50)), 3),
            "p95_ms": round(float(np.quantile(arr, 0.95)), 3),
            "max_ms": round(float(arr.max()), 3),
        }

    def _presence_filter_config_locked(self) -> dict[str, Any]:
        return {
            "enabled": self.presence_filter_enabled,
            "window_seconds": self.presence_filter_window_seconds,
            "min_motion_events": self.presence_filter_min_motion_events,
            "min_distinct_rooms": self.presence_filter_min_distinct_rooms,
            "pending_motion_events": len(self.presence_filter_events),
            "suppressed_total": self.presence_filter_suppressed_total,
        }

    def presence_filter_config(self) -> dict[str, Any]:
        return self._presence_filter_config_locked()

    async def configure_presence_filter(self, config: PresenceFilterConfigInput) -> dict[str, Any]:
        async with self.lock:
            self.presence_filter_enabled = bool(config.enabled)
            self.presence_filter_window_seconds = max(1, min(600, int(config.window_seconds)))
            self.presence_filter_min_motion_events = max(1, min(20, int(config.min_motion_events)))
            self.presence_filter_min_distinct_rooms = max(1, min(20, int(config.min_distinct_rooms)))
            self.presence_filter_events.clear()
            return self._presence_filter_config_locked()

    def _evaluate_presence_filter_locked(
        self,
        room: str,
        sensor_type: str,
        now: datetime,
    ) -> tuple[bool, dict[str, Any]]:
        debug = {
            "enabled": self.presence_filter_enabled,
            "applied": False,
            "accepted": True,
            "window_seconds": self.presence_filter_window_seconds,
            "min_motion_events": self.presence_filter_min_motion_events,
            "min_distinct_rooms": self.presence_filter_min_distinct_rooms,
        }
        if not self.presence_filter_enabled or sensor_type != "motion":
            return True, debug

        room_n = normalize_room_name(room)
        cutoff = now - timedelta(seconds=self.presence_filter_window_seconds)
        while self.presence_filter_events and self.presence_filter_events[0]["timestamp"] < cutoff:
            self.presence_filter_events.popleft()

        self.presence_filter_events.append({"timestamp": now, "room": room_n})
        candidate_rooms: list[str] = []
        candidate_count = 0
        for event in self.presence_filter_events:
            event_room = normalize_room_name(str(event.get("room") or ""))
            if not event_room:
                continue
            if event["timestamp"] < cutoff:
                continue
            if event_room == room_n or self._movement_adjacent_locked(event_room, room_n):
                candidate_count += 1
                candidate_rooms.append(event_room)

        distinct_rooms = sorted(set(candidate_rooms))
        accepted = (
            candidate_count >= self.presence_filter_min_motion_events
            and len(distinct_rooms) >= self.presence_filter_min_distinct_rooms
        )
        if not accepted:
            self.presence_filter_suppressed_total += 1

        debug.update(
            {
                "applied": True,
                "accepted": accepted,
                "events_in_window": candidate_count,
                "distinct_rooms_in_window": len(distinct_rooms),
                "rooms_in_window": distinct_rooms,
                "reason": None if accepted else "movimiento_aislado_posible_mascota",
                "suppressed_total": self.presence_filter_suppressed_total,
            }
        )
        return accepted, debug

    def _prune_inactive_rooms(self, now: datetime) -> None:
        max_delta = timedelta(seconds=self.presence_hold_seconds)
        stale_rooms = [
            room
            for room, ts in self.last_active_by_room.items()
            if now - ts > max_delta
        ]
        for room in stale_rooms:
            del self.last_active_by_room[room]
        stale_occupancy = [
            room
            for room, ts in self.occupancy_confirmed_by_room.items()
            if now - ts > max_delta
        ]
        for room in stale_occupancy:
            del self.occupancy_confirmed_by_room[room]
        filter_cutoff = now - timedelta(seconds=self.presence_filter_window_seconds)
        while self.presence_filter_events and self.presence_filter_events[0]["timestamp"] < filter_cutoff:
            self.presence_filter_events.popleft()

    def _has_adjacent_activity_since_locked(
        self,
        room: str,
        since: datetime | None,
    ) -> bool:
        room_n = normalize_room_name(room)
        if not room_n:
            return False
        self._ensure_reference_layout_locked()
        neighbors = self.reference_layout.get(room_n, [])
        if not neighbors:
            return False

        threshold = since or datetime.min.replace(tzinfo=timezone.utc)
        for nb in neighbors:
            last_ts = self.last_active_by_room.get(nb)
            if last_ts is not None and last_ts >= threshold:
                return True
        return False

    def _can_displace_presence_locked(
        self,
        previous_room: str | None,
        candidate_room: str,
        sensor_type: str,
        previous_seen_at: datetime | None,
    ) -> bool:
        if not previous_room:
            return True

        prev_n = normalize_room_name(previous_room)
        cand_n = normalize_room_name(candidate_room)
        if not prev_n or not cand_n or prev_n == cand_n:
            return True

        # Occupancy confirma presencia en el cuarto observado.
        if sensor_type == "occupancy":
            return True

        # Si la transicion observada es adyacente en el mapa aprendido/manual, permitimos el movimiento.
        if self._movement_adjacent_locked(prev_n, cand_n):
            return True

        # Si no hubo actividad intermedia en adyacentes, retenemos presencia en el cuarto previo.
        return self._has_adjacent_activity_since_locked(prev_n, previous_seen_at)

    def _ensure_presence_belief(self) -> None:
        n_rooms = len(self.ai_model.rooms)
        if n_rooms <= 0:
            self.presence_belief = np.zeros((0,), dtype=np.float32)
            return
        if self.presence_belief.shape[0] != n_rooms:
            self.presence_belief = np.full((n_rooms,), 1.0 / n_rooms, dtype=np.float32)

    def _infer_presence_with_ai(
        self,
        observed_room: str,
        sensor_type: str,
        now: datetime,
    ) -> tuple[str, float, list[str], dict[str, Any]]:
        self._ensure_presence_belief()
        if not self.ai_model.ready or observed_room not in self.ai_model.room_to_idx:
            occupancy_prediction = self.ai_model.predict_occupancy_state(list(self.sequence_history), now)
            if occupancy_prediction and occupancy_prediction.get("rooms"):
                predicted_rooms = [
                    normalize_room_name(room)
                    for room in occupancy_prediction.get("rooms", [])
                    if normalize_room_name(room)
                ]
                if observed_room not in predicted_rooms:
                    predicted_rooms.insert(0, observed_room)
                return observed_room, float(occupancy_prediction.get("confidence") or 0.5), predicted_rooms, {
                    "transformer_used": False,
                    "markov_top_room": observed_room,
                    "hybrid_top_room": observed_room,
                    "observed_room_prob": 0.5,
                    "occupancy_transformer_used": True,
                    "occupancy_transformer_rooms": predicted_rooms,
                    "occupancy_transformer_people_count": int(occupancy_prediction.get("people_count") or 0),
                    "occupancy_transformer_confidence": occupancy_prediction.get("confidence"),
                }
            return observed_room, 0.5, [observed_room], {
                "transformer_used": False,
                "markov_top_room": observed_room,
                "hybrid_top_room": observed_room,
                "observed_room_prob": 0.5,
            }

        n_rooms = len(self.ai_model.rooms)
        idx_obs = self.ai_model.room_to_idx[observed_room]

        trans = self.ai_model.transition_matrix
        markov_prior = self.presence_belief @ trans
        transformer_prior = self.ai_model.predict_next_room_probs(list(self.sequence_history), now)
        transformer_used = transformer_prior is not None and float(transformer_prior.sum()) > 0
        if transformer_prior is not None and float(transformer_prior.sum()) > 0:
            transformer_prior = transformer_prior / float(transformer_prior.sum())
            prior = (0.4 * markov_prior) + (0.6 * transformer_prior)
        else:
            prior = markov_prior

        emission = np.full((n_rooms,), 0.08, dtype=np.float32)
        reliability = SENSOR_RELIABILITY.get(sensor_type, SENSOR_RELIABILITY["other"])
        emission[idx_obs] = 0.52 + (0.42 * reliability)

        for nb in self.ai_model.neighbors(observed_room):
            idx_nb = self.ai_model.room_to_idx[nb]
            emission[idx_nb] = max(emission[idx_nb], 0.18 + (0.22 * reliability))

        posterior = prior * emission
        post_sum = float(posterior.sum())
        if post_sum > 0:
            posterior /= post_sum
        else:
            posterior = np.full((n_rooms,), 1.0 / n_rooms, dtype=np.float32)

        self.presence_belief = posterior

        best_idx = int(np.argmax(posterior))
        best_room = self.ai_model.rooms[best_idx]
        confidence = float(posterior[best_idx])
        markov_idx = int(np.argmax(markov_prior))
        transformer_idx = int(np.argmax(transformer_prior)) if transformer_used and transformer_prior is not None else None

        order = np.argsort(-posterior)
        active_rooms: list[str] = []
        thr = max(0.18, confidence * 0.45)
        for idx in order:
            prob = float(posterior[idx])
            if prob < thr and active_rooms:
                break
            active_rooms.append(self.ai_model.rooms[int(idx)])
            if len(active_rooms) >= 3:
                break

        debug = {
            "transformer_used": transformer_used,
            "markov_top_room": self.ai_model.rooms[markov_idx],
            "transformer_top_room": (
                self.ai_model.rooms[int(transformer_idx)] if transformer_idx is not None else None
            ),
            "hybrid_top_room": best_room,
            "observed_room_prob": round(float(posterior[idx_obs]), 4),
            "hybrid_top_prob": round(confidence, 4),
        }
        if transformer_used and transformer_prior is not None:
            debug["transformer_observed_room_prob"] = round(float(transformer_prior[idx_obs]), 4)
            debug["transformer_top_prob"] = round(float(transformer_prior[int(transformer_idx)]), 4)

        occupancy_prediction = self.ai_model.predict_occupancy_state(list(self.sequence_history), now)
        if occupancy_prediction and occupancy_prediction.get("rooms"):
            predicted_rooms = [
                normalize_room_name(room)
                for room in occupancy_prediction.get("rooms", [])
                if normalize_room_name(room)
            ]
            if predicted_rooms:
                active_rooms = list(dict.fromkeys(predicted_rooms + active_rooms))
                debug["occupancy_transformer_used"] = True
                debug["occupancy_transformer_rooms"] = predicted_rooms
                debug["occupancy_transformer_people_count"] = int(occupancy_prediction.get("people_count") or 0)
                debug["occupancy_transformer_confidence"] = occupancy_prediction.get("confidence")

        return best_room, confidence, active_rooms, debug

    def _build_transition(
        self,
        room: str,
        now: datetime,
    ) -> tuple[dict[str, Any] | None, bool]:
        if self.last_activation is None:
            return None, False

        prev = self.last_activation
        gap = (now - prev.timestamp).total_seconds()
        if prev.room == room:
            return (
                {
                    "from": prev.room,
                    "to": room,
                    "gap_seconds": round(gap, 3),
                    "same_room": True,
                    "rejected_by_ai": False,
                },
                False,
            )
        if (not self.include_all_state_transitions) and (
            gap < self.min_gap_seconds or gap > self.max_gap_seconds
        ):
            return None, False

        allowed = self.ai_model.are_adjacent(prev.room, room)
        if not allowed:
            self.rejected_transitions += 1
            return (
                {
                    "from": prev.room,
                    "to": room,
                    "gap_seconds": round(gap, 3),
                    "same_room": False,
                    "rejected_by_ai": True,
                    "reason": "edge_not_in_learned_map",
                },
                True,
            )

        key = edge_key(prev.room, room)
        self.edge_support[key] += 1
        self.latest_touched_edge = key

        return (
            {
                "from": prev.room,
                "to": room,
                "gap_seconds": round(gap, 3),
                "weight": 1,
                "support": int(self.edge_support[key]),
                "edge": [key[0], key[1]],
                "same_room": False,
                "rejected_by_ai": False,
            },
            False,
        )

    async def process_event(self, payload: SensorEventInput) -> dict[str, Any]:
        ingress_now = datetime.now(timezone.utc)
        now = payload.timestamp or ingress_now
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)

        entity_id = str(payload.entity_id or "").strip().lower()
        source = str(payload.source or "").strip().lower()
        is_real_ha_event = source.startswith("ha") or source.startswith("home_assistant") or source.startswith("hass")

        inferred_room = infer_room_from_entity(entity_id)
        room = normalize_room_name(payload.room or inferred_room)
        sensor_type = payload.sensor_type or classify_sensor_type(entity_id)
        state = payload.state.lower().strip()
        is_active_event = is_activation(sensor_type, state)

        async with self.lock:
            if is_real_ha_event:
                assignment = self.real_sensor_assignments.get(entity_id)
                if not assignment or (assignment and not bool(assignment.get("enabled", True))):
                    self.real_sensor_rejected_events += 1
                    self.real_sensor_last_rejected = {
                        "timestamp": to_utc_iso(now),
                        "entity_id": entity_id,
                        "source": payload.source,
                        "reason": "sensor_no_asignado" if not assignment else "sensor_deshabilitado",
                    }
                    return {
                        "status": "ignored",
                        "reason": self.real_sensor_last_rejected["reason"],
                        "entity_id": entity_id,
                        "real_sensor_config": self._real_sensor_config_locked(),
                    }
                if assignment:
                    room = normalize_room_name(str(assignment.get("room") or room))
                    assigned_type = str(assignment.get("sensor_type") or "auto").strip().lower()
                    if assigned_type != "auto":
                        sensor_type = assigned_type
                    is_active_event = is_activation(sensor_type, state)

            self.rooms.add(room)
            self._ensure_reference_layout_locked()
            presence_signal_allowed = True
            presence_filter_debug: dict[str, Any] = {}
            if is_active_event:
                presence_signal_allowed, presence_filter_debug = self._evaluate_presence_filter_locked(
                    room=room,
                    sensor_type=sensor_type,
                    now=now,
                )

            if is_active_event and presence_signal_allowed:
                self.active_sensor_types_by_room.setdefault(room, set()).add(sensor_type)
            else:
                active_types = self.active_sensor_types_by_room.get(room)
                if active_types is not None:
                    active_types.discard(sensor_type)
                    if not active_types:
                        self.active_sensor_types_by_room.pop(room, None)

            previous_room = self.current_room
            previous_seen_at = self.last_active_by_room.get(previous_room, None) if previous_room else None
            current_event = EventRecord(
                timestamp=now,
                entity_id=entity_id,
                state=state,
                sensor_type=sensor_type,
                room=room,
            )

            transition: dict[str, Any] | None = None
            confidence = 0.0
            active_rooms: list[str] = []
            relation = "desconocida"
            layout_alert: dict[str, Any] | None = None
            inference_debug: dict[str, Any] = {}
            if presence_filter_debug:
                inference_debug["presence_filter"] = presence_filter_debug
            if is_active_event and presence_signal_allowed:
                transition, _ = self._build_transition(room, now)
                # Mantiene la ultima observacion de activacion para analizar desplazamientos reales.
                self.last_activation = LastActivation(room=room, timestamp=now)
            elif sensor_type == "occupancy":
                self.occupancy_confirmed_by_room.pop(room, None)
                self.last_active_by_room.pop(room, None)
                if self.current_room == room:
                    remaining_rooms = list(self.occupancy_confirmed_by_room.keys()) + list(
                        self.active_sensor_types_by_room.keys()
                    )
                    self.current_room = normalize_room_name(remaining_rooms[0]) if remaining_rooms else None

            if is_active_event and presence_signal_allowed:
                self.last_active_by_room[room] = now

                inferred_presence_room, confidence, ai_active_rooms, ai_debug = self._infer_presence_with_ai(
                    observed_room=room,
                    sensor_type=sensor_type,
                    now=now,
                )
                inference_debug.update(ai_debug)
                resolved_presence_room = inferred_presence_room
                observed_room_forced = False

                if sensor_type == "occupancy":
                    # El sensor occupancy actua como verificacion fuerte de presencia.
                    self.occupancy_confirmed_by_room[room] = now
                    resolved_presence_room = room
                    confidence = max(confidence, 0.96)
                    if self.ai_model.ready and room in self.ai_model.room_to_idx:
                        self._ensure_presence_belief()
                        forced = np.zeros_like(self.presence_belief)
                        forced[self.ai_model.room_to_idx[room]] = 1.0
                        self.presence_belief = forced
                elif self.occupancy_confirmed_by_room and room not in self.occupancy_confirmed_by_room:
                    # Si existe occupancy activa, se usa como ancla de ground truth.
                    # Movimiento en otra habitacion suma evidencia multi-persona, pero no desplaza
                    # la presencia principal fuera de la habitacion confirmada por occupancy.
                    anchor_room = max(
                        self.occupancy_confirmed_by_room.items(),
                        key=lambda item: item[1],
                    )[0]
                    resolved_presence_room = normalize_room_name(anchor_room) or resolved_presence_room
                    confidence = max(confidence, 0.93)
                elif sensor_type == "motion" and self._can_displace_presence_locked(
                    previous_room=previous_room,
                    candidate_room=room,
                    sensor_type=sensor_type,
                    previous_seen_at=previous_seen_at,
                ):
                    # El sensor observado debe ganar sobre una prediccion que queda pegada
                    # al cuarto anterior cuando el movimiento es valido en el layout.
                    resolved_presence_room = room
                    confidence = max(confidence, 0.86)
                    observed_room_forced = True
                elif not self._can_displace_presence_locked(
                    previous_room=previous_room,
                    candidate_room=resolved_presence_room,
                    sensor_type=sensor_type,
                    previous_seen_at=previous_seen_at,
                ):
                    resolved_presence_room = normalize_room_name(previous_room) or room
                    confidence = max(confidence, 0.78)
                    # Mantiene el nodo activo cuando no hay evidencia de desplazamiento adyacente.
                    self.last_active_by_room[resolved_presence_room] = now

                self.current_room = resolved_presence_room
                if observed_room_forced:
                    active_rooms = [resolved_presence_room]
                elif ai_active_rooms:
                    active_rooms = [resolved_presence_room] + [
                        rm for rm in ai_active_rooms if rm != resolved_presence_room
                    ]
                else:
                    active_rooms = [resolved_presence_room]

            elif is_active_event:
                relation = "filtrado_ventana"
                confidence = 0.12
                if self.current_room:
                    active_rooms = [self.current_room]

            else:
                if self.current_room is None and sensor_type != "occupancy":
                    self.current_room = room
                confidence = 0.45
                if self.current_room and not (sensor_type == "occupancy" and self.current_room == room):
                    active_rooms = [self.current_room]

            if transition is not None:
                if transition.get("same_room"):
                    relation = "misma_habitacion"
                elif transition.get("rejected_by_ai"):
                    relation = "no_adyacente_modelo"
                else:
                    relation = (
                        "adyacente"
                        if int(transition.get("support", 0)) >= self.confirmed_edge_support
                        else "desconocida"
                    )

            self._prune_inactive_rooms(now)

            if not active_rooms:
                active_rooms = sorted(self.last_active_by_room.keys())
            occupancy_anchor_rooms = [
                normalize_room_name(rm)
                for rm, ts in self.occupancy_confirmed_by_room.items()
                if now - ts <= timedelta(seconds=self.presence_hold_seconds)
            ]
            active_rooms = [normalize_room_name(rm) for rm in active_rooms if normalize_room_name(rm)]
            active_rooms.extend(occupancy_anchor_rooms)
            active_sensor_rooms = sorted(
                normalize_room_name(rm)
                for rm, sensor_types in self.active_sensor_types_by_room.items()
                if normalize_room_name(rm) and sensor_types
            )
            if occupancy_anchor_rooms:
                active_rooms = occupancy_anchor_rooms + [
                    rm for rm in active_sensor_rooms if rm not in occupancy_anchor_rooms
                ]
            else:
                active_rooms.extend(active_sensor_rooms)
            if not occupancy_anchor_rooms and self.current_room and self.current_room not in active_rooms:
                active_rooms.insert(0, self.current_room)
            active_rooms = list(dict.fromkeys(active_rooms))
            self.current_active_rooms = active_rooms

            occupancy_count = len(active_rooms)
            estimated_people = self._estimate_people_locked(active_rooms)
            occupancy_transformer_count = int(inference_debug.get("occupancy_transformer_people_count") or 0)
            if occupancy_transformer_count > 0:
                estimated_people = max(estimated_people, occupancy_transformer_count)
            self.current_people_estimate = estimated_people
            self.max_people_estimate = max(self.max_people_estimate, estimated_people)

            if transition is not None:
                from_room = str(transition.get("from") or "")
                to_room = str(transition.get("to") or "")
                if from_room and to_room and not self._reference_adjacent_locked(from_room, to_room):
                    transition["reference_layout_adjacent"] = False
                    layout_alert = self._record_non_adjacent_locked(
                        timestamp=now,
                        transition=transition,
                        sensor_type=sensor_type,
                        estimated_people=estimated_people,
                        active_rooms=active_rooms,
                    )
                    relation = "no_adyacente_layout_real"
                else:
                    transition["reference_layout_adjacent"] = True

            inferred_presence = "Presente" if occupancy_count > 0 else "Ausente"

            event = {
                "index": len(self.events),
                "timestamp": to_utc_iso(now),
                "room": room,
                "sensor_type": sensor_type,
                "state": state,
                "entity_id": entity_id,
                "presence_room": self.current_room or room,
                "presence_confidence": round(confidence, 4),
                "active_rooms": active_rooms,
                "inferred_presence": inferred_presence,
                "transition": transition,
                "estimated_people": estimated_people,
                "layout_alert": layout_alert,
                "inference_debug": inference_debug,
                "presence_filter": presence_filter_debug,
                "source": payload.source,
                "ai_mode": (
                    "hf_transformer_markov" if self.ai_model.training_info.get("transformer", {}).get("enabled") else "markov_ai"
                )
                if self.ai_model.ready
                else "rule_based",
            }

            if payload.timestamp is not None:
                lag_ms = (ingress_now - now).total_seconds() * 1000.0
                if 0.0 <= lag_ms <= (30.0 * 60.0 * 1000.0):
                    self.ingestion_latency_ms.append(lag_ms)
                    event["ingestion_latency_ms"] = round(lag_ms, 3)

            processing_ms = (datetime.now(timezone.utc) - ingress_now).total_seconds() * 1000.0
            if 0.0 <= processing_ms <= 60000.0:
                self.processing_latency_ms.append(processing_ms)
            event["processing_ms"] = round(processing_ms, 3)

            self.events.append(event)
            if presence_signal_allowed or not is_active_event:
                self.sequence_history.append(current_event)
            if len(self.events) > self.max_events_buffer:
                self.events = self.events[-self.max_events_buffer :]
                for idx, evt in enumerate(self.events):
                    evt["index"] = idx

            metrics = self._evaluation_metrics_locked()

            response = {
                "presencia_inferida": inferred_presence,
                "habitacion": room,
                "habitacion_inferida_ia": self.current_room or room,
                "confianza_presencia": round(confidence, 4),
                "relacion_habitaciones": relation,
                "ocupacion_estimada": occupancy_count,
                "personas_estimadas": estimated_people,
                "habitaciones_activas": active_rooms,
                "aristas_activas": len(self.edge_support),
                "transiciones_descartadas_modelo": self.rejected_transitions,
                "modelo_ia_activo": self.ai_model.ready,
                "alerta_layout": layout_alert,
                "filtro_presencia": presence_filter_debug,
                "metricas_evaluacion": metrics,
                "event": event,
            }

        await self.broadcast_event(response)
        return response

    def snapshot(self) -> dict[str, Any]:
        self._ensure_reference_layout_locked()
        visible_rooms = self._real_map_rooms_locked()
        visible_room_set = set(visible_rooms)

        if self.ai_model.ready:
            final_edges = []
            for edge in self.ai_model.adjacency_edges:
                if edge["a"] not in visible_room_set or edge["b"] not in visible_room_set:
                    continue
                k = edge_key(edge["a"], edge["b"])
                support_live = int(self.edge_support.get(k, 0))
                final_edges.append(
                    {
                        "a": edge["a"],
                        "b": edge["b"],
                        "support": max(support_live, int(edge.get("support", 0))),
                        "score": edge.get("score", 0.0),
                    }
                )
            rooms = visible_rooms
        else:
            final_edges = [
                {"a": a, "b": b, "support": int(support)}
                for (a, b), support in self.edge_support.most_common()
                if a in visible_room_set and b in visible_room_set
            ]
            rooms = visible_rooms

        visible_edge_keys = [
            (a, b)
            for (a, b), _support in self.edge_support.items()
            if a in visible_room_set and b in visible_room_set
        ]
        inferred_live_adjacency = to_adjacency(rooms, visible_edge_keys)
        inferred_live_edges = [
            {"a": a, "b": b, "support": int(support)}
            for (a, b), support in self.edge_support.most_common()
            if a in visible_room_set and b in visible_room_set
        ]
        running = bool(self.replay_task and not self.replay_task.done())

        return {
            "schema_version": "2.0-ai-live",
            "meta": {
                "source_csv": self.ai_model.training_info.get("csv_path"),
                "input_mode": self.input_mode,
                "events_total": len(self.events),
                "activation_events_total": len(self.events),
                "rooms_total": len(rooms),
                "start": self.events[0]["timestamp"] if self.events else None,
                "end": self.events[-1]["timestamp"] if self.events else None,
                "inference_mode": (
                    "ai_probabilistic_presence"
                    if self.ai_model.ready
                    else "rule_based"
                ),
                "notes": [
                    "Snapshot generado por inferencia_hub.",
                    "Incluye adyacencia aprendida desde historico CSV cuando el modelo esta entrenado.",
                ],
            },
            "rooms": rooms,
            "events": self.events,
            "final_edges": final_edges,
            "layout_reference": self._layout_payload_locked(),
            "inferred_layout_live": {
                "adjacency": inferred_live_adjacency,
                "edges": inferred_live_edges,
                "latest_touched_edge": list(self.latest_touched_edge) if self.latest_touched_edge else None,
            },
            "presence": {
                "current_room": self.current_room,
                "active_rooms": self.current_active_rooms,
                "occupancy_ground_truth_rooms": sorted(self.occupancy_confirmed_by_room.keys()),
                "live_sensor_rooms": sorted(self.active_sensor_types_by_room.keys()),
            },
            "presence_filter": self._presence_filter_config_locked(),
            "real_sensor_config": self._real_sensor_config_locked(),
            "evaluation": self._evaluation_metrics_locked(),
            "non_adjacent_recent": self.non_adjacent_records[-40:],
            "replay": {
                "running": running,
                "mode": self.input_mode,
                "paused": self.replay_paused,
                "processed_events": self.replay_processed_events,
                "total_events": self.replay_total_events,
                "last_error": self.replay_last_error,
                "progress": (
                    round(self.replay_processed_events / self.replay_total_events, 4)
                    if self.replay_total_events > 0
                    else 0.0
                ),
            },
            "model": {
                "ready": self.ai_model.ready,
                "training_info": self.ai_model.training_info,
                "rejected_transitions": self.rejected_transitions,
            },
        }

    async def reset(self) -> None:
        async with self.lock:
            self.events.clear()
            self.rooms.clear()
            self.edge_support.clear()
            self.last_active_by_room.clear()
            self.occupancy_confirmed_by_room.clear()
            self.active_sensor_types_by_room.clear()
            self.sequence_history.clear()
            self.last_activation = None
            self.current_room = None
            self.current_active_rooms = []
            self.latest_touched_edge = None
            self.rejected_transitions = 0
            self.non_adjacent_records.clear()
            self.non_adjacent_total = 0
            self.non_adjacent_multi_person = 0
            self.non_adjacent_pet_or_noise = 0
            self.non_adjacent_sensor_error = 0
            self.current_people_estimate = 0
            self.max_people_estimate = 0
            self.presence_filter_events.clear()
            self.presence_filter_suppressed_total = 0
            self.ingestion_latency_ms.clear()
            self.processing_latency_ms.clear()
            self.input_mode = "listen"
            self.replay_paused = False
            self.replay_stop_requested = False
            self.replay_step_budget = 0
            self.replay_total_events = 0
            self.replay_processed_events = 0
            self.replay_last_error = None
            self.last_replay_config = {}
            if self.ai_model.ready:
                n_rooms = len(self.ai_model.rooms)
                if n_rooms > 0:
                    self.presence_belief = np.full((n_rooms,), 1.0 / n_rooms, dtype=np.float32)
            else:
                self.presence_belief = np.zeros((0,), dtype=np.float32)

    async def broadcast_event(self, payload: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for ws in list(self.websockets):
            try:
                await ws.send_json(payload["event"])
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.websockets.discard(ws)

    async def broadcast_snapshot(self) -> None:
        dead: list[WebSocket] = []
        payload = {"kind": "snapshot", "sim_data": self.snapshot()}
        for ws in list(self.websockets):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.websockets.discard(ws)
