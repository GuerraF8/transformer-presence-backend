"""Cálculo de métricas, alertas y estimación de personas."""

from .dependencies import *  # noqa: F401,F403


class MetricsMixin:
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
        # Una ocupación confirmada más movimiento simultáneo en otra habitación
        # proporciona evidencia de la presencia de varias personas.
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
