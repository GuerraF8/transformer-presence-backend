"""Cálculo de métricas, alertas y estimación de personas."""

from .dependencies import *  # noqa: F401,F403


class MetricsMixin:
    def _ground_truth_is_fresh_locked(self, item: dict[str, Any], now: datetime | None = None) -> bool:
        timestamp = str(item.get("timestamp") or "")
        if not timestamp:
            return False
        try:
            observed_at = parse_iso_datetime(timestamp)
        except Exception:
            return False
        current = now or datetime.now(timezone.utc)
        return current - observed_at <= timedelta(seconds=self.presence_hold_seconds)

    def _record_ground_truth_sample_locked(self, sample: dict[str, Any]) -> None:
        self.ground_truth_samples.append(sample)

    def _apply_count_ground_truth_locked(
        self,
        *,
        timestamp: datetime,
        entity_id: str,
        room: str,
        count: int,
        predicted_people: int | None = None,
    ) -> None:
        room = normalize_room_name(room)
        predicted = int(self.current_people_estimate if predicted_people is None else predicted_people)
        sample = {
            "timestamp": to_utc_iso(timestamp),
            "type": "people_count",
            "entity_id": entity_id,
            "scope": "room" if room else "global",
            "room": room,
            "ground_truth_count": int(count),
            "predicted_people": predicted,
        }
        self._record_ground_truth_sample_locked(sample)
        if room:
            self.room_count_ground_truth[room] = {
                "timestamp": sample["timestamp"],
                "entity_id": entity_id,
                "room": room,
                "count": int(count),
            }
            if count > 0:
                self.occupancy_confirmed_by_room[room] = timestamp
                self.last_active_by_room[room] = timestamp
                self.current_room = room
                if room not in self.current_active_rooms:
                    self.current_active_rooms = list(dict.fromkeys([*self.current_active_rooms, room]))
            else:
                self.occupancy_confirmed_by_room.pop(room, None)
                self.last_active_by_room.pop(room, None)
                self.current_active_rooms = [
                    active_room for active_room in self.current_active_rooms if active_room != room
                ]
                if self.current_room == room:
                    self.current_room = self.current_active_rooms[0] if self.current_active_rooms else None
            room_floor = sum(
                int(item.get("count") or 0)
                for item in self.room_count_ground_truth.values()
                if self._ground_truth_is_fresh_locked(item, timestamp)
            )
            self.current_people_estimate = max(self.current_people_estimate, room_floor)
        else:
            self.people_count_ground_truth = {
                "timestamp": sample["timestamp"],
                "entity_id": entity_id,
                "count": int(count),
            }
            self.current_people_estimate = int(count)
        self.max_people_estimate = max(self.max_people_estimate, self.current_people_estimate)

    def _record_confirmation_ground_truth_locked(
        self,
        *,
        timestamp: datetime,
        entity_id: str,
        state: str,
        training_role: str,
        room: str,
    ) -> None:
        room = normalize_room_name(room)
        if not room or str(state).lower() != "on":
            return
        expected_presence = training_role == "person_confirmation"
        sample = {
            "timestamp": to_utc_iso(timestamp),
            "type": training_role,
            "entity_id": entity_id,
            "room": room,
            "expected_presence": expected_presence,
            "predicted_room": self.current_room,
            "predicted_active_rooms": list(self.current_active_rooms),
            "predicted_people": self.current_people_estimate,
        }
        self._record_ground_truth_sample_locked(sample)
        if training_role == "person_confirmation":
            self.occupancy_confirmed_by_room[room] = timestamp
            self.last_active_by_room[room] = timestamp
            self.current_room = room
            if room not in self.current_active_rooms:
                self.current_active_rooms = list(dict.fromkeys([room, *self.current_active_rooms]))
            self.current_people_estimate = max(1, self.current_people_estimate)
            self.max_people_estimate = max(self.max_people_estimate, self.current_people_estimate)

    def _ground_truth_metrics_locked(self) -> dict[str, Any]:
        samples = list(self.ground_truth_samples)
        count_samples = [item for item in samples if item.get("type") == "people_count"]
        global_count_samples = [
            item for item in count_samples if item.get("scope") == "global"
        ]
        room_count_samples = [
            item for item in count_samples if item.get("scope") == "room"
        ]
        exact_count_matches = [
            item
            for item in global_count_samples
            if int(item.get("predicted_people") or 0) == int(item.get("ground_truth_count") or 0)
        ]
        count_errors = [
            abs(int(item.get("predicted_people") or 0) - int(item.get("ground_truth_count") or 0))
            for item in global_count_samples
        ]

        presence_samples = [
            item
            for item in samples
            if item.get("type") in {"person_confirmation", "pet_confirmation", "occupancy"}
        ]
        person_samples = [
            item
            for item in presence_samples
            if item.get("type") in {"person_confirmation", "occupancy"}
        ]
        pet_samples = [
            item
            for item in presence_samples
            if item.get("type") == "pet_confirmation"
        ]
        person_hits = [
            item
            for item in person_samples
            if normalize_room_name(str(item.get("room") or ""))
            in {
                normalize_room_name(str(room))
                for room in item.get("predicted_active_rooms", [])
            }
            or normalize_room_name(str(item.get("predicted_room") or ""))
            == normalize_room_name(str(item.get("room") or ""))
        ]
        pet_false_positives = [
            item
            for item in pet_samples
            if normalize_room_name(str(item.get("room") or ""))
            in {
                normalize_room_name(str(room))
                for room in item.get("predicted_active_rooms", [])
            }
            or normalize_room_name(str(item.get("predicted_room") or ""))
            == normalize_room_name(str(item.get("room") or ""))
        ]

        return {
            "samples_total": len(samples),
            "count": {
                "samples": len(count_samples),
                "global_samples": len(global_count_samples),
                "room_samples": len(room_count_samples),
                "count_accuracy": (
                    round(len(exact_count_matches) / len(global_count_samples), 4)
                    if global_count_samples
                    else None
                ),
                "count_mae": (
                    round(float(np.mean(count_errors)), 4)
                    if count_errors
                    else None
                ),
                "last_global": self.people_count_ground_truth,
                "last_by_room": dict(self.room_count_ground_truth),
            },
            "presence": {
                "samples": len(presence_samples),
                "person_samples": len(person_samples),
                "person_room_match_rate": (
                    round(len(person_hits) / len(person_samples), 4)
                    if person_samples
                    else None
                ),
                "pet_samples": len(pet_samples),
                "pet_false_positive_rate": (
                    round(len(pet_false_positives) / len(pet_samples), 4)
                    if pet_samples
                    else None
                ),
            },
        }

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
                "ground_truth_count": self.people_count_ground_truth,
                "ground_truth_room_counts": dict(self.room_count_ground_truth),
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
            "ground_truth": self._ground_truth_metrics_locked(),
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
