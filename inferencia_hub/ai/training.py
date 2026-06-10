"""Coordinación del entrenamiento del modelo de adyacencia."""

from .dependencies import *  # noqa: F401,F403


class TrainingMixin:
    def train_from_csv(self, req: TrainModelRequest) -> dict[str, Any]:
        return self.train_from_csv_with_reference(req, None)

    def train_from_csv_with_reference(
        self,
        req: TrainModelRequest,
        reference_layout: dict[str, list[str]] | None,
    ) -> dict[str, Any]:
        with self._train_lock:
            events = self._read_history_events(
                req.csv_path,
                req.debounce_seconds,
                req.include_all_state_transitions,
            )
            if len(events) < 30:
                raise ValueError("No hay suficientes eventos en el CSV para entrenar el modelo.")

            directed, outgoing, transitions = self._build_transition_counts(
                events,
                req.debounce_seconds,
                req.min_gap_seconds,
                req.max_gap_seconds,
            )
            rooms = sorted({evt.room for evt in events})
            if len(rooms) < 2:
                raise ValueError("El historial no contiene al menos dos habitaciones inferibles.")

            state_transitions_total = max(0, len(events) - 1)
            activation_events_total = len(self._extract_activation_events(events, req.debounce_seconds))
            room_change_transitions_total = 0
            for idx in range(1, len(events)):
                if events[idx - 1].room != events[idx].room:
                    room_change_transitions_total += 1

            count_probs = self._count_probs(directed, rooms)
            transformer_probs, transformer_meta, transformer_model, transformer_device = self._train_transformer_probs(
                events,
                rooms,
                req.debounce_seconds,
                req.min_gap_seconds,
                req.max_gap_seconds,
                req.epochs,
                req.max_samples,
            )
            blended_probs, blend_meta = self._blend_probs(
                count_probs,
                transformer_probs,
                outgoing,
                rooms,
            )
            effective_reference_layout = (
                normalize_adjacency_map(reference_layout, rooms)
                if reference_layout
                else build_scenario_templates(rooms).get("real_home", {})
            )
            neighbors, edge_list, thresholds = self._infer_graph(
                rooms,
                directed,
                blended_probs,
                req.degree_limit,
                effective_reference_layout,
            )

            ollama_review = None
            if req.use_ollama_validation:
                ollama_review = self._validate_edges_with_ollama(
                    edge_list,
                    rooms,
                    req.ollama_url,
                    req.ollama_model,
                )

            room_to_idx = {room: idx for idx, room in enumerate(rooms)}
            # Restringe transiciones a aristas aprendidas para no inventar adyacencias.
            transition_matrix = blended_probs.copy()
            for i, room in enumerate(rooms):
                allowed = set(neighbors.get(room, [])) | {room}
                mask = np.zeros((len(rooms),), dtype=np.float32)
                for other in allowed:
                    mask[room_to_idx[other]] = 1.0
                transition_matrix[i] *= mask
                row_sum = float(transition_matrix[i].sum())
                if row_sum > 0:
                    transition_matrix[i] /= row_sum
                else:
                    transition_matrix[i, i] = 1.0

            sensor_room_votes: dict[str, Counter[str]] = defaultdict(Counter)
            for evt in events:
                sensor_room_votes[evt.entity_id][evt.room] += 1
            sensor_room_map = {
                sensor: votes.most_common(1)[0][0]
                for sensor, votes in sensor_room_votes.items()
                if votes
            }

            self.ready = True
            self.rooms = rooms
            self.room_to_idx = room_to_idx
            self.transition_matrix = transition_matrix
            self.adjacency_neighbors = neighbors
            self.adjacency_edges = edge_list
            self.sensor_room_map = sensor_room_map
            self.transformer_model = transformer_model
            self.transformer_device = transformer_device
            self.training_info = {
                "events_total": len(events),
                "state_transitions_total": state_transitions_total,
                "activation_events_total": activation_events_total,
                "transitions_total": len(transitions),
                "room_change_transitions_total": room_change_transitions_total,
                "rooms_total": len(rooms),
                "directed_edges_total": len(directed),
                "include_all_state_transitions": req.include_all_state_transitions,
                "transition_filtering": (
                    "full_history_with_activation_targets"
                    if req.include_all_state_transitions
                    else "activation+debounce+gap"
                ),
                "count_model": "markov_transition",
                "transformer": transformer_meta,
                "blend": blend_meta,
                "thresholds": thresholds,
                "reference_penalty_enabled": bool(effective_reference_layout),
                "ollama_review": ollama_review,
            }

            return {
                "status": "ok",
                "csv_path": req.csv_path,
                "rooms": rooms,
                "edges": edge_list,
                "training_info": self.training_info,
            }
