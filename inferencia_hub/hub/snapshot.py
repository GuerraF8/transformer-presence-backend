"""Generación de snapshots, reinicio y publicación del estado."""

from .dependencies import *  # noqa: F401,F403


class SnapshotMixin:
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
        latest_event = self.events[-1] if self.events else None

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
            "profile": self._profile_payload_locked(),
            "events": self.events,
            "final_edges": final_edges,
            "layout_reference": self._layout_payload_locked(),
            "inferred_layout_live": {
                "adjacency": inferred_live_adjacency,
                "edges": inferred_live_edges,
                "latest_touched_edge": list(self.latest_touched_edge) if self.latest_touched_edge else None,
            },
            "presence": build_presence_snapshot(
                current_room=self.current_room,
                active_rooms=self.current_active_rooms,
                people_estimate=self.current_people_estimate,
                latest_event=latest_event,
                occupancy_ground_truth_rooms=sorted(
                    self.occupancy_confirmed_by_room.keys()
                ),
                live_sensor_rooms=sorted(self.active_sensor_types_by_room.keys()),
            ),
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
                "profile_fingerprint": self.active_profile_fingerprint,
                "compatible": self.active_profile_model_compatible,
                "training_info": self.ai_model.training_info,
                "rejected_transitions": self.rejected_transitions,
                "pet_filter": self.ai_model.pet_filter_info,
                "relative_occupancy": self.ai_model.relative_occupancy_info,
            },
        }

    def _reset_transient_locked(self) -> None:
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
                self.presence_belief = np.full(
                    (n_rooms,),
                    1.0 / n_rooms,
                    dtype=np.float32,
                )
        else:
            self.presence_belief = np.zeros((0,), dtype=np.float32)

    async def reset(self) -> None:
        async with self.lock:
            self._reset_transient_locked()

    async def broadcast_event(self, payload: dict[str, Any]) -> None:
        if self.snapshot_publisher is not None:
            await self.snapshot_publisher(payload)

    async def broadcast_snapshot(self) -> None:
        if self.snapshot_publisher is not None:
            await self.snapshot_publisher(
                {"kind": "snapshot", "sim_data": self.snapshot()}
            )
