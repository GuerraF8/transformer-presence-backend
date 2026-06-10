"""Generación de escenarios y eventos sintéticos de presencia."""

from .dependencies import *  # noqa: F401,F403


class SimulationMixin:
    def _event_feature_vector(
        self,
        event: EventRecord,
        rooms: list[str],
        room_to_idx: dict[str, int],
        previous_ts: datetime | None,
    ) -> np.ndarray:
        vec = np.zeros((len(rooms) + 11,), dtype=np.float32)
        room_idx = room_to_idx.get(event.room)
        if room_idx is not None:
            vec[room_idx] = 1.0
        sensor_slot = {
            "motion": 0,
            "door": 1,
            "occupancy": 2,
        }.get(event.sensor_type, 3)
        vec[len(rooms) + sensor_slot] = 1.0
        state_slot = {
            "active": 0,
            "inactive": 1,
            "unavailable": 2,
        }.get(classify_state_bucket(event.sensor_type, event.state), 3)
        vec[len(rooms) + 4 + state_slot] = 1.0
        vec[len(rooms) + 8] = 1.0 if is_activation(event.sensor_type, event.state) else 0.0
        vec[len(rooms) + 9] = SENSOR_RELIABILITY.get(event.sensor_type, SENSOR_RELIABILITY["other"])
        gap = 0.0 if previous_ts is None else max(0.0, (event.timestamp - previous_ts).total_seconds())
        vec[len(rooms) + 10] = min(1.0, gap / 900.0)
        return vec

    @staticmethod
    def _weighted_choice(rng: random.Random, values: list[str], weights: list[float]) -> str:
        if not values:
            return ""
        total = float(sum(max(0.0, weight) for weight in weights))
        if total <= 0.0:
            return rng.choice(values)
        cursor = rng.random() * total
        running = 0.0
        for value, weight in zip(values, weights):
            running += max(0.0, float(weight))
            if cursor <= running:
                return value
        return values[-1]

    @staticmethod
    def _counter_payload(counter: Counter[Any], limit: int = 20) -> dict[str, int]:
        return {str(key): int(value) for key, value in counter.most_common(limit)}

    def _build_real_profile(
        self,
        events: list[EventRecord],
        rooms: list[str],
        layout: dict[str, list[str]],
        max_events: int,
    ) -> dict[str, Any] | None:
        selected_events = [
            event
            for event in events[-max_events:]
            if event.room in rooms and event.sensor_type in {"motion", "door", "occupancy"}
        ]
        if len(selected_events) < 30:
            return None

        activation_events = [event for event in selected_events if is_activation(event.sensor_type, event.state)]
        room_counts: Counter[str] = Counter()
        sensor_counts: Counter[str] = Counter()
        room_sensor_counts: dict[str, Counter[str]] = defaultdict(Counter)
        hour_room_counts: dict[int, Counter[str]] = defaultdict(Counter)
        transition_counts: dict[str, Counter[str]] = defaultdict(Counter)
        coactivation_counts: dict[str, Counter[str]] = defaultdict(Counter)
        gap_by_sensor: dict[str, list[float]] = defaultdict(list)
        duration_by_sensor: dict[str, list[float]] = defaultdict(list)

        last_activation: EventRecord | None = None
        last_by_entity: dict[str, EventRecord] = {}
        recent_active_by_room: dict[str, datetime] = {}
        coactivation_window = timedelta(seconds=90)

        for event in selected_events:
            if is_activation(event.sensor_type, event.state):
                room_counts[event.room] += 1
                sensor_counts[event.sensor_type] += 1
                room_sensor_counts[event.room][event.sensor_type] += 1
                hour_room_counts[event.timestamp.hour][event.room] += 1

                if last_activation is not None:
                    gap = max(0.0, (event.timestamp - last_activation.timestamp).total_seconds())
                    gap_by_sensor[event.sensor_type].append(gap)
                    if last_activation.room != event.room and 1.0 <= gap <= 900.0:
                        transition_counts[last_activation.room][event.room] += 1
                last_activation = event

                stale = [
                    room
                    for room, ts in recent_active_by_room.items()
                    if event.timestamp - ts > coactivation_window
                ]
                for room in stale:
                    recent_active_by_room.pop(room, None)
                for other_room in recent_active_by_room.keys():
                    if other_room != event.room:
                        coactivation_counts[event.room][other_room] += 1
                        coactivation_counts[other_room][event.room] += 1
                recent_active_by_room[event.room] = event.timestamp

            previous = last_by_entity.get(event.entity_id)
            if previous is not None and is_activation(previous.sensor_type, previous.state) and not is_activation(event.sensor_type, event.state):
                duration = max(0.0, (event.timestamp - previous.timestamp).total_seconds())
                if duration <= 3600.0:
                    duration_by_sensor[event.sensor_type].append(duration)
            last_by_entity[event.entity_id] = event

        if not activation_events:
            return None

        transition_payload = {
            room: self._counter_payload(counter, limit=len(rooms))
            for room, counter in transition_counts.items()
        }
        coactivation_payload = {
            room: self._counter_payload(counter, limit=len(rooms))
            for room, counter in coactivation_counts.items()
        }
        gap_stats = {
            sensor_type: {
                "p10": round(safe_quantile(values, 0.10, 2.0), 3),
                "p50": round(safe_quantile(values, 0.50, 4.0), 3),
                "p90": round(safe_quantile(values, 0.90, 30.0), 3),
            }
            for sensor_type, values in gap_by_sensor.items()
        }
        duration_stats = {
            sensor_type: {
                "p10": round(safe_quantile(values, 0.10, 1.0), 3),
                "p50": round(safe_quantile(values, 0.50, 4.0), 3),
                "p90": round(safe_quantile(values, 0.90, 30.0), 3),
            }
            for sensor_type, values in duration_by_sensor.items()
        }

        hour_room_payload = {
            str(hour): self._counter_payload(counter, limit=len(rooms))
            for hour, counter in hour_room_counts.items()
        }
        room_sensor_payload = {
            room: self._counter_payload(counter, limit=8)
            for room, counter in room_sensor_counts.items()
        }
        transition_total = sum(sum(counter.values()) for counter in transition_counts.values())
        movement_probability = min(
            0.95,
            max(0.15, transition_total / max(1, len(activation_events) - 1)),
        )

        return {
            "enabled": True,
            "events_total": len(selected_events),
            "activation_events_total": len(activation_events),
            "rooms": rooms,
            "room_counts": self._counter_payload(room_counts, limit=len(rooms)),
            "sensor_counts": self._counter_payload(sensor_counts, limit=8),
            "room_sensor_counts": room_sensor_payload,
            "hour_room_counts": hour_room_payload,
            "transition_counts": transition_payload,
            "coactivation_counts": coactivation_payload,
            "gap_stats": gap_stats,
            "duration_stats": duration_stats,
            "movement_probability": round(float(movement_probability), 4),
            "layout_edges_total": sum(len(neighbors) for neighbors in layout.values()) // 2,
        }

    def _profile_room_weights(
        self,
        profile: dict[str, Any] | None,
        hour: int,
        rooms: list[str],
    ) -> list[float]:
        if not profile:
            return [1.0 for _room in rooms]
        hour_counts = profile.get("hour_room_counts", {}).get(str(hour), {})
        global_counts = profile.get("room_counts", {})
        weights = [
            float(hour_counts.get(room, 0)) + (0.25 * float(global_counts.get(room, 0))) + 1.0
            for room in rooms
        ]
        return weights

    def _profile_next_room(
        self,
        rng: random.Random,
        profile: dict[str, Any] | None,
        layout: dict[str, list[str]],
        current_room: str,
    ) -> str:
        neighbors = list(layout.get(current_room, []))
        if not neighbors:
            return current_room
        if not profile:
            return rng.choice(neighbors)
        transitions = profile.get("transition_counts", {}).get(current_room, {})
        weights = [float(transitions.get(room, 0)) + 1.0 for room in neighbors]
        return self._weighted_choice(rng, neighbors, weights)

    def _profile_gap_seconds(
        self,
        rng: random.Random,
        profile: dict[str, Any] | None,
        sensor_type: str,
        fallback_seconds: int,
        randomize: bool,
    ) -> float:
        if not profile:
            base = float(fallback_seconds)
        else:
            stats = profile.get("gap_stats", {}).get(sensor_type) or profile.get("gap_stats", {}).get("motion") or {}
            p10 = float(stats.get("p10", max(1.0, fallback_seconds * 0.5)))
            p50 = float(stats.get("p50", fallback_seconds))
            p90 = float(stats.get("p90", max(p50, fallback_seconds * 3.0)))
            base = rng.triangular(max(0.5, p10), max(1.0, p90), max(0.5, p50))
        if randomize:
            base *= rng.uniform(0.55, 1.8)
        return max(0.5, min(900.0, base))

    def _profile_duration_seconds(
        self,
        rng: random.Random,
        profile: dict[str, Any] | None,
        sensor_type: str,
        fallback_seconds: int,
    ) -> float:
        if not profile:
            return float(fallback_seconds)
        stats = profile.get("duration_stats", {}).get(sensor_type) or {}
        p10 = float(stats.get("p10", max(1.0, fallback_seconds * 0.5)))
        p50 = float(stats.get("p50", fallback_seconds))
        p90 = float(stats.get("p90", max(p50, fallback_seconds * 3.0)))
        return max(0.5, min(300.0, rng.triangular(max(0.5, p10), max(1.0, p90), max(0.5, p50))))

    def _weak_labeled_events_from_history(
        self,
        events: list[EventRecord],
        rooms: list[str],
        hold_seconds: int,
        max_events: int,
    ) -> list[tuple[EventRecord, set[str]]]:
        room_set = set(rooms)
        active_by_room: dict[str, datetime] = {}
        out: list[tuple[EventRecord, set[str]]] = []
        hold = timedelta(seconds=hold_seconds)

        for event in events[-max_events:]:
            if event.room not in room_set:
                continue
            if is_activation(event.sensor_type, event.state):
                active_by_room[event.room] = event.timestamp
            elif event.sensor_type == "occupancy":
                active_by_room.pop(event.room, None)

            stale = [
                room
                for room, ts in active_by_room.items()
                if event.timestamp - ts > hold
            ]
            for room in stale:
                active_by_room.pop(room, None)
            out.append((event, set(active_by_room.keys())))

        return out

    def _generate_simulated_presence_events(
        self,
        req: TrainSimulatorPresenceRequest,
        reference_layout: dict[str, list[str]] | None,
        real_profile: dict[str, Any] | None = None,
    ) -> tuple[list[tuple[EventRecord, set[str]]], list[str], dict[str, list[str]]]:
        base_rooms = sorted(normalize_adjacency_map({room: [] for room in req.rooms}).keys())
        if not base_rooms and reference_layout:
            base_rooms = sorted(normalize_adjacency_map(reference_layout).keys())
        if not base_rooms:
            base_rooms = ["bedroom", "entertainment_room", "foyer", "kitchen", "living", "sittingroom"]

        layout = build_layout_for_request(base_rooms, req.template, req.layout_edges)
        layout = normalize_adjacency_map(layout, base_rooms)
        if not any(layout.values()):
            layout = build_scenario_templates(base_rooms).get("real_home", {})
            layout = normalize_adjacency_map(layout, base_rooms)

        rooms = sorted(layout.keys())
        rng = random.Random(req.seed)
        rows: list[tuple[EventRecord, set[str]]] = []
        cursor = datetime(2026, 1, 1, tzinfo=timezone.utc)

        def occupied_rooms_from_positions(positions: list[str]) -> set[str]:
            return {room for room in positions if room}

        def advance(sensor_type: str) -> None:
            nonlocal cursor
            cursor += timedelta(
                seconds=self._profile_gap_seconds(
                    rng,
                    real_profile,
                    sensor_type,
                    req.event_interval_seconds,
                    req.domain_randomization,
                )
            )

        def emit(
            room: str,
            sensor_type: str,
            state: str,
            occupied_rooms: set[str],
            *,
            allow_false_negative: bool = True,
        ) -> None:
            nonlocal cursor
            if (
                allow_false_negative
                and req.domain_randomization
                and sensor_type in {"motion", "occupancy"}
                and rng.random() < req.false_negative_rate
            ):
                advance(sensor_type)
                return
            rows.append(
                (
                    EventRecord(
                        timestamp=cursor,
                        entity_id=f"binary_sensor.{room}_{sensor_type}_sim",
                        state=state,
                        sensor_type=sensor_type,
                        room=room,
                    ),
                    set(occupied_rooms),
                )
            )
            if sensor_type == "motion" and state == "on":
                cursor += timedelta(seconds=self._profile_duration_seconds(rng, real_profile, sensor_type, 1))
            else:
                advance(sensor_type)

        def emit_motion_pair(room: str, occupied_rooms: set[str]) -> None:
            emit(room, "motion", "on", occupied_rooms)
            emit(room, "motion", "off", occupied_rooms, allow_false_negative=False)

        def maybe_emit_false_positive(occupied_rooms: set[str]) -> None:
            if not req.domain_randomization or rng.random() >= req.false_positive_rate:
                return
            inactive_rooms = [room for room in rooms if room not in occupied_rooms]
            if not inactive_rooms:
                return
            weights = self._profile_room_weights(real_profile, cursor.hour, inactive_rooms)
            false_room = self._weighted_choice(rng, inactive_rooms, weights)
            emit_motion_pair(false_room, occupied_rooms)

        def maybe_emit_coactivation(room: str, occupied_rooms: set[str]) -> None:
            if not req.domain_randomization or not real_profile or rng.random() > 0.08:
                return
            coactive = real_profile.get("coactivation_counts", {}).get(room, {})
            candidates = [candidate for candidate in rooms if candidate != room and coactive.get(candidate, 0) > 0]
            if not candidates:
                return
            weights = [float(coactive.get(candidate, 0)) for candidate in candidates]
            coactive_room = self._weighted_choice(rng, candidates, weights)
            emit_motion_pair(coactive_room, occupied_rooms)

        for scenario_idx in range(req.scenarios):
            profile_people_ceiling = req.max_people
            if req.domain_randomization:
                profile_people_ceiling = max(1, min(req.max_people, rng.randint(1, req.max_people)))
            people_count = rng.randint(1, profile_people_ceiling)
            start_weights = self._profile_room_weights(real_profile, cursor.hour, rooms)
            positions = [self._weighted_choice(rng, rooms, start_weights) for _ in range(people_count)]
            room_counts = Counter(positions)
            occupied = occupied_rooms_from_positions(positions)

            for room in sorted(occupied):
                emit(room, "occupancy", "on", occupied, allow_false_negative=False)

            for _ in range(req.steps_per_scenario):
                person_idx = rng.randrange(people_count)
                current_room = positions[person_idx]
                neighbors = list(layout.get(current_room, []))
                movement_probability = req.movement_probability
                if real_profile:
                    movement_probability = (movement_probability + float(real_profile.get("movement_probability", movement_probability))) / 2.0
                if req.domain_randomization:
                    movement_probability = max(0.05, min(0.95, movement_probability * rng.uniform(0.55, 1.35)))
                will_move = bool(neighbors) and rng.random() <= movement_probability

                if will_move:
                    next_room = self._profile_next_room(rng, real_profile, layout, current_room)
                    previous_room = current_room
                    room_counts[previous_room] -= 1
                    if room_counts[previous_room] <= 0:
                        del room_counts[previous_room]
                    room_counts[next_room] += 1
                    positions[person_idx] = next_room
                    occupied = set(room_counts.keys())

                    if previous_room not in occupied:
                        emit(previous_room, "occupancy", "off", occupied)
                    if room_counts[next_room] == 1:
                        emit(next_room, "occupancy", "on", occupied)
                    emit_motion_pair(next_room, occupied)
                    maybe_emit_coactivation(next_room, occupied)
                else:
                    occupied = set(room_counts.keys())
                    room = current_room
                    refresh_probability = req.occupancy_refresh_probability
                    if real_profile:
                        room_sensor_counts = real_profile.get("room_sensor_counts", {}).get(room, {})
                        room_total = max(1.0, sum(float(value) for value in room_sensor_counts.values()))
                        refresh_probability = max(
                            0.02,
                            min(0.85, (refresh_probability + (float(room_sensor_counts.get("occupancy", 0)) / room_total)) / 2.0),
                        )
                    if rng.random() <= refresh_probability:
                        emit(room, "occupancy", "on", occupied)
                    emit_motion_pair(room, occupied)
                    maybe_emit_coactivation(room, occupied)
                maybe_emit_false_positive(occupied)

            cursor += timedelta(minutes=5 + scenario_idx % 11)

        return rows, rooms, layout
