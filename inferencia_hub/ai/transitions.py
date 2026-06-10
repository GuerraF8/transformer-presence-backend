"""Preparación y predicción de transiciones entre habitaciones."""

from .dependencies import *  # noqa: F401,F403


class TransitionsMixin:
    def _extract_activation_events(
        self,
        events: list[EventRecord],
        debounce_seconds: int,
    ) -> list[EventRecord]:
        last_by_entity: dict[str, datetime] = {}
        out: list[EventRecord] = []
        for evt in events:
            if not is_activation(evt.sensor_type, evt.state):
                continue
            prev_ts = last_by_entity.get(evt.entity_id)
            if prev_ts is not None and (evt.timestamp - prev_ts).total_seconds() <= debounce_seconds:
                continue
            last_by_entity[evt.entity_id] = evt.timestamp
            out.append(evt)
        return out

    def _read_history_events(
        self,
        csv_path: str,
        debounce_seconds: int,
        include_all_state_transitions: bool,
    ) -> list[EventRecord]:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"No existe CSV: {csv_path}")

        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        parsed: list[EventRecord] = []
        for row in rows:
            entity_id = str(row.get("entity_id", "")).strip()
            state = str(row.get("state", "")).strip().lower()
            ts_raw = str(row.get("last_changed", "")).strip()
            if not entity_id or not ts_raw:
                continue
            try:
                ts = parse_iso_datetime(ts_raw)
            except Exception:
                continue
            sensor_type = classify_sensor_type(entity_id)
            room = infer_room_from_entity(entity_id)
            parsed.append(
                EventRecord(
                    timestamp=ts,
                    entity_id=entity_id,
                    state=state,
                    sensor_type=sensor_type,
                    room=room,
                )
            )

        parsed.sort(key=lambda item: item.timestamp)

        if include_all_state_transitions:
            return parsed

        # Cuando se ignoran los cambios de estado inactivos, se conservan solo activaciones sin rebotes.
        return self._extract_activation_events(parsed, debounce_seconds)

    def _build_transition_counts(
        self,
        events: list[EventRecord],
        debounce_seconds: int,
        min_gap_seconds: int,
        max_gap_seconds: int,
    ) -> tuple[Counter[tuple[str, str]], Counter[str], list[tuple[EventRecord, EventRecord, float]]]:
        directed: Counter[tuple[str, str]] = Counter()
        outgoing: Counter[str] = Counter()
        transitions: list[tuple[EventRecord, EventRecord, float]] = []
        activation_events = self._extract_activation_events(events, debounce_seconds)

        for idx in range(1, len(activation_events)):
            prev = activation_events[idx - 1]
            cur = activation_events[idx]
            gap = (cur.timestamp - prev.timestamp).total_seconds()
            if prev.room == cur.room:
                continue
            if gap < min_gap_seconds or gap > max_gap_seconds:
                continue

            transitions.append((prev, cur, gap))
            if prev.room != cur.room:
                directed[(prev.room, cur.room)] += 1
                outgoing[prev.room] += 1

        return directed, outgoing, transitions

    def _count_probs(
        self,
        directed: Counter[tuple[str, str]],
        rooms: list[str],
    ) -> np.ndarray:
        room_to_idx = {room: idx for idx, room in enumerate(rooms)}
        probs = np.zeros((len(rooms), len(rooms)), dtype=np.float32)
        outgoing = defaultdict(float)
        for (orig, dst), count in directed.items():
            outgoing[orig] += float(count)
        for (orig, dst), count in directed.items():
            total = outgoing.get(orig, 0.0)
            if total <= 0:
                continue
            probs[room_to_idx[orig], room_to_idx[dst]] = float(count) / total
        return probs

    def _prepare_transformer_dataset(
        self,
        events: list[EventRecord],
        rooms: list[str],
        debounce_seconds: int,
        min_gap_seconds: int,
        max_gap_seconds: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        history_len = self.transformer_context_length
        room_to_idx = {room: idx for idx, room in enumerate(rooms)}
        n_features = len(rooms) + 11
        if len(events) <= TRANSFORMER_MIN_SAMPLES:
            return None

        raw_values: list[np.ndarray] = []
        raw_time: list[np.ndarray] = []
        prev_ts: datetime | None = None
        for event in events:
            vec = np.zeros((n_features,), dtype=np.float32)
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
            gap = 0.0 if prev_ts is None else max(0.0, (event.timestamp - prev_ts).total_seconds())
            vec[len(rooms) + 10] = min(1.0, gap / 900.0)
            prev_ts = event.timestamp
            raw_values.append(vec)
            raw_time.append(time_features_from_dt(event.timestamp))

        x_values: list[np.ndarray] = []
        x_time: list[np.ndarray] = []
        x_future: list[np.ndarray] = []
        y_target: list[int] = []
        y_origin: list[int] = []

        last_by_entity: dict[str, datetime] = {}
        last_activation: EventRecord | None = None
        for idx, event in enumerate(events):
            if not is_activation(event.sensor_type, event.state):
                continue
            prev_same_entity = last_by_entity.get(event.entity_id)
            if prev_same_entity is not None and (event.timestamp - prev_same_entity).total_seconds() <= debounce_seconds:
                continue
            last_by_entity[event.entity_id] = event.timestamp

            if last_activation is None:
                last_activation = event
                continue

            activation_gap = (event.timestamp - last_activation.timestamp).total_seconds()
            if last_activation.room == event.room:
                last_activation = event
                continue
            if activation_gap < min_gap_seconds or activation_gap > max_gap_seconds:
                last_activation = event
                continue
            if idx < history_len:
                last_activation = event
                continue

            x_values.append(np.stack(raw_values[idx - history_len : idx], axis=0))
            x_time.append(np.stack(raw_time[idx - history_len : idx], axis=0))
            x_future.append(raw_time[idx])
            y_target.append(room_to_idx[event.room])
            y_origin.append(room_to_idx[last_activation.room])
            last_activation = event

        if len(x_values) < TRANSFORMER_MIN_SAMPLES:
            return None

        return (
            np.asarray(x_values, dtype=np.float32),
            np.asarray(x_time, dtype=np.float32),
            np.asarray(x_future, dtype=np.float32),
            np.asarray(y_target, dtype=np.int64),
            np.asarray(y_origin, dtype=np.int64),
        )

    def _train_transformer_probs(
        self,
        events: list[EventRecord],
        rooms: list[str],
        debounce_seconds: int,
        min_gap_seconds: int,
        max_gap_seconds: int,
        epochs: int,
        max_samples: int,
    ) -> tuple[np.ndarray | None, dict[str, Any], Any | None, Any | None]:
        meta: dict[str, Any] = {
            "enabled": False,
            "reason": "",
            "samples": 0,
            "epochs": 0,
            "context_length": self.transformer_context_length,
        }

        if not HF_AVAILABLE:
            meta["reason"] = "torch/transformers no disponible"
            return None, meta, None, None

        dataset = self._prepare_transformer_dataset(
            events,
            rooms,
            debounce_seconds,
            min_gap_seconds,
            max_gap_seconds,
        )
        if dataset is None:
            meta["reason"] = "muestras insuficientes"
            return None, meta, None, None

        x_values, x_time, x_future, y_target, y_origin = dataset
        sample_count = int(x_values.shape[0])
        meta["samples"] = sample_count

        if sample_count > max_samples:
            idx = np.linspace(0, sample_count - 1, max_samples, dtype=int)
            x_values = x_values[idx]
            x_time = x_time[idx]
            x_future = x_future[idx]
            y_target = y_target[idx]
            y_origin = y_origin[idx]
            sample_count = int(x_values.shape[0])
            meta["samples"] = sample_count

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = NextRoomTransformer(input_size=int(x_values.shape[2]), room_count=len(rooms)).to(device)

        x_values_t = torch.tensor(x_values, dtype=torch.float32, device=device)
        x_time_t = torch.tensor(x_time, dtype=torch.float32, device=device)
        x_future_t = torch.tensor(x_future, dtype=torch.float32, device=device).unsqueeze(1)
        y_target_t = torch.tensor(y_target, dtype=torch.long, device=device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        batch_size = 256
        for _ in range(epochs):
            perm = torch.randperm(sample_count, device=device)
            for start in range(0, sample_count, batch_size):
                idx = perm[start : start + batch_size]
                logits = model(x_values_t[idx], x_time_t[idx], x_future_t[idx])
                loss = criterion(logits, y_target_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        probs_sum = np.zeros((len(rooms), len(rooms)), dtype=np.float32)
        probs_count = np.zeros((len(rooms),), dtype=np.float32)

        model.eval()
        with torch.no_grad():
            for start in range(0, sample_count, batch_size):
                end = min(start + batch_size, sample_count)
                logits = model(
                    x_values_t[start:end],
                    x_time_t[start:end],
                    x_future_t[start:end],
                )
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                origins = y_origin[start:end]
                for row_idx, origin_idx in enumerate(origins):
                    probs_sum[int(origin_idx)] += probs[row_idx]
                    probs_count[int(origin_idx)] += 1.0

        for origin_idx in range(len(rooms)):
            if probs_count[origin_idx] > 0:
                probs_sum[origin_idx] /= probs_count[origin_idx]

        meta["enabled"] = True
        meta["epochs"] = epochs
        meta["device"] = str(device)
        return probs_sum, meta, model, device

    def _blend_probs(
        self,
        count_probs: np.ndarray,
        transformer_probs: np.ndarray | None,
        outgoing: Counter[str],
        rooms: list[str],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        room_to_idx = {room: idx for idx, room in enumerate(rooms)}
        blend = count_probs.copy()
        alpha_by_room: dict[str, float] = {}

        if transformer_probs is None:
            return blend, {"transformer_used": False, "alpha_by_room": alpha_by_room}

        for room in rooms:
            idx = room_to_idx[room]
            support = float(outgoing.get(room, 0))
            alpha = min(0.62, 0.18 + (0.44 * (support / (support + 75.0))))
            alpha_by_room[room] = alpha

            c = count_probs[idx]
            t = transformer_probs[idx]
            c_sum = float(c.sum())
            t_sum = float(t.sum())
            if c_sum <= 0 and t_sum <= 0:
                continue
            if c_sum <= 0:
                row = t
            elif t_sum <= 0:
                row = c
            else:
                # Las habitaciones con más evidencia histórica reciben mayor peso del transformador.
                row = ((1.0 - alpha) * c) + (alpha * t)
            row_sum = float(row.sum())
            if row_sum > 0:
                blend[idx] = row / row_sum

        return blend, {"transformer_used": True, "alpha_by_room": alpha_by_room}

    def predict_next_room_probs(
        self,
        history_events: list[EventRecord],
        future_timestamp: datetime,
    ) -> np.ndarray | None:
        if not self.ready or not self.transformer_model or not HF_AVAILABLE:
            return None
        if len(history_events) < self.transformer_context_length:
            return None

        rooms = self.rooms
        n_features = len(rooms) + 11
        values = np.zeros((self.transformer_context_length, n_features), dtype=np.float32)
        times = np.zeros((self.transformer_context_length, 4), dtype=np.float32)
        room_to_idx = self.room_to_idx
        recent = history_events[-self.transformer_context_length :]

        prev_ts: datetime | None = None
        for idx, event in enumerate(recent):
            vec = np.zeros((n_features,), dtype=np.float32)
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
            gap = 0.0 if prev_ts is None else max(0.0, (event.timestamp - prev_ts).total_seconds())
            vec[len(rooms) + 10] = min(1.0, gap / 900.0)
            prev_ts = event.timestamp
            values[idx] = vec
            times[idx] = time_features_from_dt(event.timestamp)

        future_time = time_features_from_dt(future_timestamp)
        self.transformer_model.eval()
        with torch.no_grad():
            x_values_t = torch.tensor(values[np.newaxis, ...], dtype=torch.float32, device=self.transformer_device)
            x_time_t = torch.tensor(times[np.newaxis, ...], dtype=torch.float32, device=self.transformer_device)
            x_future_t = torch.tensor(
                future_time[np.newaxis, np.newaxis, ...],
                dtype=torch.float32,
                device=self.transformer_device,
            )
            logits = self.transformer_model(x_values_t, x_time_t, x_future_t)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        return probs.astype(np.float32)
