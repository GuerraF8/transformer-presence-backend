"""Entrenamiento y predicción de ocupación por habitación."""

from .dependencies import *  # noqa: F401,F403


class OccupancyMixin:
    def _prepare_occupancy_transformer_dataset(
        self,
        labeled_events: list[tuple[EventRecord, set[str]]],
        rooms: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        history_len = self.transformer_context_length
        if len(labeled_events) <= history_len + 10:
            return None

        room_to_idx = {room: idx for idx, room in enumerate(rooms)}
        raw_values: list[np.ndarray] = []
        raw_time: list[np.ndarray] = []
        prev_ts: datetime | None = None
        for event, _labels in labeled_events:
            raw_values.append(self._event_feature_vector(event, rooms, room_to_idx, prev_ts))
            raw_time.append(time_features_from_dt(event.timestamp))
            prev_ts = event.timestamp

        x_values: list[np.ndarray] = []
        x_time: list[np.ndarray] = []
        x_future: list[np.ndarray] = []
        y_rooms: list[np.ndarray] = []
        y_count: list[int] = []

        for idx in range(history_len, len(labeled_events)):
            _event, occupied = labeled_events[idx]
            label = np.zeros((len(rooms),), dtype=np.float32)
            for room in occupied:
                room_idx = room_to_idx.get(room)
                if room_idx is not None:
                    label[room_idx] = 1.0
            x_values.append(np.stack(raw_values[idx - history_len : idx], axis=0))
            x_time.append(np.stack(raw_time[idx - history_len : idx], axis=0))
            x_future.append(raw_time[idx])
            y_rooms.append(label)
            y_count.append(int(min(len(occupied), max(0, len(rooms)))))

        if len(x_values) < TRANSFORMER_MIN_SAMPLES:
            return None

        return (
            np.asarray(x_values, dtype=np.float32),
            np.asarray(x_time, dtype=np.float32),
            np.asarray(x_future, dtype=np.float32),
            np.asarray(y_rooms, dtype=np.float32),
            np.asarray(y_count, dtype=np.int64),
        )

    def train_occupancy_from_simulator(
        self,
        req: TrainSimulatorPresenceRequest,
        reference_layout: dict[str, list[str]] | None,
    ) -> dict[str, Any]:
        with self._train_lock:
            meta: dict[str, Any] = {
                "enabled": False,
                "reason": "",
                "samples": 0,
                "epochs": 0,
                "context_length": self.transformer_context_length,
                "objective": "multi_label_room_occupancy_and_people_count",
            }
            if not HF_AVAILABLE:
                meta["reason"] = "torch/transformers no disponible"
                self.occupancy_transformer_info = meta
                return {"status": "disabled", "training_info": meta}

            real_events: list[EventRecord] = []
            real_profile: dict[str, Any] | None = None
            self.real_profile_info = {"enabled": False}
            if req.use_real_profile and req.real_profile_csv_path:
                try:
                    real_events = self._read_history_events(
                        req.real_profile_csv_path,
                        debounce_seconds=1,
                        include_all_state_transitions=True,
                    )
                except Exception as exc:
                    meta["real_profile_warning"] = f"no se pudo leer CSV real: {exc}"

            base_rooms = sorted(normalize_adjacency_map({room: [] for room in req.rooms}).keys())
            if not base_rooms and reference_layout:
                base_rooms = sorted(normalize_adjacency_map(reference_layout).keys())
            if not base_rooms and real_events:
                base_rooms = sorted({event.room for event in real_events if event.room})

            profile_layout = build_layout_for_request(base_rooms, req.template, req.layout_edges) if base_rooms else {}
            if real_events and base_rooms:
                real_profile = self._build_real_profile(
                    real_events,
                    sorted(profile_layout.keys() or base_rooms),
                    profile_layout,
                    req.real_profile_max_events,
                )
                if real_profile:
                    self.real_profile_info = {
                        "csv_path": req.real_profile_csv_path,
                        **real_profile,
                    }
                else:
                    self.real_profile_info = {
                        "enabled": False,
                        "csv_path": req.real_profile_csv_path,
                        "reason": "muestras reales insuficientes para perfil",
                    }

            labeled_events, rooms, layout = self._generate_simulated_presence_events(req, reference_layout, real_profile)
            dataset = self._prepare_occupancy_transformer_dataset(labeled_events, rooms)
            if dataset is None:
                meta["reason"] = "muestras sinteticas insuficientes"
                self.occupancy_transformer_info = meta
                return {"status": "disabled", "rooms": rooms, "training_info": meta}

            weak_dataset = None
            weak_labeled_events: list[tuple[EventRecord, set[str]]] = []
            if req.weak_real_pretrain and real_events:
                weak_labeled_events = self._weak_labeled_events_from_history(
                    real_events,
                    rooms,
                    req.weak_presence_hold_seconds,
                    req.real_profile_max_events,
                )
                weak_dataset = self._prepare_occupancy_transformer_dataset(weak_labeled_events, rooms)

            x_values, x_time, x_future, y_rooms, y_count = dataset
            sample_count = int(x_values.shape[0])
            meta["samples"] = sample_count
            if sample_count > req.max_samples:
                idx = np.linspace(0, sample_count - 1, req.max_samples, dtype=int)
                x_values = x_values[idx]
                x_time = x_time[idx]
                x_future = x_future[idx]
                y_rooms = y_rooms[idx]
                y_count = y_count[idx]
                sample_count = int(x_values.shape[0])
                meta["samples"] = sample_count

            count_classes = int(req.max_people) + 1
            y_count = np.clip(y_count, 0, count_classes - 1)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = OccupancyTransformer(
                input_size=int(x_values.shape[2]),
                room_count=len(rooms),
                count_classes=count_classes,
            ).to(device)

            x_values_t = torch.tensor(x_values, dtype=torch.float32, device=device)
            x_time_t = torch.tensor(x_time, dtype=torch.float32, device=device)
            x_future_t = torch.tensor(x_future, dtype=torch.float32, device=device).unsqueeze(1)
            y_rooms_t = torch.tensor(y_rooms, dtype=torch.float32, device=device)
            y_count_t = torch.tensor(y_count, dtype=torch.long, device=device)

            room_loss = nn.BCEWithLogitsLoss()
            count_loss = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            batch_size = 256

            def run_batches(
                values_t: Any,
                time_t: Any,
                future_t: Any,
                rooms_t: Any,
                count_t: Any,
                samples: int,
                epochs: int,
                count_weight: float,
            ) -> float:
                last = 0.0
                for _epoch in range(max(1, epochs)):
                    perm = torch.randperm(samples, device=device)
                    for start in range(0, samples, batch_size):
                        batch_idx = perm[start : start + batch_size]
                        room_logits, count_logits = model(
                            values_t[batch_idx],
                            time_t[batch_idx],
                            future_t[batch_idx],
                        )
                        loss = room_loss(room_logits, rooms_t[batch_idx]) + (
                            count_weight * count_loss(count_logits, count_t[batch_idx])
                        )
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        last = float(loss.detach().cpu().item())
                return last

            model.train()
            last_loss = 0.0
            weak_pretrain_meta: dict[str, Any] = {
                "enabled": False,
                "samples": 0,
                "epochs": 0,
            }
            if weak_dataset is not None:
                weak_values, weak_time, weak_future, weak_rooms, weak_count = weak_dataset
                weak_sample_count = int(weak_values.shape[0])
                if weak_sample_count > req.max_samples:
                    weak_idx = np.linspace(0, weak_sample_count - 1, req.max_samples, dtype=int)
                    weak_values = weak_values[weak_idx]
                    weak_time = weak_time[weak_idx]
                    weak_future = weak_future[weak_idx]
                    weak_rooms = weak_rooms[weak_idx]
                    weak_count = weak_count[weak_idx]
                    weak_sample_count = int(weak_values.shape[0])
                weak_count = np.clip(weak_count, 0, count_classes - 1)
                weak_values_t = torch.tensor(weak_values, dtype=torch.float32, device=device)
                weak_time_t = torch.tensor(weak_time, dtype=torch.float32, device=device)
                weak_future_t = torch.tensor(weak_future, dtype=torch.float32, device=device).unsqueeze(1)
                weak_rooms_t = torch.tensor(weak_rooms, dtype=torch.float32, device=device)
                weak_count_t = torch.tensor(weak_count, dtype=torch.long, device=device)
                pretrain_epochs = max(1, min(3, req.epochs // 2))
                last_loss = run_batches(
                    weak_values_t,
                    weak_time_t,
                    weak_future_t,
                    weak_rooms_t,
                    weak_count_t,
                    weak_sample_count,
                    pretrain_epochs,
                    0.35,
                )
                weak_pretrain_meta = {
                    "enabled": True,
                    "samples": weak_sample_count,
                    "epochs": pretrain_epochs,
                    "label_strategy": "active_rooms_with_presence_hold",
                    "presence_hold_seconds": req.weak_presence_hold_seconds,
                }

            last_loss = run_batches(
                x_values_t,
                x_time_t,
                x_future_t,
                y_rooms_t,
                y_count_t,
                sample_count,
                req.epochs,
                0.65,
            )

            exact_matches = 0
            count_matches = 0
            evaluated = 0
            model.eval()
            with torch.no_grad():
                for start in range(0, sample_count, batch_size):
                    end = min(start + batch_size, sample_count)
                    room_logits, count_logits = model(x_values_t[start:end], x_time_t[start:end], x_future_t[start:end])
                    room_pred = (torch.sigmoid(room_logits) >= 0.5).float()
                    count_pred = torch.argmax(count_logits, dim=1)
                    exact_matches += int(torch.all(room_pred == y_rooms_t[start:end], dim=1).sum().detach().cpu().item())
                    count_matches += int((count_pred == y_count_t[start:end]).sum().detach().cpu().item())
                    evaluated += int(end - start)

            self.occupancy_transformer_model = model
            self.occupancy_transformer_device = device
            self.occupancy_transformer_rooms = rooms
            self.occupancy_transformer_count_classes = count_classes
            meta.update(
                {
                    "enabled": True,
                    "reason": "",
                    "epochs": req.epochs,
                    "device": str(device),
                    "rooms_total": len(rooms),
                    "synthetic_events": len(labeled_events),
                    "scenarios": req.scenarios,
                    "steps_per_scenario": req.steps_per_scenario,
                    "max_people": req.max_people,
                    "real_profile": self.real_profile_info if real_profile else {"enabled": False},
                    "weak_real_pretrain": weak_pretrain_meta,
                    "domain_randomization": {
                        "enabled": req.domain_randomization,
                        "false_positive_rate": req.false_positive_rate,
                        "false_negative_rate": req.false_negative_rate,
                    },
                    "loss": round(last_loss, 6),
                    "room_exact_match_rate": round(exact_matches / evaluated, 4) if evaluated else None,
                    "count_accuracy": round(count_matches / evaluated, 4) if evaluated else None,
                }
            )
            self.occupancy_transformer_info = meta

            if not self.rooms:
                self.rooms = rooms
                self.room_to_idx = {room: idx for idx, room in enumerate(rooms)}

            return {
                "status": "ok",
                "rooms": rooms,
                "layout": layout,
                "training_info": meta,
            }

    def predict_occupancy_state(
        self,
        history_events: list[EventRecord],
        future_timestamp: datetime,
    ) -> dict[str, Any] | None:
        from ..relative_occupancy import relative_occupancy_prediction

        relative = relative_occupancy_prediction(
            self,
            history_events,
            sorted(self.adjacency_neighbors),
            self.adjacency_neighbors,
        )
        if relative is not None:
            return relative
        if not self.occupancy_transformer_model or not HF_AVAILABLE:
            return None
        if len(history_events) < self.transformer_context_length:
            return None
        rooms = self.occupancy_transformer_rooms
        if not rooms:
            return None

        room_to_idx = {room: idx for idx, room in enumerate(rooms)}
        values = np.zeros((self.transformer_context_length, len(rooms) + 11), dtype=np.float32)
        times = np.zeros((self.transformer_context_length, 4), dtype=np.float32)
        recent = history_events[-self.transformer_context_length :]
        prev_ts: datetime | None = None
        for idx, event in enumerate(recent):
            values[idx] = self._event_feature_vector(event, rooms, room_to_idx, prev_ts)
            times[idx] = time_features_from_dt(event.timestamp)
            prev_ts = event.timestamp

        future_time = time_features_from_dt(future_timestamp)
        self.occupancy_transformer_model.eval()
        with torch.no_grad():
            x_values_t = torch.tensor(values[np.newaxis, ...], dtype=torch.float32, device=self.occupancy_transformer_device)
            x_time_t = torch.tensor(times[np.newaxis, ...], dtype=torch.float32, device=self.occupancy_transformer_device)
            x_future_t = torch.tensor(
                future_time[np.newaxis, np.newaxis, ...],
                dtype=torch.float32,
                device=self.occupancy_transformer_device,
            )
            room_logits, count_logits = self.occupancy_transformer_model(x_values_t, x_time_t, x_future_t)
            room_probs = torch.sigmoid(room_logits).detach().cpu().numpy()[0]
            count_probs = torch.softmax(count_logits, dim=1).detach().cpu().numpy()[0]

        predicted_count = int(np.argmax(count_probs))
        order = list(np.argsort(-room_probs))
        selected: list[str] = []
        if predicted_count > 0:
            for idx in order[:predicted_count]:
                if float(room_probs[int(idx)]) >= 0.25:
                    selected.append(rooms[int(idx)])
        if not selected:
            selected = [rooms[int(idx)] for idx in order if float(room_probs[int(idx)]) >= 0.55]

        return {
            "rooms": selected,
            "people_count": predicted_count,
            "confidence": round(float(np.max(count_probs)), 4),
            "room_probs": {
                room: round(float(room_probs[idx]), 4)
                for idx, room in enumerate(rooms)
            },
            "count_probs": {
                str(idx): round(float(value), 4)
                for idx, value in enumerate(count_probs)
            },
        }
