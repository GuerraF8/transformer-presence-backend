from __future__ import annotations

import asyncio
import csv
import json
import os
import random
import threading
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests

try:
    import torch
    import torch.nn as nn
    from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel

    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

try:
    from .domain import (
        SENSOR_RELIABILITY,
        TRANSFORMER_CONTEXT_LENGTH,
        TRANSFORMER_LAGS_SEQUENCE,
        TRANSFORMER_MIN_SAMPLES,
        TRANSFORMER_MODEL_CONTEXT_LENGTH,
        EventRecord,
        TrainModelRequest,
        TrainSimulatorPresenceRequest,
        build_layout_for_request,
        build_scenario_templates,
        classify_sensor_type,
        classify_state_bucket,
        edge_key,
        infer_room_from_entity,
        is_activation,
        normalize_adjacency_map,
        parse_iso_datetime,
        safe_quantile,
        shortest_path_rooms,
        time_features_from_dt,
    )
except ImportError:  # pragma: no cover - supports `uvicorn server:app` in Docker
    from domain import (
        SENSOR_RELIABILITY,
        TRANSFORMER_CONTEXT_LENGTH,
        TRANSFORMER_LAGS_SEQUENCE,
        TRANSFORMER_MIN_SAMPLES,
        TRANSFORMER_MODEL_CONTEXT_LENGTH,
        EventRecord,
        TrainModelRequest,
        TrainSimulatorPresenceRequest,
        build_layout_for_request,
        build_scenario_templates,
        classify_sensor_type,
        classify_state_bucket,
        edge_key,
        infer_room_from_entity,
        is_activation,
        normalize_adjacency_map,
        parse_iso_datetime,
        safe_quantile,
        shortest_path_rooms,
        time_features_from_dt,
    )

if HF_AVAILABLE:

    class NextRoomTransformer(nn.Module):
        def __init__(self, input_size: int, room_count: int) -> None:
            super().__init__()
            self.config = TimeSeriesTransformerConfig(
                prediction_length=1,
                context_length=TRANSFORMER_MODEL_CONTEXT_LENGTH,
                lags_sequence=TRANSFORMER_LAGS_SEQUENCE,
                input_size=input_size,
                num_time_features=4,
                d_model=40,
                encoder_layers=2,
                decoder_layers=1,
                dropout=0.1,
                attention_dropout=0.1,
            )
            self.model = TimeSeriesTransformerModel(self.config)
            self.head = nn.Linear(40, room_count)

        def forward(
            self,
            past_values: torch.Tensor,
            past_time_features: torch.Tensor,
            future_time_features: torch.Tensor,
        ) -> torch.Tensor:
            outputs = self.model(
                past_values=past_values,
                past_time_features=past_time_features,
                past_observed_mask=torch.ones_like(past_values),
                future_values=None,
                future_time_features=future_time_features,
            )
            hidden = outputs.encoder_last_hidden_state[:, -1, :]
            return self.head(hidden)


    class OccupancyTransformer(nn.Module):
        def __init__(self, input_size: int, room_count: int, count_classes: int) -> None:
            super().__init__()
            self.config = TimeSeriesTransformerConfig(
                prediction_length=1,
                context_length=TRANSFORMER_MODEL_CONTEXT_LENGTH,
                lags_sequence=TRANSFORMER_LAGS_SEQUENCE,
                input_size=input_size,
                num_time_features=4,
                d_model=48,
                encoder_layers=2,
                decoder_layers=1,
                dropout=0.12,
                attention_dropout=0.12,
            )
            self.model = TimeSeriesTransformerModel(self.config)
            self.room_head = nn.Linear(48, room_count)
            self.count_head = nn.Linear(48, count_classes)

        def forward(
            self,
            past_values: torch.Tensor,
            past_time_features: torch.Tensor,
            future_time_features: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            outputs = self.model(
                past_values=past_values,
                past_time_features=past_time_features,
                past_observed_mask=torch.ones_like(past_values),
                future_values=None,
                future_time_features=future_time_features,
            )
            hidden = outputs.encoder_last_hidden_state[:, -1, :]
            return self.room_head(hidden), self.count_head(hidden)


class AIAdjacencyModel:
    def __init__(self) -> None:
        self.ready = False
        self.rooms: list[str] = []
        self.room_to_idx: dict[str, int] = {}
        self.transition_matrix = np.zeros((0, 0), dtype=np.float32)
        self.adjacency_neighbors: dict[str, list[str]] = {}
        self.adjacency_edges: list[dict[str, Any]] = []
        self.training_info: dict[str, Any] = {}
        self.sensor_room_map: dict[str, str] = {}
        self.transformer_model: Any | None = None
        self.transformer_device: Any | None = None
        self.transformer_context_length = TRANSFORMER_CONTEXT_LENGTH
        self.occupancy_transformer_model: Any | None = None
        self.occupancy_transformer_device: Any | None = None
        self.occupancy_transformer_rooms: list[str] = []
        self.occupancy_transformer_info: dict[str, Any] = {}
        self.occupancy_transformer_count_classes = 0
        self.real_profile_info: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._train_lock = threading.Lock()

    def are_adjacent(self, a: str, b: str) -> bool:
        if not self.ready:
            return True
        if a == b:
            return True
        return b in self.adjacency_neighbors.get(a, [])

    def neighbors(self, room: str) -> list[str]:
        return self.adjacency_neighbors.get(room, [])

    def save_state(self, model_dir: str | Path) -> dict[str, Any]:
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)
        core_path = path / "model_state.json"
        transition_path = path / "transition_matrix.npy"
        transformer_path = path / "next_room_transformer.pt"
        occupancy_path = path / "occupancy_transformer.pt"

        payload = {
            "schema_version": 1,
            "ready": self.ready,
            "rooms": self.rooms,
            "adjacency_neighbors": self.adjacency_neighbors,
            "adjacency_edges": self.adjacency_edges,
            "training_info": self.training_info,
            "sensor_room_map": self.sensor_room_map,
            "transformer_context_length": self.transformer_context_length,
            "occupancy_transformer_rooms": self.occupancy_transformer_rooms,
            "occupancy_transformer_info": self.occupancy_transformer_info,
            "occupancy_transformer_count_classes": self.occupancy_transformer_count_classes,
            "real_profile_info": self.real_profile_info,
        }
        core_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(transition_path, self.transition_matrix)

        saved = {
            "core_path": str(core_path),
            "transition_path": str(transition_path),
            "transformer_path": None,
            "occupancy_transformer_path": None,
        }
        if HF_AVAILABLE and self.transformer_model is not None:
            torch.save({"state_dict": self.transformer_model.state_dict()}, transformer_path)
            saved["transformer_path"] = str(transformer_path)
        if HF_AVAILABLE and self.occupancy_transformer_model is not None:
            torch.save({"state_dict": self.occupancy_transformer_model.state_dict()}, occupancy_path)
            saved["occupancy_transformer_path"] = str(occupancy_path)
        return saved

    def load_state(self, model_dir: str | Path) -> dict[str, Any]:
        path = Path(model_dir)
        core_path = path / "model_state.json"
        transition_path = path / "transition_matrix.npy"
        transformer_path = path / "next_room_transformer.pt"
        occupancy_path = path / "occupancy_transformer.pt"
        if not core_path.exists():
            return {"loaded": False, "reason": "sin estado persistido"}

        payload = json.loads(core_path.read_text(encoding="utf-8"))
        self.ready = bool(payload.get("ready"))
        self.rooms = [str(room) for room in payload.get("rooms", [])]
        self.room_to_idx = {room: idx for idx, room in enumerate(self.rooms)}
        self.adjacency_neighbors = {
            str(room): [str(nb) for nb in neighbors]
            for room, neighbors in dict(payload.get("adjacency_neighbors", {})).items()
        }
        self.adjacency_edges = list(payload.get("adjacency_edges", []))
        self.training_info = dict(payload.get("training_info", {}))
        self.sensor_room_map = {
            str(entity_id): str(room)
            for entity_id, room in dict(payload.get("sensor_room_map", {})).items()
        }
        self.transformer_context_length = int(payload.get("transformer_context_length") or TRANSFORMER_CONTEXT_LENGTH)
        self.occupancy_transformer_rooms = [
            str(room) for room in payload.get("occupancy_transformer_rooms", [])
        ]
        self.occupancy_transformer_info = dict(payload.get("occupancy_transformer_info", {}))
        self.occupancy_transformer_count_classes = int(payload.get("occupancy_transformer_count_classes") or 0)
        self.real_profile_info = dict(payload.get("real_profile_info", {}))

        if transition_path.exists():
            self.transition_matrix = np.load(transition_path).astype(np.float32)
        elif self.rooms:
            self.transition_matrix = np.eye(len(self.rooms), dtype=np.float32)
        else:
            self.transition_matrix = np.zeros((0, 0), dtype=np.float32)

        loaded_models: list[str] = []
        if HF_AVAILABLE and transformer_path.exists() and self.rooms:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = NextRoomTransformer(input_size=len(self.rooms) + 11, room_count=len(self.rooms)).to(device)
            checkpoint = torch.load(transformer_path, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self.transformer_model = model
            self.transformer_device = device
            loaded_models.append("next_room_transformer")

        if HF_AVAILABLE and occupancy_path.exists() and self.occupancy_transformer_rooms:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            count_classes = max(1, self.occupancy_transformer_count_classes)
            rooms = self.occupancy_transformer_rooms
            model = OccupancyTransformer(
                input_size=len(rooms) + 11,
                room_count=len(rooms),
                count_classes=count_classes,
            ).to(device)
            checkpoint = torch.load(occupancy_path, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self.occupancy_transformer_model = model
            self.occupancy_transformer_device = device
            loaded_models.append("occupancy_transformer")

        return {
            "loaded": True,
            "rooms": len(self.rooms),
            "models": loaded_models,
            "core_path": str(core_path),
        }

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

        # Compatibilidad: si se desactiva el uso de todas las transiciones, reducimos a activaciones reales.
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
                # Mayor confianza del transformer para origenes con soporte alto.
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

    def _infer_graph(
        self,
        rooms: list[str],
        directed: Counter[tuple[str, str]],
        blended_probs: np.ndarray,
        degree_limit: int,
        reference_adjacency: dict[str, list[str]] | None = None,
    ) -> tuple[dict[str, list[str]], list[dict[str, Any]], dict[str, float]]:
        room_to_idx = {room: idx for idx, room in enumerate(rooms)}
        pair_candidates: list[dict[str, Any]] = []
        support_values: list[float] = []
        score_values: list[float] = []
        reference_penalized_pairs = 0
        reference_boosted_pairs = 0
        reference_vetoed_pairs = 0

        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                a = rooms[i]
                b = rooms[j]
                raw_support = float(directed.get((a, b), 0) + directed.get((b, a), 0))
                if raw_support <= 0:
                    continue
                raw_sym_score = float(blended_probs[i, j] + blended_probs[j, i])

                adjusted_support = raw_support
                adjusted_sym_score = raw_sym_score
                reference_path: list[str] = []
                reference_adjacent = None
                bridge_support = None
                penalty_factor = 0.0
                reference_veto = False

                if reference_adjacency:
                    reference_path = shortest_path_rooms(reference_adjacency, a, b)
                    if reference_path:
                        reference_adjacent = len(reference_path) == 2
                        if reference_adjacent:
                            adjusted_support *= 1.08
                            adjusted_sym_score *= 1.12
                            reference_boosted_pairs += 1
                        elif len(reference_path) > 2:
                            bridge_supports: list[float] = []
                            for src, dst in zip(reference_path[:-1], reference_path[1:]):
                                bridge_supports.append(
                                    float(directed.get((src, dst), 0) + directed.get((dst, src), 0))
                                )
                            if bridge_supports:
                                bridge_support = min(bridge_supports)
                            penalty_factor = min(0.82, 0.34 + (0.11 * (len(reference_path) - 2)))
                            if bridge_support is not None and bridge_support > 0:
                                ratio = raw_support / bridge_support
                                if ratio < 0.9:
                                    penalty_factor += 0.2
                                elif ratio < 1.15:
                                    penalty_factor += 0.1
                                # Si el mapa real ya explica el salto con aristas puente fuertes,
                                # descartamos la arista directa salvo evidencia muy superior.
                                if raw_support <= (bridge_support * 1.25):
                                    reference_veto = True
                            penalty_factor = min(0.88, penalty_factor)
                            adjusted_support *= max(0.18, 1.0 - (0.62 * penalty_factor))
                            adjusted_sym_score *= max(0.06, 1.0 - penalty_factor)
                            reference_penalized_pairs += 1
                            if reference_veto:
                                reference_vetoed_pairs += 1

                support_values.append(adjusted_support)
                score_values.append(adjusted_sym_score)
                pair_candidates.append(
                    {
                        "a": a,
                        "b": b,
                        "raw_support": raw_support,
                        "raw_score": raw_sym_score,
                        "support": adjusted_support,
                        "score": adjusted_sym_score,
                        "reference_adjacent": reference_adjacent,
                        "reference_path": reference_path,
                        "reference_bridge_support": bridge_support,
                        "reference_penalty_factor": round(penalty_factor, 4),
                        "reference_veto": reference_veto,
                    }
                )

        support_thr = max(2.0, safe_quantile(support_values, 0.35, 2.0))
        score_thr = max(0.08, safe_quantile(score_values, 0.40, 0.08))

        filtered: list[dict[str, Any]] = []
        for candidate in pair_candidates:
            if bool(candidate.get("reference_veto")):
                continue
            if float(candidate["support"]) < support_thr:
                continue
            if float(candidate["score"]) < score_thr:
                continue
            filtered.append(candidate)

        filtered.sort(key=lambda item: (float(item["support"]), float(item["score"])), reverse=True)

        neighbors: dict[str, set[str]] = {room: set() for room in rooms}
        degree = {room: 0 for room in rooms}

        for candidate in filtered:
            a = str(candidate["a"])
            b = str(candidate["b"])
            if degree[a] >= degree_limit or degree[b] >= degree_limit:
                continue
            neighbors[a].add(b)
            neighbors[b].add(a)
            degree[a] += 1
            degree[b] += 1

        # Garantiza conectividad minima sin hardcode: cada nodo queda unido al mejor vecino por evidencia.
        for room in rooms:
            if neighbors[room]:
                continue
            i = room_to_idx[room]
            best_other = None
            best_value = -1.0
            for other in rooms:
                if other == room:
                    continue
                j = room_to_idx[other]
                support = float(directed.get((room, other), 0) + directed.get((other, room), 0))
                value = support + (8.0 * float(blended_probs[i, j] + blended_probs[j, i]))
                if reference_adjacency and other in reference_adjacency.get(room, []):
                    value *= 1.18
                if value > best_value:
                    best_value = value
                    best_other = other
            if best_other is not None:
                neighbors[room].add(best_other)
                neighbors[best_other].add(room)

        edge_list: list[dict[str, Any]] = []
        seen = set()
        for a in rooms:
            for b in neighbors[a]:
                k = edge_key(a, b)
                if k in seen:
                    continue
                seen.add(k)
                raw_support = int(directed.get((a, b), 0) + directed.get((b, a), 0))
                raw_sym_score = float(
                    blended_probs[room_to_idx[a], room_to_idx[b]] + blended_probs[room_to_idx[b], room_to_idx[a]]
                )
                ref_path = shortest_path_rooms(reference_adjacency or {}, a, b) if reference_adjacency else []
                ref_adjacent = len(ref_path) == 2 if ref_path else None
                bridge_support = None
                if reference_adjacency and ref_path and len(ref_path) > 2:
                    bridge_support = min(
                        int(directed.get((src, dst), 0) + directed.get((dst, src), 0))
                        for src, dst in zip(ref_path[:-1], ref_path[1:])
                    )
                edge_list.append(
                    {
                        "a": a,
                        "b": b,
                        "support": raw_support,
                        "score": round(raw_sym_score, 4),
                        "reference_adjacent": ref_adjacent,
                        "reference_path": ref_path,
                        "reference_bridge_support": bridge_support,
                    }
                )

        edge_list.sort(key=lambda item: (item["support"], item["score"]), reverse=True)
        neighbors_sorted = {room: sorted(list(v)) for room, v in neighbors.items()}
        thresholds = {
            "support_threshold": support_thr,
            "score_threshold": score_thr,
            "reference_penalized_pairs": reference_penalized_pairs,
            "reference_boosted_pairs": reference_boosted_pairs,
            "reference_vetoed_pairs": reference_vetoed_pairs,
        }
        return neighbors_sorted, edge_list, thresholds

    def _validate_edges_with_ollama(
        self,
        edges: list[dict[str, Any]],
        rooms: list[str],
        ollama_url: str,
        ollama_model: str,
    ) -> dict[str, Any] | None:
        payload = {
            "rooms": rooms,
            "edges": edges,
            "instruction": (
                "Evalua si las adyacencias son consistentes para un hogar real. "
                "Devuelve JSON con llaves quality_score (0-1), suspicious_edges (lista de pares), notes (lista)."
            ),
        }
        body = {
            "model": ollama_model,
            "prompt": json.dumps(payload, ensure_ascii=False),
            "stream": False,
            "format": "json",
        }

        try:
            response = requests.post(
                f"{ollama_url.rstrip('/')}/api/generate",
                json=body,
                timeout=(5, 90),
            )
            response.raise_for_status()
            raw = str(response.json().get("response") or "").strip()
            if not raw:
                return None
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
        return None

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
