"""Normalización temporal y etiquetado supervisado de historiales CSV."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ..domain import (
    EventRecord,
    classify_sensor_type,
    classify_state_bucket,
    infer_room_from_entity,
    is_activation,
    normalize_room_name,
    parse_iso_datetime,
)
from .manifest import TrainingManifestStore


SPLIT_RATIOS = (0.70, 0.15, 0.15)
FILTER_FEATURE_SIZE = 12


@dataclass(slots=True)
class LabeledSequence:
    period_id: str
    split: str
    timestamp: datetime
    room: str
    context: list[EventRecord]
    human_label: float
    pet_label: float
    sample_weight: float
    label_kind: str


@dataclass(slots=True)
class PreparedSupervisedDataset:
    manifest_id: str
    fingerprint: str
    rooms: list[str]
    samples: dict[str, list[LabeledSequence]]
    periods: list[dict[str, Any]]
    files: list[dict[str, Any]]
    totals: dict[str, Any]
    context_length: int
    label_window: dict[str, int]
    weak_negative_weight: float

    def summary(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "fingerprint": self.fingerprint,
            "rooms": self.rooms,
            "periods": self.periods,
            "files": self.files,
            "totals": self.totals,
            "context_length": self.context_length,
            "label_window": self.label_window,
            "weak_negative_weight": self.weak_negative_weight,
        }


@dataclass(slots=True)
class _PeriodData:
    period_id: str
    signal_events: list[EventRecord] = field(default_factory=list)
    confirmations: dict[str, dict[str, list[tuple[datetime, datetime]]]] = field(
        default_factory=dict
    )
    unavailable: dict[str, list[tuple[datetime, datetime]]] = field(
        default_factory=dict
    )
    files: list[dict[str, Any]] = field(default_factory=list)


def _state_interval_rows(
    rows: list[dict[str, Any]],
    entity_id: str,
) -> tuple[
    list[tuple[datetime, datetime]],
    list[tuple[datetime, datetime]],
]:
    active_since: datetime | None = None
    unavailable_since: datetime | None = None
    intervals: list[tuple[datetime, datetime]] = []
    unavailable: list[tuple[datetime, datetime]] = []
    last_timestamp = rows[-1]["timestamp"] if rows else datetime.now(timezone.utc)
    for row in rows:
        if row["entity_id"] != entity_id:
            continue
        state = row["state"]
        if state == "unavailable":
            if active_since is not None:
                intervals.append((active_since, row["timestamp"]))
                active_since = None
            if unavailable_since is None:
                unavailable_since = row["timestamp"]
            continue
        if unavailable_since is not None:
            unavailable.append((unavailable_since, row["timestamp"]))
            unavailable_since = None
        if state == "on" and active_since is None:
            active_since = row["timestamp"]
        elif state != "on" and active_since is not None:
            intervals.append((active_since, row["timestamp"]))
            active_since = None
    if active_since is not None:
        intervals.append((active_since, last_timestamp))
    if unavailable_since is not None:
        unavailable.append((unavailable_since, last_timestamp))
    return intervals, unavailable


def _overlaps(
    intervals: list[tuple[datetime, datetime]],
    start: datetime,
    end: datetime,
) -> bool:
    return any(interval_start <= end and interval_end >= start for interval_start, interval_end in intervals)


def _split_bounds(
    start: datetime,
    end: datetime,
) -> dict[str, tuple[datetime, datetime]]:
    duration = end - start
    train_end = start + duration * SPLIT_RATIOS[0]
    validation_end = train_end + duration * SPLIT_RATIOS[1]
    return {
        "train": (start, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, end + timedelta(microseconds=1)),
    }


def _normalized_event_room(
    entity_id: str,
    entity_rooms: dict[str, str],
    aliases: dict[str, str],
) -> str:
    room = entity_rooms.get(entity_id) or infer_room_from_entity(entity_id)
    room = aliases.get(room, room)
    return normalize_room_name(room)


def filter_feature_vector(
    event: EventRecord,
    candidate_room: str,
    previous_timestamp: datetime | None,
    adjacency: dict[str, list[str]],
) -> np.ndarray:
    """Codifica contexto relativo sin nombres absolutos de entidad o habitación."""

    values = np.zeros((FILTER_FEATURE_SIZE,), dtype=np.float32)
    sensor_slot = {
        "motion": 0,
        "door": 1,
        "occupancy": 2,
    }.get(event.sensor_type, 3)
    values[sensor_slot] = 1.0
    state_slot = {
        "active": 0,
        "inactive": 1,
        "unavailable": 2,
    }.get(classify_state_bucket(event.sensor_type, event.state), 3)
    values[4 + state_slot] = 1.0
    values[8] = 1.0 if event.room == candidate_room else 0.0
    values[9] = (
        1.0
        if event.room in set(adjacency.get(candidate_room, []))
        else 0.0
    )
    gap = (
        0.0
        if previous_timestamp is None
        else max(0.0, (event.timestamp - previous_timestamp).total_seconds())
    )
    values[10] = min(1.0, gap / 120.0)
    values[11] = 1.0 if is_activation(event.sensor_type, event.state) else 0.0
    return values


def filter_context_matrix(
    context: list[EventRecord],
    candidate_room: str,
    adjacency: dict[str, list[str]],
    context_length: int,
) -> np.ndarray:
    matrix = np.zeros((context_length, FILTER_FEATURE_SIZE), dtype=np.float32)
    selected = context[-context_length:]
    offset = context_length - len(selected)
    previous: datetime | None = None
    for index, event in enumerate(selected):
        matrix[offset + index] = filter_feature_vector(
            event,
            candidate_room,
            previous,
            adjacency,
        )
        previous = event.timestamp
    return matrix


class SupervisedDatasetBuilder:
    """Construye secuencias etiquetadas sin modificar los CSV originales."""

    def __init__(
        self,
        store: TrainingManifestStore,
        *,
        context_length: int = 28,
    ) -> None:
        self.store = store
        self.context_length = max(4, int(context_length))

    def build(self, manifest_id: str) -> PreparedSupervisedDataset:
        validation = self.store.validate(manifest_id)
        if not validation["valid"]:
            raise ValueError("; ".join(validation["errors"]))
        manifest = validation["manifest"]
        periods: list[dict[str, Any]] = []
        samples = {"train": [], "validation": [], "test": []}
        all_rooms: set[str] = set()
        digest = hashlib.sha256()
        for file_info in validation["files"]:
            digest.update(file_info["sha256"].encode("ascii"))
        digest.update(
            str(manifest["label_window"]).encode("utf-8")
        )
        digest.update(
            str(manifest["weak_negative_weight"]).encode("ascii")
        )

        for period_config in manifest["periods"]:
            period = self._load_period(period_config)
            if not period.signal_events:
                raise ValueError(
                    f"El período {period.period_id} no contiene señales"
                )
            start = period.signal_events[0].timestamp
            end = period.signal_events[-1].timestamp
            bounds = _split_bounds(start, end)
            period_counts = {
                "id": period.period_id,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "signals": len(period.signal_events),
                "samples": {"train": 0, "validation": 0, "test": 0},
                "labels": {
                    "person": 0,
                    "pet": 0,
                    "both": 0,
                    "weak_negative": 0,
                },
            }
            before = timedelta(
                seconds=manifest["label_window"]["before_seconds"]
            )
            after = timedelta(
                seconds=manifest["label_window"]["after_seconds"]
            )
            context: list[EventRecord] = []
            for event in period.signal_events:
                context.append(event)
                if len(context) < self.context_length:
                    continue
                confirmation = period.confirmations.get(event.room)
                if not confirmation:
                    continue
                person_intervals = confirmation.get("person", [])
                pet_intervals = confirmation.get("pet", [])
                split = next(
                    (
                        name
                        for name, (split_start, split_end) in bounds.items()
                        if split_start <= event.timestamp < split_end
                    ),
                    None,
                )
                if split is None:
                    continue
                split_start, split_end = bounds[split]
                label_start = event.timestamp - before
                label_end = event.timestamp + after
                context_start = context[-self.context_length].timestamp
                if context_start < split_start or label_start < split_start or label_end >= split_end:
                    continue
                if _overlaps(
                    period.unavailable.get(event.room, []),
                    label_start,
                    label_end,
                ):
                    continue
                person = _overlaps(person_intervals, label_start, label_end)
                pet = _overlaps(pet_intervals, label_start, label_end)
                label_kind = (
                    "both"
                    if person and pet
                    else "person"
                    if person
                    else "pet"
                    if pet
                    else "weak_negative"
                )
                weight = (
                    1.0
                    if person or pet
                    else manifest["weak_negative_weight"]
                )
                samples[split].append(
                    LabeledSequence(
                        period_id=period.period_id,
                        split=split,
                        timestamp=event.timestamp,
                        room=event.room,
                        context=list(context[-self.context_length:]),
                        human_label=1.0 if person else 0.0,
                        pet_label=1.0 if pet else 0.0,
                        sample_weight=weight,
                        label_kind=label_kind,
                    )
                )
                period_counts["samples"][split] += 1
                period_counts["labels"][label_kind] += 1
                all_rooms.update(item.room for item in context if item.room)
            periods.append(period_counts)

        totals = {
            "samples": {
                split: len(split_samples)
                for split, split_samples in samples.items()
            },
            "labels": {
                kind: sum(
                    1
                    for split_samples in samples.values()
                    for sample in split_samples
                    if sample.label_kind == kind
                )
                for kind in ("person", "pet", "both", "weak_negative")
            },
        }
        return PreparedSupervisedDataset(
            manifest_id=manifest["id"],
            fingerprint=digest.hexdigest(),
            rooms=sorted(all_rooms),
            samples=samples,
            periods=periods,
            files=validation["files"],
            totals=totals,
            context_length=self.context_length,
            label_window=manifest["label_window"],
            weak_negative_weight=manifest["weak_negative_weight"],
        )

    def _load_period(self, config: dict[str, Any]) -> _PeriodData:
        all_rows: dict[tuple[str, str, datetime], dict[str, Any]] = {}
        signal_paths = {
            self.store.dataset_path(value)
            for value in config["signal_files"]
        }
        label_paths = {
            self.store.dataset_path(value)
            for value in config["label_files"]
        }
        for path in sorted(signal_paths | label_paths):
            with path.open("r", encoding="utf-8", newline="") as file:
                for row in csv.DictReader(file):
                    entity_id = str(row.get("entity_id") or "").strip().lower()
                    state = str(row.get("state") or "").strip().lower()
                    raw_timestamp = str(row.get("last_changed") or "").strip()
                    if not entity_id or not raw_timestamp:
                        continue
                    try:
                        timestamp = parse_iso_datetime(raw_timestamp)
                    except Exception:
                        continue
                    all_rows[(entity_id, state, timestamp)] = {
                        "entity_id": entity_id,
                        "state": state,
                        "timestamp": timestamp,
                        "source_path": path,
                    }
        rows = sorted(
            all_rows.values(),
            key=lambda item: (
                item["timestamp"],
                item["entity_id"],
                item["state"],
            ),
        )
        exclusions = list(config.get("exclusions") or [])

        def excluded(row: dict[str, Any]) -> bool:
            for rule in exclusions:
                if str(rule.get("entity_id") or "").lower() != row["entity_id"]:
                    continue
                after = str(rule.get("valid_from") or "")
                before = str(rule.get("valid_until") or "")
                if after and row["timestamp"] < parse_iso_datetime(after):
                    return True
                if before and row["timestamp"] > parse_iso_datetime(before):
                    return True
            return False

        roles = config["entity_roles"]
        period = _PeriodData(period_id=config["id"])
        confirmations_by_entity: dict[
            str,
            tuple[
                list[tuple[datetime, datetime]],
                list[tuple[datetime, datetime]],
            ],
        ] = {}
        for entity_id, role in roles.items():
            if role not in {"person_confirmation", "pet_confirmation"}:
                continue
            confirmations_by_entity[entity_id] = _state_interval_rows(
                rows,
                entity_id,
            )
        for entity_id, (intervals, unavailable) in confirmations_by_entity.items():
            room = _normalized_event_room(
                entity_id,
                config["entity_rooms"],
                config["room_aliases"],
            )
            role = roles[entity_id]
            room_payload = period.confirmations.setdefault(
                room,
                {"person": [], "pet": []},
            )
            room_payload[
                "person" if role == "person_confirmation" else "pet"
            ].extend(intervals)
            period.unavailable.setdefault(room, []).extend(unavailable)

        for row in rows:
            if excluded(row):
                continue
            role = roles.get(row["entity_id"], "signal")
            if role != "signal" or row["source_path"] not in signal_paths:
                continue
            room = _normalized_event_room(
                row["entity_id"],
                config["entity_rooms"],
                config["room_aliases"],
            )
            period.signal_events.append(
                EventRecord(
                    timestamp=row["timestamp"],
                    entity_id=row["entity_id"],
                    state=row["state"],
                    sensor_type=classify_sensor_type(row["entity_id"]),
                    room=room,
                )
            )
        period.files = [
            {"path": str(path)}
            for path in sorted(signal_paths | label_paths)
        ]
        return period
