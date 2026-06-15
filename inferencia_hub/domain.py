from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

ACTIVE_STATES_BY_TYPE = {
    "motion": {"on", "detected", "motion", "active", "true"},
    "door": {"on", "open", "opened", "opening", "unlocked", "true"},
    "occupancy": {"on", "occupied", "home", "present", "true"},
    "other": {"on", "true"},
}

SENSOR_RELIABILITY = {
    "motion": 0.64,
    "door": 0.48,
    "occupancy": 0.9,
    "other": 0.42,
}

STOPWORDS = {
    "binary",
    "sensor",
    "input",
    "boolean",
    "switch",
    "status",
    "state",
    "motion",
    "pir",
    "movement",
    "presence",
    "occupied",
    "occupancy",
    "contact",
    "door",
    "window",
    "detector",
    "trigger",
}

MOTION_KEYWORDS = {"motion", "pir", "movement", "presence", "detector"}
DOOR_KEYWORDS = {"door", "contact", "window", "gate", "entrance", "entry"}
OCCUPANCY_KEYWORDS = {"occupied", "occupancy", "home", "away", "present"}

# Alias minimos para mantener consistencia de nombres de habitaciones inferidas.
ROOM_ALIASES = {
    "study": "sittingroom",
    "tvroom": "entertainment_room",
    "tv_room": "entertainment_room",
}

REAL_HOME_LAYOUT_EDGES = [
    ("bedroom", "sittingroom"),
    ("sittingroom", "entertainment_room"),
    ("entertainment_room", "foyer"),
    ("foyer", "kitchen"),
    ("foyer", "living"),
]


def parse_iso_datetime(raw_value: str) -> datetime:
    value = raw_value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_utc_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def tokenize(text: str) -> list[str]:
    clean = text.replace(".", "_").replace("-", "_").lower()
    return [token for token in clean.split("_") if token]


def classify_sensor_type(entity_id: str) -> str:
    tokens = set(tokenize(entity_id))
    if tokens & DOOR_KEYWORDS:
        return "door"
    if tokens & OCCUPANCY_KEYWORDS:
        return "occupancy"
    if tokens & MOTION_KEYWORDS:
        return "motion"
    return "other"


def infer_room_from_entity(entity_id: str) -> str:
    base = entity_id.split(".", 1)[1] if "." in entity_id else entity_id
    tokens = tokenize(base)
    useful = [token for token in tokens if token not in STOPWORDS and len(token) > 1]
    room = "_".join(useful) if useful else base.lower().replace(".", "_")
    if room in ROOM_ALIASES:
        return ROOM_ALIASES[room]
    condensed = room.replace("_", "")
    return ROOM_ALIASES.get(condensed, room)


def is_activation(sensor_type: str, state: str) -> bool:
    return state.lower().strip() in ACTIVE_STATES_BY_TYPE.get(
        sensor_type, ACTIVE_STATES_BY_TYPE["other"]
    )


def classify_state_bucket(sensor_type: str, state: str) -> str:
    normalized = state.lower().strip()
    if is_activation(sensor_type, normalized):
        return "active"
    if normalized in {"off", "closed", "inactive", "idle", "false", "clear"}:
        return "inactive"
    if normalized in {"unavailable", "unknown", "none"}:
        return "unavailable"
    return "other"


def edge_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def safe_quantile(values: list[float], q: float, default: float) -> float:
    if not values:
        return float(default)
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return float(default)
    return float(np.quantile(arr, q))


def time_features_from_dt(ts: datetime) -> np.ndarray:
    hour = ts.hour + (ts.minute / 60.0) + (ts.second / 3600.0)
    hour_rad = (2.0 * math.pi * hour) / 24.0
    dow = ts.weekday() + (hour / 24.0)
    dow_rad = (2.0 * math.pi * dow) / 7.0
    return np.asarray(
        [
            math.sin(hour_rad),
            math.cos(hour_rad),
            math.sin(dow_rad),
            math.cos(dow_rad),
        ],
        dtype=np.float32,
    )


@dataclass
class LastActivation:
    room: str
    timestamp: datetime


@dataclass
class EventRecord:
    timestamp: datetime
    entity_id: str
    state: str
    sensor_type: str
    room: str


TRANSFORMER_MODEL_CONTEXT_LENGTH = 24
TRANSFORMER_LAGS_SEQUENCE = [1, 2, 3, 4]
TRANSFORMER_CONTEXT_LENGTH = TRANSFORMER_MODEL_CONTEXT_LENGTH + max(TRANSFORMER_LAGS_SEQUENCE)
TRANSFORMER_MIN_SAMPLES = TRANSFORMER_CONTEXT_LENGTH + 24


class SensorEventInput(BaseModel):
    entity_id: str
    state: str = "on"
    sensor_type: str | None = None
    room: str | None = None
    timestamp: datetime | None = None
    source: str = "ha"


class HAEntityInfo(BaseModel):
    entity_id: str
    name: str = ""
    domain: str = ""
    state: str = ""
    sensor_type: str = "other"
    room: str = ""
    device_class: str = ""
    source: str = "ha_scan"
    supported: bool = True
    last_changed: str | None = None
    area_id: str = ""
    area_name: str = ""
    area_source: str = ""
    device_id: str = ""
    unique_id: str = ""
    platform: str = ""


class HAAreaInfo(BaseModel):
    area_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    floor_id: str | None = None
    entity_ids: list[str] = Field(default_factory=list)


class HAEntityCatalogInput(BaseModel):
    source: str = "home_assistant"
    entry_id: str = ""
    scanned_at: datetime | None = None
    auto_discovery: bool = True
    tracked_entities: list[str] = Field(default_factory=list)
    areas: list[HAAreaInfo] = Field(default_factory=list)
    entities: list[HAEntityInfo] = Field(default_factory=list)


class HAActionRequestInput(BaseModel):
    action: str = Field(
        pattern=(
            "^(refresh_catalog|create_test_sensors|"
            "remove_test_sensors|remove_test_resources)$"
        )
    )
    entry_id: str | None = None
    rooms: str = "bedroom,kitchen,living"
    include_occupancy: bool = True
    initial_state: str = Field(default="off", pattern="^(on|off)$")


class RealSensorAssignmentInput(BaseModel):
    entity_id: str
    room: str
    enabled: bool = True
    sensor_type: str = Field(default="auto", pattern="^(auto|motion|door|occupancy|other)$")


class RealSensorConfigInput(BaseModel):
    rooms: list[str] = Field(default_factory=list)
    assignments: list[RealSensorAssignmentInput] = Field(default_factory=list)
    require_explicit_selection: bool = True


class ProfileRoomInput(BaseModel):
    slug: str
    name: str
    area_id: str = ""
    area_name: str = ""
    status: Literal["active", "missing"] = "active"
    warning: str = ""


class ProfileAreaInput(BaseModel):
    area_id: str
    room_slug: str
    name: str = ""


class ProfileAssignmentInput(BaseModel):
    entity_id: str
    room_slug: str
    enabled: bool = True
    sensor_type: Literal["motion", "door", "occupancy", "other"] = "other"
    training_role: Literal[
        "signal",
        "person_confirmation",
        "pet_confirmation",
    ] = "signal"
    area_id: str = ""
    area_name: str = ""
    status: Literal["active", "missing", "moved"] = "active"
    warning: str = ""
    unique_id: str = ""
    platform: str = ""


class ProfileCreateInput(BaseModel):
    name: str
    source: Literal["manual", "real_home", "detected"] = "manual"
    area_ids: list[str] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)


class ProfileUpdateInput(BaseModel):
    revision: int = Field(ge=1)
    name: str
    rooms: list[ProfileRoomInput] = Field(default_factory=list)
    areas: list[ProfileAreaInput] = Field(default_factory=list)
    assignments: list[ProfileAssignmentInput] = Field(default_factory=list)
    edges: list[list[str]] = Field(default_factory=list)


class ProfileInferLayoutInput(BaseModel):
    min_support: int = Field(default=2, ge=1, le=1000)
    max_gap_seconds: int = Field(default=600, ge=1, le=86400)


class PresenceFilterConfigInput(BaseModel):
    enabled: bool = True
    window_seconds: int = Field(default=20, ge=1, le=600)
    min_motion_events: int = Field(default=2, ge=1, le=20)
    min_distinct_rooms: int = Field(default=1, ge=1, le=20)


class LiveTrainingConfigInput(BaseModel):
    enabled: bool = True
    minimum_confirmations: int = Field(default=500, ge=20, le=100000)
    minimum_person_confirmations: int = Field(default=100, ge=1, le=50000)
    minimum_pet_confirmations: int = Field(default=100, ge=1, le=50000)
    minimum_days_between_activations: int = Field(default=7, ge=1, le=365)


class HistoryConfigInput(BaseModel):
    enabled: bool = True
    retention_days: int = Field(default=365, ge=1, le=3650)
    persisted_modes: list[Literal["listen", "replay", "simulator"]] = Field(
        default_factory=lambda: ["listen", "replay", "simulator"]
    )


class HistoryPurgeInput(BaseModel):
    confirmation: str = ""


class CsvReplayRequest(BaseModel):
    csv_path: str = Field(default="/data/history-1mes.csv")
    speed_events_per_second: float = Field(default=8.0, gt=0.0, le=200.0)
    debounce_seconds: int = Field(default=2, ge=0, le=120)
    include_all_state_transitions: bool = Field(default=True)
    max_events: int = Field(default=0, ge=0)
    use_scenario_layout: bool = Field(default=False)
    template: str = Field(default="real_home")
    layout_edges: list[list[str]] = Field(default_factory=list)
    room_mapping: dict[str, str] = Field(default_factory=dict)
    step_seconds: int = Field(default=4, ge=1, le=120)


class TrainModelRequest(BaseModel):
    csv_path: str = Field(default="/data/history-1mes.csv")
    debounce_seconds: int = Field(default=2, ge=0, le=120)
    include_all_state_transitions: bool = Field(default=True)
    min_gap_seconds: int = Field(default=2, ge=0, le=600)
    max_gap_seconds: int = Field(default=600, ge=10, le=86400)
    epochs: int = Field(default=3, ge=1, le=20)
    max_samples: int = Field(default=7000, ge=200, le=40000)
    degree_limit: int = Field(default=3, ge=1, le=8)
    use_ollama_validation: bool = Field(default=True)
    ollama_url: str = Field(default="http://127.0.0.1:11434")
    ollama_model: str = Field(default="qwen2.5:0.5b-instruct")


class TrainModelFullRequest(BaseModel):
    """Entrenamiento completo del historial con parámetros optimizados para máxima captura."""
    csv_path: str = Field(default="/data/history-1mes.csv")
    debounce_seconds: int = Field(default=1, ge=0, le=120)
    include_all_state_transitions: bool = Field(default=True)
    min_gap_seconds: int = Field(default=0, ge=0, le=600)
    max_gap_seconds: int = Field(default=900, ge=10, le=86400)
    epochs: int = Field(default=5, ge=1, le=30)
    max_samples: int = Field(default=15000, ge=500, le=50000)
    degree_limit: int = Field(default=4, ge=1, le=10)
    use_ollama_validation: bool = Field(default=False)
    skip_processed: bool = Field(default=True)


class TrainSimulatorPresenceRequest(BaseModel):
    """Entrena ocupacion multi-habitacion con escenarios sinteticos del simulador."""
    rooms: list[str] = Field(default_factory=list)
    template: str = Field(default="real_home")
    layout_edges: list[list[str]] = Field(default_factory=list)
    scenarios: int = Field(default=240, ge=20, le=5000)
    steps_per_scenario: int = Field(default=90, ge=20, le=1000)
    max_people: int = Field(default=2, ge=1, le=6)
    event_interval_seconds: int = Field(default=4, ge=1, le=120)
    movement_probability: float = Field(default=0.7, ge=0.0, le=1.0)
    occupancy_refresh_probability: float = Field(default=0.25, ge=0.0, le=1.0)
    epochs: int = Field(default=5, ge=1, le=30)
    max_samples: int = Field(default=15000, ge=500, le=80000)
    seed: int = Field(default=42, ge=0)
    use_real_profile: bool = True
    real_profile_csv_path: str = ""
    real_profile_max_events: int = Field(default=50000, ge=500, le=300000)
    domain_randomization: bool = True
    false_positive_rate: float = Field(default=0.03, ge=0.0, le=0.4)
    false_negative_rate: float = Field(default=0.08, ge=0.0, le=0.6)
    weak_real_pretrain: bool = True
    weak_presence_hold_seconds: int = Field(default=180, ge=10, le=3600)


class TrainingManifestInput(BaseModel):
    manifest_id: str = Field(min_length=1, max_length=120)


class TrainPresenceSupervisedRequest(BaseModel):
    manifest_id: str = Field(default="person_pet_foyer", min_length=1, max_length=120)
    epochs: int = Field(default=5, ge=1, le=30)
    seed: int = Field(default=42, ge=0)
    min_human_recall: float = Field(default=0.98, ge=0.5, le=1.0)
    synthetic_scenarios: int = Field(default=120, ge=20, le=1000)
    synthetic_steps: int = Field(default=60, ge=20, le=500)
    max_samples: int = Field(default=15000, ge=500, le=80000)


class LayoutReferenceInput(BaseModel):
    adjacency_text: str = ""
    adjacency: dict[str, list[str]] = Field(default_factory=dict)
    rooms: list[str] = Field(default_factory=list)


class ReplayControlInput(BaseModel):
    action: str = Field(default="start", pattern="^(start|pause|step|reset)$")


class RuntimeModeInput(BaseModel):
    mode: str = Field(default="listen", pattern="^(listen|replay|simulator)$")


def normalize_room_name(value: str | None) -> str:
    if not value:
        return ""
    room = value.strip().lower().replace(" ", "_")
    return ROOM_ALIASES.get(room, room)


def add_undirected_edge(adjacency: dict[str, set[str]], a: str, b: str) -> None:
    if not a or not b or a == b:
        return
    adjacency.setdefault(a, set()).add(b)
    adjacency.setdefault(b, set()).add(a)


def to_adjacency(rooms: list[str], edges: list[tuple[str, str]]) -> dict[str, list[str]]:
    adjacency: dict[str, set[str]] = {room: set() for room in rooms}
    for a_raw, b_raw in edges:
        a = normalize_room_name(a_raw)
        b = normalize_room_name(b_raw)
        if a and b:
            add_undirected_edge(adjacency, a, b)
    return {room: sorted(list(neighbors)) for room, neighbors in adjacency.items()}


def edge_list_from_adjacency(adjacency: dict[str, list[str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen = set()
    for room, neighbors in adjacency.items():
        for nb in neighbors:
            key = edge_key(room, nb)
            if key in seen:
                continue
            seen.add(key)
            out.append({"a": key[0], "b": key[1]})
    out.sort(key=lambda item: (item["a"], item["b"]))
    return out


def shortest_path_rooms(adjacency: dict[str, list[str]], start: str, goal: str) -> list[str]:
    start_n = normalize_room_name(start)
    goal_n = normalize_room_name(goal)
    if not start_n or not goal_n:
        return []
    if start_n == goal_n:
        return [start_n]
    if start_n not in adjacency or goal_n not in adjacency:
        return []

    queue: deque[list[str]] = deque([[start_n]])
    visited = {start_n}
    while queue:
        path = queue.popleft()
        node = path[-1]
        for nb in adjacency.get(node, []):
            if nb in visited:
                continue
            new_path = path + [nb]
            if nb == goal_n:
                return new_path
            visited.add(nb)
            queue.append(new_path)
    return []


def build_scenario_templates(rooms: list[str]) -> dict[str, dict[str, list[str]]]:
    normalized_rooms = sorted({normalize_room_name(room) for room in rooms if normalize_room_name(room)})
    if not normalized_rooms:
        return {"real_home": {}}

    real_home_edges = [
        (a, b)
        for (a, b) in REAL_HOME_LAYOUT_EDGES
        if a in normalized_rooms and b in normalized_rooms
    ]

    line_edges: list[tuple[str, str]] = []
    for idx in range(1, len(normalized_rooms)):
        line_edges.append((normalized_rooms[idx - 1], normalized_rooms[idx]))

    ring_edges = list(line_edges)
    if len(normalized_rooms) > 2:
        ring_edges.append((normalized_rooms[0], normalized_rooms[-1]))

    center_room = "foyer" if "foyer" in normalized_rooms else normalized_rooms[0]
    star_edges = [(center_room, room) for room in normalized_rooms if room != center_room]

    return {
        "real_home": to_adjacency(normalized_rooms, real_home_edges),
        "lineal": to_adjacency(normalized_rooms, line_edges),
        "anillo": to_adjacency(normalized_rooms, ring_edges),
        "estrella": to_adjacency(normalized_rooms, star_edges),
    }


def build_layout_for_request(
    rooms: list[str],
    template: str,
    custom_edges: list[list[str]],
) -> dict[str, list[str]]:
    normalized_rooms = sorted({normalize_room_name(room) for room in rooms if normalize_room_name(room)})
    if not normalized_rooms:
        return {}

    templates = build_scenario_templates(normalized_rooms)
    if template == "personalizado" and custom_edges:
        parsed_edges: list[tuple[str, str]] = []
        for pair in custom_edges:
            if len(pair) != 2:
                continue
            parsed_edges.append((pair[0], pair[1]))
        custom_layout = to_adjacency(normalized_rooms, parsed_edges)
        if any(custom_layout.values()):
            return custom_layout

    selected = templates.get(template, templates.get("real_home", {}))
    if selected:
        return selected

    # Utiliza un mapa lineal cuando la plantilla seleccionada no contiene aristas.
    fallback_edges: list[tuple[str, str]] = []
    for idx in range(1, len(normalized_rooms)):
        fallback_edges.append((normalized_rooms[idx - 1], normalized_rooms[idx]))
    return to_adjacency(normalized_rooms, fallback_edges)


def normalize_adjacency_map(
    adjacency: dict[str, list[str]],
    rooms: list[str] | None = None,
) -> dict[str, list[str]]:
    normalized_rooms = {
        normalize_room_name(room)
        for room in (rooms or [])
        if normalize_room_name(room)
    }
    for room, neighbors in adjacency.items():
        room_n = normalize_room_name(room)
        if room_n:
            normalized_rooms.add(room_n)
        for nb in neighbors:
            nb_n = normalize_room_name(nb)
            if nb_n:
                normalized_rooms.add(nb_n)

    adjacency_set: dict[str, set[str]] = {room: set() for room in sorted(normalized_rooms)}
    for room, neighbors in adjacency.items():
        room_n = normalize_room_name(room)
        if not room_n:
            continue
        adjacency_set.setdefault(room_n, set())
        for nb in neighbors:
            nb_n = normalize_room_name(nb)
            if not nb_n or nb_n == room_n:
                continue
            adjacency_set.setdefault(nb_n, set())
            add_undirected_edge(adjacency_set, room_n, nb_n)

    return {
        room: sorted(list(neighbors))
        for room, neighbors in sorted(adjacency_set.items(), key=lambda item: item[0])
    }


def parse_adjacency_text(text: str) -> dict[str, list[str]]:
    adjacency: dict[str, list[str]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        room_raw, neighbors_raw = line.split(":", 1)
        room = normalize_room_name(room_raw)
        if not room:
            continue
        neighbors: list[str] = []
        for nb_raw in neighbors_raw.split(","):
            nb = normalize_room_name(nb_raw)
            if nb and nb != room:
                neighbors.append(nb)
        adjacency[room] = neighbors
    return normalize_adjacency_map(adjacency)


def adjacency_to_text(adjacency: dict[str, list[str]]) -> str:
    lines: list[str] = []
    for room in sorted(adjacency.keys()):
        neighbors = sorted(set(adjacency.get(room, [])))
        lines.append(f"{room}: {', '.join(neighbors)}" if neighbors else f"{room}:")
    return "\n".join(lines)


def adjacency_edge_set(adjacency: dict[str, list[str]]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for room, neighbors in adjacency.items():
        for nb in neighbors:
            out.add(edge_key(room, nb))
    return out
