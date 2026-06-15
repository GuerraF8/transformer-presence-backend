from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket

from .history_store import HistoryStore, history_store_from_env
from .live_training_store import (
    LiveTrainingStore,
    live_training_store_from_env,
)
from .hub_state import InferenceHubState
from .profile_store import PresenceProfileStore, profile_store_from_env
from .supervised.manifest import TrainingManifestStore


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class HAEntityCatalog:
    """Catálogo de Home Assistant indexado por identificador de entidad."""

    def __init__(self) -> None:
        self._payload: dict[str, Any] = {
            "source": None,
            "entry_id": None,
            "scanned_at": None,
            "received_at": None,
            "auto_discovery": True,
            "tracked_entities": [],
            "areas": [],
            "entities": [],
            "entities_total": 0,
            "supported_total": 0,
        }
        self._by_entity_id: dict[str, dict[str, Any]] = {}

    def replace(self, payload: dict[str, Any]) -> None:
        entities = payload.get("entities")
        normalized_entities = entities if isinstance(entities, list) else []
        self._payload = {**payload, "entities": normalized_entities}
        self._by_entity_id = {
            str(item.get("entity_id") or "").strip().lower(): item
            for item in normalized_entities
            if isinstance(item, dict) and str(item.get("entity_id") or "").strip()
        }

    def as_dict(self) -> dict[str, Any]:
        return self._payload

    def get(self, key: str, default: Any = None) -> Any:
        return self._payload.get(key, default)

    def has_entity(self, entity_id: str) -> bool:
        return str(entity_id or "").strip().lower() in self._by_entity_id

    def sensor_name(self, entity_id: str) -> str:
        item = self._by_entity_id.get(str(entity_id or "").strip().lower())
        return str(item.get("name") or entity_id) if item else entity_id


class HAActionQueue:
    def __init__(self) -> None:
        self.pending: deque[dict[str, Any]] = deque(maxlen=100)
        self.recent_results: deque[dict[str, Any]] = deque(maxlen=50)
        self.sequence = 0
        self.integration_status: dict[str, Any] = {
            "last_seen_at": None,
            "entries": {},
        }

    def request(self, action: str, entry_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.sequence += 1
        request_id = (
            f"ha-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{self.sequence}"
        )
        item = {
            "request_id": request_id,
            "status": "pending",
            "action": action,
            "entry_id": entry_id,
            "requested_at": utc_iso(),
            "payload": payload,
        }
        self.pending.append(item)
        return {"status": "queued", **item, "pending_total": len(self.pending)}

    def claim(self, entry_id: str) -> dict[str, Any]:
        selected = next(
            (
                item
                for item in self.pending
                if not item.get("entry_id")
                or not entry_id
                or item.get("entry_id") == entry_id
            ),
            None,
        )
        if selected is None:
            return {"status": "empty", "action": None}
        self.pending.remove(selected)
        return {
            **selected,
            "status": "claimed",
            "claimed_at": utc_iso(),
            "claimed_by": entry_id,
        }

    def complete(self, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        result = {"request_id": request_id, "received_at": utc_iso(), **payload}
        self.recent_results.appendleft(result)
        return result


class WebSocketBroker:
    """Administra conexiones WebSocket y publica mensajes a los clientes."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)

    async def publish(self, payload: dict[str, Any]) -> None:
        for websocket in list(self._connections):
            try:
                await websocket.send_json(payload)
            except Exception:
                self._connections.discard(websocket)


def default_training_status() -> dict[str, dict[str, Any]]:
    return {
        "historical": {
            "state": "idle",
            "label": "Historico CSV",
            "started_at": None,
            "finished_at": None,
            "message": "sin entrenamiento iniciado desde UI",
        },
        "presence": {
            "state": "idle",
            "label": "Presencia simulador",
            "started_at": None,
            "finished_at": None,
            "message": "sin entrenamiento iniciado desde UI",
        },
        "supervised": {
            "state": "idle",
            "label": "Presencia supervisada",
            "started_at": None,
            "finished_at": None,
            "message": "sin entrenamiento supervisado iniciado",
        },
    }


@dataclass(slots=True)
class ApplicationContext:
    hub: InferenceHubState = field(default_factory=InferenceHubState)
    history: HistoryStore = field(default_factory=history_store_from_env)
    profiles: PresenceProfileStore = field(default_factory=profile_store_from_env)
    catalog: HAEntityCatalog = field(default_factory=HAEntityCatalog)
    actions: HAActionQueue = field(default_factory=HAActionQueue)
    websocket: WebSocketBroker = field(default_factory=WebSocketBroker)
    manifests: TrainingManifestStore = field(
        default_factory=TrainingManifestStore
    )
    live_training: LiveTrainingStore = field(
        default_factory=live_training_store_from_env
    )
    training_status: dict[str, dict[str, Any]] = field(
        default_factory=default_training_status
    )

    def __post_init__(self) -> None:
        self.hub.snapshot_publisher = self.websocket.publish
