"""Persistencia atomica de perfiles de presencia."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4


SCHEMA_VERSION = 1
TRAINING_ROLE_SIGNAL = "signal"
TRAINING_ROLE_PERSON_CONFIRMATION = "person_confirmation"
TRAINING_ROLE_PET_CONFIRMATION = "pet_confirmation"
TRAINING_ROLE_PEOPLE_COUNT_CONFIRMATION = "people_count_confirmation"
TRAINING_ROLES = {
    TRAINING_ROLE_SIGNAL,
    TRAINING_ROLE_PERSON_CONFIRMATION,
    TRAINING_ROLE_PET_CONFIRMATION,
    TRAINING_ROLE_PEOPLE_COUNT_CONFIRMATION,
}


class ProfileNotFoundError(KeyError):
    """Indica que el perfil solicitado no existe."""


class ProfileRevisionError(ValueError):
    """Indica que el perfil fue modificado por otro cliente."""


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def profile_fingerprint(profile: dict[str, Any]) -> str:
    structural = {
        "rooms": sorted(
            str(room.get("slug") or "")
            for room in profile.get("rooms", [])
            if isinstance(room, dict)
        ),
        "edges": sorted(
            sorted((str(edge[0]), str(edge[1])))
            for edge in profile.get("edges", [])
            if isinstance(edge, list) and len(edge) == 2
        ),
    }
    encoded = json.dumps(
        structural,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_training_role(value: Any) -> str:
    role = str(value or TRAINING_ROLE_SIGNAL)
    return role if role in TRAINING_ROLES else TRAINING_ROLE_SIGNAL


def profile_validation_errors(profile: dict[str, Any]) -> list[str]:
    rooms = {
        str(room.get("slug") or "").strip().lower()
        for room in profile.get("rooms", [])
        if isinstance(room, dict)
    }
    errors: list[str] = []
    for item in profile.get("assignments", []):
        if not isinstance(item, dict):
            continue
        entity_id = str(item.get("entity_id") or "").strip().lower()
        role_raw = str(item.get("training_role") or TRAINING_ROLE_SIGNAL)
        role = normalized_training_role(role_raw)
        room_slug = str(item.get("room_slug") or "").strip().lower()
        if role_raw not in TRAINING_ROLES:
            errors.append(f"{entity_id or 'Entidad'} tiene un uso no valido: {role_raw}")
            continue
        if role == TRAINING_ROLE_PEOPLE_COUNT_CONFIRMATION:
            if room_slug and room_slug not in rooms:
                errors.append(f"{entity_id} no puede asignarse: habitacion desconocida {room_slug}")
            continue
        if not room_slug:
            errors.append(f"{entity_id} requiere una habitacion asignada")
        elif room_slug not in rooms:
            errors.append(f"{entity_id} no puede asignarse: habitacion desconocida {room_slug}")
    return errors


def normalize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(profile)
    normalized["id"] = str(normalized.get("id") or uuid4())
    normalized["name"] = str(normalized.get("name") or "Perfil sin nombre").strip()
    normalized["source"] = str(normalized.get("source") or "manual")
    normalized["revision"] = max(1, int(normalized.get("revision") or 1))
    normalized["rooms"] = [
        {
            "slug": str(room.get("slug") or "").strip().lower(),
            "name": str(room.get("name") or room.get("slug") or "").strip(),
            "area_id": str(room.get("area_id") or ""),
            "area_name": str(room.get("area_name") or ""),
            "status": str(room.get("status") or "active"),
            "warning": str(room.get("warning") or ""),
        }
        for room in normalized.get("rooms", [])
        if isinstance(room, dict) and str(room.get("slug") or "").strip()
    ]
    room_slugs = {room["slug"] for room in normalized["rooms"]}
    normalized["areas"] = [
        {
            "area_id": str(area.get("area_id") or ""),
            "room_slug": str(area.get("room_slug") or "").strip().lower(),
            "name": str(area.get("name") or ""),
        }
        for area in normalized.get("areas", [])
        if isinstance(area, dict)
        and str(area.get("area_id") or "")
        and str(area.get("room_slug") or "").strip().lower() in room_slugs
    ]
    assignments = []
    for item in normalized.get("assignments", []):
        if not isinstance(item, dict):
            continue
        entity_id = str(item.get("entity_id") or "").strip().lower()
        room_slug = str(item.get("room_slug") or "").strip().lower()
        training_role = normalized_training_role(item.get("training_role"))
        if not entity_id:
            continue
        if training_role == TRAINING_ROLE_PEOPLE_COUNT_CONFIRMATION:
            if room_slug and room_slug not in room_slugs:
                continue
        elif room_slug not in room_slugs:
            continue
        assignments.append(
            {
                "entity_id": entity_id,
                "room_slug": room_slug,
                "enabled": bool(item.get("enabled", True)),
                "sensor_type": str(item.get("sensor_type") or "other"),
                "training_role": training_role,
                "area_id": str(item.get("area_id") or ""),
                "area_name": str(item.get("area_name") or ""),
                "status": str(item.get("status") or "active"),
                "warning": str(item.get("warning") or ""),
                "unique_id": str(item.get("unique_id") or ""),
                "platform": str(item.get("platform") or ""),
            }
        )
    normalized["assignments"] = assignments
    edges: set[tuple[str, str]] = set()
    for edge in normalized.get("edges", []):
        if not isinstance(edge, list) or len(edge) != 2:
            continue
        left = str(edge[0] or "").strip().lower()
        right = str(edge[1] or "").strip().lower()
        if left in room_slugs and right in room_slugs and left != right:
            edges.add(tuple(sorted((left, right))))
    normalized["edges"] = [list(edge) for edge in sorted(edges)]
    normalized["created_at"] = str(normalized.get("created_at") or utc_iso())
    normalized["updated_at"] = str(normalized.get("updated_at") or utc_iso())
    normalized["fingerprint"] = profile_fingerprint(normalized)
    return normalized


class PresenceProfileStore:
    """Administra perfiles persistentes y el perfil activo."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = RLock()
        self._data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "active_profile_id": None,
            "profiles": [],
        }

    def load(self) -> dict[str, Any]:
        with self._lock:
            if not self.path.exists():
                return self.snapshot()
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            profiles = [
                normalize_profile(item)
                for item in payload.get("profiles", [])
                if isinstance(item, dict)
            ]
            profile_ids = {profile["id"] for profile in profiles}
            active = payload.get("active_profile_id")
            self._data = {
                "schema_version": SCHEMA_VERSION,
                "active_profile_id": active if active in profile_ids else None,
                "profiles": profiles,
            }
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._data)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def list_profiles(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._data["profiles"])

    def get(self, profile_id: str) -> dict[str, Any]:
        with self._lock:
            profile = next(
                (
                    item
                    for item in self._data["profiles"]
                    if item.get("id") == profile_id
                ),
                None,
            )
            if profile is None:
                raise ProfileNotFoundError(profile_id)
            return deepcopy(profile)

    def active(self) -> dict[str, Any] | None:
        with self._lock:
            profile_id = self._data.get("active_profile_id")
            return self.get(profile_id) if profile_id else None

    def create(self, profile: dict[str, Any], *, activate: bool = False) -> dict[str, Any]:
        with self._lock:
            normalized = normalize_profile(profile)
            if any(
                item.get("id") == normalized["id"]
                for item in self._data["profiles"]
            ):
                normalized["id"] = str(uuid4())
            self._data["profiles"].append(normalized)
            if activate:
                self._data["active_profile_id"] = normalized["id"]
            self._save()
            return deepcopy(normalized)

    def update(
        self,
        profile_id: str,
        profile: dict[str, Any],
        *,
        expected_revision: int,
    ) -> dict[str, Any]:
        with self._lock:
            current = self.get(profile_id)
            if int(current["revision"]) != int(expected_revision):
                raise ProfileRevisionError(profile_id)
            updated = normalize_profile(
                {
                    **current,
                    **profile,
                    "id": profile_id,
                    "revision": int(current["revision"]) + 1,
                    "created_at": current["created_at"],
                    "updated_at": utc_iso(),
                }
            )
            self._data["profiles"] = [
                updated if item.get("id") == profile_id else item
                for item in self._data["profiles"]
            ]
            self._save()
            return deepcopy(updated)

    def activate(self, profile_id: str) -> dict[str, Any]:
        with self._lock:
            profile = self.get(profile_id)
            self._data["active_profile_id"] = profile_id
            self._save()
            return profile

    def delete(self, profile_id: str) -> tuple[dict[str, Any], bool]:
        with self._lock:
            profile = self.get(profile_id)
            was_active = self._data.get("active_profile_id") == profile_id
            self._data["profiles"] = [
                item
                for item in self._data["profiles"]
                if item.get("id") != profile_id
            ]
            if was_active:
                self._data["active_profile_id"] = None
            self._save()
            return profile, was_active


def profile_store_from_env() -> PresenceProfileStore:
    return PresenceProfileStore(
        os.getenv(
            "PRESENCE_PROFILES_PATH",
            "/app/data/presence_profiles.json",
        )
    )
