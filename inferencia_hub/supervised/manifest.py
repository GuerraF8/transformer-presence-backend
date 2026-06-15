"""Carga y validación de manifiestos locales de entrenamiento."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from threading import Lock
from typing import Any


SCHEMA_VERSION = 1
ALLOWED_ROLES = {"signal", "person_confirmation", "pet_confirmation", "ignore"}


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TrainingManifestStore:
    """Administra manifiestos y restringe el acceso a rutas de datos permitidas."""

    def __init__(
        self,
        manifest_dir: str | Path | None = None,
        dataset_root: str | Path | None = None,
        defaults_dir: str | Path | None = None,
    ) -> None:
        package_root = Path(__file__).resolve().parents[1]
        self.manifest_dir = Path(
            manifest_dir
            or os.getenv(
                "TRAINING_MANIFEST_DIR",
                "/app/data/training_manifests",
            )
        ).resolve()
        self.dataset_root = Path(
            dataset_root or os.getenv("TRAINING_DATASET_ROOT", "/data")
        ).resolve()
        self.defaults_dir = Path(
            defaults_dir or package_root / "defaults" / "training_manifests"
        ).resolve()
        self.report_dir = Path(
            os.getenv(
                "TRAINING_REPORT_DIR",
                str(self.manifest_dir.parent / "training_reports"),
            )
        ).resolve()
        self._confirmation_entities: set[str] = set()
        self._initialized = False
        self._initialization_lock = Lock()

    def initialize(self) -> None:
        """Prepara los directorios y manifiestos incluidos cuando se necesitan."""

        if self._initialized:
            return
        with self._initialization_lock:
            if self._initialized:
                return
            self._ensure_directories()
            self._install_defaults()
            self._refresh_confirmation_entities()
            self._initialized = True

    def _ensure_directories(self) -> None:
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def _install_defaults(self) -> None:
        if not self.defaults_dir.exists():
            return
        for source in self.defaults_dir.glob("*.json"):
            target = self.manifest_dir / source.name
            if not target.exists():
                shutil.copyfile(source, target)

    @staticmethod
    def _safe_id(manifest_id: str) -> str:
        value = str(manifest_id or "").strip()
        if not value or Path(value).name != value:
            raise ValueError("Identificador de manifiesto inválido")
        if value.endswith(".json"):
            value = value[:-5]
        if not value or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in value):
            raise ValueError("Identificador de manifiesto inválido")
        return value

    def manifest_path(self, manifest_id: str) -> Path:
        safe_id = self._safe_id(manifest_id)
        path = (self.manifest_dir / f"{safe_id}.json").resolve()
        if self.manifest_dir not in path.parents:
            raise ValueError("El manifiesto está fuera del directorio permitido")
        return path

    def dataset_path(self, value: str) -> Path:
        raw = Path(str(value or "").strip())
        path = (
            raw.resolve()
            if raw.is_absolute()
            else (self.dataset_root / raw).resolve()
        )
        if path != self.dataset_root and self.dataset_root not in path.parents:
            raise ValueError(f"CSV fuera del directorio permitido: {value}")
        if path.suffix.lower() != ".csv":
            raise ValueError(f"El archivo no es CSV: {value}")
        return path

    def load(self, manifest_id: str) -> dict[str, Any]:
        self.initialize()
        path = self.manifest_path(manifest_id)
        if not path.exists():
            raise FileNotFoundError(f"No existe el manifiesto: {manifest_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return self._normalize(payload, manifest_id=path.stem)

    def list(self) -> list[dict[str, Any]]:
        self.initialize()
        self._refresh_confirmation_entities()
        manifests: list[dict[str, Any]] = []
        for path in sorted(self.manifest_dir.glob("*.json")):
            try:
                payload = self._normalize(
                    json.loads(path.read_text(encoding="utf-8")),
                    manifest_id=path.stem,
                )
                manifests.append(
                    {
                        "id": payload["id"],
                        "name": payload["name"],
                        "description": payload.get("description", ""),
                        "periods": len(payload["periods"]),
                        "path": str(path),
                    }
                )
            except Exception as exc:
                manifests.append(
                    {
                        "id": path.stem,
                        "name": path.stem,
                        "valid": False,
                        "error": str(exc),
                        "path": str(path),
                    }
                )
        return manifests

    def validate(self, manifest_id: str) -> dict[str, Any]:
        self.initialize()
        manifest = self.load(manifest_id)
        self._refresh_confirmation_entities()
        files: list[dict[str, Any]] = []
        errors: list[str] = []
        seen: set[Path] = set()
        for period in manifest["periods"]:
            for role in ("signal_files", "label_files"):
                for value in period[role]:
                    try:
                        path = self.dataset_path(value)
                        if path in seen:
                            continue
                        seen.add(path)
                        if not path.exists():
                            errors.append(f"No existe {path}")
                            continue
                        digest = _sha256(path)
                        expected = str(
                            manifest.get("file_hashes", {}).get(path.name) or ""
                        )
                        hash_matches = not expected or expected == digest
                        if not hash_matches:
                            errors.append(f"Hash SHA-256 inesperado para {path.name}")
                        files.append(
                            {
                                "name": path.name,
                                "path": str(path),
                                "size": path.stat().st_size,
                                "sha256": digest,
                                "expected_sha256": expected or None,
                                "hash_matches": hash_matches,
                            }
                        )
                    except Exception as exc:
                        errors.append(str(exc))
        return {
            "id": manifest["id"],
            "name": manifest["name"],
            "valid": not errors,
            "validated_at": _utc_iso(),
            "files": files,
            "errors": errors,
            "manifest": manifest,
        }

    def is_confirmation_entity(self, entity_id: str) -> bool:
        self.initialize()
        return (
            str(entity_id or "").strip().lower()
            in self._confirmation_entities
        )

    def _refresh_confirmation_entities(self) -> None:
        entities: set[str] = set()
        for path in self.manifest_dir.glob("*.json"):
            try:
                payload = self._normalize(
                    json.loads(path.read_text(encoding="utf-8")),
                    manifest_id=path.stem,
                )
            except Exception:
                continue
            for period in payload["periods"]:
                entities.update(
                    entity_id
                    for entity_id, role in period["entity_roles"].items()
                    if role
                    in {"person_confirmation", "pet_confirmation"}
                )
        self._confirmation_entities = entities

    def report_path(self, run_id: str) -> Path:
        safe_id = self._safe_id(run_id)
        path = (self.report_dir / f"{safe_id}.json").resolve()
        if self.report_dir not in path.parents:
            raise ValueError("Reporte fuera del directorio permitido")
        return path

    def save_report(self, run_id: str, payload: dict[str, Any]) -> Path:
        self.initialize()
        path = self.report_path(run_id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return path

    def load_report(self, run_id: str) -> dict[str, Any]:
        self.initialize()
        path = self.report_path(run_id)
        if not path.exists():
            raise FileNotFoundError(f"No existe el reporte: {run_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _normalize(
        self,
        payload: dict[str, Any],
        *,
        manifest_id: str,
    ) -> dict[str, Any]:
        if int(payload.get("schema_version") or 0) != SCHEMA_VERSION:
            raise ValueError("Versión de manifiesto no soportada")
        normalized = deepcopy(payload)
        normalized["id"] = self._safe_id(
            str(normalized.get("id") or manifest_id)
        )
        normalized["name"] = str(normalized.get("name") or normalized["id"])
        normalized["description"] = str(normalized.get("description") or "")
        normalized["label_window"] = {
            "before_seconds": max(
                0,
                int(
                    normalized.get("label_window", {}).get(
                        "before_seconds",
                        10,
                    )
                ),
            ),
            "after_seconds": max(
                0,
                int(
                    normalized.get("label_window", {}).get(
                        "after_seconds",
                        20,
                    )
                ),
            ),
        }
        normalized["weak_negative_weight"] = max(
            0.0,
            min(1.0, float(normalized.get("weak_negative_weight", 0.15))),
        )
        periods: list[dict[str, Any]] = []
        for index, raw_period in enumerate(normalized.get("periods", [])):
            if not isinstance(raw_period, dict):
                continue
            roles = {
                str(entity).strip().lower(): str(role).strip().lower()
                for entity, role in dict(
                    raw_period.get("entity_roles") or {}
                ).items()
            }
            invalid_roles = sorted(set(roles.values()) - ALLOWED_ROLES)
            if invalid_roles:
                raise ValueError(
                    f"Roles no soportados: {', '.join(invalid_roles)}"
                )
            periods.append(
                {
                    "id": str(raw_period.get("id") or f"period_{index + 1}"),
                    "expected_range": dict(
                        raw_period.get("expected_range") or {}
                    ),
                    "rooms": [
                        str(room).strip().lower()
                        for room in raw_period.get("rooms", [])
                        if str(room).strip()
                    ],
                    "signal_files": [
                        str(item)
                        for item in raw_period.get("signal_files", [])
                    ],
                    "label_files": [
                        str(item)
                        for item in raw_period.get("label_files", [])
                    ],
                    "entity_roles": roles,
                    "entity_rooms": {
                        str(entity).strip().lower(): str(room).strip().lower()
                        for entity, room in dict(
                            raw_period.get("entity_rooms") or {}
                        ).items()
                    },
                    "room_aliases": {
                        str(room).strip().lower(): str(alias).strip().lower()
                        for room, alias in dict(
                            raw_period.get("room_aliases") or {}
                        ).items()
                    },
                    "exclusions": list(raw_period.get("exclusions") or []),
                }
            )
        if not periods:
            raise ValueError("El manifiesto no contiene períodos")
        normalized["periods"] = periods
        normalized["file_hashes"] = {
            str(name): str(value).lower()
            for name, value in dict(
                normalized.get("file_hashes") or {}
            ).items()
        }
        return normalized
