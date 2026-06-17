"""Persistencia de confirmaciones y ejecuciones de aprendizaje en vivo."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_iso(value: datetime | None = None) -> str:
    current = value or datetime.now(timezone.utc)
    return current.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


DEFAULT_CONFIG = {
    "enabled": True,
    "minimum_confirmations": 500,
    "minimum_person_confirmations": 100,
    "minimum_pet_confirmations": 100,
    "minimum_days_between_activations": 7,
}


class LiveTrainingStore:
    """Administra etiquetas Frigate y resultados de adaptación del modelo."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS live_training_config (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    enabled INTEGER NOT NULL,
                    minimum_confirmations INTEGER NOT NULL,
                    minimum_person_confirmations INTEGER NOT NULL,
                    minimum_pet_confirmations INTEGER NOT NULL,
                    minimum_days_between_activations INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS training_confirmations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_timestamp TEXT NOT NULL,
                    stored_at TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    training_role TEXT NOT NULL,
                    room TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    profile_revision INTEGER NOT NULL,
                    profile_fingerprint TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_training_confirmations_profile_time
                    ON training_confirmations(
                        profile_id, profile_fingerprint, event_timestamp, id
                    );
                CREATE INDEX IF NOT EXISTS idx_training_confirmations_role
                    ON training_confirmations(training_role, state);

                CREATE TABLE IF NOT EXISTS live_training_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_timestamp TEXT NOT NULL,
                    stored_at TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    sensor_type TEXT NOT NULL,
                    room TEXT NOT NULL,
                    state TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    profile_revision INTEGER NOT NULL,
                    profile_fingerprint TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_live_training_signals_profile_time
                    ON live_training_signals(
                        profile_id, profile_fingerprint, event_timestamp, id
                    );

                CREATE TABLE IF NOT EXISTS live_training_runs (
                    run_id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    profile_fingerprint TEXT NOT NULL,
                    state TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    confirmation_cutoff_id INTEGER,
                    activated_components_json TEXT NOT NULL DEFAULT '[]',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    message TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_live_training_runs_profile
                    ON live_training_runs(profile_id, started_at DESC);
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(training_confirmations)")
            }
            if "numeric_value" not in columns:
                connection.execute(
                    "ALTER TABLE training_confirmations ADD COLUMN numeric_value REAL"
                )
            connection.execute(
                """
                INSERT OR IGNORE INTO live_training_config (
                    id, enabled, minimum_confirmations,
                    minimum_person_confirmations,
                    minimum_pet_confirmations,
                    minimum_days_between_activations,
                    updated_at
                ) VALUES (1, 1, 500, 100, 100, 7, ?)
                """,
                (utc_iso(),),
            )
            connection.commit()

    def config(self) -> dict[str, Any]:
        self.initialize()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM live_training_config WHERE id = 1"
            ).fetchone()
        return {
            "enabled": bool(row["enabled"]),
            "minimum_confirmations": int(row["minimum_confirmations"]),
            "minimum_person_confirmations": int(
                row["minimum_person_confirmations"]
            ),
            "minimum_pet_confirmations": int(row["minimum_pet_confirmations"]),
            "minimum_days_between_activations": int(
                row["minimum_days_between_activations"]
            ),
            "updated_at": row["updated_at"],
        }

    def update_config(self, values: dict[str, Any]) -> dict[str, Any]:
        config = {**DEFAULT_CONFIG, **values}
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE live_training_config
                SET enabled = ?,
                    minimum_confirmations = ?,
                    minimum_person_confirmations = ?,
                    minimum_pet_confirmations = ?,
                    minimum_days_between_activations = ?,
                    updated_at = ?
                WHERE id = 1
                """,
                (
                    int(bool(config["enabled"])),
                    int(config["minimum_confirmations"]),
                    int(config["minimum_person_confirmations"]),
                    int(config["minimum_pet_confirmations"]),
                    int(config["minimum_days_between_activations"]),
                    utc_iso(),
                ),
            )
            connection.commit()
        return self.config()

    def record_confirmation(
        self,
        *,
        timestamp: str,
        entity_id: str,
        state: str,
        training_role: str,
        room: str,
        profile_id: str,
        profile_revision: int,
        profile_fingerprint: str,
        numeric_value: float | None = None,
    ) -> int:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO training_confirmations (
                    event_timestamp, stored_at, entity_id, state,
                    training_role, room, profile_id, profile_revision,
                    profile_fingerprint, numeric_value
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    utc_iso(),
                    entity_id,
                    state,
                    training_role,
                    room,
                    profile_id,
                    int(profile_revision),
                    profile_fingerprint,
                    numeric_value,
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def record_signal(
        self,
        *,
        timestamp: str,
        entity_id: str,
        sensor_type: str,
        room: str,
        state: str,
        profile_id: str,
        profile_revision: int,
        profile_fingerprint: str,
    ) -> int:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO live_training_signals (
                    event_timestamp, stored_at, entity_id, sensor_type,
                    room, state, profile_id, profile_revision,
                    profile_fingerprint
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    utc_iso(),
                    entity_id,
                    sensor_type,
                    room,
                    state,
                    profile_id,
                    int(profile_revision),
                    profile_fingerprint,
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def _last_activated_run(
        self,
        profile_id: str,
        profile_fingerprint: str,
    ) -> sqlite3.Row | None:
        with self._connection() as connection:
            return connection.execute(
                """
                SELECT * FROM live_training_runs
                WHERE profile_id = ?
                  AND profile_fingerprint = ?
                  AND state = 'activated'
                ORDER BY finished_at DESC
                LIMIT 1
                """,
                (profile_id, profile_fingerprint),
            ).fetchone()

    def confirmation_counts(
        self,
        profile_id: str,
        profile_fingerprint: str,
        *,
        only_new: bool = True,
    ) -> dict[str, int]:
        last = self._last_activated_run(profile_id, profile_fingerprint)
        cutoff = (
            int(last["confirmation_cutoff_id"] or 0)
            if only_new and last is not None
            else 0
        )
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT training_role, COUNT(*) AS total, MAX(id) AS max_id
                FROM training_confirmations
                WHERE profile_id = ?
                  AND profile_fingerprint = ?
                  AND (state = 'on' OR training_role = 'people_count_confirmation')
                  AND id > ?
                GROUP BY training_role
                """,
                (profile_id, profile_fingerprint, cutoff),
            ).fetchall()
        counts = {
            "total": 0,
            "person": 0,
            "pet": 0,
            "count": 0,
            "maximum_id": cutoff,
        }
        for row in rows:
            total = int(row["total"])
            role = str(row["training_role"])
            counts["total"] += total
            counts["maximum_id"] = max(
                counts["maximum_id"],
                int(row["max_id"] or cutoff),
            )
            if role == "person_confirmation":
                counts["person"] += total
            elif role == "pet_confirmation":
                counts["pet"] += total
            elif role == "people_count_confirmation":
                counts["count"] += total
        return counts

    def confirmations(
        self,
        profile_id: str,
        profile_fingerprint: str,
    ) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM training_confirmations
                WHERE profile_id = ? AND profile_fingerprint = ?
                ORDER BY event_timestamp, id
                """,
                (profile_id, profile_fingerprint),
            ).fetchall()
        return [dict(row) for row in rows]

    def signal_events(
        self,
        profile_id: str,
        profile_fingerprint: str,
        *,
        from_timestamp: str,
        to_timestamp: str,
    ) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT event_timestamp, entity_id, sensor_type, room, state
                FROM live_training_signals
                WHERE profile_id = ?
                  AND profile_fingerprint = ?
                  AND event_timestamp >= ?
                  AND event_timestamp <= ?
                ORDER BY event_timestamp, id
                """,
                (
                    profile_id,
                    profile_fingerprint,
                    from_timestamp,
                    to_timestamp,
                ),
            ).fetchall()
        return [dict(row) for row in rows]

    def begin_run(
        self,
        profile_id: str,
        profile_fingerprint: str,
        trigger: str,
    ) -> str:
        run_id = uuid4().hex
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO live_training_runs (
                    run_id, profile_id, profile_fingerprint, state,
                    trigger, started_at
                ) VALUES (?, ?, ?, 'running', ?, ?)
                """,
                (
                    run_id,
                    profile_id,
                    profile_fingerprint,
                    trigger,
                    utc_iso(),
                ),
            )
            connection.commit()
        return run_id

    def finish_run(
        self,
        run_id: str,
        *,
        state: str,
        message: str,
        confirmation_cutoff_id: int,
        activated_components: list[str],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE live_training_runs
                SET state = ?, finished_at = ?, message = ?,
                    confirmation_cutoff_id = ?,
                    activated_components_json = ?,
                    metrics_json = ?
                WHERE run_id = ?
                """,
                (
                    state,
                    utc_iso(),
                    message,
                    int(confirmation_cutoff_id),
                    json.dumps(activated_components, ensure_ascii=False),
                    json.dumps(metrics, ensure_ascii=False),
                    run_id,
                ),
            )
            connection.commit()
        return self.run(run_id)

    def run(self, run_id: str) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM live_training_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        payload = dict(row)
        payload["activated_components"] = json.loads(
            payload.pop("activated_components_json") or "[]"
        )
        payload["metrics"] = json.loads(payload.pop("metrics_json") or "{}")
        return payload

    def latest_run(
        self,
        profile_id: str,
        profile_fingerprint: str,
    ) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT run_id FROM live_training_runs
                WHERE profile_id = ? AND profile_fingerprint = ?
                ORDER BY started_at DESC LIMIT 1
                """,
                (profile_id, profile_fingerprint),
            ).fetchone()
        return self.run(str(row["run_id"])) if row else None

    def last_activated_run(
        self,
        profile_id: str,
        profile_fingerprint: str,
    ) -> dict[str, Any] | None:
        row = self._last_activated_run(profile_id, profile_fingerprint)
        return self.run(str(row["run_id"])) if row else None


def live_training_store_from_env() -> LiveTrainingStore:
    import os

    data_dir = Path(os.getenv("INFERENCIA_DATA_DIR", "/app/data"))
    path = Path(
        os.getenv(
            "HISTORY_DB_PATH",
            str(data_dir / "presence_history.sqlite3"),
        )
    )
    return LiveTrainingStore(path)
