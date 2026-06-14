from __future__ import annotations

import asyncio
from contextlib import contextmanager
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
VALID_MODES = {"listen", "replay", "simulator"}


def utc_iso(value: datetime | None = None) -> str:
    current = value or datetime.now(timezone.utc)
    return current.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _parse_json(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


class HistoryStore:
    def __init__(
        self,
        path: str | Path,
        *,
        default_enabled: bool = True,
        default_retention_days: int = 365,
        default_modes: set[str] | None = None,
        batch_size: int = 100,
        batch_delay_seconds: float = 0.05,
    ) -> None:
        self.path = Path(path)
        self.default_enabled = bool(default_enabled)
        self.default_retention_days = max(1, int(default_retention_days))
        self.default_modes = sorted((default_modes or VALID_MODES) & VALID_MODES)
        self.batch_size = max(1, batch_size)
        self.batch_delay_seconds = max(0.0, batch_delay_seconds)
        self.queue: asyncio.Queue[tuple[dict[str, Any], asyncio.Future | None] | None] = (
            asyncio.Queue()
        )
        self.worker_task: asyncio.Task | None = None
        self.last_error: str | None = None
        self.last_cleanup_at: str | None = None
        self._config: dict[str, Any] = {
            "enabled": self.default_enabled,
            "retention_days": self.default_retention_days,
            "persisted_modes": self.default_modes,
        }

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
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
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version < 1:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS history_config (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        enabled INTEGER NOT NULL,
                        retention_days INTEGER NOT NULL,
                        persisted_modes_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS presence_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_timestamp TEXT NOT NULL,
                        stored_at TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        sensor_name TEXT NOT NULL,
                        sensor_type TEXT NOT NULL,
                        room TEXT NOT NULL,
                        state TEXT NOT NULL,
                        source TEXT NOT NULL,
                        input_mode TEXT NOT NULL,
                        inferred_presence INTEGER NOT NULL,
                        inferred_room TEXT NOT NULL,
                        confidence REAL,
                        estimated_people INTEGER NOT NULL,
                        active_rooms_json TEXT NOT NULL,
                        layout_alert_json TEXT,
                        raw_payload_json TEXT NOT NULL,
                        inference_payload_json TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_presence_events_timestamp
                        ON presence_events(event_timestamp DESC, id DESC);
                    CREATE INDEX IF NOT EXISTS idx_presence_events_entity
                        ON presence_events(entity_id);
                    CREATE INDEX IF NOT EXISTS idx_presence_events_sensor_type
                        ON presence_events(sensor_type);
                    CREATE INDEX IF NOT EXISTS idx_presence_events_room
                        ON presence_events(room);
                    CREATE INDEX IF NOT EXISTS idx_presence_events_input_mode
                        ON presence_events(input_mode);
                    CREATE INDEX IF NOT EXISTS idx_presence_events_stored_at
                        ON presence_events(stored_at);
                    """
                )
                version = 1

            if version < 2:
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_presence_events_alert_timestamp
                    ON presence_events(event_timestamp DESC, id DESC)
                    WHERE layout_alert_json IS NOT NULL
                    """
                )
                version = 2

            connection.execute(f"PRAGMA user_version={version}")

            connection.execute(
                """
                INSERT OR IGNORE INTO history_config (
                    id, enabled, retention_days, persisted_modes_json, updated_at
                ) VALUES (1, ?, ?, ?, ?)
                """,
                (
                    int(self.default_enabled),
                    self.default_retention_days,
                    _json(self.default_modes),
                    utc_iso(),
                ),
            )
            connection.commit()
        self._config = self._read_config()

    async def start(self) -> None:
        await asyncio.to_thread(self.initialize)
        await asyncio.to_thread(self.cleanup)
        self.worker_task = asyncio.create_task(
            self._writer_loop(),
            name="presence_history_writer",
        )

    async def stop(self) -> None:
        if not self.worker_task:
            return
        await self.queue.put(None)
        await self.worker_task
        self.worker_task = None

    def _read_config(self) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT enabled, retention_days, persisted_modes_json, updated_at "
                "FROM history_config WHERE id = 1"
            ).fetchone()
        if row is None:
            return {
                "enabled": self.default_enabled,
                "retention_days": self.default_retention_days,
                "persisted_modes": self.default_modes,
                "updated_at": None,
            }
        modes = [
            mode
            for mode in _parse_json(row["persisted_modes_json"], [])
            if mode in VALID_MODES
        ]
        return {
            "enabled": bool(row["enabled"]),
            "retention_days": int(row["retention_days"]),
            "persisted_modes": modes,
            "updated_at": row["updated_at"],
        }

    def get_config(self) -> dict[str, Any]:
        return dict(self._config)

    def update_config(
        self,
        *,
        enabled: bool,
        retention_days: int,
        persisted_modes: list[str],
    ) -> dict[str, Any]:
        modes = sorted({mode for mode in persisted_modes if mode in VALID_MODES})
        if not modes:
            raise ValueError("Debe seleccionarse al menos un modo de persistencia")
        retention = max(1, min(int(retention_days), 3650))
        updated_at = utc_iso()
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE history_config
                SET enabled = ?, retention_days = ?, persisted_modes_json = ?, updated_at = ?
                WHERE id = 1
                """,
                (int(enabled), retention, _json(modes), updated_at),
            )
            connection.commit()
        self._config = {
            "enabled": bool(enabled),
            "retention_days": retention,
            "persisted_modes": modes,
            "updated_at": updated_at,
        }
        self.cleanup()
        return self.get_config()

    def should_persist(self, input_mode: str) -> bool:
        return bool(self._config.get("enabled")) and input_mode in set(
            self._config.get("persisted_modes", [])
        )

    async def enqueue(self, record: dict[str, Any], *, wait: bool = False) -> bool:
        if not self.should_persist(str(record.get("input_mode") or "")):
            return False
        future = asyncio.get_running_loop().create_future() if wait else None
        await self.queue.put((record, future))
        if future is None:
            return True
        return bool(await future)

    async def _writer_loop(self) -> None:
        while True:
            first = await self.queue.get()
            if first is None:
                return
            batch = [first]
            if self.batch_delay_seconds:
                await asyncio.sleep(self.batch_delay_seconds)
            while len(batch) < self.batch_size:
                try:
                    item = self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    await self._write_batch(batch)
                    return
                batch.append(item)
            await self._write_batch(batch)

    async def _write_batch(
        self,
        batch: list[tuple[dict[str, Any], asyncio.Future | None]],
    ) -> None:
        success = True
        try:
            await asyncio.to_thread(
                self.insert_many,
                [record for record, _future in batch],
            )
            self.last_error = None
            await asyncio.to_thread(self.cleanup_if_due)
        except Exception as err:  # noqa: BLE001
            success = False
            self.last_error = str(err)
        for _record, future in batch:
            if future is not None and not future.done():
                future.set_result(success)

    def insert_many(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        values = []
        for record in records:
            values.append(
                (
                    record["event_timestamp"],
                    record.get("stored_at") or utc_iso(),
                    record["entity_id"],
                    record.get("sensor_name") or record["entity_id"],
                    record.get("sensor_type") or "other",
                    record.get("room") or "",
                    record.get("state") or "",
                    record.get("source") or "",
                    record.get("input_mode") or "listen",
                    int(bool(record.get("inferred_presence"))),
                    record.get("inferred_room") or "",
                    record.get("confidence"),
                    max(0, int(record.get("estimated_people") or 0)),
                    _json(record.get("active_rooms") or []),
                    _json(record["layout_alert"])
                    if record.get("layout_alert") is not None
                    else None,
                    _json(record.get("raw_payload") or {}),
                    _json(record.get("inference_payload") or {}),
                )
            )
        with self._connection() as connection:
            connection.executemany(
                """
                INSERT INTO presence_events (
                    event_timestamp, stored_at, entity_id, sensor_name,
                    sensor_type, room, state, source, input_mode,
                    inferred_presence, inferred_room, confidence,
                    estimated_people, active_rooms_json, layout_alert_json,
                    raw_payload_json, inference_payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            connection.commit()

    def cleanup_if_due(self) -> int:
        if self.last_cleanup_at:
            previous = datetime.fromisoformat(self.last_cleanup_at.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) - previous < timedelta(days=1):
                return 0
        return self.cleanup()

    def cleanup(self) -> int:
        cutoff = utc_iso(
            datetime.now(timezone.utc)
            - timedelta(days=int(self._config.get("retention_days", 365)))
        )
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM presence_events WHERE stored_at < ?",
                (cutoff,),
            )
            connection.commit()
            deleted = int(cursor.rowcount or 0)
        self.last_cleanup_at = utc_iso()
        return deleted

    def purge(self) -> int:
        with self._connection() as connection:
            total = int(
                connection.execute("SELECT COUNT(*) FROM presence_events").fetchone()[0]
            )
            connection.execute("DELETE FROM presence_events")
            connection.commit()
        return total

    def status(self) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       MIN(event_timestamp) AS first_timestamp,
                       MAX(event_timestamp) AS last_timestamp
                FROM presence_events
                """
            ).fetchone()
        return {
            **self.get_config(),
            "database_path": str(self.path),
            "database_size_bytes": self.path.stat().st_size if self.path.exists() else 0,
            "events_total": int(row["total"] or 0),
            "first_timestamp": row["first_timestamp"],
            "last_timestamp": row["last_timestamp"],
            "last_cleanup_at": self.last_cleanup_at,
            "last_error": self.last_error,
            "queue_size": self.queue.qsize(),
        }

    @staticmethod
    def _filters(
        *,
        query: str = "",
        sensor_type: str = "",
        room: str = "",
        input_mode: str = "",
        from_ts: str = "",
        to_ts: str = "",
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if query.strip():
            pattern = f"%{query.strip().lower()}%"
            clauses.append(
                "(LOWER(sensor_name) LIKE ? OR LOWER(entity_id) LIKE ?)"
            )
            params.extend([pattern, pattern])
        for column, value in (
            ("sensor_type", sensor_type),
            ("room", room),
            ("input_mode", input_mode),
        ):
            if value:
                clauses.append(f"{column} = ?")
                params.append(value)
        if from_ts:
            clauses.append("event_timestamp >= ?")
            params.append(from_ts)
        if to_ts:
            clauses.append("event_timestamp <= ?")
            params.append(to_ts)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "event_timestamp": row["event_timestamp"],
            "stored_at": row["stored_at"],
            "entity_id": row["entity_id"],
            "sensor_name": row["sensor_name"],
            "sensor_type": row["sensor_type"],
            "room": row["room"],
            "state": row["state"],
            "source": row["source"],
            "input_mode": row["input_mode"],
            "inferred_presence": bool(row["inferred_presence"]),
            "inferred_room": row["inferred_room"],
            "confidence": row["confidence"],
            "estimated_people": int(row["estimated_people"]),
            "active_rooms": _parse_json(row["active_rooms_json"], []),
            "layout_alert": _parse_json(row["layout_alert_json"], None),
            "raw_payload": _parse_json(row["raw_payload_json"], {}),
            "inference_payload": _parse_json(row["inference_payload_json"], {}),
        }

    def query_events(
        self,
        *,
        query: str = "",
        sensor_type: str = "",
        room: str = "",
        input_mode: str = "",
        from_ts: str = "",
        to_ts: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        page = max(1, int(page))
        page_size = max(1, min(int(page_size), 200))
        where, params = self._filters(
            query=query,
            sensor_type=sensor_type,
            room=room,
            input_mode=input_mode,
            from_ts=from_ts,
            to_ts=to_ts,
        )
        with self._connection() as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM presence_events{where}",
                    params,
                ).fetchone()[0]
            )
            pages = max(1, (total + page_size - 1) // page_size)
            page = min(page, pages)
            rows = connection.execute(
                f"""
                SELECT * FROM presence_events{where}
                ORDER BY event_timestamp DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, page_size, (page - 1) * page_size],
            ).fetchall()
            sensors = connection.execute(
                "SELECT DISTINCT entity_id, sensor_name FROM presence_events "
                "ORDER BY sensor_name, entity_id"
            ).fetchall()
            sensor_types = connection.execute(
                "SELECT DISTINCT sensor_type FROM presence_events ORDER BY sensor_type"
            ).fetchall()
            rooms = connection.execute(
                "SELECT DISTINCT room FROM presence_events WHERE room <> '' ORDER BY room"
            ).fetchall()
            modes = connection.execute(
                "SELECT DISTINCT input_mode FROM presence_events ORDER BY input_mode"
            ).fetchall()
        return {
            "items": [self._row_to_event(row) for row in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": pages,
            "options": {
                "sensors": [
                    {
                        "entity_id": row["entity_id"],
                        "sensor_name": row["sensor_name"],
                    }
                    for row in sensors
                ],
                "sensor_types": [row["sensor_type"] for row in sensor_types],
                "rooms": [row["room"] for row in rooms],
                "input_modes": [row["input_mode"] for row in modes],
            },
        }

    def query_alerts(
        self,
        *,
        query: str = "",
        sensor_type: str = "",
        room: str = "",
        input_mode: str = "",
        from_ts: str = "",
        to_ts: str = "",
        page: int = 1,
        page_size: int = 25,
    ) -> dict[str, Any]:
        page = max(1, int(page))
        page_size = max(1, min(int(page_size), 200))
        where, params = self._filters(
            query=query,
            sensor_type=sensor_type,
            room=room,
            input_mode=input_mode,
            from_ts=from_ts,
            to_ts=to_ts,
        )
        alert_where = (
            f"{where} AND layout_alert_json IS NOT NULL"
            if where
            else " WHERE layout_alert_json IS NOT NULL"
        )
        with self._connection() as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM presence_events{alert_where}",
                    params,
                ).fetchone()[0]
            )
            pages = max(1, (total + page_size - 1) // page_size)
            page = min(page, pages)
            rows = connection.execute(
                f"""
                SELECT * FROM presence_events{alert_where}
                ORDER BY event_timestamp DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, page_size, (page - 1) * page_size],
            ).fetchall()
        return {
            "items": [self._row_to_event(row) for row in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": pages,
        }

    def query_presence(
        self,
        *,
        query: str = "",
        sensor_type: str = "",
        room: str = "",
        input_mode: str = "",
        from_ts: str = "",
        to_ts: str = "",
        max_points: int = 1000,
    ) -> dict[str, Any]:
        limit = max(10, min(int(max_points), 5000))
        where, params = self._filters(
            query=query,
            sensor_type=sensor_type,
            room=room,
            input_mode=input_mode,
            from_ts=from_ts,
            to_ts=to_ts,
        )
        with self._connection() as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM presence_events{where}",
                    params,
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT event_timestamp, inferred_presence, inferred_room,
                       estimated_people, confidence
                FROM presence_events{where}
                ORDER BY event_timestamp ASC, id ASC
                LIMIT ?
                """,
                [*params, limit],
            ).fetchall()
        points: list[dict[str, Any]] = []
        previous: tuple[Any, ...] | None = None
        for row in rows:
            current = (
                bool(row["inferred_presence"]),
                row["inferred_room"],
                int(row["estimated_people"]),
            )
            if current == previous:
                continue
            previous = current
            points.append(
                {
                    "timestamp": row["event_timestamp"],
                    "presence": current[0],
                    "room": current[1],
                    "people": current[2],
                    "confidence": row["confidence"],
                }
            )
        return {
            "points": points,
            "source_events": total,
            "truncated": total > limit,
            "max_points": limit,
        }

    def transition_support(
        self,
        entity_ids: list[str],
        *,
        max_gap_seconds: int = 600,
    ) -> list[dict[str, Any]]:
        normalized = sorted(
            {
                str(entity_id or "").strip().lower()
                for entity_id in entity_ids
                if str(entity_id or "").strip()
            }
        )
        if not normalized:
            return []
        placeholders = ",".join("?" for _item in normalized)
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT event_timestamp, entity_id, room, state
                FROM presence_events
                WHERE LOWER(entity_id) IN ({placeholders})
                ORDER BY event_timestamp ASC, id ASC
                """,
                normalized,
            ).fetchall()
        active_states = {
            "on",
            "open",
            "opened",
            "occupied",
            "home",
            "present",
            "detected",
            "motion",
            "active",
            "true",
        }
        previous: tuple[datetime, str] | None = None
        supports: dict[tuple[str, str], int] = {}
        for row in rows:
            room = str(row["room"] or "").strip().lower()
            if not room or str(row["state"] or "").strip().lower() not in active_states:
                continue
            try:
                timestamp = datetime.fromisoformat(
                    str(row["event_timestamp"]).replace("Z", "+00:00")
                )
            except ValueError:
                continue
            if previous is not None:
                previous_time, previous_room = previous
                gap = (timestamp - previous_time).total_seconds()
                if (
                    previous_room != room
                    and 0 <= gap <= max(1, int(max_gap_seconds))
                ):
                    edge = tuple(sorted((previous_room, room)))
                    supports[edge] = supports.get(edge, 0) + 1
            previous = (timestamp, room)
        total = sum(supports.values())
        return [
            {
                "a": edge[0],
                "b": edge[1],
                "support": support,
                "confidence": round(support / total, 4) if total else 0.0,
            }
            for edge, support in sorted(
                supports.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]


def history_store_from_env() -> HistoryStore:
    modes = {
        item.strip().lower()
        for item in os.getenv(
            "HISTORY_PERSIST_MODES",
            "listen,replay,simulator",
        ).split(",")
        if item.strip().lower() in VALID_MODES
    }
    enabled = os.getenv("HISTORY_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    return HistoryStore(
        os.getenv(
            "HISTORY_DB_PATH",
            "/app/data/presence_history.sqlite3",
        ),
        default_enabled=enabled,
        default_retention_days=int(os.getenv("HISTORY_RETENTION_DAYS", "365")),
        default_modes=modes or VALID_MODES,
    )
