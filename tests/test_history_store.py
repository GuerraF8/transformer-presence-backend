from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from inferencia_hub.history_store import HistoryStore, utc_iso


def record(
    entity_id: str = "binary_sensor.kitchen_motion",
    *,
    mode: str = "listen",
    room: str = "kitchen",
    sensor_type: str = "motion",
    stored_at: str | None = None,
    event_timestamp: str = "2026-06-07T10:00:00Z",
    state: str = "on",
    layout_alert: dict | None = None,
) -> dict:
    return {
        "event_timestamp": event_timestamp,
        "stored_at": stored_at or "2026-06-07T10:00:01Z",
        "entity_id": entity_id,
        "sensor_name": "Movimiento cocina",
        "sensor_type": sensor_type,
        "room": room,
        "state": state,
        "source": "ha_state_change",
        "input_mode": mode,
        "inferred_presence": True,
        "inferred_room": room,
        "confidence": 0.91,
        "estimated_people": 1,
        "active_rooms": [room],
        "layout_alert": layout_alert,
        "raw_payload": {"entity_id": entity_id},
        "inference_payload": {"presencia_inferida": "Presente"},
    }


class HistoryStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "history.sqlite3"
        self.store = HistoryStore(self.path)
        self.store.initialize()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_initializes_and_reopens_persistent_database(self) -> None:
        self.store.insert_many([record()])

        reopened = HistoryStore(self.path)
        reopened.initialize()

        self.assertEqual(reopened.status()["events_total"], 1)
        self.assertEqual(reopened.get_config()["retention_days"], 365)

    def test_filters_pagination_and_presence_series(self) -> None:
        self.store.insert_many(
            [
                record(),
                record(
                    "binary_sensor.bedroom_occupancy",
                    mode="simulator",
                    room="bedroom",
                    sensor_type="occupancy",
                ),
                record(
                    "binary_sensor.foyer_door",
                    mode="replay",
                    room="foyer",
                    sensor_type="door",
                ),
            ]
        )

        filtered = self.store.query_events(
            query="cocina",
            sensor_type="motion",
            room="kitchen",
            input_mode="listen",
            page=1,
            page_size=1,
        )
        series = self.store.query_presence(input_mode="listen")

        self.assertEqual(filtered["total"], 1)
        self.assertEqual(filtered["items"][0]["sensor_name"], "Movimiento cocina")
        self.assertEqual(filtered["items"][0]["raw_payload"]["entity_id"], "binary_sensor.kitchen_motion")
        self.assertEqual(len(series["points"]), 1)
        self.assertFalse(series["truncated"])

    def test_configuration_modes_and_purge(self) -> None:
        config = self.store.update_config(
            enabled=True,
            retention_days=90,
            persisted_modes=["listen", "replay"],
        )
        self.store.insert_many([record()])

        self.assertEqual(config["retention_days"], 90)
        self.assertTrue(self.store.should_persist("listen"))
        self.assertFalse(self.store.should_persist("simulator"))
        self.assertEqual(self.store.purge(), 1)
        self.assertEqual(self.store.status()["events_total"], 0)

    def test_date_range_and_stable_order(self) -> None:
        self.store.insert_many(
            [
                record(event_timestamp="2026-06-07T09:00:00Z"),
                record(
                    "binary_sensor.kitchen_motion_2",
                    event_timestamp="2026-06-07T10:00:00Z",
                ),
                record(
                    "binary_sensor.kitchen_motion_3",
                    event_timestamp="2026-06-07T10:00:00Z",
                ),
            ]
        )

        result = self.store.query_events(
            from_ts="2026-06-07T10:00:00Z",
            to_ts="2026-06-07T10:00:00Z",
        )

        self.assertEqual(result["total"], 2)
        self.assertEqual(
            [item["entity_id"] for item in result["items"]],
            [
                "binary_sensor.kitchen_motion_3",
                "binary_sensor.kitchen_motion_2",
            ],
        )

    def test_retention_removes_old_rows(self) -> None:
        old = utc_iso(datetime.now(timezone.utc) - timedelta(days=400))
        self.store.insert_many([record(stored_at=old)])

        deleted = self.store.cleanup()

        self.assertEqual(deleted, 1)

    def test_alerts_are_persistent_filtered_and_paginated(self) -> None:
        first_alert = {
            "from": "kitchen",
            "to": "bedroom",
            "cause": "multi_person_probable",
            "gap_seconds": 3,
        }
        second_alert = {
            "from": "bedroom",
            "to": "foyer",
            "cause": "sensor_or_data_error",
            "gap_seconds": 1,
        }
        self.store.insert_many(
            [
                record(),
                record(
                    "binary_sensor.kitchen_motion_2",
                    event_timestamp="2026-06-07T11:00:00Z",
                    layout_alert=first_alert,
                ),
                record(
                    "binary_sensor.bedroom_occupancy",
                    room="bedroom",
                    sensor_type="occupancy",
                    event_timestamp="2026-06-07T12:00:00Z",
                    layout_alert=second_alert,
                ),
            ]
        )

        first_page = self.store.query_alerts(page=1, page_size=1)
        filtered = self.store.query_alerts(
            sensor_type="motion",
            room="kitchen",
            page=1,
            page_size=25,
        )
        reopened = HistoryStore(self.path)
        reopened.initialize()

        self.assertEqual(first_page["total"], 2)
        self.assertEqual(first_page["pages"], 2)
        self.assertEqual(
            first_page["items"][0]["layout_alert"]["cause"],
            "sensor_or_data_error",
        )
        self.assertEqual(filtered["total"], 1)
        self.assertEqual(filtered["items"][0]["layout_alert"], first_alert)
        self.assertEqual(reopened.query_alerts()["total"], 2)

    def test_migrates_version_one_database_with_alert_index(self) -> None:
        with self.store._connection() as connection:
            connection.execute("DROP INDEX idx_presence_events_alert_timestamp")
            connection.execute("PRAGMA user_version=1")
            connection.commit()

        self.store.initialize()

        with self.store._connection() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            index = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='index' AND name='idx_presence_events_alert_timestamp'"
            ).fetchone()
        self.assertEqual(version, 2)
        self.assertIsNotNone(index)


class HistoryStoreAsyncTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = HistoryStore(
            Path(self.temp_dir.name) / "history.sqlite3",
            batch_delay_seconds=0,
        )
        await self.store.start()

    async def asyncTearDown(self) -> None:
        await self.store.stop()
        self.temp_dir.cleanup()

    async def test_queue_batches_all_modes_and_waits_for_listen(self) -> None:
        listen_saved = await self.store.enqueue(record(), wait=True)
        await self.store.enqueue(record(mode="replay"), wait=False)
        await self.store.enqueue(record(mode="simulator"), wait=False)
        await asyncio.sleep(0.05)

        self.assertTrue(listen_saved)
        self.assertEqual(self.store.status()["events_total"], 3)

    async def test_disabled_store_skips_events(self) -> None:
        self.store.update_config(
            enabled=False,
            retention_days=365,
            persisted_modes=["listen"],
        )

        saved = await self.store.enqueue(record(), wait=True)

        self.assertFalse(saved)
        self.assertEqual(self.store.status()["events_total"], 0)

    async def test_writer_recovers_after_sqlite_error(self) -> None:
        original_insert_many = self.store.insert_many
        attempts = 0

        def flaky_insert_many(records: list[dict]) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("sqlite temporalmente no disponible")
            original_insert_many(records)

        self.store.insert_many = flaky_insert_many  # type: ignore[method-assign]

        first = await self.store.enqueue(record(), wait=True)
        second = await self.store.enqueue(
            record("binary_sensor.bedroom_occupancy", room="bedroom"),
            wait=True,
        )

        self.assertFalse(first)
        self.assertTrue(second)
        self.assertIsNone(self.store.last_error)
        self.assertEqual(self.store.status()["events_total"], 1)
