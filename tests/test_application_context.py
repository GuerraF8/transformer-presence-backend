from __future__ import annotations

import asyncio
import unittest

from inferencia_hub.app_context import (
    HAActionQueue,
    HAEntityCatalog,
    WebSocketBroker,
)
from inferencia_hub.server import app, create_app


class FakeWebSocket:
    def __init__(self) -> None:
        self.accepted = False
        self.payloads: list[dict] = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, payload: dict) -> None:
        self.payloads.append(payload)


class ApplicationContextTest(unittest.TestCase):
    def test_public_routes_and_version_remain_available(self) -> None:
        paths = {getattr(route, "path", "") for route in app.routes}
        expected = {
            "/api/health",
            "/api/events",
            "/api/sim_data",
            "/api/history/events",
            "/api/history/alerts",
            "/api/ha_entities",
            "/api/profiles",
            "/api/profiles/{profile_id}",
            "/api/train_model",
            "/api/train_presence_supervised",
            "/api/training/manifests",
            "/api/model/rollback",
            "/api/replay_csv",
            "/presencia",
        }
        self.assertEqual(app.version, "0.6.0")
        self.assertIs(create_app(), app)
        self.assertTrue(expected.issubset(paths))
        self.assertIs(app.state.context.hub, app.state.context.hub)

    def test_catalog_replaces_payload_and_indexes_entities(self) -> None:
        catalog = HAEntityCatalog()
        entities = [
            {"entity_id": f"binary_sensor.room_{index}", "name": f"Sensor {index}"}
            for index in range(10_000)
        ]
        catalog.replace({"entities": entities, "entities_total": len(entities)})
        self.assertTrue(catalog.has_entity("binary_sensor.room_9999"))
        self.assertEqual(
            catalog.sensor_name("binary_sensor.room_9999"),
            "Sensor 9999",
        )
        self.assertFalse(catalog.has_entity("binary_sensor.missing"))

    def test_action_queue_preserves_request_claim_result_contract(self) -> None:
        queue = HAActionQueue()
        requested = queue.request("refresh_catalog", "entry-1", {"rooms": ""})
        claimed = queue.claim("entry-1")
        result = queue.complete(claimed["request_id"], {"status": "ok"})
        # La solicitud conserva el estado pendiente hasta que un cliente la reclama.
        self.assertEqual(requested["status"], "pending")
        self.assertEqual(claimed["status"], "claimed")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(queue.claim("entry-1")["status"], "empty")

    def test_websocket_broker_owns_transport_connections(self) -> None:
        async def exercise() -> None:
            broker = WebSocketBroker()
            websocket = FakeWebSocket()
            await broker.connect(websocket)  # type: ignore[arg-type]
            await broker.publish({"kind": "snapshot"})
            broker.disconnect(websocket)  # type: ignore[arg-type]
            await broker.publish({"kind": "event"})
            self.assertTrue(websocket.accepted)
            self.assertEqual(websocket.payloads, [{"kind": "snapshot"}])

        asyncio.run(exercise())
