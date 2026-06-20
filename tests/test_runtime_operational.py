from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from inferencia_hub.runtime import history, home_assistant, layout, presence


class HistoryStoreStub:
    def __init__(self):
        self.config = {"enabled": True}

    def status(self):
        return self.config

    def update_config(self, **kwargs):
        self.config.update(kwargs)

    def query_events(self, **kwargs):
        return {"kind": "events", **kwargs}

    def query_alerts(self, **kwargs):
        return {"kind": "alerts", **kwargs}

    def query_presence(self, **kwargs):
        return {"kind": "presence", **kwargs}

    def purge(self):
        return {"events": 3}


@pytest.mark.asyncio
async def test_history_handlers_validate_and_delegate():
    store = HistoryStoreStub()
    with patch.object(history, "history_store", store):
        assert (await history.get_history_config())["enabled"] is True
        config = history.HistoryConfigInput(
            enabled=False, retention_days=30, persisted_modes=["listen"]
        )
        assert (await history.update_history_config(config))["retention_days"] == 30
        events = await history.get_history_events(
            from_ts="2026-01-01T00:00:00Z", to_ts="2026-01-02T00:00:00Z"
        )
        assert events["kind"] == "events"
        assert (await history.get_history_alerts())["kind"] == "alerts"
        assert (await history.get_history_presence(max_points=20))["max_points"] == 20
        assert (await history.purge_history(history.HistoryPurgeInput(confirmation="BORRAR")))["deleted"] == {"events": 3}

    for function in (history.get_history_events, history.get_history_alerts, history.get_history_presence):
        with pytest.raises(HTTPException) as exc:
            await function(from_ts="2026-02-01T00:00:00Z", to_ts="2026-01-01T00:00:00Z")
        assert exc.value.status_code == 422
    with pytest.raises(HTTPException):
        history.normalize_history_timestamp("invalid", "from_ts")
    with pytest.raises(HTTPException):
        await history.purge_history(history.HistoryPurgeInput(confirmation="NO"))


def test_home_assistant_payload_and_action_queue_contracts():
    catalog = SimpleNamespace(
        as_dict=lambda: {"entities": []},
        get=lambda key, default=None: {
            "entities": [{"entity_id": "binary_sensor.motion"}],
            "entities_total": 1,
            "supported_total": 1,
        }.get(key, default),
    )
    hub = SimpleNamespace(
        real_sensor_config=lambda: {
            "rooms": ["kitchen"],
            "assignments": [{"entity_id": "binary_sensor.motion", "room": "kitchen", "enabled": True}],
        },
        _profile_payload_locked=lambda: {"id": "p1"},
    )
    action_queue = SimpleNamespace(
        pending=[], recent_results=[], integration_status={"entries": {}},
        request=lambda *args: {"status": "queued", "args": args},
        claim=lambda entry: {"status": "empty", "entry": entry},
        complete=lambda request_id, payload: {"request_id": request_id, **payload},
    )
    with (
        patch.object(home_assistant, "ha_entity_catalog", catalog),
        patch.object(home_assistant, "hub_state", hub),
        patch.object(home_assistant.context, "actions", action_queue),
    ):
        assert home_assistant.get_ha_entities() == {"entities": []}
        result = home_assistant.get_real_sensor_config()
        assert result["room_sensor_counts"]["kitchen"] == 1
        status = home_assistant.update_ha_integration_status({"entry_id": "e1", "ok": True})
        assert status["integration_status"]["entries"]["e1"]["ok"] is True
        request = home_assistant.request_ha_action(
            home_assistant.HAActionRequestInput(action="refresh_catalog")
        )
        assert request["status"] == "queued"
        assert home_assistant.claim_ha_action("e1")["status"] == "empty"
        assert home_assistant.complete_ha_action("r1", {"status": "ok"})["result"]["request_id"] == "r1"


@pytest.mark.asyncio
async def test_catalog_update_normalizes_and_broadcasts():
    catalog = MagicMock()
    hub = SimpleNamespace(broadcast_snapshot=AsyncMock())
    payload = home_assistant.HAEntityCatalogInput(
        source="ha",
        entry_id="e1",
        tracked_entities=["b", "a", "a"],
        areas=[{"area_id": "kitchen", "name": "Kitchen"}],
        entities=[{
            "entity_id": "binary_sensor.motion", "name": "Motion",
            "domain": "binary_sensor", "supported": True,
        }],
    )
    with (
        patch.object(home_assistant, "ha_entity_catalog", catalog),
        patch.object(home_assistant, "hub_state", hub),
        patch.object(home_assistant, "reconcile_profiles_with_catalog", AsyncMock(return_value={"updated": 1})),
    ):
        result = await home_assistant.update_ha_entities(payload)
    assert result["tracked_entities"] == ["a", "b"]
    assert result["supported_total"] == 1
    catalog.replace.assert_called_once()
    hub.broadcast_snapshot.assert_awaited_once()


@pytest.mark.asyncio
async def test_layout_queries_and_profile_requirement():
    fake_lock = asyncio.Lock()
    hub = SimpleNamespace(
        ai_model=SimpleNamespace(rooms=["kitchen"]),
        rooms={"office"},
        reference_layout={"kitchen": {"office"}},
        lock=fake_lock,
        _layout_payload_locked=lambda: {"rooms": ["kitchen", "office"]},
        _evaluation_metrics_locked=lambda: {"non_adjacent": {"recent": list(range(10))}},
    )
    with patch.object(layout, "hub_state", hub):
        templates = layout.scenario_templates()
        assert "lineal" in templates["templates"]
        assert (await layout.get_layout_reference())["layout_reference"]["rooms"]
        metrics = await layout.evaluation_metrics(limit=2)
        assert metrics["metrics"]["non_adjacent"]["recent"] == [8, 9]
    with patch.object(layout.profile_store, "active", return_value=None):
        with pytest.raises(HTTPException) as exc:
            await layout.set_layout_reference(layout.LayoutReferenceInput(rooms=["kitchen"]))
        assert exc.value.status_code == 409


def test_presence_parsing_and_read_models():
    assert presence._parse_people_count_state("2") == 2
    assert presence._parse_people_count_state("2.5") is None
    assert presence._parse_people_count_state("bad") is None
    assert presence._parse_people_count_state("-1") is None
    hub = SimpleNamespace(
        replay_task=None, events=[1], rooms={"kitchen"}, input_mode="listen",
        replay_paused=False, ai_model=SimpleNamespace(
            ready=True, training_info={"transformer": {"enabled": True}},
            occupancy_transformer_info={"enabled": True}, relative_occupancy_info={},
            rooms=["kitchen"], adjacency_edges=[], occupancy_transformer_rooms=["kitchen"],
            pet_filter_info={},
        ),
        _profile_payload_locked=lambda: {"id": "p1"},
        presence_filter_config=lambda: {"enabled": True},
        real_sensor_config=lambda: {"assignments": [], "enabled_entities": [], "rejected_events": 0},
        snapshot=lambda: {"status": "ok"},
    )
    catalog = SimpleNamespace(get=lambda key, default=None: default, as_dict=lambda: {})
    store = SimpleNamespace(status=lambda: {"enabled": True})
    with (
        patch.object(presence, "hub_state", hub),
        patch.object(presence, "ha_entity_catalog", catalog),
        patch.object(presence, "history_store", store),
        patch.object(presence, "live_training_status", return_value={"state": "idle"}),
    ):
        assert presence.health()["model_ready"] is True
        assert presence.get_sim_data()["live_training"]["state"] == "idle"
        assert presence.get_presence_filter()["enabled"] is True
        assert presence.model_info()["ready"] is True
        assert presence.get_input_mode()["mode"] == "listen"


@pytest.mark.asyncio
async def test_ingest_rejects_invalid_sources_and_modes():
    hub = SimpleNamespace(active_profile_id="p1", input_mode="listen")
    with patch.object(presence, "hub_state", hub):
        unknown = await presence.ingest_event(
            presence.SensorEventInput(entity_id="x", source="unknown")
        )
        assert unknown["reason"] == "unknown_event_source"
        replay = await presence.ingest_event(
            presence.SensorEventInput(entity_id="x", source="csv_replay")
        )
        assert replay["reason"] == "replay_not_active"
        simulator = await presence.ingest_event(
            presence.SensorEventInput(entity_id="x", source="sensor_simulator")
        )
        assert simulator["reason"] == "simulator_not_active"
    with patch.object(presence, "hub_state", SimpleNamespace(active_profile_id=None, input_mode="listen")):
        result = await presence.ingest_event(presence.SensorEventInput(entity_id="x"))
        assert result["reason"] == "no_active_profile"


@pytest.mark.asyncio
async def test_real_sensor_configuration_success_and_validation_error():
    profile = {
        "id": "p1", "revision": 1, "name": "Home",
        "rooms": [{"slug": "kitchen", "name": "Kitchen"}],
        "assignments": [], "areas": [], "edges": [],
    }
    updated = {**profile, "revision": 2}
    store = SimpleNamespace(active=lambda: profile, update=MagicMock(return_value=updated))
    catalog = [{
        "entity_id": "binary_sensor.motion", "sensor_type": "motion",
        "area_id": "a1", "area_name": "Kitchen", "unique_id": "u1", "platform": "zha",
    }]
    hub = SimpleNamespace(
        real_sensor_config=lambda: {
            "rooms": ["kitchen", "office"],
            "assignments": [{"entity_id": "binary_sensor.motion", "room": "kitchen", "enabled": True}],
            "enabled_entities": ["binary_sensor.motion"],
        },
        _profile_payload_locked=lambda: updated,
    )
    request = home_assistant.RealSensorConfigInput(
        rooms=["Office"],
        assignments=[
            {"entity_id": "BINARY_SENSOR.MOTION", "room": "Kitchen", "sensor_type": "auto", "enabled": True},
            {"entity_id": "", "room": "Office"},
        ],
    )
    with (
        patch.object(home_assistant, "profile_store", store),
        patch.object(home_assistant, "_catalog_entities", return_value=catalog),
        patch.object(home_assistant, "profile_validation_errors", return_value=[]),
        patch.object(home_assistant, "_apply_profile", AsyncMock()),
        patch.object(home_assistant, "activate_listen_mode", AsyncMock()) as activate,
        patch.object(home_assistant, "hub_state", hub),
        patch.object(home_assistant, "ha_entity_catalog", SimpleNamespace(get=lambda key, default=None: default)),
    ):
        result = await home_assistant.set_real_sensor_config(request)
    assert result["room_sensor_counts"]["kitchen"] == 1
    activate.assert_awaited_once()

    with (
        patch.object(home_assistant, "profile_store", store),
        patch.object(home_assistant, "_catalog_entities", return_value=catalog),
        patch.object(home_assistant, "profile_validation_errors", return_value=["bad"]),
    ):
        with pytest.raises(HTTPException) as exc:
            await home_assistant.set_real_sensor_config(request)
        assert exc.value.status_code == 422


@pytest.mark.asyncio
async def test_layout_update_persists_profile_and_applies_it():
    profile = {
        "id": "p1", "revision": 1,
        "rooms": [{"slug": "kitchen", "name": "Kitchen"}],
        "edges": [], "areas": [], "assignments": [],
    }
    result = {
        "layout_reference": {
            "rooms": ["kitchen", "office"],
            "edges": [{"a": "kitchen", "b": "office"}],
        },
        "metrics": {},
    }
    hub = SimpleNamespace(configure_reference_layout=AsyncMock(return_value=result))
    store = SimpleNamespace(active=lambda: profile, update=MagicMock(side_effect=lambda _id, payload, **_kw: {**payload, "id": "p1", "revision": 2}))
    with (
        patch.object(layout, "profile_store", store),
        patch.object(layout, "hub_state", hub),
        patch.object(layout, "_apply_profile", AsyncMock()) as apply,
    ):
        response = await layout.set_layout_reference(
            layout.LayoutReferenceInput(rooms=["kitchen", "office"])
        )
    assert response["profile"]["revision"] == 2
    apply.assert_awaited_once()


@pytest.mark.asyncio
async def test_presence_runtime_controls_rejections_and_socket_cleanup():
    assert presence._parse_people_count_state("") is None
    hub = SimpleNamespace(
        active_profile_id="p1", input_mode="listen",
        replay_task=None, replay_stop_requested=False,
        real_sensor_assignments={},
        configure_presence_filter=AsyncMock(return_value={"enabled": True}),
        broadcast_snapshot=AsyncMock(), reset=AsyncMock(),
    )
    with (
        patch.object(presence, "hub_state", hub),
        patch.object(presence, "catalog_has_entity", return_value=False),
    ):
        rejected = await presence.ingest_event(
            presence.SensorEventInput(entity_id="binary_sensor.missing", source="ha")
        )
        assert rejected["reason"] == "sensor_not_in_ha_catalog"
        assert (await presence.set_presence_filter(presence.PresenceFilterConfigInput()))["enabled"]
        await presence.set_input_mode(presence.RuntimeModeInput(mode="simulator"))
        assert hub.input_mode == "simulator"
        with patch.object(presence, "activate_listen_mode", AsyncMock()) as activate:
            await presence.set_input_mode(presence.RuntimeModeInput(mode="listen"))
            activate.assert_awaited_once()
        assert await presence.reset_state() == {"status": "ok"}

    websocket = SimpleNamespace(
        send_json=AsyncMock(), receive_text=AsyncMock(side_effect=RuntimeError("closed"))
    )
    broker = SimpleNamespace(connect=AsyncMock(), disconnect=MagicMock())
    socket_hub = SimpleNamespace(snapshot=lambda: {"status": "ok"})
    with (
        patch.object(presence.context, "websocket", broker),
        patch.object(presence, "hub_state", socket_hub),
    ):
        await presence.presencia_socket(websocket)
    broker.connect.assert_awaited_once_with(websocket)
    broker.disconnect.assert_called_once_with(websocket)
