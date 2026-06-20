from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inferencia_hub.runtime import lifecycle


def test_runtime_paths_and_json_persistence(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCIA_DATA_DIR", str(tmp_path))
    assert lifecycle.data_dir() == tmp_path
    assert lifecycle.model_state_dir() == tmp_path / "model_state"
    assert lifecycle.training_status_path().parent == tmp_path
    assert lifecycle.real_sensor_config_path().parent == tmp_path

    hub = SimpleNamespace(active_profile_id="profile-1")
    with patch.object(lifecycle, "hub_state", hub):
        assert lifecycle.active_model_state_dir().name == "profile-1"

    lifecycle.training_status.clear()
    lifecycle.training_status["historical"] = {"state": "running"}
    lifecycle.persist_training_status()
    lifecycle.training_status.clear()
    lifecycle.load_training_status()
    assert lifecycle.training_status["historical"]["state"] == "error"


def test_training_artifact_resolution_and_status(tmp_path, monkeypatch):
    csv_path = tmp_path / "events.csv"
    csv_path.write_text("entity_id,state\n", encoding="utf-8")
    monkeypatch.setenv("TRAINING_CSV_PATH", str(csv_path))
    monkeypatch.setenv("TRAINING_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("TRAINING_STATUS_PATH", str(tmp_path / "status.json"))
    assert lifecycle.resolve_training_csv() == str(csv_path)
    artifact = lifecycle.store_training_artifact({"status": "ok"})
    assert artifact is not None
    assert json.loads((tmp_path / "artifacts" / "latest_training.json").read_text())["status"] == "ok"

    lifecycle.training_status.clear()
    lifecycle.mark_training_status("historical", "running", "started", request={"epochs": 1})
    lifecycle.mark_training_status(
        "historical", "success", "done",
        result={
            "status": "ok",
            "training_info": {"samples": 10, "transformer": {"enabled": True}},
            "simulated_sensor_csv": {"url": "/file", "rows": 10},
            "model_state": {"saved": True},
        },
    )
    assert lifecycle.training_status["historical"]["result_summary"]["samples"] == 10


def test_model_state_save_atomic_and_rollback(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_STATE_DIR", str(tmp_path / "models"))

    class Model:
        def __init__(self):
            self.training_info = {}

        def save_state(self, path):
            path.mkdir(parents=True, exist_ok=True)
            target = path / "model.json"
            target.write_text("{}", encoding="utf-8")
            return {"model": str(target)}

        def load_state(self, path):
            return {"loaded": (path / "model.json").exists()}

    hub = SimpleNamespace(
        active_profile_id="profile-1",
        active_profile_fingerprint="fingerprint",
        active_profile_model_compatible=False,
        ai_model=Model(),
    )
    with patch.object(lifecycle, "hub_state", hub):
        saved = lifecycle.persist_model_state()
        assert saved and hub.active_profile_model_compatible
        first = lifecycle.persist_model_state_atomic()
        assert first and first["active_dir"]
        (lifecycle.active_model_state_dir() / "model.json").write_text('{"version":2}', encoding="utf-8")
        second = lifecycle.persist_model_state_atomic()
        assert second and second["previous_dir"]
        rolled = lifecycle.rollback_model_state()
        assert rolled["status"] == "ok"


@pytest.mark.asyncio
async def test_history_sink_listen_mode_and_activation():
    store = SimpleNamespace(
        enqueue=AsyncMock(return_value=False),
        should_persist=lambda mode: True,
    )
    catalog = SimpleNamespace(sensor_name=lambda entity_id: "Motion")
    payload = lifecycle.SensorEventInput(
        entity_id="binary_sensor.motion", state="on", sensor_type="motion",
        room="kitchen", source="ha",
    )
    event = {
        "timestamp": "2026-01-01T00:00:00Z", "entity_id": payload.entity_id,
        "sensor_type": "motion", "room": "kitchen", "state": "on",
        "input_mode": "listen", "inferred_presence": "presente",
    }
    hub = SimpleNamespace(
        input_mode="replay", replay_stop_requested=False, replay_step_budget=2,
        replay_task=None, replay_paused=True, broadcast_snapshot=AsyncMock(),
    )
    with (
        patch.object(lifecycle, "history_store", store),
        patch.object(lifecycle, "ha_entity_catalog", catalog),
        patch.object(lifecycle, "hub_state", hub),
    ):
        await lifecycle.persist_history_event(payload, event, {"status": "ok"})
        await lifecycle.activate_listen_mode()
    assert store.enqueue.await_args.kwargs["wait"] is True
    assert hub.input_mode == "listen"
    assert hub.replay_paused is False


@pytest.mark.asyncio
async def test_shutdown_closes_scheduler_and_history():
    history_store = SimpleNamespace(stop=AsyncMock())
    with (
        patch.object(lifecycle, "history_store", history_store),
        patch("inferencia_hub.runtime.live_training.stop_live_training_scheduler", AsyncMock()) as stop,
    ):
        await lifecycle.shutdown_history_store()
    stop.assert_awaited_once()
    history_store.stop.assert_awaited_once()


def test_export_simulated_sensor_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINING_EXPORT_DIR", str(tmp_path))
    event = SimpleNamespace(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        entity_id="binary_sensor.motion", state="on", sensor_type="motion", room="kitchen",
    )
    model = SimpleNamespace(
        real_profile_info={"enabled": False},
        _generate_simulated_presence_events=MagicMock(
            return_value=([(event, {"kitchen"})], ["kitchen"], {})
        ),
    )
    hub = SimpleNamespace(ai_model=model)
    request = lifecycle.TrainSimulatorPresenceRequest(rooms=["kitchen"], scenarios=20, steps_per_scenario=20)
    with patch.object(lifecycle, "hub_state", hub):
        result = lifecycle.export_simulated_sensor_csv(request, {"kitchen": []})
    assert result["rows"] == 1
    assert Path(result["path"]).exists()


@pytest.mark.asyncio
async def test_startup_initializes_stores_and_respects_model_flags(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINING_STATUS_PATH", str(tmp_path / "status.json"))
    history = SimpleNamespace(start=AsyncMock())
    manifests = SimpleNamespace(initialize=MagicMock())
    live = SimpleNamespace(initialize=MagicMock())
    hub = SimpleNamespace(
        active_profile_model_compatible=True,
        active_profile_id="p1", active_profile_fingerprint="fp",
    )
    profile = {"id": "p1"}
    with (
        patch.object(lifecycle, "history_store", history),
        patch.object(lifecycle, "training_manifests", manifests),
        patch.object(lifecycle, "live_training_store", live),
        patch.object(lifecycle, "hub_state", hub),
        patch("inferencia_hub.runtime.profiles.initialize_profiles", return_value=profile),
        patch("inferencia_hub.runtime.profiles._apply_profile", AsyncMock()) as apply,
        patch("inferencia_hub.runtime.live_training.start_live_training_scheduler") as scheduler,
    ):
        await lifecycle.startup_train_model()
    history.start.assert_awaited_once()
    scheduler.assert_called_once()
    apply.assert_awaited_once_with(profile)
    assert lifecycle.training_status["model_state"]["state"] == "loaded"

    hub.active_profile_model_compatible = False
    monkeypatch.setenv("AUTO_TRAIN_ON_START", "0")
    with (
        patch.object(lifecycle, "history_store", history),
        patch.object(lifecycle, "training_manifests", manifests),
        patch.object(lifecycle, "live_training_store", live),
        patch.object(lifecycle, "hub_state", hub),
        patch("inferencia_hub.runtime.profiles.initialize_profiles", return_value=None),
        patch("inferencia_hub.runtime.live_training.start_live_training_scheduler"),
    ):
        await lifecycle.startup_train_model()


@pytest.mark.asyncio
async def test_startup_runs_background_training_when_csv_is_available(tmp_path, monkeypatch):
    csv_path = tmp_path / "events.csv"
    csv_path.write_text("entity_id,state\n", encoding="utf-8")
    monkeypatch.setenv("TRAINING_CSV_PATH", str(csv_path))
    monkeypatch.setenv("AUTO_TRAIN_ON_START", "1")
    monkeypatch.setenv("FORCE_AUTO_TRAIN_ON_START", "1")
    model = SimpleNamespace(
        rooms=["kitchen"],
        train_from_csv_with_reference=MagicMock(return_value={"status": "ok"}),
    )
    hub = SimpleNamespace(
        active_profile_model_compatible=False,
        active_profile_id="p1", active_profile_fingerprint="fp",
        lock=__import__("asyncio").Lock(), reference_layout={"kitchen": []},
        ai_model=model, rooms=set(), _ensure_reference_layout_locked=MagicMock(),
    )
    history = SimpleNamespace(start=AsyncMock())
    tasks = []
    original_create_task = __import__("asyncio").create_task

    def capture(coro):
        task = original_create_task(coro)
        tasks.append(task)
        return task

    with (
        patch.object(lifecycle, "history_store", history),
        patch.object(lifecycle, "training_manifests", SimpleNamespace(initialize=MagicMock())),
        patch.object(lifecycle, "live_training_store", SimpleNamespace(initialize=MagicMock())),
        patch.object(lifecycle, "hub_state", hub),
        patch.object(lifecycle, "persist_model_state", MagicMock()),
        patch.object(lifecycle.asyncio, "create_task", side_effect=capture),
        patch("inferencia_hub.runtime.profiles.initialize_profiles", return_value={"id": "p1"}),
        patch("inferencia_hub.runtime.profiles._apply_profile", AsyncMock()),
        patch("inferencia_hub.runtime.live_training.start_live_training_scheduler"),
    ):
        await lifecycle.startup_train_model()
        await __import__("asyncio").gather(*tasks)
    model.train_from_csv_with_reference.assert_called_once()
    assert "kitchen" in hub.rooms
