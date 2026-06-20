from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from inferencia_hub.domain import (
    LayoutReferenceInput,
    PresenceFilterConfigInput,
    SensorEventInput,
)
from inferencia_hub.hub.dependencies import EventRecord, LastActivation
from inferencia_hub.hub_state import InferenceHubState


NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_layout_sensor_and_filter_configuration_paths():
    hub = InferenceHubState()
    assert hub._edge_quality({("a", "b")}, {("a", "b"), ("b", "c")})["recall"] == 0.5
    with pytest.raises(ValueError):
        await hub.configure_reference_layout(LayoutReferenceInput())

    hub.real_sensor_rooms = {"kitchen", "office"}
    configured = await hub.configure_reference_layout(
        LayoutReferenceInput(adjacency_text="kitchen: office", rooms=["kitchen", "office"])
    )
    assert configured["layout_reference"]["source"] == "manual"
    assert hub._movement_adjacent_locked("kitchen", "office") is True
    assert hub._movement_adjacent_locked("kitchen", "garage") is False
    assert hub._reference_adjacent_locked("", "office") is True

    config = await hub.configure_presence_filter(
        PresenceFilterConfigInput(enabled=True, window_seconds=2, min_motion_events=2, min_distinct_rooms=1)
    )
    assert config["window_seconds"] == 2
    hub.ai_model.predict_human_motion = MagicMock(return_value=None)
    accepted, debug = hub._evaluate_presence_filter_locked("kitchen", "door", NOW)
    assert accepted and not debug["applied"]
    accepted, _ = hub._evaluate_presence_filter_locked("kitchen", "motion", NOW)
    assert accepted is False
    accepted, _ = hub._evaluate_presence_filter_locked("kitchen", "motion", NOW + timedelta(seconds=1))
    assert accepted is True
    hub.ai_model.predict_human_motion = MagicMock(
        return_value={"suppression_enabled": True, "accepted": False}
    )
    accepted, debug = hub._evaluate_presence_filter_locked("office", "motion", NOW)
    assert not accepted and debug["reason"] == "movimiento_clasificado_como_mascota"


def test_filter_pruning_and_displacement_rules():
    hub = InferenceHubState()
    hub.presence_hold_seconds = 10
    hub.reference_layout = {"kitchen": ["hall"], "hall": ["kitchen"]}
    hub.reference_layout_source = "manual"
    hub.last_active_by_room = {"old": NOW - timedelta(seconds=20), "hall": NOW}
    hub.occupancy_confirmed_by_room = {"old": NOW - timedelta(seconds=20)}
    hub.presence_filter_events.append({"timestamp": NOW - timedelta(seconds=30), "room": "old"})
    hub._prune_inactive_rooms(NOW)
    assert "old" not in hub.last_active_by_room
    assert hub._has_adjacent_activity_since_locked("kitchen", NOW - timedelta(seconds=1))
    assert hub._has_adjacent_activity_since_locked("unknown", NOW) is False
    assert hub._can_displace_presence_locked(None, "office", "motion", None)
    assert hub._can_displace_presence_locked("kitchen", "office", "occupancy", NOW)
    assert hub._can_displace_presence_locked("kitchen", "hall", "motion", NOW)
    assert hub._can_displace_presence_locked("kitchen", "office", "motion", NOW) is True


def test_ground_truth_people_and_quality_metrics():
    hub = InferenceHubState()
    hub.reference_layout = {"kitchen": ["hall"], "hall": ["kitchen"], "office": []}
    hub.reference_layout_source = "manual"
    hub.real_sensor_rooms = {"kitchen", "hall", "office"}
    hub._apply_count_ground_truth_locked(
        timestamp=NOW, entity_id="sensor.count", room="", count=2, predicted_people=2
    )
    hub._apply_count_ground_truth_locked(
        timestamp=NOW, entity_id="sensor.kitchen", room="kitchen", count=1, predicted_people=2
    )
    hub._apply_count_ground_truth_locked(
        timestamp=NOW + timedelta(seconds=1), entity_id="sensor.kitchen", room="kitchen", count=0
    )
    hub.current_room = "office"
    hub.current_active_rooms = ["office"]
    hub._record_confirmation_ground_truth_locked(
        timestamp=NOW, entity_id="person", state="on", training_role="person_confirmation", room="office"
    )
    hub._record_confirmation_ground_truth_locked(
        timestamp=NOW, entity_id="pet", state="on", training_role="pet_confirmation", room="office"
    )
    hub._record_confirmation_ground_truth_locked(
        timestamp=NOW, entity_id="ignored", state="off", training_role="person_confirmation", room="office"
    )
    metrics = hub._ground_truth_metrics_locked()
    assert metrics["count"]["count_accuracy"] == 1.0
    assert metrics["presence"]["person_room_match_rate"] == 1.0

    hub.occupancy_confirmed_by_room = {"kitchen": NOW}
    hub.active_sensor_types_by_room = {"kitchen": {"occupancy"}, "office": {"motion"}}
    assert hub._estimate_people_locked(["kitchen", "hall", "office"]) >= 2
    assert hub._estimate_people_locked([]) == 0

    for people, rooms, gap, expected in [
        (2, ["a", "b"], 30, "multiples_personas_probable"),
        (1, ["a"], 2, "mascota_o_ruido"),
        (1, ["a"], 30, "error_sensor_o_datos"),
    ]:
        record = hub._record_non_adjacent_locked(
            timestamp=NOW, transition={"from": "a", "to": "b", "gap_seconds": gap},
            sensor_type="motion", estimated_people=people, active_rooms=rooms,
        )
        assert record["cause"] == expected

    hub.events = [{
        "sensor_type": "occupancy", "state": "on", "room": "kitchen",
        "presence_room": "kitchen", "presence_confidence": 0.8,
        "ai_mode": "markov_ai", "transition": {"same_room": False, "rejected_by_ai": False},
        "inference_debug": {
            "transformer_used": True, "hybrid_top_room": "kitchen",
            "transformer_top_room": "kitchen", "transformer_observed_room_prob": 0.7,
        },
    }]
    quality = hub._inference_quality_metrics_locked()
    assert quality["transformer_usage_rate"] == 1.0
    hub.ingestion_latency_ms.extend([1, 2, 3])
    assert hub._summarize_latency(hub.ingestion_latency_ms)["p50_ms"] == 2.0
    assert hub._summarize_latency(deque())["count"] == 0
    assert "map" in hub.evaluation_metrics()
    assert "f1" in hub.training_map_validation_locked()


def test_probabilistic_inference_and_transition_branches():
    hub = InferenceHubState()
    hub.ai_model.ready = False
    hub.ai_model.predict_occupancy_state = MagicMock(return_value=None)
    room, confidence, active, debug = hub._infer_presence_with_ai("kitchen", "motion", NOW)
    assert (room, confidence, active) == ("kitchen", 0.5, ["kitchen"])
    hub.ai_model.predict_occupancy_state = MagicMock(
        return_value={"rooms": ["office"], "confidence": 0.9, "people_count": 1}
    )
    assert "office" in hub._infer_presence_with_ai("kitchen", "motion", NOW)[2]

    hub.ai_model.ready = True
    hub.ai_model.rooms = ["kitchen", "hall"]
    hub.ai_model.room_to_idx = {"kitchen": 0, "hall": 1}
    hub.ai_model.transition_matrix = np.asarray([[0.7, 0.3], [0.2, 0.8]], dtype=np.float32)
    hub.ai_model.predict_next_room_probs = MagicMock(return_value=np.asarray([0.2, 0.8], dtype=np.float32))
    hub.ai_model.predict_occupancy_state = MagicMock(return_value={"rooms": ["hall"], "confidence": 0.8})
    hub.ai_model.neighbors = MagicMock(return_value=["hall"])
    inferred = hub._infer_presence_with_ai("kitchen", "motion", NOW)
    assert inferred[3]["transformer_used"] is True

    assert hub._build_transition("kitchen", NOW) == (None, False)
    hub.last_activation = LastActivation(room="kitchen", timestamp=NOW)
    same, rejected = hub._build_transition("kitchen", NOW + timedelta(seconds=1))
    assert same["same_room"] and not rejected
    hub.ai_model.are_adjacent = MagicMock(return_value=False)
    transition, rejected = hub._build_transition("office", NOW + timedelta(seconds=2))
    assert rejected and transition["rejected_by_ai"]
    hub.ai_model.are_adjacent = MagicMock(return_value=True)
    transition, rejected = hub._build_transition("hall", NOW + timedelta(seconds=3))
    assert transition["support"] == 1 and not rejected


@pytest.mark.asyncio
async def test_event_pipeline_real_sensor_rejections_and_state_transitions():
    hub = InferenceHubState()
    hub.snapshot_publisher = AsyncMock()
    ignored = await hub.process_event(
        SensorEventInput(entity_id="binary_sensor.unknown", state="on", source="ha")
    )
    assert ignored["reason"] == "sensor_no_asignado"
    hub.real_sensor_assignments["binary_sensor.disabled"] = {
        "room": "kitchen", "enabled": False, "sensor_type": "motion"
    }
    ignored = await hub.process_event(
        SensorEventInput(entity_id="binary_sensor.disabled", state="on", source="ha")
    )
    assert ignored["reason"] == "sensor_deshabilitado"

    hub.presence_filter_enabled = False
    hub.real_sensor_rooms = {"kitchen", "office"}
    hub.reference_layout = {"kitchen": ["office"], "office": ["kitchen"]}
    hub.reference_layout_source = "manual"
    hub.real_sensor_assignments.update({
        "binary_sensor.motion": {"room": "kitchen", "enabled": True, "sensor_type": "motion"},
        "binary_sensor.occupancy": {"room": "office", "enabled": True, "sensor_type": "occupancy"},
    })
    hub.event_sink = AsyncMock(side_effect=RuntimeError("disk full"))
    motion = await hub.process_event(
        SensorEventInput(entity_id="binary_sensor.motion", state="on", source="ha", timestamp=NOW)
    )
    assert motion["presencia_inferida"] == "Presente"
    occupancy = await hub.process_event(
        SensorEventInput(entity_id="binary_sensor.occupancy", state="on", source="ha", timestamp=NOW + timedelta(seconds=1))
    )
    assert occupancy["habitacion_inferida_ia"] == "office"
    off = await hub.process_event(
        SensorEventInput(entity_id="binary_sensor.occupancy", state="off", source="ha", timestamp=NOW + timedelta(seconds=2))
    )
    assert off["event"]["state"] == "off"
    assert len(hub.events) == 3


@pytest.mark.asyncio
async def test_event_pipeline_filtered_motion_and_buffer_compaction():
    hub = InferenceHubState()
    hub.snapshot_publisher = AsyncMock()
    hub.max_events_buffer = 1
    hub.presence_filter_enabled = True
    hub.presence_filter_min_motion_events = 2
    hub.ai_model.predict_human_motion = MagicMock(return_value=None)
    first = await hub.process_event(
        SensorEventInput(entity_id="binary_sensor.kitchen_motion", room="kitchen", sensor_type="motion", state="on", source="sensor_simulator", timestamp=NOW)
    )
    assert first["relacion_habitaciones"] == "filtrado_ventana"
    await hub.process_event(
        SensorEventInput(entity_id="binary_sensor.kitchen_motion", room="kitchen", sensor_type="motion", state="on", source="sensor_simulator", timestamp=NOW + timedelta(seconds=1))
    )
    assert len(hub.events) == 1
    assert hub.events[0]["index"] == 0
