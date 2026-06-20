from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from inferencia_hub.domain import (
    ProfileCreateInput,
    ProfileInferLayoutInput,
    ProfileUpdateInput,
)
from inferencia_hub.profile_store import PresenceProfileStore
from inferencia_hub.runtime import profiles


class Catalog:
    def __init__(self, entities=None, areas=None):
        self.payload = {"entities": entities or [], "areas": areas or []}

    def get(self, key, default=None):
        return self.payload.get(key, default)


def detected_request():
    return ProfileCreateInput(
        name="Casa detectada", source="detected", area_ids=[], entity_ids=[]
    )


def test_profile_builders_model_status_and_legacy_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_STATE_DIR", str(tmp_path / "models"))
    catalog = Catalog(
        areas=[
            {"area_id": "a1", "name": "Office"},
            {"area_id": "a2", "name": "Office"},
        ],
        entities=[
            {"entity_id": "binary_sensor.motion", "area_id": "a1", "sensor_type": "motion", "platform": "zha", "unique_id": "u1"},
            {"entity_id": "binary_sensor.loose", "area_id": "", "room": "Garage", "sensor_type": "invalid"},
        ],
    )
    with patch.object(profiles, "ha_entity_catalog", catalog):
        built = profiles._build_profile(detected_request())
        loose = profiles._build_profile(ProfileCreateInput(
            name="Loose", source="detected", entity_ids=["binary_sensor.loose"]
        ))
    assert {room["slug"] for room in built["rooms"]} >= {"office", "office_2"}
    assert loose["assignments"][0]["room_slug"] == "garage"
    assert loose["assignments"][0]["sensor_type"] == "other"
    assert profiles._build_profile(ProfileCreateInput(name="Manual"))["source"] == "manual"
    assert profiles._build_profile(ProfileCreateInput(name="Real", source="real_home"))["edges"]

    profile = {"id": "p1", "fingerprint": "fp"}
    assert profiles._profile_model_status(profile)["available"] is False
    model_dir = profiles._profile_model_dir("p1")
    model_dir.mkdir(parents=True)
    (model_dir / "model_state.json").write_text("invalid", encoding="utf-8")
    assert profiles._profile_model_status(profile)["compatible"] is False
    (model_dir / "model_state.json").write_text(
        '{"training_info":{"profile_fingerprint":"fp"}}', encoding="utf-8"
    )
    assert profiles._profile_model_status(profile)["compatible"] is True

    legacy = profiles._legacy_profile_payload({
        "rooms": ["Kitchen", "Office"],
        "assignments": [
            {"entity_id": "BINARY_SENSOR.MOTION", "room": "Kitchen", "sensor_type": "bad", "training_role": "bad"},
            "invalid",
            {"entity_id": "", "room": "Office"},
        ],
    })
    assert legacy["assignments"][0]["sensor_type"] == "other"


@pytest.mark.asyncio
async def test_profile_crud_inference_and_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_STATE_DIR", str(tmp_path / "models"))
    store = PresenceProfileStore(tmp_path / "profiles.json")
    store.load()
    with (
        patch.object(profiles, "profile_store", store),
        patch.object(profiles, "ha_entity_catalog", Catalog()),
        patch.object(profiles, "_apply_profile", AsyncMock()),
    ):
        created = await profiles.create_profile(ProfileCreateInput(name="Manual"))
        profile_id = created["id"]
        assert profiles.list_profiles()["profiles"]
        assert profiles.get_profile(profile_id)["id"] == profile_id
        with pytest.raises(HTTPException) as exc:
            profiles.get_profile("missing")
        assert exc.value.status_code == 404

        active = await profiles.activate_profile(profile_id)
        assert active["active"] is True
        req = ProfileUpdateInput(
            revision=active["revision"], name="Updated",
            rooms=[{"slug": "kitchen", "name": "Kitchen"}],
            areas=[], assignments=[], edges=[],
        )
        updated = await profiles.update_profile(profile_id, req)
        assert updated["name"] == "Updated"
        with pytest.raises(HTTPException) as exc:
            await profiles.update_profile(profile_id, req)
        assert exc.value.status_code == 409

        with patch.object(
            profiles.history_store, "transition_support",
            return_value=[
                {"a": "kitchen", "b": "office", "support": 5},
                {"a": "kitchen", "b": "kitchen", "support": 1},
            ],
        ):
            inferred = await profiles.infer_profile_layout(
                profile_id, ProfileInferLayoutInput(min_support=2, max_gap_seconds=60)
            )
        assert inferred["proposals"] == []
        deleted = await profiles.delete_profile(profile_id)
        assert deleted["was_active"] is True
        with pytest.raises(HTTPException):
            await profiles.activate_profile("missing")
        with pytest.raises(HTTPException):
            await profiles.delete_profile("missing")


@pytest.mark.asyncio
async def test_catalog_reconciliation_updates_moved_missing_and_renamed(tmp_path):
    store = PresenceProfileStore(tmp_path / "profiles.json")
    store.load()
    profile = store.create({
        "name": "Profile", "source": "detected",
        "rooms": [{"slug": "office", "name": "Old", "area_id": "a1", "area_name": "Old"}],
        "areas": [{"area_id": "a1", "room_slug": "office", "name": "Old"}],
        "assignments": [
            {"entity_id": "binary_sensor.old", "room_slug": "office", "enabled": True, "sensor_type": "motion", "status": "active", "area_id": "a1", "platform": "zha", "unique_id": "u1"},
            {"entity_id": "binary_sensor.missing", "room_slug": "office", "enabled": True, "sensor_type": "motion", "status": "active"},
        ],
        "edges": [],
    }, activate=True)
    catalog = Catalog(
        areas=[{"area_id": "a1", "name": "New Office"}],
        entities=[{
            "entity_id": "binary_sensor.new", "area_id": "a2", "area_name": "Other",
            "platform": "zha", "unique_id": "u1",
        }],
    )
    with (
        patch.object(profiles, "profile_store", store),
        patch.object(profiles, "ha_entity_catalog", catalog),
        patch.object(profiles, "_apply_profile", AsyncMock()) as apply,
    ):
        result = await profiles.reconcile_profiles_with_catalog()
    assert result["profiles_changed"] == 1
    assert result["assignments_disabled"] == 2
    apply.assert_awaited_once()


def test_initialize_profiles_migrates_legacy_config(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_STATE_DIR", str(tmp_path / "models"))
    store = PresenceProfileStore(tmp_path / "profiles.json")
    legacy = tmp_path / "real_sensor_config.json"
    legacy.write_text('{"rooms":["kitchen"],"assignments":[]}', encoding="utf-8")
    with patch.object(profiles, "profile_store", store):
        migrated = profiles.initialize_profiles(legacy)
        assert migrated["source"] == "migrated"
        assert profiles.initialize_profiles(legacy)["id"] == migrated["id"]
