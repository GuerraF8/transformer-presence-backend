from __future__ import annotations

import json
import unittest

import analizar_sostenibilidad_inferencias as benchmark
from inferencia_hub.ai import AIAdjacencyModel
from inferencia_hub.ai.dependencies import TORCH_AVAILABLE


def test_generate_synthetic_events_is_deterministic() -> None:
    first = benchmark.generate_synthetic_events(12)
    second = benchmark.generate_synthetic_events(12)

    assert first == second
    assert first[0].room == "bedroom"
    assert first[0].sensor_type == "motion"
    assert first[1].sensor_type == "occupancy"
    assert first[-1].timestamp > first[0].timestamp


def test_synthetic_profile_and_adjacency_are_consistent() -> None:
    profile = benchmark.synthetic_profile()
    adjacency = benchmark.synthetic_adjacency()

    assert sorted(adjacency) == sorted(room["slug"] for room in profile["rooms"])
    assert "sittingroom" in adjacency["bedroom"]
    assert "bedroom" in adjacency["sittingroom"]
    assert ["foyer", "kitchen"] in profile["edges"]


def test_workload_uses_stable_contexts_and_candidate_rooms() -> None:
    events = benchmark.generate_synthetic_events(72)
    workload = benchmark.build_workload(events, contexts=8, context_size=32)

    assert len(workload) == 8
    assert len(workload[0].history) == 32
    assert workload[0].candidate_room == "bedroom"
    assert workload[6].candidate_room == "bedroom"
    assert workload[0].timestamp > workload[0].history[-1].timestamp


def test_normalize_emissions_handles_missing_and_valid_values() -> None:
    missing = benchmark.normalize_emissions(None, 100)
    valid = benchmark.normalize_emissions(0.002, 1000)

    assert missing["kg_per_1m_inferences"] is None
    assert valid["kg_per_inference"] == 0.000002
    assert valid["kg_per_1000_inferences"] == 0.002
    assert valid["kg_per_1m_inferences"] == 2.0


def test_scenario_metrics_include_latency_and_throughput() -> None:
    metrics = benchmark.scenario_metrics(
        scenario="pet_filter",
        iterations=4,
        calls_per_iteration=1,
        wall_time_seconds=2.0,
        process_time_seconds=1.5,
        latencies_seconds=[0.1, 0.2, 0.3, 0.4],
        emissions_kg=0.004,
        codecarbon_file="codecarbon_pet_filter.csv",
    )

    assert metrics["inference_calls_total"] == 4
    assert metrics["inferences_per_second"] == 2.0
    assert metrics["latency_ms"]["mean_per_iteration"] == 250.0
    assert metrics["codecarbon"]["kg_per_1m_inferences"] == 1000.0


class FakeModel:
    def __init__(self) -> None:
        self.adjacency_neighbors = benchmark.synthetic_adjacency()
        self.relative_occupancy_device = "cpu"
        self.pet_filter_device = "cpu"
        self.relative_occupancy_info = {"artifact_id": "fake-relative"}
        self.pet_filter_info = {"artifact_id": "fake-pet"}

    def predict_occupancy_state(self, history, timestamp):
        return {
            "rooms": [history[-1].room],
            "people_count": 1,
            "confidence": 0.9,
        }

    def predict_human_motion(self, history, candidate_room, timestamp, adjacency):
        return {
            "human_probability": 0.8,
            "accepted": True,
            "strategy": "supervised_transformer",
        }


def test_main_writes_outputs_without_codecarbon(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "load_benchmark_model", FakeModel)
    monkeypatch.setattr(
        benchmark,
        "device_metadata",
        lambda model: {
            "torch_version": "test",
            "cuda_available": False,
            "cuda_device_count": 0,
            "relative_occupancy_device": "cpu",
            "pet_filter_device": "cpu",
            "relative_occupancy_artifact": "fake-relative",
            "pet_filter_artifact": "fake-pet",
        },
    )

    result = benchmark.main(
        [
            "--iterations",
            "2",
            "--warmup",
            "1",
            "--min-duration-seconds",
            "0",
            "--output-dir",
            str(tmp_path),
            "--no-codecarbon",
        ]
    )

    assert result == 0
    payload = json.loads(
        (tmp_path / "sostenibilidad_inferencias.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["codecarbon"]["enabled"] is False
    assert [item["scenario"] for item in payload["results"]] == [
        "relative_occupancy",
        "pet_filter",
        "combined_pipeline",
    ]
    assert (tmp_path / "sostenibilidad_inferencias.md").exists()


class SustainabilityArtifactLoadTest(unittest.TestCase):
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch no instalado")
    def test_real_packaged_artifacts_load_for_benchmark(self) -> None:
        model = AIAdjacencyModel()
        relative = model.load_packaged_relative_occupancy()
        pet = model.load_packaged_pet_filter()
        benchmark.configure_model_layout(model)

        self.assertTrue(relative["loaded"])
        self.assertTrue(pet["loaded"])
        self.assertTrue(model.ready)
        self.assertIn("foyer", model.adjacency_neighbors)
