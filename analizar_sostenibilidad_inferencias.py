from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time as perf_time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from inferencia_hub.ai import AIAdjacencyModel
from inferencia_hub.domain import EventRecord


DEFAULT_OUTPUT_DIR = Path("outputs") / "sostenibilidad_inferencias"
DEFAULT_COUNTRY = "CHL"
ROOMS = (
    "bedroom",
    "sittingroom",
    "entertainment_room",
    "foyer",
    "kitchen",
    "living",
)
EDGES = (
    ("bedroom", "sittingroom"),
    ("sittingroom", "entertainment_room"),
    ("entertainment_room", "foyer"),
    ("foyer", "kitchen"),
    ("foyer", "living"),
)


@dataclass(frozen=True)
class SyntheticWorkItem:
    history: tuple[EventRecord, ...]
    candidate_room: str
    timestamp: datetime


@dataclass(frozen=True)
class Scenario:
    name: str
    calls_per_iteration: int
    run: Callable[[int], Any]


def synthetic_adjacency() -> dict[str, list[str]]:
    adjacency = {room: [] for room in ROOMS}
    for left, right in EDGES:
        adjacency[left].append(right)
        adjacency[right].append(left)
    return {room: sorted(neighbors) for room, neighbors in adjacency.items()}


def synthetic_profile() -> dict[str, Any]:
    return {
        "id": "sostenibilidad_sintetica",
        "name": "Sostenibilidad sintética",
        "revision": 1,
        "fingerprint": "sostenibilidad-sintetica-v1",
        "rooms": [
            {"slug": room, "name": room.replace("_", " ").title()}
            for room in ROOMS
        ],
        "areas": [],
        "assignments": [],
        "edges": [list(edge) for edge in EDGES],
    }


def generate_synthetic_events(count: int = 96) -> list[EventRecord]:
    if count <= 0:
        return []
    start = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
    events: list[EventRecord] = []
    sensor_cycle = ("motion", "occupancy", "motion", "door")
    state_by_type = {
        "motion": "on",
        "occupancy": "occupied",
        "door": "open",
    }
    room_path = (
        "bedroom",
        "sittingroom",
        "entertainment_room",
        "foyer",
        "kitchen",
        "foyer",
        "living",
        "foyer",
        "entertainment_room",
        "sittingroom",
    )
    for index in range(count):
        room = room_path[index % len(room_path)]
        sensor_type = sensor_cycle[index % len(sensor_cycle)]
        events.append(
            EventRecord(
                timestamp=start + timedelta(seconds=index * 11),
                entity_id=f"binary_sensor.{room}_{sensor_type}_{index % 4}",
                state=state_by_type[sensor_type],
                sensor_type=sensor_type,
                room=room,
            )
        )
    return events


def build_workload(
    events: list[EventRecord],
    *,
    contexts: int = 32,
    context_size: int = 48,
) -> list[SyntheticWorkItem]:
    if len(events) < context_size:
        raise ValueError(
            f"Se requieren al menos {context_size} eventos sintéticos; hay {len(events)}."
        )
    items: list[SyntheticWorkItem] = []
    max_start = max(1, len(events) - context_size)
    for index in range(contexts):
        start = index % max_start
        history = tuple(events[start : start + context_size])
        candidate_room = ROOMS[index % len(ROOMS)]
        timestamp = history[-1].timestamp + timedelta(seconds=7)
        items.append(
            SyntheticWorkItem(
                history=history,
                candidate_room=candidate_room,
                timestamp=timestamp,
            )
        )
    return items


def configure_model_layout(model: AIAdjacencyModel) -> None:
    adjacency = synthetic_adjacency()
    model.rooms = sorted(adjacency)
    model.room_to_idx = {
        room: index for index, room in enumerate(model.rooms)
    }
    model.adjacency_neighbors = adjacency
    model.ready = True


def require_torch() -> Any:
    try:
        import torch
    except Exception as exc:
        raise SystemExit(
            "PyTorch no está instalado. Instale dependencias ML con "
            "`pip install -r inferencia_hub/requirements-ml.txt` o use la imagen "
            "con `INSTALL_ML=1`."
        ) from exc
    return torch


def load_benchmark_model() -> AIAdjacencyModel:
    require_torch()
    model = AIAdjacencyModel()
    pet = model.load_packaged_pet_filter()
    if not pet.get("loaded"):
        raise SystemExit(
            "No se pudo cargar pet_motion_transformer.pt: "
            f"{pet.get('reason', 'sin motivo informado')}"
        )
    relative = model.load_packaged_relative_occupancy()
    if not relative.get("loaded"):
        raise SystemExit(
            "No se pudo cargar relative_occupancy_transformer.pt: "
            f"{relative.get('reason', 'sin motivo informado')}"
        )
    configure_model_layout(model)
    return model


def create_tracker(
    *,
    enabled: bool,
    output_dir: Path,
    output_file: str,
    country: str,
):
    if not enabled:
        return None
    try:
        from codecarbon import OfflineEmissionsTracker
    except ImportError as exc:
        raise SystemExit(
            "CodeCarbon no está instalado. Ejecute: pip install codecarbon "
            "o use --no-codecarbon para una ejecución de diagnóstico."
        ) from exc
    return OfflineEmissionsTracker(
        country_iso_code=country,
        output_dir=str(output_dir),
        output_file=output_file,
        project_name="sostenibilidad-inferencias-pytorch",
        log_level="error",
    )


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil((len(ordered) * fraction) - 1)),
    )
    return ordered[index]


def normalize_emissions(
    emissions_kg: float | None,
    inference_calls: int,
) -> dict[str, float | None]:
    if emissions_kg is None or inference_calls <= 0:
        return {
            "kg_per_inference": None,
            "kg_per_1000_inferences": None,
            "kg_per_1m_inferences": None,
        }
    per_call = emissions_kg / inference_calls
    return {
        "kg_per_inference": per_call,
        "kg_per_1000_inferences": per_call * 1_000,
        "kg_per_1m_inferences": per_call * 1_000_000,
    }


def scenario_metrics(
    *,
    scenario: str,
    iterations: int,
    calls_per_iteration: int,
    wall_time_seconds: float,
    process_time_seconds: float,
    latencies_seconds: list[float],
    emissions_kg: float | None,
    codecarbon_file: str | None,
) -> dict[str, Any]:
    inference_calls = iterations * calls_per_iteration
    latency_avg = (
        statistics.fmean(latencies_seconds)
        if latencies_seconds
        else 0.0
    )
    normalized = normalize_emissions(emissions_kg, inference_calls)
    return {
        "scenario": scenario,
        "iterations": iterations,
        "calls_per_iteration": calls_per_iteration,
        "inference_calls_total": inference_calls,
        "wall_time_seconds": wall_time_seconds,
        "process_time_seconds": process_time_seconds,
        "iterations_per_second": (
            iterations / wall_time_seconds if wall_time_seconds > 0 else None
        ),
        "inferences_per_second": (
            inference_calls / wall_time_seconds
            if wall_time_seconds > 0
            else None
        ),
        "latency_ms": {
            "mean_per_iteration": latency_avg * 1000,
            "mean_per_inference": (
                latency_avg * 1000 / calls_per_iteration
                if calls_per_iteration
                else None
            ),
            "p50_per_iteration": (
                percentile(latencies_seconds, 0.50) or 0.0
            )
            * 1000,
            "p95_per_iteration": (
                percentile(latencies_seconds, 0.95) or 0.0
            )
            * 1000,
        },
        "codecarbon": {
            "emissions_kg": emissions_kg,
            "output_file": codecarbon_file,
            **normalized,
        },
    }


def run_scenario(
    scenario: Scenario,
    *,
    warmup: int,
    iterations: int,
    min_duration_seconds: float,
    output_dir: Path,
    country: str,
    codecarbon_enabled: bool,
) -> dict[str, Any]:
    for index in range(warmup):
        scenario.run(index)

    output_file = f"codecarbon_{scenario.name}.csv"
    tracker = create_tracker(
        enabled=codecarbon_enabled,
        output_dir=output_dir,
        output_file=output_file,
        country=country,
    )
    if tracker:
        tracker.start()

    completed = 0
    latencies: list[float] = []
    start_wall = perf_time.perf_counter()
    start_cpu = perf_time.process_time()
    emissions_kg = None
    try:
        while completed < iterations or (
            min_duration_seconds > 0
            and perf_time.perf_counter() - start_wall < min_duration_seconds
        ):
            item_start = perf_time.perf_counter()
            scenario.run(completed)
            latencies.append(perf_time.perf_counter() - item_start)
            completed += 1
    finally:
        wall_time_seconds = perf_time.perf_counter() - start_wall
        process_time_seconds = perf_time.process_time() - start_cpu
        if tracker:
            emissions_kg = tracker.stop()

    return scenario_metrics(
        scenario=scenario.name,
        iterations=completed,
        calls_per_iteration=scenario.calls_per_iteration,
        wall_time_seconds=wall_time_seconds,
        process_time_seconds=process_time_seconds,
        latencies_seconds=latencies,
        emissions_kg=emissions_kg,
        codecarbon_file=output_file if codecarbon_enabled else None,
    )


def build_scenarios(
    model: AIAdjacencyModel,
    workload: list[SyntheticWorkItem],
) -> list[Scenario]:
    if not workload:
        raise ValueError("La carga sintética no contiene casos de inferencia.")

    def item(index: int) -> SyntheticWorkItem:
        return workload[index % len(workload)]

    def relative(index: int) -> Any:
        current = item(index)
        return model.predict_occupancy_state(
            list(current.history),
            current.timestamp,
        )

    def pet_filter(index: int) -> Any:
        current = item(index)
        return model.predict_human_motion(
            list(current.history),
            current.candidate_room,
            current.timestamp,
            model.adjacency_neighbors,
        )

    def combined(index: int) -> tuple[Any, Any]:
        return relative(index), pet_filter(index)

    return [
        Scenario("relative_occupancy", 1, relative),
        Scenario("pet_filter", 1, pet_filter),
        Scenario("combined_pipeline", 2, combined),
    ]


def device_metadata(model: AIAdjacencyModel) -> dict[str, Any]:
    torch = require_torch()
    return {
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": (
            int(torch.cuda.device_count())
            if torch.cuda.is_available()
            else 0
        ),
        "relative_occupancy_device": str(model.relative_occupancy_device),
        "pet_filter_device": str(model.pet_filter_device),
        "relative_occupancy_artifact": model.relative_occupancy_info.get(
            "artifact_id"
        ),
        "pet_filter_artifact": model.pet_filter_info.get("artifact_id"),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Análisis de sostenibilidad de inferencias PyTorch",
        "",
        "## Entorno",
        "",
        f"- PyTorch: `{payload['device'].get('torch_version')}`",
        f"- CUDA disponible: `{payload['device'].get('cuda_available')}`",
        f"- Dispositivo ocupación relativa: `{payload['device'].get('relative_occupancy_device')}`",
        f"- Dispositivo filtro persona/mascota: `{payload['device'].get('pet_filter_device')}`",
        f"- CodeCarbon: `{payload['codecarbon']['enabled']}`",
        f"- País offline: `{payload['codecarbon']['country_iso_code']}`",
        "",
        "## Resultados",
        "",
        "| Escenario | Inferencias | s pared | inf/s | kgCO2e | kgCO2e / 1M inf |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in payload["results"]:
        carbon = result["codecarbon"]
        emissions = carbon.get("emissions_kg")
        per_million = carbon.get("kg_per_1m_inferences")
        lines.append(
            "| {scenario} | {calls} | {wall:.3f} | {throughput:.3f} | {emissions} | {per_million} |".format(
                scenario=result["scenario"],
                calls=result["inference_calls_total"],
                wall=result["wall_time_seconds"],
                throughput=result["inferences_per_second"] or 0.0,
                emissions=(
                    f"{emissions:.10f}" if emissions is not None else "n/a"
                ),
                per_million=(
                    f"{per_million:.10f}"
                    if per_million is not None
                    else "n/a"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Nota de interpretación",
            "",
            "CodeCarbon entrega una estimación dependiente del hardware, carga de fondo, configuración energética y disponibilidad de mediciones del sistema. Para comparar ejecuciones, use la misma máquina, el mismo modo de energía y la misma versión de dependencias.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "sostenibilidad_inferencias.json"
    md_path = output_dir / "sostenibilidad_inferencias.md"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(payload), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Mide con CodeCarbon el costo de inferencias PyTorch reales del backend."
        )
    )
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--min-duration-seconds", type=float, default=60.0)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        "--offline-emissions-country",
        default=DEFAULT_COUNTRY,
    )
    parser.add_argument(
        "--no-codecarbon",
        action="store_true",
        help="Ejecuta el benchmark sin medir emisiones; útil para tests.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.iterations <= 0:
        raise SystemExit("--iterations debe ser mayor que cero.")
    if args.warmup < 0:
        raise SystemExit("--warmup no puede ser negativo.")
    if args.min_duration_seconds < 0:
        raise SystemExit("--min-duration-seconds no puede ser negativo.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_benchmark_model()
    events = generate_synthetic_events()
    workload = build_workload(events)
    scenarios = build_scenarios(model, workload)

    results = [
        run_scenario(
            scenario,
            warmup=args.warmup,
            iterations=args.iterations,
            min_duration_seconds=args.min_duration_seconds,
            output_dir=output_dir,
            country=args.offline_emissions_country,
            codecarbon_enabled=not args.no_codecarbon,
        )
        for scenario in scenarios
    ]
    payload = {
        "metadata": {
            "script": Path(__file__).name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "workload": {
                "type": "synthetic_stable",
                "profile": synthetic_profile(),
                "events": len(events),
                "contexts": len(workload),
            },
        },
        "device": device_metadata(model),
        "codecarbon": {
            "enabled": not args.no_codecarbon,
            "country_iso_code": args.offline_emissions_country,
        },
        "config": {
            "iterations": args.iterations,
            "warmup": args.warmup,
            "min_duration_seconds": args.min_duration_seconds,
        },
        "results": results,
    }
    write_outputs(payload, output_dir)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
