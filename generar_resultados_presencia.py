"""Calcula métricas históricas de presencia desde CSV exportados.

El script permite dos usos:

1. Procesar CSV externos con estructura equivalente a un historial de estados.
2. Reproducir las evidencias privadas del informe solo cuando se active
   explícitamente el modo ``--use-local-study``.

El formato recomendado es una tabla de eventos con columnas equivalentes a:

* ``entity_id``: entidad de Home Assistant, nombre visible o identificador
  estable definido por quien reproduce el experimento.
* ``state``: estado observado.
* ``last_changed``: timestamp del cambio de estado.

También se acepta un formato ancho con una columna temporal y una columna por
entidad. En ese caso, cada celda no vacía se interpreta como el estado vigente
de esa entidad en el timestamp de la fila.

Uso genérico:

    python generar_resultados_presencia.py \
        --input mi_historial.csv \
        --count-inferred sensor.inferencia_de_presencia_2 \
        --count-reference sensor.num_in_house \
        --room binary_sensor.inferencia_de_presencia_occupancy_6=Kitchen \
        --confirmation-reference input_boolean.kitchen_occupied="Kitchen occ."

Uso con medición de rendimiento y emisiones:

    python generar_resultados_presencia.py \
        --input mi_historial.csv \
        --track-emissions \
        --offline-emissions-country CHL

Salidas principales:

* ``metricas_presencia.json``: métricas, metadatos de entrada, rendimiento y,
  si se activa, medición de CodeCarbon.
* ``rendimiento_metricas_presencia.json``: resumen del tiempo de ejecución,
  filas procesadas y emisiones estimadas.
* figuras PNG, cuando Pillow está instalado y no se usa ``--no-figures``.

Las métricas se ponderan por duración. Cada estado se interpreta como vigente
hasta el siguiente cambio. Los estados nulos se excluyen hasta encontrar la
primera observación válida de cada serie.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time as perf_time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "InformeMemoria2026" / "datos"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent

DEFAULT_COUNT_INFERRED = "sensor.inferencia_de_presencia_2"
DEFAULT_COUNT_REFERENCE = "sensor.num_in_house"
GENERAL_OCCUPANCY = "binary_sensor.inferencia_de_presencia_occupancy"

DEFAULT_INFERRED_ROOMS = {
    "binary_sensor.inferencia_de_presencia_occupancy_2": "Bedroom",
    "binary_sensor.inferencia_de_presencia_occupancy_3": "Entertainment",
    "binary_sensor.inferencia_de_presencia_occupancy_4": "Foyer",
    "binary_sensor.inferencia_de_presencia_occupancy_5": "Guest",
    "binary_sensor.inferencia_de_presencia_occupancy_6": "Kitchen",
    "binary_sensor.inferencia_de_presencia_occupancy_7": "Living",
    "binary_sensor.inferencia_de_presencia_occupancy_8": "Office",
    "binary_sensor.inferencia_de_presencia_occupancy_9": "Study/Sitting",
}

DEFAULT_CONFIRMATION_REFERENCES = {
    "input_boolean.kitchen_occupied": "Kitchen occ.",
    "input_boolean.office_occupied": "Office occ.",
    "input_boolean.study_occupied": "Study occ.",
    "input_boolean.tvroom_occupied": "TV occ.",
    "binary_sensor.hall_person_occupancy": "Hall camera",
    "binary_sensor.sittingroom_person_occupancy": "Sitting camera",
    "binary_sensor.chair_occupied": "Chair",
}

DEFAULT_MOTION_REFERENCES = {
    "binary_sensor.bedroom_sensor_motion": "Bedroom motion",
    "binary_sensor.entertainment_room_sensor_motion": "Entertainment motion",
    "binary_sensor.foyer_motion_motion": "Foyer motion",
    "binary_sensor.kitchen_sensor_motion": "Kitchen motion",
    "binary_sensor.office_sensor_motion": "Office motion",
}

DEFAULT_DIRECT_COMPARISONS = {
    "Bedroom": ["binary_sensor.bedroom_sensor_motion"],
    "Entertainment": [
        "binary_sensor.entertainment_room_sensor_motion",
        "input_boolean.tvroom_occupied",
    ],
    "Foyer": [
        "binary_sensor.hall_person_occupancy",
        "binary_sensor.foyer_motion_motion",
    ],
    "Kitchen": [
        "input_boolean.kitchen_occupied",
        "binary_sensor.kitchen_sensor_motion",
    ],
    "Office": [
        "input_boolean.office_occupied",
        "binary_sensor.office_sensor_motion",
    ],
    "Study/Sitting": [
        "input_boolean.study_occupied",
        "binary_sensor.sittingroom_person_occupancy",
        "binary_sensor.chair_occupied",
    ],
}

NULL_STATES = {"", "none", "null", "nan", "unknown", "unavailable"}
TIMESTAMP_COLUMNS = ("last_changed", "last_updated", "time", "timestamp", "datetime", "date")
ENTITY_COLUMNS = ("entity_id", "entity", "entity_name", "friendly_name", "name")
STATE_COLUMNS = ("state", "value", "new_state", "status")


@dataclass(frozen=True)
class History:
    key: str
    label: str
    csv_path: Path
    expected_rows: int | None = None


def default_histories() -> list[History]:
    return [
        History(
            key="historial_anterior",
            label="17--18 jun. sin confirmaciones",
            csv_path=DATA_DIR / "presenceJunio17.csv",
            expected_rows=2062,
        ),
        History(
            key="historial_extenso",
            label="18--25 jun. con confirmaciones",
            csv_path=DATA_DIR / "history17a25Junio.csv",
            expected_rows=43359,
        ),
    ]


def parse_timestamp(value: Any, naive_tz: ZoneInfo | timezone = timezone.utc) -> datetime:
    text = str(value).strip()
    if not text:
        raise ValueError("timestamp vacío")
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
            try:
                parsed = datetime.strptime(text, pattern)
                break
            except ValueError:
                continue
        else:
            raise
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=naive_tz)
    return parsed.astimezone(timezone.utc)


def local_iso(value: datetime, tz: ZoneInfo) -> str:
    return value.astimezone(tz).isoformat(timespec="seconds")


def normalize_count(value: Any) -> float | None:
    try:
        text = str(value).strip().lower()
        if text in NULL_STATES:
            return None
        return float(text.replace(",", "."))
    except (TypeError, ValueError):
        return None


def normalize_binary(value: Any) -> int | None:
    text = str(value).strip().lower()
    if text in NULL_STATES:
        return None
    return {
        "off": 0,
        "false": 0,
        "0": 0,
        "closed": 0,
        "clear": 0,
        "idle": 0,
        "no": 0,
        "on": 1,
        "true": 1,
        "1": 1,
        "open": 1,
        "detected": 1,
        "active": 1,
        "yes": 1,
    }.get(text)


def choose_column(fieldnames: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {name.strip().lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def load_history(
    history: History,
    *,
    naive_tz: ZoneInfo | timezone = timezone.utc,
) -> tuple[list[dict[str, str]], dict[str, list[tuple[datetime, str]]], dict[str, str]]:
    with history.csv_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if history.expected_rows is not None and len(rows) != history.expected_rows:
        raise AssertionError(
            f"{history.csv_path.name}: se esperaban {history.expected_rows} filas y hay {len(rows)}"
        )
    if not rows:
        raise AssertionError(f"{history.csv_path} no contiene filas")

    timestamp_col = choose_column(fieldnames, TIMESTAMP_COLUMNS)
    if timestamp_col is None:
        raise AssertionError(
            f"{history.csv_path.name}: no se encontró columna temporal. "
            f"Use una de: {', '.join(TIMESTAMP_COLUMNS)}"
        )

    entity_col = choose_column(fieldnames, ENTITY_COLUMNS)
    state_col = choose_column(fieldnames, STATE_COLUMNS)
    events: dict[str, list[tuple[datetime, str]]] = defaultdict(list)

    if entity_col and state_col:
        for row in rows:
            entity = str(row.get(entity_col, "")).strip()
            state = str(row.get(state_col, "")).strip()
            if not entity:
                continue
            events[entity].append((parse_timestamp(row[timestamp_col], naive_tz), state))
        input_format = "long"
    else:
        entity_columns = [name for name in fieldnames if name != timestamp_col]
        if not entity_columns:
            raise AssertionError(
                f"{history.csv_path.name}: el formato ancho requiere una columna temporal y columnas de entidades"
            )
        for row in rows:
            timestamp = parse_timestamp(row[timestamp_col], naive_tz)
            for entity in entity_columns:
                state = str(row.get(entity, "")).strip()
                if state and state.lower() not in NULL_STATES:
                    events[entity].append((timestamp, state))
        input_format = "wide"

    for values in events.values():
        values.sort(key=lambda item: item[0])
    return rows, events, {"timestamp_column": timestamp_col, "format": input_format}


def first_valid_timestamp(
    events_by_entity: dict[str, list[tuple[datetime, str]]],
    entity: str,
    normalizer: Callable[[Any], object | None],
) -> datetime | None:
    for timestamp, state in events_by_entity.get(entity, []):
        if normalizer(state) is not None:
            return timestamp
    return None


def generic_bounds(events_by_entity: dict[str, list[tuple[datetime, str]]]) -> tuple[datetime, datetime]:
    timestamps = [timestamp for values in events_by_entity.values() for timestamp, _ in values]
    if not timestamps:
        raise AssertionError("No hay eventos válidos para analizar")
    return min(timestamps), max(timestamps)


def step_value(events: list[tuple[datetime, str]], instant: datetime, normalizer: Callable[[Any], object | None]):
    value = None
    for timestamp, state in events:
        if timestamp > instant:
            break
        normalized = normalizer(state)
        if normalized is not None:
            value = normalized
    return value


def event_times(
    events_by_entity: dict[str, list[tuple[datetime, str]]],
    entities: list[str],
    start: datetime,
    end: datetime,
) -> list[datetime]:
    points = {start, end}
    for entity in entities:
        points.update(timestamp for timestamp, _ in events_by_entity.get(entity, []) if start < timestamp < end)
    return sorted(points)


def paired_intervals(
    events_by_entity: dict[str, list[tuple[datetime, str]]],
    inferred_id: str,
    inferred_normalizer: Callable[[Any], object | None],
    actual_id: str,
    actual_normalizer: Callable[[Any], object | None],
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    """Reconstruye intervalos de dos series usando cursores lineales."""
    if inferred_id not in events_by_entity or actual_id not in events_by_entity:
        return []

    points = event_times(events_by_entity, [inferred_id, actual_id], start, end)
    inferred_events = events_by_entity[inferred_id]
    actual_events = events_by_entity[actual_id]
    inferred_index = 0
    actual_index = 0
    inferred_value = None
    actual_value = None
    segments = []

    for left, right in zip(points, points[1:]):
        while inferred_index < len(inferred_events) and inferred_events[inferred_index][0] <= left:
            candidate = inferred_normalizer(inferred_events[inferred_index][1])
            if candidate is not None:
                inferred_value = candidate
            inferred_index += 1
        while actual_index < len(actual_events) and actual_events[actual_index][0] <= left:
            candidate = actual_normalizer(actual_events[actual_index][1])
            if candidate is not None:
                actual_value = candidate
            actual_index += 1
        if inferred_value is None or actual_value is None:
            continue
        segments.append({"start": left, "end": right, "actual": actual_value, "inferred": inferred_value})

    return segments


def split_at_local_midnight(start: datetime, end: datetime, tz: ZoneInfo):
    cursor = start
    while cursor < end:
        local = cursor.astimezone(tz)
        next_date = local.date() + timedelta(days=1)
        midnight = datetime.combine(next_date, time.min, tz).astimezone(timezone.utc)
        interval_end = min(end, midnight)
        yield cursor, interval_end, local.date().isoformat()
        cursor = interval_end


def build_count_segments(
    events_by_entity: dict[str, list[tuple[datetime, str]]],
    count_reference: str,
    count_inferred: str,
    start: datetime,
    end: datetime,
    tz: ZoneInfo,
) -> list[dict[str, Any]]:
    segments = []
    raw_segments = paired_intervals(
        events_by_entity,
        count_inferred,
        normalize_count,
        count_reference,
        normalize_count,
        start,
        end,
    )
    for item in raw_segments:
        for part_start, part_end, day in split_at_local_midnight(item["start"], item["end"], tz):
            segments.append(
                {
                    "start": part_start,
                    "end": part_end,
                    "day": day,
                    "actual": item["actual"],
                    "inferred": item["inferred"],
                }
            )
    return segments


def build_binary_segments(
    events_by_entity: dict[str, list[tuple[datetime, str]]],
    inferred_id: str,
    reference_id: str,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]] | None:
    if inferred_id not in events_by_entity or reference_id not in events_by_entity:
        return None
    return paired_intervals(
        events_by_entity,
        inferred_id,
        normalize_binary,
        reference_id,
        normalize_binary,
        start,
        end,
    )


def count_metrics(segments: list[dict[str, Any]]) -> dict[str, float] | None:
    if not segments:
        return None
    total = sum((item["end"] - item["start"]).total_seconds() for item in segments)
    if total <= 0:
        return None

    def weighted(function: Callable[[dict[str, Any]], float | bool]) -> float:
        return sum((item["end"] - item["start"]).total_seconds() * float(function(item)) for item in segments)

    errors = [item["inferred"] - item["actual"] for item in segments]
    return {
        "duration_seconds": total,
        "duration_hours": total / 3600,
        "mae": weighted(lambda item: abs(item["inferred"] - item["actual"])) / total,
        "rmse": math.sqrt(weighted(lambda item: (item["inferred"] - item["actual"]) ** 2) / total),
        "bias": weighted(lambda item: item["inferred"] - item["actual"]) / total,
        "max_error": max(abs(error) for error in errors),
        "exact": weighted(lambda item: item["inferred"] == item["actual"]) / total,
        "within_one": weighted(lambda item: abs(item["inferred"] - item["actual"]) <= 1) / total,
        "over": weighted(lambda item: item["inferred"] > item["actual"]) / total,
        "under": weighted(lambda item: item["inferred"] < item["actual"]) / total,
    }


def binary_metrics(segments: list[dict[str, Any]] | None) -> dict[str, float | None] | None:
    if not segments:
        return None
    totals = {"tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
    inferred_seconds = 0.0
    reference_seconds = 0.0
    for item in segments:
        duration = (item["end"] - item["start"]).total_seconds()
        inferred_seconds += duration if item["inferred"] == 1 else 0.0
        reference_seconds += duration if item["actual"] == 1 else 0.0
        key = {
            (1, 1): "tp",
            (0, 0): "tn",
            (1, 0): "fp",
            (0, 1): "fn",
        }[(item["inferred"], item["actual"])]
        totals[key] += duration
    tp, tn, fp, fn = (totals[key] for key in ("tp", "tn", "fp", "fn"))

    def ratio(numerator: float, denominator: float) -> float | None:
        return numerator / denominator if denominator else None

    totals.update(
        duration_seconds=tp + tn + fp + fn,
        inferred_hours=inferred_seconds / 3600,
        reference_hours=reference_seconds / 3600,
        accuracy=ratio(tp + tn, tp + tn + fp + fn),
        precision=ratio(tp, tp + fp),
        recall=ratio(tp, tp + fn),
        specificity=ratio(tn, tn + fp),
        f1=ratio(2 * tp, 2 * tp + fp + fn) or 0.0,
        iou=ratio(tp, tp + fp + fn),
    )
    return totals


def safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text.strip("_") or "historial"


def portable_path(path: Path) -> str:
    """Evita guardar rutas locales personales en los artefactos exportados."""
    return path.name


def source_label(history: History) -> str:
    """Etiqueta una fuente privada sin exponer la ruta ni el nombre del CSV."""
    try:
        history.csv_path.resolve().relative_to(DATA_DIR.resolve())
        return history.key
    except ValueError:
        return portable_path(history.csv_path)


def load_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("Para generar figuras instale Pillow o ejecute con --no-figures") from exc
    return Image, ImageDraw, ImageFont


def font(size: int, bold: bool = False):
    _, _, ImageFont = load_pillow()
    filename = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(filename, size)
    except OSError:
        return ImageFont.load_default(size=size)


def draw_count_chart(
    segments: list[dict[str, Any]],
    start: datetime,
    end: datetime,
    title: str,
    output_path: Path,
    tz: ZoneInfo,
) -> None:
    if not segments:
        return
    Image, ImageDraw, _ = load_pillow()
    width, height = 1800, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 150, 120, 1740, 735
    navy, orange, grid, text = "#17324D", "#E67E22", "#D7E0E8", "#263746"
    draw.text((left, 35), title, fill=text, font=font(36, True))
    draw.text((left, 82), f"Estados vigentes hasta el siguiente cambio, hora local {tz.key}", fill="#536878", font=font(23))
    maximum = max(4, int(max(max(item["actual"], item["inferred"]) for item in segments)))

    def x_position(timestamp: datetime) -> float:
        return left + (timestamp - start).total_seconds() / (end - start).total_seconds() * (right - left)

    def y_position(value: float) -> float:
        return bottom - value / maximum * (bottom - top)

    for value in range(maximum + 1):
        y = y_position(value)
        draw.line((left, y, right, y), fill=grid, width=2)
        draw.text((105, y - 14), str(value), fill=text, font=font(23))
    draw.line((left, top, left, bottom), fill=text, width=3)
    draw.line((left, bottom, right, bottom), fill=text, width=3)
    draw.text((25, 385), "Personas", fill=text, font=font(25, True))

    tick_count = 7
    for index in range(tick_count):
        timestamp = start + (end - start) * index / (tick_count - 1)
        x = x_position(timestamp)
        draw.line((x, bottom, x, bottom + 10), fill=text, width=2)
        line1, line2 = timestamp.astimezone(tz).strftime("%d-%m\n%H:%M").split("\n")
        draw.text((x - 31, bottom + 18), line1, fill=text, font=font(20))
        draw.text((x - 29, bottom + 43), line2, fill=text, font=font(20))

    for field, color in (("actual", navy), ("inferred", orange)):
        points = []
        for item in segments:
            x1, x2 = x_position(item["start"]), x_position(item["end"])
            y = y_position(item[field])
            if points and points[-1][1] != y:
                points.append((x1, points[-1][1]))
                points.append((x1, y))
            elif not points:
                points.append((x1, y))
            points.append((x2, y))
        draw.line(points, fill=color, width=5)

    legend_y = 820
    for legend_x, color, label in ((left, navy, "Conteo real"), (left + 300, orange, "Conteo inferido")):
        draw.line((legend_x, legend_y, legend_x + 65, legend_y), fill=color, width=7)
        draw.text((legend_x + 80, legend_y - 17), label, fill=text, font=font(24))
    image.save(output_path, dpi=(180, 180))


def heat_color(value: float):
    low, middle, high = (197, 73, 73), (245, 202, 92), (60, 141, 90)
    if value <= 0.5:
        fraction, start, end = value / 0.5, low, middle
    else:
        fraction, start, end = (value - 0.5) / 0.5, middle, high
    return tuple(round(start[i] + fraction * (end[i] - start[i])) for i in range(3))


def draw_heatmap(
    title: str,
    subtitle: str,
    row_labels: list[str],
    column_labels: list[str],
    matrix: list[list[float | None]],
    output_path: Path,
) -> None:
    if not row_labels or not column_labels:
        return
    Image, ImageDraw, _ = load_pillow()
    cell_w, cell_h = 170, 70
    left, top = 300, 155
    width = left + cell_w * len(column_labels) + 90
    height = top + cell_h * len(row_labels) + 120
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    text = "#263746"
    draw.text((50, 35), title, fill=text, font=font(30, True))
    draw.text((50, 78), subtitle, fill="#536878", font=font(19))

    for column, label in enumerate(column_labels):
        x = left + column * cell_w
        draw.text((x + 7, top - 55), label, fill=text, font=font(15, True))

    for row_index, label in enumerate(row_labels):
        y = top + row_index * cell_h
        draw.text((50, y + 20), label, fill=text, font=font(18, True))
        for column, value in enumerate(matrix[row_index]):
            x = left + column * cell_w
            if value is None:
                fill = "#EEF2F5"
                percentage = "s/d"
                label_color = "#6B7C8D"
            else:
                fill = heat_color(value)
                percentage = f"{value * 100:.1f}%".replace(".", ",")
                label_color = "white" if value < 0.25 or value > 0.62 else text
            draw.rounded_rectangle((x, y, x + cell_w - 8, y + cell_h - 8), radius=8, fill=fill, outline="white", width=3)
            bbox = draw.textbbox((0, 0), percentage, font=font(17, True))
            draw.text((x + (cell_w - 8 - (bbox[2] - bbox[0])) / 2, y + 20), percentage, fill=label_color, font=font(17, True))
    image.save(output_path, dpi=(180, 180))


def parse_mapping(values: list[str] | None, defaults: dict[str, str]) -> dict[str, str]:
    mapping = dict(defaults)
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"Mapeo inválido '{item}'. Use entidad=Etiqueta.")
        key, value = item.split("=", 1)
        mapping[key.strip()] = value.strip().strip('"')
    return mapping


def parse_direct(values: list[str] | None) -> dict[str, list[str]]:
    direct = {room: list(entities) for room, entities in DEFAULT_DIRECT_COMPARISONS.items()}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"Comparación directa inválida '{item}'. Use Habitación=entidad1,entidad2.")
        room, entities = item.split("=", 1)
        direct[room.strip()] = [entity.strip() for entity in entities.split(",") if entity.strip()]
    return direct


def parse_expected_rows(values: list[str] | None) -> dict[str, int]:
    expected = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"Valor inválido '{item}'. Use clave=filas.")
        key, value = item.split("=", 1)
        expected[key.strip()] = int(value)
    return expected


def parse_inputs(values: list[str] | None, expected_rows: dict[str, int], use_local_study: bool) -> list[History]:
    if use_local_study:
        histories = default_histories()
        missing = [history.csv_path for history in histories if not history.csv_path.exists()]
        if missing:
            raise SystemExit("No se encontraron los CSV privados del estudio local.")
        return histories
    if not values:
        raise SystemExit("Debe indicar al menos un CSV con --input. Los historiales del estudio no se entregan por defecto.")

    histories = []
    for index, item in enumerate(values, start=1):
        if "=" in item:
            key, raw_path = item.split("=", 1)
        else:
            raw_path = item
            key = f"historial_{index}"
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"No existe el CSV indicado: {path}")
        histories.append(
            History(
                key=safe_name(key),
                label=safe_name(key).replace("_", " "),
                csv_path=path,
                expected_rows=expected_rows.get(safe_name(key)),
            )
        )
    return histories


def evaluate_history(
    history: History,
    *,
    count_inferred: str,
    count_reference: str,
    inferred_rooms: dict[str, str],
    confirmation_references: dict[str, str],
    motion_references: dict[str, str],
    direct_comparisons: dict[str, list[str]],
    tz: ZoneInfo,
    naive_tz: ZoneInfo | timezone,
) -> tuple[dict[str, Any], list[dict[str, Any]], tuple[datetime, datetime]]:
    rows, events, input_metadata = load_history(history, naive_tz=naive_tz)
    start, end = generic_bounds(events)
    count_start = first_valid_timestamp(events, count_inferred, normalize_count)
    if count_start is not None:
        start = count_start

    count_segments = build_count_segments(events, count_reference, count_inferred, start, end, tz)
    overall = count_metrics(count_segments)
    daily = {}
    if count_segments:
        daily = {
            day: count_metrics([item for item in count_segments if item["day"] == day])
            for day in sorted({item["day"] for item in count_segments})
        }

    confirmation_refs = {key: value for key, value in confirmation_references.items() if key in events}
    motion_refs = {key: value for key, value in motion_references.items() if key in events}
    all_refs = confirmation_refs | motion_refs
    present_rooms = {entity: room for entity, room in inferred_rooms.items() if entity in events}
    missing_rooms = {entity: room for entity, room in inferred_rooms.items() if entity not in events}

    matrices = {}
    for matrix_key, refs in (("confirmation", confirmation_refs), ("with_motion", all_refs)):
        rows_matrix = []
        values = {}
        for inferred_id, room in present_rooms.items():
            row = []
            values[room] = {}
            for reference_id, reference_label in refs.items():
                metrics = binary_metrics(build_binary_segments(events, inferred_id, reference_id, start, end))
                values[room][reference_label] = metrics
                row.append(None if metrics is None else metrics["f1"])
            rows_matrix.append(row)
        matrices[matrix_key] = {
            "row_labels": list(present_rooms.values()),
            "column_labels": list(refs.values()),
            "values": values,
            "f1_matrix": rows_matrix,
        }

    reference_labels = confirmation_references | motion_references
    direct = {}
    for inferred_id, room in present_rooms.items():
        direct[room] = {}
        for reference_id in direct_comparisons.get(room, []):
            metrics = binary_metrics(build_binary_segments(events, inferred_id, reference_id, start, end))
            if metrics is not None:
                direct[room][reference_labels.get(reference_id, reference_id)] = metrics

    result = {
        "source_file": source_label(history),
        "source_rows": len(rows),
        "entity_count": len(events),
        "input_format": input_metadata["format"],
        "timestamp_column": input_metadata["timestamp_column"],
        "timezone": tz.key,
        "analysis_start_local": local_iso(start, tz),
        "analysis_end_local": local_iso(end, tz),
        "count_entities": {
            "inferred": count_inferred,
            "reference": count_reference,
            "available": bool(count_segments),
        },
        "overall_count": overall,
        "daily_count": daily,
        "present_rooms": present_rooms,
        "missing_rooms": missing_rooms,
        "general_occupancy_present": GENERAL_OCCUPANCY in events,
        "matrices": matrices,
        "direct_comparisons": direct,
    }
    return result, count_segments, (start, end)


def assert_expected(results: dict[str, Any]) -> None:
    if "historial_anterior" not in results or "historial_extenso" not in results:
        return
    short = results["historial_anterior"].get("overall_count")
    long = results["historial_extenso"].get("overall_count")
    if not short or not long:
        return
    expected = {
        "short_duration": (short["duration_seconds"], 90022.169, 0.002),
        "short_mae": (short["mae"], 0.5923016, 5e-6),
        "short_rmse": (short["rmse"], 0.7746900, 5e-6),
        "short_bias": (short["bias"], -0.3708541, 5e-6),
        "long_hours": (long["duration_hours"], 167.1903731, 5e-6),
        "long_mae": (long["mae"], 0.7690075, 5e-6),
        "long_rmse": (long["rmse"], 0.8861730, 5e-6),
        "long_bias": (long["bias"], -0.3744710, 5e-6),
    }
    for key, (actual, expected_value, tolerance) in expected.items():
        if not math.isclose(actual, expected_value, rel_tol=0, abs_tol=tolerance):
            raise AssertionError(f"Métrica divergente {key}: {actual} != {expected_value}")


def create_tracker(enabled: bool, output_dir: Path, country: str | None):
    if not enabled:
        return None
    try:
        if country:
            from codecarbon import OfflineEmissionsTracker

            return OfflineEmissionsTracker(
                country_iso_code=country,
                output_dir=str(output_dir),
                output_file="codecarbon_metricas_presencia.csv",
                project_name="metricas-presencia",
                log_level="error",
            )
        from codecarbon import EmissionsTracker

        return EmissionsTracker(
            output_dir=str(output_dir),
            output_file="codecarbon_metricas_presencia.csv",
            project_name="metricas-presencia",
            log_level="error",
        )
    except ImportError as exc:
        raise SystemExit("CodeCarbon no está instalado. Ejecute: pip install codecarbon") from exc


def write_template(path: Path) -> None:
    template = {
        "description": "Ejemplo de parámetros equivalentes para ejecutar el script con CSV propio.",
        "command": [
            "python",
            "generar_resultados_presencia.py",
            "--input",
            "mi_historial.csv",
            "--count-inferred",
            "sensor.inferencia_de_presencia_2",
            "--count-reference",
            "sensor.num_in_house",
            "--room",
            "binary_sensor.inferencia_de_presencia_occupancy_6=Kitchen",
            "--confirmation-reference",
            "input_boolean.kitchen_occupied=Kitchen occ.",
            "--motion-reference",
            "binary_sensor.kitchen_sensor_motion=Kitchen motion",
        ],
        "accepted_csv_formats": {
            "long": ["entity_id", "state", "last_changed"],
            "wide": ["timestamp", "sensor.a", "binary_sensor.b"],
        },
    }
    path.write_text(json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calcula métricas temporales de presencia desde CSV.")
    parser.add_argument("--input", action="append", help="CSV de entrada. Formato: ruta.csv o clave=ruta.csv.")
    parser.add_argument("--use-local-study", action="store_true", help="Usa los CSV privados del estudio local si están disponibles.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directorio de salida.")
    parser.add_argument("--metrics-json", default="metricas_presencia.json", help="Nombre del JSON de métricas.")
    parser.add_argument("--timezone", default="America/Santiago", help="Zona horaria local para reportes diarios.")
    parser.add_argument("--naive-timezone", default="UTC", help="Zona horaria asumida si el CSV no trae offset.")
    parser.add_argument("--count-inferred", default=DEFAULT_COUNT_INFERRED, help="Entidad de conteo inferido.")
    parser.add_argument("--count-reference", default=DEFAULT_COUNT_REFERENCE, help="Entidad de conteo real o referencia.")
    parser.add_argument("--room", action="append", help="Entidad binaria inferida y etiqueta: entidad=Habitación.")
    parser.add_argument("--confirmation-reference", action="append", help="Referencia de ocupación/cámara/silla: entidad=Etiqueta.")
    parser.add_argument("--motion-reference", action="append", help="Referencia de movimiento: entidad=Etiqueta.")
    parser.add_argument("--direct-comparison", action="append", help="Comparación directa: Habitación=entidad1,entidad2.")
    parser.add_argument("--expected-rows", action="append", help="Validación opcional de filas: clave=numero.")
    parser.add_argument("--validate-known", action="store_true", help="Activa aserciones de métricas conocidas del estudio.")
    parser.add_argument("--no-figures", action="store_true", help="Calcula solo JSON, sin generar PNG.")
    parser.add_argument("--track-emissions", action="store_true", help="Mide tiempo y emisiones con CodeCarbon.")
    parser.add_argument("--offline-emissions-country", help="Código ISO de país para CodeCarbon offline, por ejemplo CHL.")
    parser.add_argument("--write-template", help="Escribe un JSON de ejemplo de ejecución y termina.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.write_template:
        write_template(Path(args.write_template))
        return 0

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tz = ZoneInfo(args.timezone)
    naive_tz: ZoneInfo | timezone = timezone.utc if args.naive_timezone.upper() == "UTC" else ZoneInfo(args.naive_timezone)
    expected_rows = parse_expected_rows(args.expected_rows)
    histories = parse_inputs(args.input, expected_rows, args.use_local_study)

    inferred_rooms = parse_mapping(args.room, DEFAULT_INFERRED_ROOMS)
    confirmation_references = parse_mapping(args.confirmation_reference, DEFAULT_CONFIRMATION_REFERENCES)
    motion_references = parse_mapping(args.motion_reference, DEFAULT_MOTION_REFERENCES)
    direct_comparisons = parse_direct(args.direct_comparison)

    tracker = create_tracker(args.track_emissions, output_dir, args.offline_emissions_country)
    if tracker:
        tracker.start()
    start_wall = perf_time.perf_counter()
    start_cpu = perf_time.process_time()
    emissions_kg = None
    generated_files: list[str] = []

    try:
        results: dict[str, Any] = {}
        count_payload: dict[str, tuple[list[dict[str, Any]], tuple[datetime, datetime], str]] = {}
        total_rows = 0
        for history in histories:
            result, count_segments, bounds = evaluate_history(
                history,
                count_inferred=args.count_inferred,
                count_reference=args.count_reference,
                inferred_rooms=inferred_rooms,
                confirmation_references=confirmation_references,
                motion_references=motion_references,
                direct_comparisons=direct_comparisons,
                tz=tz,
                naive_tz=naive_tz,
            )
            results[history.key] = result
            total_rows += result["source_rows"]
            count_payload[history.key] = (count_segments, bounds, history.label)

        if args.validate_known or args.use_local_study:
            assert_expected(results)

        if not args.no_figures:
            for key, (segments, (start, end), label) in count_payload.items():
                if not segments:
                    continue
                suffix = "dos_dias" if key == "historial_anterior" else safe_name(key)
                if key == "historial_extenso":
                    suffix = "historial_extenso"
                figure_path = output_dir / f"conteo_presencia_{suffix}.png"
                draw_count_chart(segments, start, end, f"Conteo real e inferido - {label}", figure_path, tz)
                generated_files.append(figure_path.name)

            if "historial_extenso" in results:
                heatmap_source_key = "historial_extenso"
            else:
                heatmap_source_key = next(iter(results))
            matrices = results[heatmap_source_key]["matrices"]
            confirmation_path = output_dir / "matriz_f1_ocupacion_confirmacion.png"
            draw_heatmap(
                "F1 entre habitaciones inferidas y confirmaciones",
                "Cámaras y ocupación directa, sin sensores de movimiento.",
                matrices["confirmation"]["row_labels"],
                matrices["confirmation"]["column_labels"],
                matrices["confirmation"]["f1_matrix"],
                confirmation_path,
            )
            generated_files.append(confirmation_path.name)

            motion_path = output_dir / "matriz_f1_ocupacion_movimiento.png"
            draw_heatmap(
                "F1 entre habitaciones inferidas, confirmaciones y movimiento",
                "Movimiento usado como proxy secundario, no como presencia sostenida.",
                matrices["with_motion"]["row_labels"],
                matrices["with_motion"]["column_labels"],
                matrices["with_motion"]["f1_matrix"],
                motion_path,
            )
            generated_files.append(motion_path.name)

        runtime = {
            "wall_time_seconds": perf_time.perf_counter() - start_wall,
            "process_time_seconds": perf_time.process_time() - start_cpu,
            "input_files": [source_label(history) for history in histories],
            "histories": len(histories),
            "rows_processed": total_rows,
            "figures_enabled": not args.no_figures,
            "codecarbon": {
                "enabled": bool(tracker),
                "country_iso_code": args.offline_emissions_country,
                "emissions_kg": None,
                "output_file": "codecarbon_metricas_presencia.csv" if tracker else None,
            },
        }
        payload = {"metadata": {"script": Path(__file__).name, "generated_files": generated_files}, "runtime": runtime, "results": results}
    finally:
        if tracker:
            emissions_kg = tracker.stop()

    if emissions_kg is not None:
        payload["runtime"]["codecarbon"]["emissions_kg"] = emissions_kg

    metrics_path = output_dir / args.metrics_json
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    runtime_path = output_dir / "rendimiento_metricas_presencia.json"
    runtime_path.write_text(json.dumps(payload["runtime"], indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
