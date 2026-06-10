from __future__ import annotations

import asyncio
import os
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np

try:
    from ..ai_model import AIAdjacencyModel
    from ..domain import (
        EventRecord,
        LastActivation,
        LayoutReferenceInput,
        PresenceFilterConfigInput,
        RealSensorConfigInput,
        SENSOR_RELIABILITY,
        SensorEventInput,
        adjacency_edge_set,
        adjacency_to_text,
        build_scenario_templates,
        classify_sensor_type,
        edge_key,
        edge_list_from_adjacency,
        infer_room_from_entity,
        is_activation,
        normalize_adjacency_map,
        normalize_room_name,
        parse_adjacency_text,
        to_adjacency,
        to_utc_iso,
    )
    from ..presence_contract import build_presence_snapshot
except ImportError:  # pragma: no cover - permite ejecutar `uvicorn server:app` en Docker
    from ai_model import AIAdjacencyModel
    from domain import (
        EventRecord,
        LastActivation,
        LayoutReferenceInput,
        PresenceFilterConfigInput,
        RealSensorConfigInput,
        SENSOR_RELIABILITY,
        SensorEventInput,
        adjacency_edge_set,
        adjacency_to_text,
        build_scenario_templates,
        classify_sensor_type,
        edge_key,
        edge_list_from_adjacency,
        infer_room_from_entity,
        is_activation,
        normalize_adjacency_map,
        normalize_room_name,
        parse_adjacency_text,
        to_adjacency,
        to_utc_iso,
    )
    from presence_contract import build_presence_snapshot
