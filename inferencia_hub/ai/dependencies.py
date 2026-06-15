from __future__ import annotations

import asyncio
import csv
import json
import os
import random
import threading
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

try:
    from ..models.transformers import NextRoomTransformer, OccupancyTransformer

    HF_AVAILABLE = TORCH_AVAILABLE
except Exception:
    NextRoomTransformer = None
    OccupancyTransformer = None
    HF_AVAILABLE = False

try:
    from ..domain import (
        SENSOR_RELIABILITY,
        TRANSFORMER_CONTEXT_LENGTH,
        TRANSFORMER_LAGS_SEQUENCE,
        TRANSFORMER_MIN_SAMPLES,
        TRANSFORMER_MODEL_CONTEXT_LENGTH,
        EventRecord,
        TrainModelRequest,
        TrainSimulatorPresenceRequest,
        build_layout_for_request,
        build_scenario_templates,
        classify_sensor_type,
        classify_state_bucket,
        edge_key,
        infer_room_from_entity,
        is_activation,
        normalize_adjacency_map,
        parse_iso_datetime,
        safe_quantile,
        shortest_path_rooms,
        time_features_from_dt,
    )
except ImportError:  # pragma: no cover - permite ejecutar `uvicorn server:app` en Docker
    from domain import (
        SENSOR_RELIABILITY,
        TRANSFORMER_CONTEXT_LENGTH,
        TRANSFORMER_LAGS_SEQUENCE,
        TRANSFORMER_MIN_SAMPLES,
        TRANSFORMER_MODEL_CONTEXT_LENGTH,
        EventRecord,
        TrainModelRequest,
        TrainSimulatorPresenceRequest,
        build_layout_for_request,
        build_scenario_templates,
        classify_sensor_type,
        classify_state_bucket,
        edge_key,
        infer_room_from_entity,
        is_activation,
        normalize_adjacency_map,
        parse_iso_datetime,
        safe_quantile,
        shortest_path_rooms,
        time_features_from_dt,
    )
