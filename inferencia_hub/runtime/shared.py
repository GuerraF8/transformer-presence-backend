"""Dependencias y estado compartidos por los controladores."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from ..app_context import ApplicationContext
from ..domain import (
    CsvReplayRequest,
    HAActionRequestInput,
    HAEntityCatalogInput,
    HistoryConfigInput,
    HistoryPurgeInput,
    LayoutReferenceInput,
    PresenceFilterConfigInput,
    RealSensorConfigInput,
    ReplayControlInput,
    RuntimeModeInput,
    SensorEventInput,
    TrainModelFullRequest,
    TrainModelRequest,
    TrainSimulatorPresenceRequest,
    build_layout_for_request,
    build_scenario_templates,
    classify_sensor_type,
    edge_list_from_adjacency,
    infer_room_from_entity,
    is_activation,
    normalize_room_name,
    parse_iso_datetime,
    shortest_path_rooms,
    to_utc_iso,
)

context = ApplicationContext()
hub_state = context.hub
history_store = context.history
ha_entity_catalog = context.catalog
training_status = context.training_status
LOGGER = logging.getLogger("inferencia_hub")
