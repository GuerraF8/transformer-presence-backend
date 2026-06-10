"""Carga, ejecución y control de reproducciones históricas."""

from .shared import *  # noqa: F401,F403


async def _load_csv_events(
    csv_path: str,
    debounce_seconds: int,
    include_all_state_transitions: bool,
) -> list[SensorEventInput]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe CSV: {csv_path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    parsed_rows: list[tuple[datetime, SensorEventInput]] = []
    for row in rows:
        entity_id = str(row.get("entity_id", "")).strip()
        state = str(row.get("state", "")).strip().lower()
        ts_raw = str(row.get("last_changed", "")).strip()

        if not entity_id or not ts_raw:
            continue

        try:
            timestamp = parse_iso_datetime(ts_raw)
        except Exception:
            continue

        sensor_type = classify_sensor_type(entity_id)
        room = normalize_room_name(infer_room_from_entity(entity_id))

        parsed_rows.append(
            (
                timestamp,
                SensorEventInput(
                    entity_id=entity_id,
                    state=state,
                    sensor_type=sensor_type,
                    room=room,
                    timestamp=timestamp,
                    source="csv_replay",
                ),
            )
        )

    parsed_rows.sort(key=lambda item: item[0])

    if include_all_state_transitions:
        return [event for _, event in parsed_rows]

    last_by_entity: dict[str, datetime] = {}
    out: list[SensorEventInput] = []
    for ts, event in parsed_rows:
        if not is_activation(event.sensor_type or "other", event.state):
            continue
        prev = last_by_entity.get(event.entity_id)
        if prev is not None and (ts - prev).total_seconds() <= debounce_seconds:
            continue
        last_by_entity[event.entity_id] = ts
        out.append(event)

    return out


def _normalize_room_mapping(mapping: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for source, target in mapping.items():
        source_room = normalize_room_name(source)
        target_room = normalize_room_name(target)
        if source_room and target_room:
            out[source_room] = target_room
    return out


def _build_scenario_events(
    events: list[SensorEventInput],
    req: CsvReplayRequest,
) -> tuple[list[SensorEventInput], dict[str, list[str]]]:
    if not events:
        return events, {}

    room_mapping = _normalize_room_mapping(req.room_mapping)
    base_rooms = sorted(
        {
            normalize_room_name(event.room)
            for event in events
            if normalize_room_name(event.room)
        }
    )
    mapped_rooms = sorted({room_mapping.get(room, room) for room in base_rooms})
    layout = build_layout_for_request(mapped_rooms, req.template, req.layout_edges)

    scenario_events: list[SensorEventInput] = []
    current_room = ""
    cursor_time = events[0].timestamp or datetime.now(timezone.utc)
    if cursor_time.tzinfo is None:
        cursor_time = cursor_time.replace(tzinfo=timezone.utc)

    for event in events:
        original_room = normalize_room_name(event.room)
        mapped_room = room_mapping.get(original_room, original_room)

        event_time = event.timestamp or cursor_time
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
        else:
            event_time = event_time.astimezone(timezone.utc)
        if event_time <= cursor_time:
            event_time = cursor_time + timedelta(seconds=1)

        if current_room and mapped_room and mapped_room != current_room:
            path = shortest_path_rooms(layout, current_room, mapped_room)
            if path and len(path) > 2:
                for intermediate in path[1:-1]:
                    event_time = event_time + timedelta(seconds=req.step_seconds)
                    scenario_events.append(
                        SensorEventInput(
                            entity_id=f"simulated_sensor.{intermediate}_scenario",
                            state="on",
                            sensor_type="motion",
                            room=intermediate,
                            timestamp=event_time,
                            source=f"csv_scenario_intermediate:{req.template}",
                        )
                    )

        scenario_events.append(
            SensorEventInput(
                entity_id=event.entity_id,
                state=event.state,
                sensor_type=event.sensor_type,
                room=mapped_room,
                timestamp=event_time,
                source=f"csv_scenario:{req.template}",
            )
        )
        cursor_time = event_time

        if is_activation(event.sensor_type or "other", event.state):
            current_room = mapped_room

    return scenario_events, layout


async def _run_csv_replay(req: CsvReplayRequest) -> None:
    await hub_state.reset()
    hub_state.input_mode = "replay"
    hub_state.replay_paused = False
    hub_state.replay_stop_requested = False
    hub_state.replay_step_budget = 0
    hub_state.replay_last_error = None

    if not hub_state.ai_model.ready and os.getenv("AUTO_TRAIN_ON_REPLAY", "1") != "0":
        train_req = TrainModelRequest(
            csv_path=req.csv_path,
            debounce_seconds=req.debounce_seconds,
            include_all_state_transitions=req.include_all_state_transitions,
        )
        try:
            await asyncio.to_thread(hub_state.ai_model.train_from_csv, train_req)
            await hub_state.reset()
        except Exception:
            # Si falla el entrenamiento, la reproducción continúa con inferencia por reglas.
            pass

    hub_state.input_mode = "replay"

    try:
        events = await _load_csv_events(
            req.csv_path,
            req.debounce_seconds,
            req.include_all_state_transitions,
        )
    except Exception:
        hub_state.replay_task = None
        raise

    if req.max_events > 0:
        events = events[: req.max_events]

    scenario_layout: dict[str, list[str]] = {}
    if req.use_scenario_layout:
        events, scenario_layout = _build_scenario_events(events, req)

    hub_state.replay_total_events = len(events)
    hub_state.replay_processed_events = 0

    hub_state.last_replay_config = {
        "csv_path": req.csv_path,
        "speed_events_per_second": req.speed_events_per_second,
        "debounce_seconds": req.debounce_seconds,
        "include_all_state_transitions": req.include_all_state_transitions,
        "max_events": req.max_events,
        "use_scenario_layout": req.use_scenario_layout,
        "template": req.template,
        "layout_edges": edge_list_from_adjacency(scenario_layout),
        "room_mapping": req.room_mapping,
        "step_seconds": req.step_seconds,
        "events_loaded": len(events),
    }

    delay = 1.0 / req.speed_events_per_second
    for event in events:
        if hub_state.replay_stop_requested:
            break
        while hub_state.replay_paused and not hub_state.replay_stop_requested:
            if hub_state.replay_step_budget > 0:
                hub_state.replay_step_budget -= 1
                break
            await asyncio.sleep(0.2)
        if hub_state.replay_stop_requested:
            break

        await hub_state.process_event(event)
        hub_state.replay_processed_events += 1
        await asyncio.sleep(delay)

    hub_state.replay_paused = False
    hub_state.replay_stop_requested = False
    hub_state.replay_step_budget = 0
    hub_state.replay_task = None
    await hub_state.broadcast_snapshot()


async def replay_csv(req: CsvReplayRequest) -> dict[str, Any]:
    if hub_state.replay_task and not hub_state.replay_task.done():
        raise HTTPException(status_code=409, detail="Ya hay una simulacion en ejecucion")

    async def _runner() -> None:
        try:
            await _run_csv_replay(req)
        except asyncio.CancelledError:
            hub_state.replay_task = None
            hub_state.replay_paused = False
            hub_state.replay_stop_requested = False
            raise
        except Exception as exc:
            hub_state.replay_last_error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("CSV replay failed after %s/%s events", hub_state.replay_processed_events, hub_state.replay_total_events)
            hub_state.replay_task = None
            hub_state.replay_paused = False
            hub_state.replay_stop_requested = False
            await hub_state.broadcast_snapshot()

    hub_state.input_mode = "replay"
    hub_state.replay_task = asyncio.create_task(_runner())

    return {
        "status": "started",
        "csv_path": req.csv_path,
        "speed_events_per_second": req.speed_events_per_second,
        "debounce_seconds": req.debounce_seconds,
        "include_all_state_transitions": req.include_all_state_transitions,
        "max_events": req.max_events,
        "use_scenario_layout": req.use_scenario_layout,
        "template": req.template,
        "step_seconds": req.step_seconds,
        "input_mode": hub_state.input_mode,
    }


async def replay_control(req: ReplayControlInput) -> dict[str, Any]:
    running = bool(hub_state.replay_task and not hub_state.replay_task.done())

    if req.action == "pause":
        if not running:
            raise HTTPException(status_code=409, detail="No hay replay activo para pausar")
        hub_state.replay_paused = True

    elif req.action == "start":
        if not running:
            raise HTTPException(status_code=409, detail="No hay replay activo. Inicia uno nuevo con /api/replay_csv")
        hub_state.replay_paused = False
        hub_state.replay_step_budget = 0

    elif req.action == "step":
        if not running:
            raise HTTPException(status_code=409, detail="No hay replay activo para avanzar paso a paso")
        if not hub_state.replay_paused:
            raise HTTPException(status_code=409, detail="Step solo disponible cuando el replay esta pausado")
        hub_state.replay_step_budget += 1

    elif req.action == "reset":
        hub_state.replay_stop_requested = True
        hub_state.replay_step_budget = 0
        if hub_state.replay_task and not hub_state.replay_task.done():
            hub_state.replay_task.cancel()
        await hub_state.reset()
        await hub_state.broadcast_snapshot()

    return replay_status()


def replay_status() -> dict[str, Any]:
    running = bool(hub_state.replay_task and not hub_state.replay_task.done())
    return {
        "running": running,
        "mode": hub_state.input_mode,
        "paused": hub_state.replay_paused,
        "step_budget": hub_state.replay_step_budget,
        "events": len(hub_state.events),
        "rooms": len(hub_state.rooms),
        "model_ready": hub_state.ai_model.ready,
        "processed_events": hub_state.replay_processed_events,
        "total_events": hub_state.replay_total_events,
        "last_error": hub_state.replay_last_error,
        "progress": (
            round(hub_state.replay_processed_events / hub_state.replay_total_events, 4)
            if hub_state.replay_total_events > 0
            else 0.0
        ),
        "last_replay_config": hub_state.last_replay_config,
    }
