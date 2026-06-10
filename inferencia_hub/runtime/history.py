"""Configuración y consultas del historial persistente."""

from .shared import *  # noqa: F401,F403


def normalize_history_timestamp(value: str, field_name: str) -> str:
    if not value:
        return ""
    try:
        return to_utc_iso(parse_iso_datetime(value))
    except (TypeError, ValueError) as err:
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} debe ser una fecha ISO valida",
        ) from err


async def get_history_config() -> dict[str, Any]:
    return await asyncio.to_thread(history_store.status)


async def update_history_config(config: HistoryConfigInput) -> dict[str, Any]:
    try:
        await asyncio.to_thread(
            history_store.update_config,
            enabled=config.enabled,
            retention_days=config.retention_days,
            persisted_modes=list(config.persisted_modes),
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    return await asyncio.to_thread(history_store.status)


async def get_history_events(
    query: str = "",
    sensor_type: str = "",
    room: str = "",
    input_mode: str = "",
    from_ts: str = "",
    to_ts: str = "",
    page: int = 1,
    page_size: int = 50,
) -> dict[str, Any]:
    normalized_from = normalize_history_timestamp(from_ts, "from_ts")
    normalized_to = normalize_history_timestamp(to_ts, "to_ts")
    if normalized_from and normalized_to and normalized_from > normalized_to:
        raise HTTPException(status_code=422, detail="from_ts no puede ser posterior a to_ts")
    return await asyncio.to_thread(
        history_store.query_events,
        query=query,
        sensor_type=sensor_type,
        room=room,
        input_mode=input_mode,
        from_ts=normalized_from,
        to_ts=normalized_to,
        page=page,
        page_size=page_size,
    )


async def get_history_presence(
    query: str = "",
    sensor_type: str = "",
    room: str = "",
    input_mode: str = "",
    from_ts: str = "",
    to_ts: str = "",
    max_points: int = 1000,
) -> dict[str, Any]:
    normalized_from = normalize_history_timestamp(from_ts, "from_ts")
    normalized_to = normalize_history_timestamp(to_ts, "to_ts")
    if normalized_from and normalized_to and normalized_from > normalized_to:
        raise HTTPException(status_code=422, detail="from_ts no puede ser posterior a to_ts")
    return await asyncio.to_thread(
        history_store.query_presence,
        query=query,
        sensor_type=sensor_type,
        room=room,
        input_mode=input_mode,
        from_ts=normalized_from,
        to_ts=normalized_to,
        max_points=max_points,
    )


async def purge_history(payload: HistoryPurgeInput) -> dict[str, Any]:
    if payload.confirmation != "BORRAR":
        raise HTTPException(
            status_code=400,
            detail='La confirmacion debe ser exactamente "BORRAR"',
        )
    deleted = await asyncio.to_thread(history_store.purge)
    return {"status": "ok", "deleted": deleted}
