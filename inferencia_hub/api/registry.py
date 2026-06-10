"""Registro de routers organizados por dominio."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from fastapi import FastAPI

from .common import HandlerMap
from .history import build_router as build_history_router
from .home_assistant import build_router as build_home_assistant_router
from .layout import build_router as build_layout_router
from .presence import build_router as build_presence_router
from .replay import build_router as build_replay_router
from .training import build_router as build_training_router


def include_domain_routers(
    app: FastAPI, handlers: Mapping[str, Callable[..., Any]]
) -> None:
    """Registra las rutas HTTP y WebSocket de cada dominio."""

    handler_map: HandlerMap = handlers
    for router in (
        build_presence_router(handler_map),
        build_history_router(handler_map),
        build_home_assistant_router(handler_map),
        build_layout_router(handler_map),
        build_training_router(handler_map),
        build_replay_router(handler_map),
    ):
        app.include_router(router)
