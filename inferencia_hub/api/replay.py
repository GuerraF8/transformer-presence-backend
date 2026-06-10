"""Rutas de ejecución y control de reproducción histórica."""

from fastapi import APIRouter

from .common import HandlerMap, endpoint

PATHS = {"/api/replay_csv", "/api/replay_control", "/api/replay_status"}


def build_router(handlers: HandlerMap) -> APIRouter:
    router = APIRouter(tags=["07 Replay"])
    router.add_api_route(
        "/api/replay_csv",
        endpoint(handlers, "replay_csv"),
        methods=["POST"],
        summary="Iniciar replay de CSV",
    )
    router.add_api_route(
        "/api/replay_control",
        endpoint(handlers, "replay_control"),
        methods=["POST"],
        summary="Controlar replay",
    )
    router.add_api_route(
        "/api/replay_status",
        endpoint(handlers, "replay_status"),
        methods=["GET"],
        summary="Estado del replay",
    )
    return router
