"""Rutas de configuración y consulta del historial persistente."""

from fastapi import APIRouter

from .common import HandlerMap, endpoint

PATHS = {
    "/api/history/config",
    "/api/history/events",
    "/api/history/presence",
    "/api/history/purge",
}


def build_router(handlers: HandlerMap) -> APIRouter:
    router = APIRouter(tags=["04 Historial"])
    router.add_api_route(
        "/api/history/config",
        endpoint(handlers, "get_history_config"),
        methods=["GET"],
        summary="Obtener configuracion de historial",
    )
    router.add_api_route(
        "/api/history/config",
        endpoint(handlers, "update_history_config"),
        methods=["PUT"],
        summary="Actualizar configuracion de historial",
    )
    router.add_api_route(
        "/api/history/events",
        endpoint(handlers, "get_history_events"),
        methods=["GET"],
        summary="Consultar eventos historicos",
    )
    router.add_api_route(
        "/api/history/presence",
        endpoint(handlers, "get_history_presence"),
        methods=["GET"],
        summary="Consultar serie historica de presencia",
    )
    router.add_api_route(
        "/api/history/purge",
        endpoint(handlers, "purge_history"),
        methods=["POST"],
        summary="Borrar historial persistido",
    )
    return router
