"""Rutas de ingesta, estado y consultas de presencia."""

from fastapi import APIRouter

from .common import HandlerMap, endpoint

PATHS = {
    "/api/health",
    "/api/events",
    "/api/sim_data",
    "/api/presence_filter",
    "/api/model_info",
    "/api/input_mode",
    "/api/reset",
    "/presencia",
}


def build_router(handlers: HandlerMap) -> APIRouter:
    router = APIRouter()
    router.add_api_route(
        "/api/health",
        endpoint(handlers, "health"),
        methods=["GET"],
        tags=["01 Estado"],
        summary="Estado del backend",
    )
    router.add_api_route(
        "/api/events",
        endpoint(handlers, "ingest_event"),
        methods=["POST"],
        tags=["02 Ingesta"],
        summary="Ingestar evento de sensor",
    )
    router.add_api_route(
        "/api/sim_data",
        endpoint(handlers, "get_sim_data"),
        methods=["GET"],
        tags=["03 Presencia"],
        summary="Snapshot completo para la UI",
    )
    router.add_api_route(
        "/api/presence_filter",
        endpoint(handlers, "get_presence_filter"),
        methods=["GET"],
        tags=["03 Presencia"],
        summary="Obtener filtro de presencia",
    )
    router.add_api_route(
        "/api/presence_filter",
        endpoint(handlers, "set_presence_filter"),
        methods=["POST"],
        tags=["03 Presencia"],
        summary="Actualizar filtro de presencia",
    )
    router.add_api_route(
        "/api/model_info",
        endpoint(handlers, "model_info"),
        methods=["GET"],
        tags=["05 Modelo"],
        summary="Informacion del modelo",
    )
    router.add_api_route(
        "/api/input_mode",
        endpoint(handlers, "get_input_mode"),
        methods=["GET"],
        tags=["11 Sistema"],
        summary="Obtener modo de entrada",
    )
    router.add_api_route(
        "/api/input_mode",
        endpoint(handlers, "set_input_mode"),
        methods=["POST"],
        tags=["11 Sistema"],
        summary="Cambiar modo de entrada",
    )
    router.add_api_route(
        "/api/reset",
        endpoint(handlers, "reset_state"),
        methods=["POST"],
        tags=["11 Sistema"],
        summary="Reiniciar estado runtime",
    )
    router.add_api_websocket_route(
        "/presencia", endpoint(handlers, "presencia_socket")
    )
    return router
