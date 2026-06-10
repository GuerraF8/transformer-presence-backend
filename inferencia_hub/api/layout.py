"""Rutas del mapa de referencia, escenarios y evaluación."""

from fastapi import APIRouter

from .common import HandlerMap, endpoint

PATHS = {
    "/api/scenario_templates",
    "/api/layout_reference",
    "/api/evaluation_metrics",
}


def build_router(handlers: HandlerMap) -> APIRouter:
    router = APIRouter()
    router.add_api_route(
        "/api/scenario_templates",
        endpoint(handlers, "scenario_templates"),
        methods=["GET"],
        tags=["09 Escenarios"],
        summary="Listar plantillas de escenario",
    )
    router.add_api_route(
        "/api/layout_reference",
        endpoint(handlers, "get_layout_reference"),
        methods=["GET"],
        tags=["08 Layout y metricas"],
        summary="Obtener layout de referencia",
    )
    router.add_api_route(
        "/api/layout_reference",
        endpoint(handlers, "set_layout_reference"),
        methods=["POST"],
        tags=["08 Layout y metricas"],
        summary="Actualizar layout de referencia",
    )
    router.add_api_route(
        "/api/evaluation_metrics",
        endpoint(handlers, "evaluation_metrics"),
        methods=["GET"],
        tags=["08 Layout y metricas"],
        summary="Obtener metricas de evaluacion",
    )
    return router
