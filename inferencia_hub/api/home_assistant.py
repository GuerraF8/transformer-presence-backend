"""Rutas del catálogo, configuración y acciones de Home Assistant."""

from fastapi import APIRouter

from .common import HandlerMap, endpoint

PATHS = {
    "/api/ha_entities",
    "/api/real_sensor_config",
    "/api/ha_actions",
    "/api/ha_integration_status",
    "/api/ha_actions/pending",
    "/api/ha_actions/{request_id}/result",
}


def build_router(handlers: HandlerMap) -> APIRouter:
    router = APIRouter(tags=["11 Sistema"])
    routes = [
        ("GET", "/api/ha_entities", "get_ha_entities", "Listar entidades Home Assistant"),
        ("POST", "/api/ha_entities", "update_ha_entities", "Actualizar entidades Home Assistant"),
        ("GET", "/api/real_sensor_config", "get_real_sensor_config", "Configuracion de sensores reales"),
        ("POST", "/api/real_sensor_config", "set_real_sensor_config", "Actualizar sensores reales"),
        ("GET", "/api/ha_actions", "list_ha_actions", "Estado de acciones Home Assistant"),
        ("POST", "/api/ha_integration_status", "update_ha_integration_status", "Actualizar estado integracion Home Assistant"),
        ("POST", "/api/ha_actions", "request_ha_action", "Solicitar accion Home Assistant"),
        ("GET", "/api/ha_actions/pending", "claim_ha_action", "Tomar accion pendiente Home Assistant"),
        ("POST", "/api/ha_actions/{request_id}/result", "complete_ha_action", "Registrar resultado Home Assistant"),
    ]
    for method, path, handler, summary in routes:
        router.add_api_route(
            path,
            endpoint(handlers, handler),
            methods=[method],
            summary=summary,
        )
    return router
