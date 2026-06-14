"""Rutas de perfiles de presencia."""

from fastapi import APIRouter

from .common import HandlerMap, endpoint

PATHS = {
    "/api/profiles",
    "/api/profiles/{profile_id}",
    "/api/profiles/{profile_id}/activate",
    "/api/profiles/{profile_id}/infer-layout",
}


def build_router(handlers: HandlerMap) -> APIRouter:
    router = APIRouter(tags=["08 Layout y metricas"])
    router.add_api_route(
        "/api/profiles",
        endpoint(handlers, "list_profiles"),
        methods=["GET"],
        summary="Listar perfiles",
    )
    router.add_api_route(
        "/api/profiles",
        endpoint(handlers, "create_profile"),
        methods=["POST"],
        summary="Crear perfil",
    )
    router.add_api_route(
        "/api/profiles/{profile_id}",
        endpoint(handlers, "get_profile"),
        methods=["GET"],
        summary="Obtener perfil",
    )
    router.add_api_route(
        "/api/profiles/{profile_id}",
        endpoint(handlers, "update_profile"),
        methods=["PUT"],
        summary="Actualizar perfil",
    )
    router.add_api_route(
        "/api/profiles/{profile_id}",
        endpoint(handlers, "delete_profile"),
        methods=["DELETE"],
        summary="Eliminar perfil",
    )
    router.add_api_route(
        "/api/profiles/{profile_id}/activate",
        endpoint(handlers, "activate_profile"),
        methods=["POST"],
        summary="Activar perfil",
    )
    router.add_api_route(
        "/api/profiles/{profile_id}/infer-layout",
        endpoint(handlers, "infer_profile_layout"),
        methods=["POST"],
        summary="Proponer adyacencia desde historial",
    )
    return router
