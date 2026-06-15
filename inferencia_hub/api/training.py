"""Rutas de entrenamiento y descarga de artefactos."""

from fastapi import APIRouter

from .common import HandlerMap, endpoint

PATHS = {
    "/api/train_model",
    "/api/train_model_full",
    "/api/train_presence_simulator",
    "/api/train_presence_supervised",
    "/api/training/manifests",
    "/api/training/manifests/validate",
    "/api/training/reports/{run_id}",
    "/api/model/rollback",
    "/api/live_training/status",
    "/api/live_training/config",
    "/api/live_training/run",
    "/api/training_exports/{filename}",
}


def build_router(handlers: HandlerMap) -> APIRouter:
    router = APIRouter()
    for path, handler, summary in (
        ("/api/train_model", "train_model", "Entrenar modelo desde CSV"),
        ("/api/train_model_full", "train_model_full", "Entrenar modelo completo desde CSV"),
        ("/api/train_presence_simulator", "train_presence_simulator", "Entrenar presencia desde simulador"),
        ("/api/train_presence_supervised", "train_presence_supervised", "Entrenar presencia con confirmaciones"),
        ("/api/training/manifests/validate", "validate_training_manifest", "Validar manifiesto de entrenamiento"),
        ("/api/model/rollback", "rollback_model", "Restaurar modelo anterior"),
    ):
        router.add_api_route(
            path,
            endpoint(handlers, handler),
            methods=["POST"],
            tags=["06 Entrenamiento"],
            summary=summary,
        )
    router.add_api_route(
        "/api/training/manifests",
        endpoint(handlers, "list_training_manifests"),
        methods=["GET"],
        tags=["06 Entrenamiento"],
        summary="Listar manifiestos de entrenamiento",
    )
    router.add_api_route(
        "/api/live_training/status",
        endpoint(handlers, "live_training_status"),
        methods=["GET"],
        tags=["06 Entrenamiento"],
        summary="Consultar aprendizaje en vivo",
    )
    router.add_api_route(
        "/api/live_training/config",
        endpoint(handlers, "get_live_training_config"),
        methods=["GET"],
        tags=["06 Entrenamiento"],
        summary="Consultar configuración del aprendizaje en vivo",
    )
    router.add_api_route(
        "/api/live_training/config",
        endpoint(handlers, "update_live_training_config"),
        methods=["PUT"],
        tags=["06 Entrenamiento"],
        summary="Actualizar configuración del aprendizaje en vivo",
    )
    router.add_api_route(
        "/api/live_training/run",
        endpoint(handlers, "run_live_training"),
        methods=["POST"],
        tags=["06 Entrenamiento"],
        summary="Evaluar y adaptar el modelo con confirmaciones",
    )
    router.add_api_route(
        "/api/training/reports/{run_id}",
        endpoint(handlers, "get_training_report"),
        methods=["GET"],
        tags=["06 Entrenamiento"],
        summary="Consultar reporte de entrenamiento",
    )
    router.add_api_route(
        "/api/training_exports/{filename}",
        endpoint(handlers, "download_training_export"),
        methods=["GET"],
        tags=["10 Descargas"],
        summary="Descargar CSV sintetico",
    )
    return router
