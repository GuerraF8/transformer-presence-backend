"""Rutas de entrenamiento y descarga de artefactos."""

from fastapi import APIRouter

from .common import HandlerMap, endpoint

PATHS = {
    "/api/train_model",
    "/api/train_model_full",
    "/api/train_presence_simulator",
    "/api/training_exports/{filename}",
}


def build_router(handlers: HandlerMap) -> APIRouter:
    router = APIRouter()
    for path, handler, summary in (
        ("/api/train_model", "train_model", "Entrenar modelo desde CSV"),
        ("/api/train_model_full", "train_model_full", "Entrenar modelo completo desde CSV"),
        ("/api/train_presence_simulator", "train_presence_simulator", "Entrenar presencia desde simulador"),
    ):
        router.add_api_route(
            path,
            endpoint(handlers, handler),
            methods=["POST"],
            tags=["06 Entrenamiento"],
            summary=summary,
        )
    router.add_api_route(
        "/api/training_exports/{filename}",
        endpoint(handlers, "download_training_export"),
        methods=["GET"],
        tags=["10 Descargas"],
        summary="Descargar CSV sintetico",
    )
    return router
