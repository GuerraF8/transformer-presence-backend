"""Configuración y ensamblaje de la aplicación FastAPI."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.registry import include_domain_routers
from .runtime import *  # noqa: F401,F403
from .runtime import HANDLERS, shutdown_history_store, startup_train_model
from .runtime.shared import context
from .web_paths import resolve_web_dir

TAG_STATUS = "01 Estado"
TAG_INGESTION = "02 Ingesta"
TAG_PRESENCE = "03 Presencia"
TAG_HISTORY = "04 Historial"
TAG_MODEL = "05 Modelo"
TAG_TRAINING = "06 Entrenamiento"
TAG_REPLAY = "07 Replay"
TAG_LAYOUT = "08 Layout y metricas"
TAG_SCENARIOS = "09 Escenarios"
TAG_DOWNLOADS = "10 Descargas"
TAG_SYSTEM = "11 Sistema"

OPENAPI_TAGS = [
    {"name": TAG_STATUS, "description": "Salud del backend y estado operativo general."},
    {"name": TAG_INGESTION, "description": "Recepcion de eventos normalizados desde Home Assistant o simulador."},
    {"name": TAG_PRESENCE, "description": "Snapshot de presencia, filtros temporales y estado inferido."},
    {"name": TAG_HISTORY, "description": "Persistencia y consulta historica de eventos e inferencias."},
    {"name": TAG_MODEL, "description": "Metadata del modelo de inferencia y transformadores entrenados."},
    {"name": TAG_TRAINING, "description": "Entrenamiento desde historico CSV o datos sinteticos del simulador."},
    {"name": TAG_REPLAY, "description": "Replay de historicos CSV y control de ejecucion paso a paso."},
    {"name": TAG_LAYOUT, "description": "Mapa de referencia, adyacencia y metricas de evaluacion."},
    {"name": TAG_SCENARIOS, "description": "Plantillas de layouts para simulacion y entrenamiento."},
    {"name": TAG_DOWNLOADS, "description": "Descarga de artefactos generados por entrenamiento."},
    {"name": TAG_SYSTEM, "description": "Operaciones administrativas del runtime."},
]

LOGGER = logging.getLogger("inferencia_hub")
WEB_DIR = resolve_web_dir()


def _create_base_app() -> FastAPI:
    application = FastAPI(
        title="Inferencia Presencia Hub",
        version="0.3.1",
        openapi_tags=OPENAPI_TAGS,
    )
    application.state.context = context
    cors_origins_raw = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
    cors_origins = (
        ["*"]
        if cors_origins_raw == "*"
        else [origin.strip() for origin in cors_origins_raw.split(",") if origin.strip()]
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return application


def _mount_web(application: FastAPI) -> None:
    configured_web_dir = os.getenv("WEB_DIR", "").strip()
    if configured_web_dir and Path(configured_web_dir).expanduser().resolve() != WEB_DIR:
        LOGGER.warning(
            "WEB_DIR=%s no existe; se usaran los recursos incluidos en %s",
            configured_web_dir,
            WEB_DIR,
        )
    if WEB_DIR.exists():
        application.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
        return

    @application.get("/", tags=[TAG_SYSTEM], summary="Fallback de interfaz web")
    def root_fallback() -> dict[str, str]:
        return {
            "message": "inferencia_hub operativo, pero WEB_DIR no existe",
            "web_dir": str(WEB_DIR),
        }


app = _create_base_app()
app.add_event_handler("startup", startup_train_model)
app.add_event_handler("shutdown", shutdown_history_store)
include_domain_routers(app, HANDLERS)
_mount_web(app)

# Referencias directas a los servicios compartidos por los controladores.
hub_state = context.hub
history_store = context.history
ha_entity_catalog = context.catalog
training_status = context.training_status


def create_app() -> FastAPI:
    """Devuelve la aplicación ASGI configurada."""

    return app
