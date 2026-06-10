"""Punto de entrada ASGI utilizado por el contenedor."""

from inferencia_hub.server import app, create_app

__all__ = ["app", "create_app"]
