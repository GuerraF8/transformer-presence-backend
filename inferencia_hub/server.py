"""Punto de entrada ASGI que publica la aplicación y su función de creación."""

from .application import app, create_app

__all__ = ["app", "create_app"]
