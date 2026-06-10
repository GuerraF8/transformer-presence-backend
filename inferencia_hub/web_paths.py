"""Resolución de la ruta que contiene los recursos del panel web."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_WEB_DIR = Path(__file__).resolve().parent / "web"


def resolve_web_dir(configured: str | None = None) -> Path:
    """Devuelve la ruta configurada o los recursos incluidos en el paquete."""

    raw_path = configured
    if raw_path is None:
        raw_path = os.getenv("WEB_DIR", "").strip()

    if raw_path:
        requested = Path(raw_path).expanduser().resolve()
        if requested.is_dir():
            return requested

    return PACKAGE_WEB_DIR.resolve()
