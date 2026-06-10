"""Utilidades para registrar controladores en los routers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


HandlerMap = Mapping[str, Callable[..., Any]]


def endpoint(handlers: HandlerMap, name: str) -> Callable[..., Any]:
    try:
        return handlers[name]
    except KeyError as err:
        raise RuntimeError(f"Handler API no registrado: {name}") from err
