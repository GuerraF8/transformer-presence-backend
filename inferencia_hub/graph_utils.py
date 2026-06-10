"""Utilidades de mapas y adyacencia utilizadas por la inferencia."""

from .domain import (
    add_undirected_edge,
    adjacency_edge_set,
    adjacency_to_text,
    build_layout_for_request,
    build_scenario_templates,
    edge_key,
    edge_list_from_adjacency,
    normalize_adjacency_map,
    parse_adjacency_text,
    shortest_path_rooms,
    to_adjacency,
)

__all__ = [
    "add_undirected_edge",
    "adjacency_edge_set",
    "adjacency_to_text",
    "build_layout_for_request",
    "build_scenario_templates",
    "edge_key",
    "edge_list_from_adjacency",
    "normalize_adjacency_map",
    "parse_adjacency_text",
    "shortest_path_rooms",
    "to_adjacency",
]
