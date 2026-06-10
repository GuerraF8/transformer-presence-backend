"""Construcción y validación del grafo de adyacencia."""

from .dependencies import *  # noqa: F401,F403


class GraphMixin:
    def _infer_graph(
        self,
        rooms: list[str],
        directed: Counter[tuple[str, str]],
        blended_probs: np.ndarray,
        degree_limit: int,
        reference_adjacency: dict[str, list[str]] | None = None,
    ) -> tuple[dict[str, list[str]], list[dict[str, Any]], dict[str, float]]:
        room_to_idx = {room: idx for idx, room in enumerate(rooms)}
        pair_candidates: list[dict[str, Any]] = []
        support_values: list[float] = []
        score_values: list[float] = []
        reference_penalized_pairs = 0
        reference_boosted_pairs = 0
        reference_vetoed_pairs = 0

        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                a = rooms[i]
                b = rooms[j]
                raw_support = float(directed.get((a, b), 0) + directed.get((b, a), 0))
                if raw_support <= 0:
                    continue
                raw_sym_score = float(blended_probs[i, j] + blended_probs[j, i])

                adjusted_support = raw_support
                adjusted_sym_score = raw_sym_score
                reference_path: list[str] = []
                reference_adjacent = None
                bridge_support = None
                penalty_factor = 0.0
                reference_veto = False

                if reference_adjacency:
                    reference_path = shortest_path_rooms(reference_adjacency, a, b)
                    if reference_path:
                        reference_adjacent = len(reference_path) == 2
                        if reference_adjacent:
                            adjusted_support *= 1.08
                            adjusted_sym_score *= 1.12
                            reference_boosted_pairs += 1
                        elif len(reference_path) > 2:
                            bridge_supports: list[float] = []
                            for src, dst in zip(reference_path[:-1], reference_path[1:]):
                                bridge_supports.append(
                                    float(directed.get((src, dst), 0) + directed.get((dst, src), 0))
                                )
                            if bridge_supports:
                                bridge_support = min(bridge_supports)
                            penalty_factor = min(0.82, 0.34 + (0.11 * (len(reference_path) - 2)))
                            if bridge_support is not None and bridge_support > 0:
                                ratio = raw_support / bridge_support
                                if ratio < 0.9:
                                    penalty_factor += 0.2
                                elif ratio < 1.15:
                                    penalty_factor += 0.1
                                # Si el mapa real ya explica el salto con aristas puente fuertes,
                                # descartamos la arista directa salvo evidencia muy superior.
                                if raw_support <= (bridge_support * 1.25):
                                    reference_veto = True
                            penalty_factor = min(0.88, penalty_factor)
                            adjusted_support *= max(0.18, 1.0 - (0.62 * penalty_factor))
                            adjusted_sym_score *= max(0.06, 1.0 - penalty_factor)
                            reference_penalized_pairs += 1
                            if reference_veto:
                                reference_vetoed_pairs += 1

                support_values.append(adjusted_support)
                score_values.append(adjusted_sym_score)
                pair_candidates.append(
                    {
                        "a": a,
                        "b": b,
                        "raw_support": raw_support,
                        "raw_score": raw_sym_score,
                        "support": adjusted_support,
                        "score": adjusted_sym_score,
                        "reference_adjacent": reference_adjacent,
                        "reference_path": reference_path,
                        "reference_bridge_support": bridge_support,
                        "reference_penalty_factor": round(penalty_factor, 4),
                        "reference_veto": reference_veto,
                    }
                )

        support_thr = max(2.0, safe_quantile(support_values, 0.35, 2.0))
        score_thr = max(0.08, safe_quantile(score_values, 0.40, 0.08))

        filtered: list[dict[str, Any]] = []
        for candidate in pair_candidates:
            if bool(candidate.get("reference_veto")):
                continue
            if float(candidate["support"]) < support_thr:
                continue
            if float(candidate["score"]) < score_thr:
                continue
            filtered.append(candidate)

        filtered.sort(key=lambda item: (float(item["support"]), float(item["score"])), reverse=True)

        neighbors: dict[str, set[str]] = {room: set() for room in rooms}
        degree = {room: 0 for room in rooms}

        for candidate in filtered:
            a = str(candidate["a"])
            b = str(candidate["b"])
            if degree[a] >= degree_limit or degree[b] >= degree_limit:
                continue
            neighbors[a].add(b)
            neighbors[b].add(a)
            degree[a] += 1
            degree[b] += 1

        # Garantiza conectividad mínima: cada nodo se une al vecino con mayor evidencia.
        for room in rooms:
            if neighbors[room]:
                continue
            i = room_to_idx[room]
            best_other = None
            best_value = -1.0
            for other in rooms:
                if other == room:
                    continue
                j = room_to_idx[other]
                support = float(directed.get((room, other), 0) + directed.get((other, room), 0))
                value = support + (8.0 * float(blended_probs[i, j] + blended_probs[j, i]))
                if reference_adjacency and other in reference_adjacency.get(room, []):
                    value *= 1.18
                if value > best_value:
                    best_value = value
                    best_other = other
            if best_other is not None:
                neighbors[room].add(best_other)
                neighbors[best_other].add(room)

        edge_list: list[dict[str, Any]] = []
        seen = set()
        for a in rooms:
            for b in neighbors[a]:
                k = edge_key(a, b)
                if k in seen:
                    continue
                seen.add(k)
                raw_support = int(directed.get((a, b), 0) + directed.get((b, a), 0))
                raw_sym_score = float(
                    blended_probs[room_to_idx[a], room_to_idx[b]] + blended_probs[room_to_idx[b], room_to_idx[a]]
                )
                ref_path = shortest_path_rooms(reference_adjacency or {}, a, b) if reference_adjacency else []
                ref_adjacent = len(ref_path) == 2 if ref_path else None
                bridge_support = None
                if reference_adjacency and ref_path and len(ref_path) > 2:
                    bridge_support = min(
                        int(directed.get((src, dst), 0) + directed.get((dst, src), 0))
                        for src, dst in zip(ref_path[:-1], ref_path[1:])
                    )
                edge_list.append(
                    {
                        "a": a,
                        "b": b,
                        "support": raw_support,
                        "score": round(raw_sym_score, 4),
                        "reference_adjacent": ref_adjacent,
                        "reference_path": ref_path,
                        "reference_bridge_support": bridge_support,
                    }
                )

        edge_list.sort(key=lambda item: (item["support"], item["score"]), reverse=True)
        neighbors_sorted = {room: sorted(list(v)) for room, v in neighbors.items()}
        thresholds = {
            "support_threshold": support_thr,
            "score_threshold": score_thr,
            "reference_penalized_pairs": reference_penalized_pairs,
            "reference_boosted_pairs": reference_boosted_pairs,
            "reference_vetoed_pairs": reference_vetoed_pairs,
        }
        return neighbors_sorted, edge_list, thresholds

    def _validate_edges_with_ollama(
        self,
        edges: list[dict[str, Any]],
        rooms: list[str],
        ollama_url: str,
        ollama_model: str,
    ) -> dict[str, Any] | None:
        payload = {
            "rooms": rooms,
            "edges": edges,
            "instruction": (
                "Evalua si las adyacencias son consistentes para un hogar real. "
                "Devuelve JSON con llaves quality_score (0-1), suspicious_edges (lista de pares), notes (lista)."
            ),
        }
        body = {
            "model": ollama_model,
            "prompt": json.dumps(payload, ensure_ascii=False),
            "stream": False,
            "format": "json",
        }

        try:
            response = requests.post(
                f"{ollama_url.rstrip('/')}/api/generate",
                json=body,
                timeout=(5, 90),
            )
            response.raise_for_status()
            raw = str(response.json().get("response") or "").strip()
            if not raw:
                return None
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
        return None
