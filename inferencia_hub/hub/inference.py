"""Inferencia probabilística de habitación y ocupación."""

from .dependencies import *  # noqa: F401,F403


class InferenceMixin:
    def _ensure_presence_belief(self) -> None:
        n_rooms = len(self.ai_model.rooms)
        if n_rooms <= 0:
            self.presence_belief = np.zeros((0,), dtype=np.float32)
            return
        if self.presence_belief.shape[0] != n_rooms:
            self.presence_belief = np.full((n_rooms,), 1.0 / n_rooms, dtype=np.float32)

    def _infer_presence_with_ai(
        self,
        observed_room: str,
        sensor_type: str,
        now: datetime,
    ) -> tuple[str, float, list[str], dict[str, Any]]:
        self._ensure_presence_belief()
        if not self.ai_model.ready or observed_room not in self.ai_model.room_to_idx:
            occupancy_prediction = self.ai_model.predict_occupancy_state(list(self.sequence_history), now)
            if occupancy_prediction and occupancy_prediction.get("rooms"):
                predicted_rooms = [
                    normalize_room_name(room)
                    for room in occupancy_prediction.get("rooms", [])
                    if normalize_room_name(room)
                ]
                if observed_room not in predicted_rooms:
                    predicted_rooms.insert(0, observed_room)
                return observed_room, float(occupancy_prediction.get("confidence") or 0.5), predicted_rooms, {
                    "transformer_used": False,
                    "markov_top_room": observed_room,
                    "hybrid_top_room": observed_room,
                    "observed_room_prob": 0.5,
                    "occupancy_transformer_used": True,
                    "occupancy_transformer_rooms": predicted_rooms,
                    "occupancy_transformer_people_count": int(occupancy_prediction.get("people_count") or 0),
                    "occupancy_transformer_confidence": occupancy_prediction.get("confidence"),
                }
            return observed_room, 0.5, [observed_room], {
                "transformer_used": False,
                "markov_top_room": observed_room,
                "hybrid_top_room": observed_room,
                "observed_room_prob": 0.5,
            }

        n_rooms = len(self.ai_model.rooms)
        idx_obs = self.ai_model.room_to_idx[observed_room]

        trans = self.ai_model.transition_matrix
        markov_prior = self.presence_belief @ trans
        transformer_prior = self.ai_model.predict_next_room_probs(list(self.sequence_history), now)
        transformer_used = transformer_prior is not None and float(transformer_prior.sum()) > 0
        if transformer_prior is not None and float(transformer_prior.sum()) > 0:
            transformer_prior = transformer_prior / float(transformer_prior.sum())
            prior = (0.4 * markov_prior) + (0.6 * transformer_prior)
        else:
            prior = markov_prior

        emission = np.full((n_rooms,), 0.08, dtype=np.float32)
        reliability = SENSOR_RELIABILITY.get(sensor_type, SENSOR_RELIABILITY["other"])
        emission[idx_obs] = 0.52 + (0.42 * reliability)

        for nb in self.ai_model.neighbors(observed_room):
            idx_nb = self.ai_model.room_to_idx[nb]
            emission[idx_nb] = max(emission[idx_nb], 0.18 + (0.22 * reliability))

        posterior = prior * emission
        post_sum = float(posterior.sum())
        if post_sum > 0:
            posterior /= post_sum
        else:
            posterior = np.full((n_rooms,), 1.0 / n_rooms, dtype=np.float32)

        self.presence_belief = posterior

        best_idx = int(np.argmax(posterior))
        best_room = self.ai_model.rooms[best_idx]
        confidence = float(posterior[best_idx])
        markov_idx = int(np.argmax(markov_prior))
        transformer_idx = int(np.argmax(transformer_prior)) if transformer_used and transformer_prior is not None else None

        order = np.argsort(-posterior)
        active_rooms: list[str] = []
        thr = max(0.18, confidence * 0.45)
        for idx in order:
            prob = float(posterior[idx])
            if prob < thr and active_rooms:
                break
            active_rooms.append(self.ai_model.rooms[int(idx)])
            if len(active_rooms) >= 3:
                break

        debug = {
            "transformer_used": transformer_used,
            "markov_top_room": self.ai_model.rooms[markov_idx],
            "transformer_top_room": (
                self.ai_model.rooms[int(transformer_idx)] if transformer_idx is not None else None
            ),
            "hybrid_top_room": best_room,
            "observed_room_prob": round(float(posterior[idx_obs]), 4),
            "hybrid_top_prob": round(confidence, 4),
        }
        if transformer_used and transformer_prior is not None:
            debug["transformer_observed_room_prob"] = round(float(transformer_prior[idx_obs]), 4)
            debug["transformer_top_prob"] = round(float(transformer_prior[int(transformer_idx)]), 4)

        occupancy_prediction = self.ai_model.predict_occupancy_state(list(self.sequence_history), now)
        if occupancy_prediction and occupancy_prediction.get("rooms"):
            predicted_rooms = [
                normalize_room_name(room)
                for room in occupancy_prediction.get("rooms", [])
                if normalize_room_name(room)
            ]
            if predicted_rooms:
                active_rooms = list(dict.fromkeys(predicted_rooms + active_rooms))
                debug["occupancy_transformer_used"] = True
                debug["occupancy_transformer_rooms"] = predicted_rooms
                debug["occupancy_transformer_people_count"] = int(occupancy_prediction.get("people_count") or 0)
                debug["occupancy_transformer_confidence"] = occupancy_prediction.get("confidence")

        return best_room, confidence, active_rooms, debug

    def _build_transition(
        self,
        room: str,
        now: datetime,
    ) -> tuple[dict[str, Any] | None, bool]:
        if self.last_activation is None:
            return None, False

        prev = self.last_activation
        gap = (now - prev.timestamp).total_seconds()
        if prev.room == room:
            return (
                {
                    "from": prev.room,
                    "to": room,
                    "gap_seconds": round(gap, 3),
                    "same_room": True,
                    "rejected_by_ai": False,
                },
                False,
            )
        if (not self.include_all_state_transitions) and (
            gap < self.min_gap_seconds or gap > self.max_gap_seconds
        ):
            return None, False

        allowed = self.ai_model.are_adjacent(prev.room, room)
        if not allowed:
            self.rejected_transitions += 1
            return (
                {
                    "from": prev.room,
                    "to": room,
                    "gap_seconds": round(gap, 3),
                    "same_room": False,
                    "rejected_by_ai": True,
                    "reason": "edge_not_in_learned_map",
                },
                True,
            )

        key = edge_key(prev.room, room)
        self.edge_support[key] += 1
        self.latest_touched_edge = key

        return (
            {
                "from": prev.room,
                "to": room,
                "gap_seconds": round(gap, 3),
                "weight": 1,
                "support": int(self.edge_support[key]),
                "edge": [key[0], key[1]],
                "same_room": False,
                "rejected_by_ai": False,
            },
            False,
        )
