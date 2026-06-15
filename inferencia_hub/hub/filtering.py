"""Filtro temporal de presencia y reglas de desplazamiento."""

from .dependencies import *  # noqa: F401,F403


class FilteringMixin:
    def _presence_filter_config_locked(self) -> dict[str, Any]:
        learned = dict(self.ai_model.pet_filter_info or {})
        return {
            "enabled": self.presence_filter_enabled,
            "window_seconds": self.presence_filter_window_seconds,
            "min_motion_events": self.presence_filter_min_motion_events,
            "min_distinct_rooms": self.presence_filter_min_distinct_rooms,
            "pending_motion_events": len(self.presence_filter_events),
            "suppressed_total": self.presence_filter_suppressed_total,
            "strategy": (
                "supervised_transformer"
                if learned.get("enabled")
                and learned.get("suppression_enabled")
                else "temporal_rules"
            ),
            "supervised": learned,
        }

    def presence_filter_config(self) -> dict[str, Any]:
        return self._presence_filter_config_locked()

    async def configure_presence_filter(self, config: PresenceFilterConfigInput) -> dict[str, Any]:
        async with self.lock:
            self.presence_filter_enabled = bool(config.enabled)
            self.presence_filter_window_seconds = max(1, min(600, int(config.window_seconds)))
            self.presence_filter_min_motion_events = max(1, min(20, int(config.min_motion_events)))
            self.presence_filter_min_distinct_rooms = max(1, min(20, int(config.min_distinct_rooms)))
            self.presence_filter_events.clear()
            return self._presence_filter_config_locked()

    def _evaluate_presence_filter_locked(
        self,
        room: str,
        sensor_type: str,
        now: datetime,
    ) -> tuple[bool, dict[str, Any]]:
        debug = {
            "enabled": self.presence_filter_enabled,
            "applied": False,
            "accepted": True,
            "window_seconds": self.presence_filter_window_seconds,
            "min_motion_events": self.presence_filter_min_motion_events,
            "min_distinct_rooms": self.presence_filter_min_distinct_rooms,
        }
        if not self.presence_filter_enabled or sensor_type != "motion":
            return True, debug

        supervised = self.ai_model.predict_human_motion(
            list(self.sequence_history),
            room,
            now,
            self.reference_layout,
        )
        if supervised is not None:
            debug.update(supervised)
            debug["applied"] = True
            if supervised.get("suppression_enabled"):
                accepted = bool(supervised.get("accepted"))
                debug["accepted"] = accepted
                debug["reason"] = (
                    None
                    if accepted
                    else "movimiento_clasificado_como_mascota"
                )
                if not accepted:
                    self.presence_filter_suppressed_total += 1
                debug["suppressed_total"] = self.presence_filter_suppressed_total
                return accepted, debug

        room_n = normalize_room_name(room)
        cutoff = now - timedelta(seconds=self.presence_filter_window_seconds)
        while self.presence_filter_events and self.presence_filter_events[0]["timestamp"] < cutoff:
            self.presence_filter_events.popleft()

        self.presence_filter_events.append({"timestamp": now, "room": room_n})
        candidate_rooms: list[str] = []
        candidate_count = 0
        for event in self.presence_filter_events:
            event_room = normalize_room_name(str(event.get("room") or ""))
            if not event_room:
                continue
            if event["timestamp"] < cutoff:
                continue
            if event_room == room_n or self._movement_adjacent_locked(event_room, room_n):
                candidate_count += 1
                candidate_rooms.append(event_room)

        distinct_rooms = sorted(set(candidate_rooms))
        accepted = (
            candidate_count >= self.presence_filter_min_motion_events
            and len(distinct_rooms) >= self.presence_filter_min_distinct_rooms
        )
        if not accepted:
            self.presence_filter_suppressed_total += 1

        debug.update(
            {
                "applied": True,
                "accepted": accepted,
                "events_in_window": candidate_count,
                "distinct_rooms_in_window": len(distinct_rooms),
                "rooms_in_window": distinct_rooms,
                "reason": None if accepted else "movimiento_aislado_posible_mascota",
                "suppressed_total": self.presence_filter_suppressed_total,
            }
        )
        return accepted, debug

    def _prune_inactive_rooms(self, now: datetime) -> None:
        max_delta = timedelta(seconds=self.presence_hold_seconds)
        stale_rooms = [
            room
            for room, ts in self.last_active_by_room.items()
            if now - ts > max_delta
        ]
        for room in stale_rooms:
            del self.last_active_by_room[room]
        stale_occupancy = [
            room
            for room, ts in self.occupancy_confirmed_by_room.items()
            if now - ts > max_delta
        ]
        for room in stale_occupancy:
            del self.occupancy_confirmed_by_room[room]
        filter_cutoff = now - timedelta(seconds=self.presence_filter_window_seconds)
        while self.presence_filter_events and self.presence_filter_events[0]["timestamp"] < filter_cutoff:
            self.presence_filter_events.popleft()

    def _has_adjacent_activity_since_locked(
        self,
        room: str,
        since: datetime | None,
    ) -> bool:
        room_n = normalize_room_name(room)
        if not room_n:
            return False
        self._ensure_reference_layout_locked()
        neighbors = self.reference_layout.get(room_n, [])
        if not neighbors:
            return False

        threshold = since or datetime.min.replace(tzinfo=timezone.utc)
        for nb in neighbors:
            last_ts = self.last_active_by_room.get(nb)
            if last_ts is not None and last_ts >= threshold:
                return True
        return False

    def _can_displace_presence_locked(
        self,
        previous_room: str | None,
        candidate_room: str,
        sensor_type: str,
        previous_seen_at: datetime | None,
    ) -> bool:
        if not previous_room:
            return True

        prev_n = normalize_room_name(previous_room)
        cand_n = normalize_room_name(candidate_room)
        if not prev_n or not cand_n or prev_n == cand_n:
            return True

        # La ocupación confirma presencia en la habitación observada.
        if sensor_type == "occupancy":
            return True

        # Si la transicion observada es adyacente en el mapa aprendido/manual, permitimos el movimiento.
        if self._movement_adjacent_locked(prev_n, cand_n):
            return True

        # Si no hubo actividad intermedia en adyacentes, retenemos presencia en el cuarto previo.
        return self._has_adjacent_activity_since_locked(prev_n, previous_seen_at)
