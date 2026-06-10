"""Procesamiento de eventos y actualización del estado inferido."""

from .dependencies import *  # noqa: F401,F403


class EventsMixin:
    async def process_event(self, payload: SensorEventInput) -> dict[str, Any]:
        ingress_now = datetime.now(timezone.utc)
        now = payload.timestamp or ingress_now
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)

        entity_id = str(payload.entity_id or "").strip().lower()
        source = str(payload.source or "").strip().lower()
        is_real_ha_event = source.startswith("ha") or source.startswith("home_assistant") or source.startswith("hass")

        inferred_room = infer_room_from_entity(entity_id)
        room = normalize_room_name(payload.room or inferred_room)
        sensor_type = payload.sensor_type or classify_sensor_type(entity_id)
        state = payload.state.lower().strip()
        is_active_event = is_activation(sensor_type, state)

        async with self.lock:
            if is_real_ha_event:
                assignment = self.real_sensor_assignments.get(entity_id)
                if not assignment or (assignment and not bool(assignment.get("enabled", True))):
                    self.real_sensor_rejected_events += 1
                    self.real_sensor_last_rejected = {
                        "timestamp": to_utc_iso(now),
                        "entity_id": entity_id,
                        "source": payload.source,
                        "reason": "sensor_no_asignado" if not assignment else "sensor_deshabilitado",
                    }
                    return {
                        "status": "ignored",
                        "reason": self.real_sensor_last_rejected["reason"],
                        "entity_id": entity_id,
                        "real_sensor_config": self._real_sensor_config_locked(),
                    }
                if assignment:
                    room = normalize_room_name(str(assignment.get("room") or room))
                    assigned_type = str(assignment.get("sensor_type") or "auto").strip().lower()
                    if assigned_type != "auto":
                        sensor_type = assigned_type
                    is_active_event = is_activation(sensor_type, state)

            self.rooms.add(room)
            self._ensure_reference_layout_locked()
            presence_signal_allowed = True
            presence_filter_debug: dict[str, Any] = {}
            if is_active_event:
                presence_signal_allowed, presence_filter_debug = self._evaluate_presence_filter_locked(
                    room=room,
                    sensor_type=sensor_type,
                    now=now,
                )

            if is_active_event and presence_signal_allowed:
                self.active_sensor_types_by_room.setdefault(room, set()).add(sensor_type)
            else:
                active_types = self.active_sensor_types_by_room.get(room)
                if active_types is not None:
                    active_types.discard(sensor_type)
                    if not active_types:
                        self.active_sensor_types_by_room.pop(room, None)

            previous_room = self.current_room
            previous_seen_at = self.last_active_by_room.get(previous_room, None) if previous_room else None
            current_event = EventRecord(
                timestamp=now,
                entity_id=entity_id,
                state=state,
                sensor_type=sensor_type,
                room=room,
            )

            transition: dict[str, Any] | None = None
            confidence = 0.0
            active_rooms: list[str] = []
            relation = "desconocida"
            layout_alert: dict[str, Any] | None = None
            inference_debug: dict[str, Any] = {}
            if presence_filter_debug:
                inference_debug["presence_filter"] = presence_filter_debug
            if is_active_event and presence_signal_allowed:
                transition, _ = self._build_transition(room, now)
                # Mantiene la ultima observacion de activacion para analizar desplazamientos reales.
                self.last_activation = LastActivation(room=room, timestamp=now)
            elif sensor_type == "occupancy":
                self.occupancy_confirmed_by_room.pop(room, None)
                self.last_active_by_room.pop(room, None)
                if self.current_room == room:
                    remaining_rooms = list(self.occupancy_confirmed_by_room.keys()) + list(
                        self.active_sensor_types_by_room.keys()
                    )
                    self.current_room = normalize_room_name(remaining_rooms[0]) if remaining_rooms else None

            if is_active_event and presence_signal_allowed:
                self.last_active_by_room[room] = now

                inferred_presence_room, confidence, ai_active_rooms, ai_debug = self._infer_presence_with_ai(
                    observed_room=room,
                    sensor_type=sensor_type,
                    now=now,
                )
                inference_debug.update(ai_debug)
                resolved_presence_room = inferred_presence_room
                observed_room_forced = False

                if sensor_type == "occupancy":
                    # El sensor de ocupación actúa como confirmación directa de presencia.
                    self.occupancy_confirmed_by_room[room] = now
                    resolved_presence_room = room
                    confidence = max(confidence, 0.96)
                    if self.ai_model.ready and room in self.ai_model.room_to_idx:
                        self._ensure_presence_belief()
                        forced = np.zeros_like(self.presence_belief)
                        forced[self.ai_model.room_to_idx[room]] = 1.0
                        self.presence_belief = forced
                elif self.occupancy_confirmed_by_room and room not in self.occupancy_confirmed_by_room:
                    # Una ocupación activa fija la habitación principal con evidencia directa.
                    # Movimiento en otra habitacion suma evidencia multi-persona, pero no desplaza
                    # la presencia principal fuera de la habitación confirmada por ocupación.
                    anchor_room = max(
                        self.occupancy_confirmed_by_room.items(),
                        key=lambda item: item[1],
                    )[0]
                    resolved_presence_room = normalize_room_name(anchor_room) or resolved_presence_room
                    confidence = max(confidence, 0.93)
                elif sensor_type == "motion" and self._can_displace_presence_locked(
                    previous_room=previous_room,
                    candidate_room=room,
                    sensor_type=sensor_type,
                    previous_seen_at=previous_seen_at,
                ):
                    # El sensor observado debe ganar sobre una prediccion que queda pegada
                    # a la habitación anterior cuando el movimiento es válido en el mapa.
                    resolved_presence_room = room
                    confidence = max(confidence, 0.86)
                    observed_room_forced = True
                elif not self._can_displace_presence_locked(
                    previous_room=previous_room,
                    candidate_room=resolved_presence_room,
                    sensor_type=sensor_type,
                    previous_seen_at=previous_seen_at,
                ):
                    resolved_presence_room = normalize_room_name(previous_room) or room
                    confidence = max(confidence, 0.78)
                    # Mantiene el nodo activo cuando no hay evidencia de desplazamiento adyacente.
                    self.last_active_by_room[resolved_presence_room] = now

                self.current_room = resolved_presence_room
                if observed_room_forced:
                    active_rooms = [resolved_presence_room]
                elif ai_active_rooms:
                    active_rooms = [resolved_presence_room] + [
                        rm for rm in ai_active_rooms if rm != resolved_presence_room
                    ]
                else:
                    active_rooms = [resolved_presence_room]

            elif is_active_event:
                relation = "filtrado_ventana"
                confidence = 0.12
                if self.current_room:
                    active_rooms = [self.current_room]

            else:
                if self.current_room is None and sensor_type != "occupancy":
                    self.current_room = room
                confidence = 0.45
                if self.current_room and not (sensor_type == "occupancy" and self.current_room == room):
                    active_rooms = [self.current_room]

            if transition is not None:
                if transition.get("same_room"):
                    relation = "misma_habitacion"
                elif transition.get("rejected_by_ai"):
                    relation = "no_adyacente_modelo"
                else:
                    relation = (
                        "adyacente"
                        if int(transition.get("support", 0)) >= self.confirmed_edge_support
                        else "desconocida"
                    )

            self._prune_inactive_rooms(now)

            if not active_rooms:
                active_rooms = sorted(self.last_active_by_room.keys())
            occupancy_anchor_rooms = [
                normalize_room_name(rm)
                for rm, ts in self.occupancy_confirmed_by_room.items()
                if now - ts <= timedelta(seconds=self.presence_hold_seconds)
            ]
            active_rooms = [normalize_room_name(rm) for rm in active_rooms if normalize_room_name(rm)]
            active_rooms.extend(occupancy_anchor_rooms)
            active_sensor_rooms = sorted(
                normalize_room_name(rm)
                for rm, sensor_types in self.active_sensor_types_by_room.items()
                if normalize_room_name(rm) and sensor_types
            )
            if occupancy_anchor_rooms:
                active_rooms = occupancy_anchor_rooms + [
                    rm for rm in active_sensor_rooms if rm not in occupancy_anchor_rooms
                ]
            else:
                active_rooms.extend(active_sensor_rooms)
            if not occupancy_anchor_rooms and self.current_room and self.current_room not in active_rooms:
                active_rooms.insert(0, self.current_room)
            active_rooms = list(dict.fromkeys(active_rooms))
            self.current_active_rooms = active_rooms

            occupancy_count = len(active_rooms)
            estimated_people = self._estimate_people_locked(active_rooms)
            occupancy_transformer_count = int(inference_debug.get("occupancy_transformer_people_count") or 0)
            if occupancy_transformer_count > 0:
                estimated_people = max(estimated_people, occupancy_transformer_count)
            self.current_people_estimate = estimated_people
            self.max_people_estimate = max(self.max_people_estimate, estimated_people)

            if transition is not None:
                from_room = str(transition.get("from") or "")
                to_room = str(transition.get("to") or "")
                if from_room and to_room and not self._reference_adjacent_locked(from_room, to_room):
                    transition["reference_layout_adjacent"] = False
                    layout_alert = self._record_non_adjacent_locked(
                        timestamp=now,
                        transition=transition,
                        sensor_type=sensor_type,
                        estimated_people=estimated_people,
                        active_rooms=active_rooms,
                    )
                    relation = "no_adyacente_layout_real"
                else:
                    transition["reference_layout_adjacent"] = True

            inferred_presence = "Presente" if occupancy_count > 0 else "Ausente"

            event = {
                "index": len(self.events),
                "timestamp": to_utc_iso(now),
                "room": room,
                "sensor_type": sensor_type,
                "state": state,
                "entity_id": entity_id,
                "presence_room": self.current_room or room,
                "presence_confidence": round(confidence, 4),
                "active_rooms": active_rooms,
                "inferred_presence": inferred_presence,
                "transition": transition,
                "estimated_people": estimated_people,
                "layout_alert": layout_alert,
                "inference_debug": inference_debug,
                "presence_filter": presence_filter_debug,
                "source": payload.source,
                "input_mode": self.input_mode,
                "ai_mode": (
                    "hf_transformer_markov" if self.ai_model.training_info.get("transformer", {}).get("enabled") else "markov_ai"
                )
                if self.ai_model.ready
                else "rule_based",
            }

            if payload.timestamp is not None:
                lag_ms = (ingress_now - now).total_seconds() * 1000.0
                if 0.0 <= lag_ms <= (30.0 * 60.0 * 1000.0):
                    self.ingestion_latency_ms.append(lag_ms)
                    event["ingestion_latency_ms"] = round(lag_ms, 3)

            processing_ms = (datetime.now(timezone.utc) - ingress_now).total_seconds() * 1000.0
            if 0.0 <= processing_ms <= 60000.0:
                self.processing_latency_ms.append(processing_ms)
            event["processing_ms"] = round(processing_ms, 3)

            self.events.append(event)
            if presence_signal_allowed or not is_active_event:
                self.sequence_history.append(current_event)
            if len(self.events) > self.max_events_buffer:
                self.events = self.events[-self.max_events_buffer :]
                for idx, evt in enumerate(self.events):
                    evt["index"] = idx

            metrics = self._evaluation_metrics_locked()

            response = {
                "presencia_inferida": inferred_presence,
                "habitacion": room,
                "habitacion_inferida_ia": self.current_room or room,
                "confianza_presencia": round(confidence, 4),
                "input_mode": self.input_mode,
                "updated_at": event["timestamp"],
                "relacion_habitaciones": relation,
                "ocupacion_estimada": occupancy_count,
                "personas_estimadas": estimated_people,
                "habitaciones_activas": active_rooms,
                "aristas_activas": len(self.edge_support),
                "transiciones_descartadas_modelo": self.rejected_transitions,
                "modelo_ia_activo": self.ai_model.ready,
                "alerta_layout": layout_alert,
                "filtro_presencia": presence_filter_debug,
                "metricas_evaluacion": metrics,
                "event": event,
            }

        if self.event_sink is not None:
            try:
                await self.event_sink(payload, event, response)
            except Exception:
                # La persistencia es complementaria y no debe interrumpir inferencia.
                pass
        await self.broadcast_event(response)
        return response
