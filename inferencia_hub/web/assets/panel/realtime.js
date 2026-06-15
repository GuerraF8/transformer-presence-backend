import { fetchJson, resolveBackendWebSocketUrl } from "./api.js";
import { edgeKey } from "./format.js";

export function applySnapshotState(state, simData) {
  if (!simData || typeof simData !== "object") return;
  if (Array.isArray(simData.rooms)) {
    state.rooms = [...new Set(simData.rooms.map(String).filter(Boolean))].sort();
  }
  if (Array.isArray(simData.events)) {
    state.events = simData.events.slice(-30000);
  }
  if (simData.layout_reference) {
    state.reference = { ...state.reference, ...simData.layout_reference };
  }
  state.inferredEdges.clear();
  const edges =
    simData.inferred_layout_live?.edges || simData.final_edges || [];
  for (const edge of edges) {
    if (edge?.a && edge?.b) {
      state.inferredEdges.set(
        edgeKey(edge.a, edge.b),
        Number(edge.support || 0),
      );
    }
  }
  const touched = simData.inferred_layout_live?.latest_touched_edge;
  if (Array.isArray(touched) && touched.length === 2) {
    state.latestEdge = edgeKey(touched[0], touched[1]);
  }
  if (simData.evaluation) state.metrics = simData.evaluation;
  if (simData.presence_filter) {
    state.presenceFilter = {
      ...state.presenceFilter,
      ...simData.presence_filter,
    };
  }
  if (simData.ha_entity_catalog) {
    state.haEntityCatalog = {
      ...state.haEntityCatalog,
      ...simData.ha_entity_catalog,
    };
  }
  if (simData.replay) state.replay = { ...state.replay, ...simData.replay };
  if (simData.profile?.room_labels) {
    state.roomLabels = { ...simData.profile.room_labels };
  }
  if (simData.meta?.input_mode) state.replay.mode = simData.meta.input_mode;
  if (simData.meta?.backend_version) {
    state.backendVersion = String(simData.meta.backend_version);
  }
  const presence = simData.presence;
  if (presence && typeof presence === "object") {
    const peopleEstimate = Number(presence.people_estimate || 0);
    state.peopleEstimate = Number.isFinite(peopleEstimate)
      ? Math.max(0, peopleEstimate)
      : 0;
    if (presence.current_room) state.currentRoom = String(presence.current_room);
    if (Array.isArray(presence.active_rooms)) {
      state.activeRooms = presence.active_rooms.map(String).filter(Boolean);
    }
    if (Array.isArray(presence.occupancy_ground_truth_rooms)) {
      state.occupancyRooms =
        presence.occupancy_ground_truth_rooms.map(String).filter(Boolean);
    }
    if (Array.isArray(presence.live_sensor_rooms)) {
      state.liveSensorRooms =
        presence.live_sensor_rooms.map(String).filter(Boolean);
    }
  }
  const latest = state.events.at(-1);
  state.currentRoom ||=
    latest?.presence_room || latest?.room || null;
  if (!state.activeRooms.length && latest?.active_rooms?.length) {
    state.activeRooms = latest.active_rooms.map(String).filter(Boolean);
  }
  if (!state.activeRooms.length && state.currentRoom) {
    state.activeRooms = [state.currentRoom];
  }
  if (!state.latestEdge && latest?.transition?.edge?.length === 2) {
    state.latestEdge = edgeKey(...latest.transition.edge);
  }
}

export function applyEventState(state, event) {
  state.events.push(event);
  if (state.events.length > 30000) state.events.shift();
  const room = String(event.room || "");
  if (room && !state.rooms.includes(room)) {
    state.rooms.push(room);
    state.rooms.sort();
  }
  state.currentRoom = event.presence_room || event.room || state.currentRoom;
  if (Number.isFinite(Number(event.estimated_people))) {
    state.peopleEstimate = Math.max(0, Number(event.estimated_people));
  }
  state.activeRooms = event.active_rooms?.length
    ? event.active_rooms.map(String).filter(Boolean)
    : state.currentRoom
      ? [state.currentRoom]
      : [];
  if (event.sensor_type === "occupancy" && room) {
    const occupancy = new Set(state.occupancyRooms);
    if (String(event.state).toLowerCase() === "on") occupancy.add(room);
    else occupancy.delete(room);
    state.occupancyRooms = [...occupancy];
  }
  if (event.transition?.edge?.length === 2) {
    const key = edgeKey(...event.transition.edge);
    state.inferredEdges.set(key, Number(event.transition.support || 0));
    state.latestEdge = key;
  }
}

export function createRealtimeController({
  state,
  el,
  renderAll,
  setRealSensorConfig,
  scheduleHistoryRefresh,
  setTopStatus,
  windowRef = window,
  WebSocketClass = WebSocket,
}) {
  function applySnapshot(simData) {
    applySnapshotState(state, simData);
    if (simData.real_sensor_config) {
      setRealSensorConfig(simData.real_sensor_config);
    }
    el.modelState.textContent = simData.model?.ready
      ? simData.model.training_info?.transformer?.enabled
        ? "hf_transformer_markov"
        : "markov_ai"
      : "rule_based";
    renderAll();
  }

  function applyEvent(event) {
    applyEventState(state, event);
    if (event.ai_mode) el.modelState.textContent = String(event.ai_mode);
    renderAll();
    scheduleHistoryRefresh(event);
  }

  async function fetchSnapshot() {
    const snapshot =
      (await fetchJson("/api/sim_data", { cache: "no-store" })) || {};
    applySnapshot(snapshot);
    return snapshot;
  }

  function connect() {
    state.ws = new WebSocketClass(
      resolveBackendWebSocketUrl("presencia", windowRef.document.baseURI),
    );
    state.ws.onopen = () => {
      el.wsState.textContent = "conectado";
      setTopStatus("conexion live activa", false);
    };
    state.ws.onerror = () => {
      el.wsState.textContent = "error";
      setTopStatus("error websocket", true);
    };
    state.ws.onclose = () => {
      el.wsState.textContent = "desconectado";
      setTopStatus("conexion cerrada, reconectando", true);
      windowRef.setTimeout(connect, 2200);
    };
    state.ws.onmessage = (message) => {
      try {
        const payload = JSON.parse(message.data);
        if (payload?.kind === "snapshot" && payload.sim_data) {
          applySnapshot(payload.sim_data);
          setTopStatus("snapshot live", false);
        } else {
          applyEvent(payload);
        }
      } catch {
        setTopStatus("mensaje websocket invalido", true);
      }
    };
  }

  return { connect, fetchSnapshot };
}
