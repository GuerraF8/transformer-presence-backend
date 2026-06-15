import { fetchJson } from "./api.js";
import { roomLabel, toMs, toPercent } from "./format.js";
import { adjacencyToEdges, drawMap } from "./map.js";
import { numberFromSelect } from "./replay-training.js";
import { collectRooms } from "./rooms.js";
import { setMiniStatus } from "./ui.js";

function selectOption(select, value) {
  const stringValue = String(value);
  if ([...select.options].some((option) => option.value === stringValue)) {
    select.value = stringValue;
  }
}

export function normalizePresenceFilterDraft(value) {
  const source = value && typeof value === "object" ? value : {};
  return {
    enabled: source.enabled !== false,
    window_seconds: Number(source.window_seconds || 20),
    min_motion_events: Number(source.min_motion_events || 2),
    min_distinct_rooms: Number(source.min_distinct_rooms || 1),
  };
}

export function createDashboardController({
  state,
  el,
  renderAll,
  documentRef = document,
}) {
  function renderKpis() {
    el.totalEvents.textContent = String(state.events.length);
    el.totalRooms.textContent = String(collectRooms(state).length);
    const people = state.metrics?.people || {};
    el.peopleNow.textContent = String(Number(people.current_estimate || 0));
    el.peopleMax.textContent = String(Number(people.max_observed || 0));
    el.peopleSummary.textContent = String(Number(state.peopleEstimate || 0));
    el.backendVersion.textContent = String(state.backendVersion || "-");
  }

  function renderMetrics() {
    const metrics = state.metrics || {};
    const map = metrics.map || {};
    const quality = map.live_confirmed_quality || {};
    const nonAdjacent = metrics.non_adjacent || {};
    const latency = metrics.latency || {};
    el.mapPrecision.textContent = toPercent(quality.precision);
    el.mapRecall.textContent = toPercent(quality.recall);
    el.mapF1.textContent = toPercent(quality.f1);
    el.mapTpFpFn.textContent =
      `TP ${quality.tp || 0} | FP ${quality.fp || 0} | FN ${quality.fn || 0}`;
    el.mapSupportSummary.textContent =
      `${map.live_edges_confirmed || 0} confirmadas / ` +
      `${map.reference_edges || 0} reales`;
    el.nonAdjTotal.textContent = String(nonAdjacent.total || 0);
    el.nonAdjBreakdown.textContent =
      `m${nonAdjacent.multi_person_probable || 0} ` +
      `p${nonAdjacent.pet_or_noise || 0} ` +
      `e${nonAdjacent.sensor_or_data_error || 0}`;
    el.latInP95.textContent = toMs(latency.ingestion?.p95_ms);
    el.latProcP95.textContent = toMs(latency.processing?.p95_ms);
  }

  function renderPresenceFilter() {
    const remote = state.presenceFilter || {};
    const config = state.presenceFilterDirty
      ? state.presenceFilterDraft
      : remote;
    const draft = normalizePresenceFilterDraft(config);
    el.petFilterEnabled.value = draft.enabled ? "true" : "false";
    selectOption(el.petFilterWindowInput, draft.window_seconds);
    selectOption(el.petFilterMinEventsInput, draft.min_motion_events);
    selectOption(el.petFilterMinRoomsInput, draft.min_distinct_rooms);
    setMiniStatus(
      el.petFilterStatus,
      state.presenceFilterDirty
        ? "cambios pendientes: pulsa Aplicar filtro"
        : `filtrados: ${remote.suppressed_total || 0} | ` +
          `ventana activa: ${remote.pending_motion_events || 0}`,
      false,
    );
  }

  function updatePresenceFilterDraft() {
    state.presenceFilterDraft = {
      enabled: el.petFilterEnabled.value !== "false",
      window_seconds: numberFromSelect(el.petFilterWindowInput, 20),
      min_motion_events: numberFromSelect(el.petFilterMinEventsInput, 2),
      min_distinct_rooms: numberFromSelect(el.petFilterMinRoomsInput, 1),
    };
    state.presenceFilterDirty = true;
    renderPresenceFilter();
  }

  function renderMaps() {
    const rooms = collectRooms(state);
    const referenceEdges = state.reference.edges?.length
      ? state.reference.edges.map((edge) => ({
          a: edge.a,
          b: edge.b,
          support: 1,
        }))
      : adjacencyToEdges(state.reference.adjacency || {});
    const inferredEdges = [...state.inferredEdges.entries()].map(
      ([key, support]) => {
        const [a, b] = key.split("|");
        return { a, b, support };
      },
    );
    const roomLabels = {
      ...(state.reference.room_labels || {}),
      ...state.roomLabels,
    };
    const options = {
      activeRoom: state.currentRoom,
      activeRooms: state.activeRooms,
      occupancyRooms: state.occupancyRooms,
      roomLabels,
    };
    drawMap(el.realGraph, rooms, referenceEdges, {
      ...options,
      mode: "reference",
    });
    drawMap(el.inferredGraph, rooms, inferredEdges, {
      ...options,
      mode: "inferred",
      latestEdge: state.latestEdge,
    });
    const presence = state.activeRooms.length
      ? state.activeRooms
          .map((room) => roomLabels[room] || roomLabel(room))
          .join(", ")
      : "sin presencia";
    el.realMapMeta.textContent =
      `version ${state.reference.version || 0} | ` +
      `source ${state.reference.source || "-"} | presencia ${presence}`;
    el.inferredMapMeta.textContent =
      `${inferredEdges.length} aristas | personas ` +
      `${state.metrics?.people?.current_estimate || 0} | presencia ${presence}`;
  }

  function render() {
    renderKpis();
    renderMetrics();
    renderPresenceFilter();
    renderMaps();
  }

  function setMapTab(tab) {
    state.activeMapTab = tab === "live" ? "live" : "fixed";
    const live = state.activeMapTab === "live";
    el.mapTabFixed.classList.toggle("active", !live);
    el.mapTabLive.classList.toggle("active", live);
    el.mapTabFixed.setAttribute("aria-selected", String(!live));
    el.mapTabLive.setAttribute("aria-selected", String(live));
    el.fixedMapPanel.hidden = live;
    el.liveMapPanel.hidden = !live;
    el.fixedMapPanel.classList.toggle("active", !live);
    el.liveMapPanel.classList.toggle("active", live);
  }

  async function fetchPresenceFilter() {
    state.presenceFilter =
      (await fetchJson("/api/presence_filter", { cache: "no-store" })) || {};
    if (!state.presenceFilterDirty) {
      state.presenceFilterDraft = normalizePresenceFilterDraft(
        state.presenceFilter,
      );
    }
    renderPresenceFilter();
    return state.presenceFilter;
  }

  async function applyPresenceFilter() {
    const payload = normalizePresenceFilterDraft(
      state.presenceFilterDraft || state.presenceFilter,
    );
    try {
      state.presenceFilter =
        (await fetchJson("/api/presence_filter", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        })) || payload;
      state.presenceFilterDraft = normalizePresenceFilterDraft(
        state.presenceFilter,
      );
      state.presenceFilterDirty = false;
      renderPresenceFilter();
      setMiniStatus(el.petFilterStatus, "filtro aplicado", false);
    } catch (error) {
      setMiniStatus(el.petFilterStatus, String(error.message || error), true);
    }
  }

  async function fetchEvaluationMetrics() {
    const payload = await fetchJson("/api/evaluation_metrics?limit=120", {
      cache: "no-store",
    });
    if (payload?.metrics) state.metrics = payload.metrics;
    if (payload?.layout_reference) {
      state.reference = {
        ...state.reference,
        ...payload.layout_reference,
      };
    }
    renderAll();
  }

  function registerActions() {
    el.petFilterApplyBtn.addEventListener("click", applyPresenceFilter);
    for (const select of [
      el.petFilterEnabled,
      el.petFilterWindowInput,
      el.petFilterMinEventsInput,
      el.petFilterMinRoomsInput,
    ]) {
      select.addEventListener("change", updatePresenceFilterDraft);
    }
    el.mapTabFixed.addEventListener("click", () => setMapTab("fixed"));
    el.mapTabLive.addEventListener("click", () => setMapTab("live"));
  }

  return {
    fetchEvaluationMetrics,
    fetchPresenceFilter,
    registerActions,
    render,
    setMapTab,
  };
}
