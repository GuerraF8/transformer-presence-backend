import { fetchJson } from "./api.js";
import {
  formatBytes,
  formatInteger,
  formatTime,
  isoToLocalInput,
  localInputToIso,
  roomLabel,
} from "./format.js";

const SVG_NS = "http://www.w3.org/2000/svg";
export const HISTORY_FILTER_STORAGE_KEY =
  "inferencia-presencia.history-filters.v1";

export function defaultHistoryFilters() {
  return {
    query: "",
    sensorType: "",
    room: "",
    inputMode: "",
    fromTs: "",
    toTs: "",
  };
}

export function normalizeHistoryFilters(value) {
  const source = value && typeof value === "object" ? value : {};
  return {
    query: String(source.query || ""),
    sensorType: String(source.sensorType || ""),
    room: String(source.room || ""),
    inputMode: ["", "listen", "replay", "simulator"].includes(
      String(source.inputMode || ""),
    )
      ? String(source.inputMode || "")
      : "",
    fromTs: String(source.fromTs || ""),
    toTs: String(source.toTs || ""),
  };
}

export function parseHistoryFilters(rawValue) {
  try {
    return rawValue
      ? normalizeHistoryFilters(JSON.parse(rawValue))
      : defaultHistoryFilters();
  } catch (_error) {
    return defaultHistoryFilters();
  }
}

export function loadHistoryFilters(storage) {
  if (!storage) return defaultHistoryFilters();
  try {
    return parseHistoryFilters(storage.getItem(HISTORY_FILTER_STORAGE_KEY));
  } catch (_error) {
    return defaultHistoryFilters();
  }
}

export function saveHistoryFilters(storage, filters) {
  const normalized = normalizeHistoryFilters(filters);
  if (!storage) return normalized;
  try {
    storage.setItem(HISTORY_FILTER_STORAGE_KEY, JSON.stringify(normalized));
  } catch (_error) {
    // El historial sigue operativo cuando el navegador bloquea el almacenamiento.
  }
  return normalized;
}

export function hasActiveHistoryFilters(filters) {
  return Object.values(normalizeHistoryFilters(filters)).some(Boolean);
}

export function buildHistorySearchParams(filters) {
  return new URLSearchParams({
    query: filters.query,
    sensor_type: filters.sensorType,
    room: filters.room,
    input_mode: filters.inputMode,
    from_ts: filters.fromTs,
    to_ts: filters.toTs,
  });
}

export function normalizeHistoryConfigDraft(value) {
  const source = value && typeof value === "object" ? value : {};
  const validModes = new Set(["listen", "replay", "simulator"]);
  return {
    enabled: source.enabled !== false,
    retention_days: Number(source.retention_days || 365),
    persisted_modes: Array.isArray(source.persisted_modes)
      ? source.persisted_modes.filter((mode) => validModes.has(mode))
      : [],
  };
}

export function createHistoryController({
  state,
  el,
  setMiniStatus,
  appendCell,
  appendBadgeCell,
  documentRef = document,
  windowRef = window,
}) {
  let filterInputTimer = null;

  function getStorage() {
    try {
      return windowRef.localStorage;
    } catch (_error) {
      return null;
    }
  }

  function renderConfig() {
    const remote = state.history.config || {};
    const config = state.history.configDirty
      ? state.history.configDraft
      : remote;
    const draft = normalizeHistoryConfigDraft(config);
    const modes = new Set(
      draft.persisted_modes,
    );
    el.historyEnabled.value = String(draft.enabled);
    el.historyRetentionDays.value = String(draft.retention_days);
    el.historyModeListen.checked = modes.has("listen");
    el.historyModeReplay.checked = modes.has("replay");
    el.historyModeSimulator.checked = modes.has("simulator");
    el.historyConfigTotal.textContent = formatInteger(remote.events_total || 0);
    el.historyConfigSize.textContent = formatBytes(remote.database_size_bytes);
    el.historyConfigRange.textContent = remote.first_timestamp
      ? formatTime(remote.first_timestamp) +
        " - " +
        formatTime(remote.last_timestamp)
      : "Sin eventos";
    el.historyConfigPath.textContent = String(remote.database_path || "-");
    setMiniStatus(
      el.historyConfigStatus,
      state.history.configDirty
        ? "cambios pendientes: pulsa Guardar historial"
        : remote.last_error
          ? "SQLite: " + remote.last_error
          : remote.enabled === false
          ? "persistencia desactivada"
          : "historial operativo",
      !!remote.last_error && !state.history.configDirty,
    );
  }

  function updateConfigDraft() {
    state.history.configDraft = normalizeHistoryConfigDraft({
      enabled: el.historyEnabled.value === "true",
      retention_days: Number(el.historyRetentionDays.value || 365),
      persisted_modes: [
        el.historyModeListen,
        el.historyModeReplay,
        el.historyModeSimulator,
      ]
        .filter((input) => input.checked)
        .map((input) => input.value),
    });
    state.history.configDirty = true;
    renderConfig();
  }

  function populateSelect(
    select,
    values,
    emptyLabel,
    labelFormatter,
    selectedValue = select.value,
  ) {
    select.innerHTML = "";
    const empty = documentRef.createElement("option");
    empty.value = "";
    empty.textContent = emptyLabel;
    select.appendChild(empty);
    (values || []).forEach((value) => {
      const option = documentRef.createElement("option");
      option.value = String(value);
      option.textContent = labelFormatter
        ? labelFormatter(value)
        : roomLabel(value);
      select.appendChild(option);
    });
    select.value = [...select.options].some(
      (option) => option.value === selectedValue,
    )
      ? selectedValue
      : "";
  }

  function renderOptions() {
    const options = state.history.options || {};
    populateSelect(
      el.historySensorType,
      options.sensor_types,
      "Todos",
      undefined,
      state.history.filters.sensorType,
    );
    populateSelect(
      el.historyRoom,
      options.rooms,
      "Todas",
      undefined,
      state.history.filters.room,
    );
    const modes = [
      ...new Set([
        ...(options.input_modes || []),
        "listen",
        "replay",
        "simulator",
      ]),
    ];
    populateSelect(
      el.historyInputMode,
      modes,
      "Todos",
      (mode) => {
        if (mode === "listen") return "Escucha";
        if (mode === "replay") return "Replay";
        if (mode === "simulator") return "Simulador";
        return String(mode);
      },
      state.history.filters.inputMode,
    );
    el.historySensorOptions.innerHTML = "";
    (options.sensors || []).forEach((sensor) => {
      const option = documentRef.createElement("option");
      option.value = String(sensor.entity_id || "");
      option.label = String(sensor.sensor_name || sensor.entity_id || "");
      el.historySensorOptions.appendChild(option);
    });
  }

  function renderEvents() {
    el.eventList.innerHTML = "";
    const eventNoun = state.history.total === 1 ? " evento" : " eventos";
    el.eventSummary.textContent =
      formatInteger(state.history.total) +
      eventNoun +
      (hasActiveHistoryFilters(state.history.filters)
        ? " filtrados"
        : "");
    if (!state.history.items.length) {
      const row = documentRef.createElement("tr");
      const cell = documentRef.createElement("td");
      cell.colSpan = 9;
      cell.textContent = hasActiveHistoryFilters(state.history.filters)
        ? "No hay eventos para los filtros seleccionados"
        : "No hay eventos registrados";
      row.appendChild(cell);
      el.eventList.appendChild(row);
    }
    state.history.items.forEach((event) => {
      const row = documentRef.createElement("tr");
      appendCell(row, "#" + String(event.id || "-"));
      appendCell(row, formatTime(event.event_timestamp));
      appendCell(row, roomLabel(event.room));
      const sensorCell = appendCell(
        row,
        String(event.sensor_name || event.entity_id || "-"),
      );
      sensorCell.title = String(event.entity_id || "");
      appendCell(row, String(event.sensor_type || "other"));
      appendBadgeCell(
        row,
        String(event.state || "-"),
        String(event.state || "").toLowerCase() === "on" ? "on" : "off",
      );
      appendBadgeCell(
        row,
        event.inferred_presence
          ? "Presente: " + roomLabel(event.inferred_room)
          : "Ausente",
        event.inferred_presence ? "on" : "off",
      );
      appendCell(row, String(event.estimated_people || 0));
      appendBadgeCell(
        row,
        event.layout_alert
          ? String(event.layout_alert.cause || "no_adyacente")
          : "ok",
        event.layout_alert ? "alert" : "ok",
      );
      el.eventList.appendChild(row);
    });
    el.historyPageStatus.textContent =
      "Página " + state.history.page + " de " + state.history.pages;
    el.historyPrevBtn.disabled = state.history.page <= 1;
    el.historyNextBtn.disabled = state.history.page >= state.history.pages;
    el.historyNewEventsBtn.hidden = !state.history.newEvents;
  }

  function renderAlerts() {
    const alerts = state.history.alerts;
    el.alertList.innerHTML = "";
    if (!alerts.items.length) {
      const row = documentRef.createElement("tr");
      const cell = documentRef.createElement("td");
      cell.colSpan = 5;
      cell.textContent = hasActiveHistoryFilters(state.history.filters)
        ? "No hay alertas para los filtros seleccionados"
        : "Sin alertas no adyacentes registradas";
      row.appendChild(cell);
      el.alertList.appendChild(row);
    }
    alerts.items.forEach((event) => {
      const alert = event.layout_alert || {};
      const row = documentRef.createElement("tr");
      appendCell(
        row,
        `${roomLabel(alert.from)} -> ${roomLabel(alert.to)}`,
      );
      appendCell(row, formatTime(event.event_timestamp));
      appendBadgeCell(
        row,
        String(alert.cause || "no_adyacente"),
        "alert",
      );
      appendCell(row, String(event.estimated_people || 0));
      appendCell(row, `${Number(alert.gap_seconds || 0)}s`);
      el.alertList.appendChild(row);
    });
    el.alertPageStatus.textContent =
      "Página " +
      alerts.page +
      " de " +
      alerts.pages +
      " · " +
      formatInteger(alerts.total) +
      (alerts.total === 1 ? " alerta" : " alertas");
    el.alertPrevBtn.disabled = alerts.page <= 1;
    el.alertNextBtn.disabled = alerts.page >= alerts.pages;
    el.alertNewEventsBtn.hidden = !alerts.newAlerts;
  }

  function svgElement(name, attributes) {
    const node = documentRef.createElementNS(SVG_NS, name);
    Object.entries(attributes || {}).forEach(([key, value]) => {
      node.setAttribute(key, String(value));
    });
    return node;
  }

  function renderChart() {
    const points = state.history.points || [];
    el.historyChart.innerHTML = "";
    const title = svgElement("title", { id: "historyChartTitle" });
    title.textContent = "Gráfico histórico de presencia y personas estimadas";
    el.historyChart.appendChild(title);
    const description = svgElement("desc", {
      id: "historyChartDescription",
    });
    el.historyChart.appendChild(description);
    if (!points.length) {
      description.textContent =
        "Sin datos históricos para los filtros seleccionados.";
      const text = svgElement("text", {
        x: 450,
        y: 112,
        class: "history-chart-empty",
      });
      text.textContent = "Sin datos históricos";
      el.historyChart.appendChild(text);
      setMiniStatus(el.historyChartStatus, "sin puntos", false);
      return;
    }

    const width = 900;
    const height = 220;
    const left = 48;
    const right = 18;
    const top = 18;
    const bottom = 34;
    const start = new Date(points[0].timestamp).getTime();
    const end = new Date(points.at(-1).timestamp).getTime();
    const duration = Math.max(1, end - start);
    const maxPeople = Math.max(
      1,
      ...points.map((point) => Number(point.people || 0)),
    );
    const x = (timestamp) =>
      left +
      ((new Date(timestamp).getTime() - start) / duration) *
        (width - left - right);
    const presenceY = (present) =>
      present ? top + 34 : height - bottom - 34;
    const peopleY = (people) =>
      height -
      bottom -
      (Number(people || 0) / maxPeople) * (height - top - bottom);

    for (const attributes of [
      {
        x1: left,
        y1: height - bottom,
        x2: width - right,
        y2: height - bottom,
        class: "history-chart-axis",
      },
      {
        x1: left,
        y1: top,
        x2: left,
        y2: height - bottom,
        class: "history-chart-axis",
      },
    ]) {
      el.historyChart.appendChild(svgElement("line", attributes));
    }
    let presencePath = "";
    points.forEach((point, index) => {
      const pointX = x(point.timestamp);
      const pointY = presenceY(point.presence);
      presencePath +=
        index === 0
          ? `M ${pointX} ${pointY}`
          : ` H ${pointX} V ${pointY}`;
    });
    const peoplePath = points
      .map(
        (point, index) =>
          `${index === 0 ? "M" : "L"} ${x(point.timestamp)} ${peopleY(point.people)}`,
      )
      .join(" ");
    el.historyChart.appendChild(
      svgElement("path", {
        d: presencePath,
        class: "history-presence-line",
      }),
    );
    el.historyChart.appendChild(
      svgElement("path", { d: peoplePath, class: "history-people-line" }),
    );
    [
      { x: 10, y: presenceY(true) + 4, text: "ON" },
      { x: 10, y: presenceY(false) + 4, text: "OFF" },
      { x: left, y: height - 10, text: formatTime(points[0].timestamp) },
      {
        x: width - right,
        y: height - 10,
        text: formatTime(points.at(-1).timestamp),
        anchor: "end",
      },
    ].forEach((label) => {
      const node = svgElement("text", {
        x: label.x,
        y: label.y,
        class: "history-chart-label",
        "text-anchor": label.anchor || "start",
      });
      node.textContent = label.text;
      el.historyChart.appendChild(node);
    });
    description.textContent =
      `${points.length} cambios de presencia. ` +
      `Máximo de personas estimadas: ${maxPeople}.`;
    setMiniStatus(
      el.historyChartStatus,
      state.history.truncated
        ? `serie truncada a ${points.length} cambios`
        : `${points.length} cambios`,
      state.history.truncated,
    );
  }

  function render() {
    renderOptions();
    renderEvents();
    renderAlerts();
    renderChart();
  }

  async function fetchConfig() {
    state.history.config = await fetchJson("/api/history/config", {
      cache: "no-store",
    });
    if (!state.history.configDirty) {
      state.history.configDraft = normalizeHistoryConfigDraft(
        state.history.config,
      );
    }
    renderConfig();
    return state.history.config;
  }

  async function fetchHistory({
    includeEvents = true,
    includePresence = true,
    includeAlerts = true,
  } = {}) {
    const params = buildHistorySearchParams(state.history.filters);
    params.set("page", String(state.history.page));
    params.set("page_size", String(state.history.pageSize));
    const alertParams = buildHistorySearchParams(state.history.filters);
    alertParams.set("page", String(state.history.alerts.page));
    alertParams.set("page_size", String(state.history.alerts.pageSize));
    const presenceParams = buildHistorySearchParams(state.history.filters);
    presenceParams.set("max_points", "1000");
    const [eventsResult, presenceResult, alertsResult] = await Promise.all([
      includeEvents
        ? fetchJson("/api/history/events?" + params, { cache: "no-store" })
        : null,
      includePresence
        ? fetchJson("/api/history/presence?" + presenceParams, {
            cache: "no-store",
          })
        : null,
      includeAlerts
        ? fetchJson("/api/history/alerts?" + alertParams, {
            cache: "no-store",
          })
        : null,
    ]);
    if (eventsResult) {
      state.history.items = Array.isArray(eventsResult.items)
        ? eventsResult.items
        : [];
      state.history.total = Number(eventsResult.total || 0);
      state.history.page = Number(eventsResult.page || 1);
      state.history.pages = Number(eventsResult.pages || 1);
      state.history.options = eventsResult.options || state.history.options;
      state.history.newEvents = false;
    }
    if (presenceResult) {
      state.history.points = Array.isArray(presenceResult.points)
        ? presenceResult.points
        : [];
      state.history.sourceEvents = Number(presenceResult.source_events || 0);
      state.history.truncated = !!presenceResult.truncated;
    }
    if (alertsResult) {
      state.history.alerts.items = Array.isArray(alertsResult.items)
        ? alertsResult.items
        : [];
      state.history.alerts.total = Number(alertsResult.total || 0);
      state.history.alerts.page = Number(alertsResult.page || 1);
      state.history.alerts.pages = Number(alertsResult.pages || 1);
      state.history.alerts.newAlerts = false;
    }
    render();
  }

  async function saveConfig() {
    const draft = normalizeHistoryConfigDraft(
      state.history.configDraft || state.history.config,
    );
    const modes = draft.persisted_modes;
    if (!modes.length) {
      setMiniStatus(
        el.historyConfigStatus,
        "selecciona al menos un modo",
        true,
      );
      return;
    }
    try {
      el.historyConfigSaveBtn.disabled = true;
      state.history.config = await fetchJson("/api/history/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          enabled: draft.enabled,
          retention_days: draft.retention_days,
          persisted_modes: modes,
        }),
      });
      state.history.configDraft = normalizeHistoryConfigDraft(
        state.history.config,
      );
      state.history.configDirty = false;
      renderConfig();
      setMiniStatus(
        el.historyConfigStatus,
        "configuración guardada",
        false,
      );
      await fetchHistory();
    } catch (error) {
      setMiniStatus(
        el.historyConfigStatus,
        String(error.message || error),
        true,
      );
    } finally {
      el.historyConfigSaveBtn.disabled = false;
    }
  }

  async function purge() {
    if (el.historyPurgeConfirmation.value !== "BORRAR") return;
    try {
      el.historyPurgeBtn.disabled = true;
      const result = await fetchJson("/api/history/purge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirmation: "BORRAR" }),
      });
      el.historyPurgeConfirmation.value = "";
      setMiniStatus(
        el.historyConfigStatus,
        `historial borrado: ${formatInteger(result.deleted || 0)} eventos`,
        false,
      );
      await Promise.all([fetchConfig(), fetchHistory()]);
    } catch (error) {
      setMiniStatus(
        el.historyConfigStatus,
        String(error.message || error),
        true,
      );
    } finally {
      el.historyPurgeBtn.disabled =
        el.historyPurgeConfirmation.value !== "BORRAR";
    }
  }

  function filtersFromControls() {
    return normalizeHistoryFilters({
      query: el.historyQuery.value.trim(),
      sensorType: el.historySensorType.value,
      room: el.historyRoom.value,
      inputMode: el.historyInputMode.value,
      fromTs: localInputToIso(el.historyFrom.value),
      toTs: localInputToIso(el.historyTo.value),
    });
  }

  function renderFilterControls() {
    const filters = state.history.filters;
    el.historyQuery.value = filters.query;
    el.historySensorType.value = filters.sensorType;
    el.historyRoom.value = filters.room;
    el.historyInputMode.value = filters.inputMode;
    el.historyFrom.value = isoToLocalInput(filters.fromTs);
    el.historyTo.value = isoToLocalInput(filters.toTs);
  }

  function applyFilters() {
    if (filterInputTimer) {
      windowRef.clearTimeout(filterInputTimer);
      filterInputTimer = null;
    }
    state.history.filters = saveHistoryFilters(
      getStorage(),
      filtersFromControls(),
    );
    state.history.page = 1;
    state.history.alerts.page = 1;
    fetchHistory().catch((error) =>
      setMiniStatus(
        el.historyChartStatus,
        String(error.message || error),
        true,
      ),
    );
  }

  function clearFilters() {
    state.history.filters = saveHistoryFilters(
      getStorage(),
      defaultHistoryFilters(),
    );
    renderFilterControls();
    state.history.page = 1;
    state.history.alerts.page = 1;
    fetchHistory().catch((error) =>
      setMiniStatus(
        el.historyChartStatus,
        String(error.message || error),
        true,
      ),
    );
  }

  function scheduleFilterApply() {
    if (filterInputTimer) {
      windowRef.clearTimeout(filterInputTimer);
    }
    filterInputTimer = windowRef.setTimeout(() => {
      filterInputTimer = null;
      applyFilters();
    }, 350);
  }

  function scheduleRefresh(event = null) {
    const refreshEvents = state.history.page === 1;
    const isAlert = !!event?.layout_alert;
    const refreshAlerts = isAlert && state.history.alerts.page === 1;
    if (!refreshEvents) {
      state.history.newEvents = true;
      el.historyNewEventsBtn.hidden = false;
    }
    if (isAlert && !refreshAlerts) {
      state.history.alerts.newAlerts = true;
      el.alertNewEventsBtn.hidden = false;
    }
    if (!refreshEvents && !refreshAlerts) {
      return;
    }
    if (state.history.refreshTimer) {
      windowRef.clearTimeout(state.history.refreshTimer);
    }
    state.history.refreshTimer = windowRef.setTimeout(() => {
      state.history.refreshTimer = null;
      Promise.all([
        fetchHistory({
          includeEvents: refreshEvents,
          includePresence: refreshEvents,
          includeAlerts: refreshAlerts,
        }),
        fetchConfig(),
      ]).catch(() => {});
    }, 500);
  }

  function registerActions() {
    el.historyConfigSaveBtn.addEventListener("click", saveConfig);
    for (const control of [
      el.historyEnabled,
      el.historyRetentionDays,
      el.historyModeListen,
      el.historyModeReplay,
      el.historyModeSimulator,
    ]) {
      control.addEventListener("change", updateConfigDraft);
    }
    el.historyRetentionDays.addEventListener("input", updateConfigDraft);
    el.historyPurgeConfirmation.addEventListener("input", () => {
      el.historyPurgeBtn.disabled =
        el.historyPurgeConfirmation.value !== "BORRAR";
    });
    el.historyPurgeBtn.addEventListener("click", purge);
    el.historyFilterForm.addEventListener("submit", (event) => {
      event.preventDefault();
      applyFilters();
    });
    for (const select of [
      el.historySensorType,
      el.historyRoom,
      el.historyInputMode,
    ]) {
      select.addEventListener("change", applyFilters);
    }
    for (const dateInput of [el.historyFrom, el.historyTo]) {
      dateInput.addEventListener("change", applyFilters);
    }
    el.historyQuery.addEventListener("input", scheduleFilterApply);
    el.historyClearBtn.addEventListener("click", clearFilters);
    el.historyPrevBtn.addEventListener("click", () => {
      if (state.history.page <= 1) return;
      state.history.page -= 1;
      fetchHistory({
        includePresence: false,
        includeAlerts: false,
      }).catch((error) =>
        setMiniStatus(
          el.historyChartStatus,
          String(error.message || error),
          true,
        ),
      );
    });
    el.historyNextBtn.addEventListener("click", () => {
      if (state.history.page >= state.history.pages) return;
      state.history.page += 1;
      fetchHistory({
        includePresence: false,
        includeAlerts: false,
      }).catch((error) =>
        setMiniStatus(
          el.historyChartStatus,
          String(error.message || error),
          true,
        ),
      );
    });
    el.historyNewEventsBtn.addEventListener("click", () => {
      state.history.page = 1;
      fetchHistory({ includeAlerts: false }).catch((error) =>
        setMiniStatus(
          el.historyChartStatus,
          String(error.message || error),
          true,
        ),
      );
    });
    el.alertPrevBtn.addEventListener("click", () => {
      if (state.history.alerts.page <= 1) return;
      state.history.alerts.page -= 1;
      fetchHistory({
        includeEvents: false,
        includePresence: false,
      }).catch((error) =>
        setMiniStatus(
          el.historyChartStatus,
          String(error.message || error),
          true,
        ),
      );
    });
    el.alertNextBtn.addEventListener("click", () => {
      if (state.history.alerts.page >= state.history.alerts.pages) return;
      state.history.alerts.page += 1;
      fetchHistory({
        includeEvents: false,
        includePresence: false,
      }).catch((error) =>
        setMiniStatus(
          el.historyChartStatus,
          String(error.message || error),
          true,
        ),
      );
    });
    el.alertNewEventsBtn.addEventListener("click", () => {
      state.history.alerts.page = 1;
      fetchHistory({
        includeEvents: false,
        includePresence: false,
      }).catch((error) =>
        setMiniStatus(
          el.historyChartStatus,
          String(error.message || error),
          true,
        ),
      );
    });
    if (typeof windowRef.addEventListener === "function") {
      windowRef.addEventListener("storage", (event) => {
        if (event.key !== HISTORY_FILTER_STORAGE_KEY) return;
        state.history.filters = parseHistoryFilters(event.newValue);
        renderFilterControls();
        state.history.page = 1;
        state.history.alerts.page = 1;
        fetchHistory().catch((error) =>
          setMiniStatus(
            el.historyChartStatus,
            String(error.message || error),
            true,
          ),
        );
      });
    }
  }

  function initializeFilters() {
    state.history.filters = loadHistoryFilters(getStorage());
    renderFilterControls();
  }

  return {
    fetch: fetchHistory,
    fetchConfig,
    initializeFilters,
    registerActions,
    render,
    renderConfig,
    scheduleRefresh,
  };
}
