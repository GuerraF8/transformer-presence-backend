import { fetchJson } from "./api.js";
import { formatTime, roomLabel } from "./format.js";
import {
  cloneRealSensorConfig,
  realSensorAssignmentChanged,
} from "./real-sensors.js";
import { collectRooms } from "./rooms.js";

export function createHomeAssistantController({
  state,
  el,
  appendBadgeCell,
  setMiniStatus,
  renderAll,
  sensorSelectionController = null,
  documentRef = document,
  windowRef = window,
}) {
  function workingConfig() {
    if (!state.realSensorDraft) {
      state.realSensorDraft = cloneRealSensorConfig(state.realSensorConfig);
    }
    return state.realSensorDraft;
  }

  function setConfig(config) {
    state.realSensorConfig = cloneRealSensorConfig({
      ...state.realSensorConfig,
      ...(config || {}),
    });
    if (!state.realSensorDirty) {
      state.realSensorDraft = cloneRealSensorConfig(state.realSensorConfig);
    }
  }

  function savedAssignments() {
    return new Map(
      (state.realSensorConfig.assignments || [])
        .filter((item) => item?.entity_id)
        .map((item) => [String(item.entity_id), item]),
    );
  }

  function markDirty(message) {
    state.realSensorDirty = true;
    if (message) setMiniStatus(el.realSensorStatus, message, false);
  }

  function upsertAssignment(entityId) {
    const id = String(entityId || "").trim();
    if (!id) return null;
    const config = workingConfig();
    config.assignments ||= [];
    let assignment = config.assignments.find((item) => item?.entity_id === id);
    if (!assignment) {
      assignment = {
        entity_id: id,
        room: "",
        enabled: false,
        sensor_type: "auto",
      };
      config.assignments.push(assignment);
    }
    return assignment;
  }

  function configuredRooms() {
    return [
      ...new Set([...(workingConfig().rooms || []), ...collectRooms(state)]),
    ]
      .map((room) => String(room || "").trim())
      .filter(Boolean)
      .sort();
  }

  function wireCatalogControls() {
    if (!el.haSensorList) return;
    el.haSensorList
      .querySelectorAll("[data-real-sensor-enabled]")
      .forEach((input) => {
        input.addEventListener("change", () => {
          const entityId = input.dataset.realSensorEnabled || "";
          const assignment = upsertAssignment(entityId);
          if (!assignment) return;
          const escaped = CSS.escape(entityId);
          const room = el.haSensorList.querySelector(
            `[data-real-sensor-room='${escaped}']`,
          );
          const type = el.haSensorList.querySelector(
            `[data-real-sensor-type='${escaped}']`,
          );
          assignment.enabled = input.checked;
          assignment.room = room?.value || assignment.room;
          assignment.sensor_type =
            type?.value || assignment.sensor_type || "auto";
          if (assignment.enabled && !assignment.room) {
            assignment.room = configuredRooms()[0] || "";
          }
          markDirty("cambio pendiente: confirma para aplicar sensores reales");
          renderCatalog();
        });
      });
    el.haSensorList
      .querySelectorAll("[data-real-sensor-room]")
      .forEach((select) => {
        select.addEventListener("change", () => {
          const assignment = upsertAssignment(
            select.dataset.realSensorRoom || "",
          );
          if (!assignment) return;
          assignment.room = select.value;
          assignment.enabled = !!select.value;
          markDirty("cambio pendiente: confirma para aplicar sensores reales");
          renderCatalog();
        });
      });
    el.haSensorList
      .querySelectorAll("[data-real-sensor-type]")
      .forEach((select) => {
        select.addEventListener("change", () => {
          const assignment = upsertAssignment(
            select.dataset.realSensorType || "",
          );
          if (!assignment) return;
          assignment.sensor_type = select.value || "auto";
          markDirty("cambio pendiente: confirma para aplicar sensores reales");
          renderCatalog();
        });
      });
  }

  function renderCatalog() {
    if (sensorSelectionController) {
      sensorSelectionController.renderEntities();
      const catalog = state.haEntityCatalog || {};
      el.haSensorStatus.textContent =
        `${catalog.auto_discovery === false ? "lista explícita" : "catálogo completo"} | ` +
        (catalog.scanned_at ? formatTime(catalog.scanned_at) : "sin escaneo");
      return;
    }
    const catalog = state.haEntityCatalog || {};
    const config = workingConfig();
    const entities = Array.isArray(catalog.entities) ? catalog.entities : [];
    const supported = entities.filter((entity) => entity?.supported !== false);
    const assignments = config.assignments || [];
    const assignmentByEntity = new Map(
      assignments
        .filter((item) => item?.entity_id)
        .map((item) => [String(item.entity_id), item]),
    );
    const saved = savedAssignments();
    const rooms = configuredRooms();
    const enabled = assignments.filter(
      (item) => item && item.enabled !== false && item.room,
    );
    el.haSensorSummary.textContent =
      `${supported.length} compatibles / ${entities.length} detectadas`;
    const pending = assignments.filter((item) =>
      realSensorAssignmentChanged(item, saved.get(String(item?.entity_id))),
    ).length;
    el.realSensorSummary.textContent =
      `${enabled.length} sensores activos / ${rooms.length} habitaciones reales` +
      (state.realSensorDirty ? ` | ${pending} cambios pendientes` : "");
    el.realSensorRequireSelect.value =
      config.require_explicit_selection === false ? "false" : "true";
    el.realSensorSearchInput.value = state.realSensorSearch;
    el.realSensorApplyBtn.disabled = !state.realSensorDirty;
    el.realSensorResetBtn.disabled = !state.realSensorDirty;
    el.haSensorStatus.textContent =
      `${catalog.auto_discovery === false ? "lista explicita" : "auto discovery"} | ` +
      (catalog.scanned_at ? formatTime(catalog.scanned_at) : "sin escaneo");
    el.haSensorList.innerHTML = "";

    const query = state.realSensorSearch.trim().toLowerCase();
    const filtered = entities.filter((entity) => {
      if (!query) return true;
      return [
        entity.entity_id,
        entity.name,
        entity.domain,
        entity.state,
        entity.sensor_type,
        entity.room,
        entity.device_class,
      ]
        .map((value) => String(value || "").toLowerCase())
        .join(" ")
        .includes(query);
    });
    if (!filtered.length) {
      const row = documentRef.createElement("tr");
      const cell = documentRef.createElement("td");
      cell.colSpan = 5;
      cell.textContent = entities.length
        ? "Sin entidades para la busqueda actual"
        : "Sin catalogo recibido desde Home Assistant";
      row.appendChild(cell);
      el.haSensorList.appendChild(row);
      return;
    }
    filtered.slice(0, 160).forEach((entity) => {
      const entityId = String(entity.entity_id || "");
      const assignment = assignmentByEntity.get(entityId) || {};
      const selected = assignment.enabled !== false && !!assignment.room;
      const changed = realSensorAssignmentChanged(
        assignment,
        saved.get(entityId),
      );
      const row = documentRef.createElement("tr");
      row.classList.toggle("real-sensor-selected", selected);
      row.classList.toggle("real-sensor-pending", changed);

      const useCell = documentRef.createElement("td");
      const checkbox = documentRef.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = selected;
      checkbox.dataset.realSensorEnabled = entityId;
      checkbox.setAttribute("aria-label", `Usar ${entityId}`);
      useCell.appendChild(checkbox);
      row.appendChild(useCell);

      const name = documentRef.createElement("th");
      name.scope = "row";
      name.textContent = entityId || "-";
      row.appendChild(name);

      const typeCell = documentRef.createElement("td");
      const type = documentRef.createElement("select");
      type.dataset.realSensorType = entityId;
      type.setAttribute("aria-label", `Tipo de sensor para ${entityId}`);
      for (const value of ["auto", "motion", "door", "occupancy", "other"]) {
        type.add(new Option(value, value));
      }
      type.value = String(
        assignment.sensor_type || entity.sensor_type || "auto",
      );
      typeCell.appendChild(type);
      row.appendChild(typeCell);

      const roomCell = documentRef.createElement("td");
      const room = documentRef.createElement("select");
      room.dataset.realSensorRoom = entityId;
      room.setAttribute("aria-label", `Habitación asignada a ${entityId}`);
      room.add(new Option("Sin asignar", ""));
      for (const value of rooms) room.add(new Option(roomLabel(value), value));
      room.value = String(assignment.room || "");
      roomCell.appendChild(room);
      row.appendChild(roomCell);
      appendBadgeCell(
        row,
        String(entity.state || "-"),
        String(entity.state || "").toLowerCase() === "on" ? "on" : "off",
      );
      el.haSensorList.appendChild(row);
    });
    wireCatalogControls();
  }

  function renderDiagnostics() {
    const catalog = state.haEntityCatalog || {};
    const actions = state.haActions || {};
    const entities = catalog.entities || [];
    const supported = entities.filter((entity) => entity?.supported !== false);
    const pending = actions.pending || [];
    const latest = (actions.recent_results || [])[0];
    const entries = Object.values(actions.integration_status?.entries || {});
    const heartbeat = entries.sort(
      (a, b) =>
        (Date.parse(b.last_seen_at || "") || 0) -
        (Date.parse(a.last_seen_at || "") || 0),
    )[0];
    el.haDiagReceivedAt.textContent = catalog.received_at
      ? formatTime(catalog.received_at)
      : "-";
    el.haDiagScannedAt.textContent = catalog.scanned_at
      ? formatTime(catalog.scanned_at)
      : "-";
    el.haDiagSource.textContent = String(catalog.source || "-");
    el.haDiagPending.textContent = String(pending.length);
    el.haDiagHeartbeat.textContent = heartbeat?.last_seen_at
      ? formatTime(heartbeat.last_seen_at)
      : "-";
    el.haDiagLastResult.textContent = latest
      ? `${latest.status || "-"} | ${latest.action || latest.request_id || "accion HA"}`
      : "-";
    el.haDiagEntry.textContent = String(
      heartbeat?.entry_id || catalog.entry_id || "-",
    );
    let message =
      `Catálogo recibido: ${entities.length} entidades; ` +
      `${supported.length} con categoría autodetectada.`;
    let error = false;
    if (!heartbeat && !catalog.received_at) {
      message =
        "El backend no ve heartbeat ni catálogo de la integración Home Assistant.";
      error = true;
    } else if (!entities.length) {
      message = "Home Assistant publicó un catálogo vacío.";
      error = true;
    }
    if (pending.length) {
      message += ` Hay ${pending.length} acción(es) pendientes.`;
      error = true;
    }
    if (heartbeat?.last_error) {
      message += ` Último error HA: ${heartbeat.last_error}`;
      error = true;
    }
    if (latest?.action === "create_test_sensors") {
      message =
        `Recursos de prueba: ${(latest.created_sensors || []).length} sensores y ` +
        `${(latest.created_areas || []).length} áreas nuevas.`;
      error = false;
    } else if (
      ["remove_test_sensors", "remove_test_resources"].includes(latest?.action)
    ) {
      message =
        `Recursos eliminados: ${(latest.removed_sensors || []).length} sensores y ` +
        `${(latest.removed_areas || []).length} áreas. ` +
        `Áreas conservadas: ${(latest.preserved_areas || []).length}.`;
      error = latest?.status === "error";
    }
    setMiniStatus(el.haDiagnosticStatus, message, error);
  }

  function render() {
    renderCatalog();
    renderDiagnostics();
  }

  async function fetchCatalog() {
    state.haEntityCatalog =
      (await fetchJson("/api/ha_entities", { cache: "no-store" })) || {};
    render();
    return state.haEntityCatalog;
  }

  async function fetchActions() {
    state.haActions =
      (await fetchJson("/api/ha_actions", { cache: "no-store" })) || {};
    renderDiagnostics();
    return state.haActions;
  }

  async function fetchConfig() {
    const payload = await fetchJson("/api/real_sensor_config", {
      cache: "no-store",
    });
    if (payload?.config) setConfig(payload.config);
    if (payload?.catalog) state.haEntityCatalog = payload.catalog;
    render();
    return payload;
  }

  async function refreshDiagnostics() {
    await Promise.allSettled([fetchCatalog(), fetchActions(), fetchConfig()]);
    renderDiagnostics();
  }

  async function requestAction(action, payload = {}) {
    return fetchJson("/api/ha_actions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, ...payload }),
    });
  }

  function scheduleRefresh() {
    [1200, 3200, 6200].forEach((delay) => {
      windowRef.setTimeout(() => refreshDiagnostics().catch(() => {}), delay);
    });
  }

  async function refreshCatalog() {
    try {
      const result = await requestAction("refresh_catalog");
      setMiniStatus(
        el.haSensorStatus,
        `refresco solicitado a Home Assistant | ${result.request_id || ""}`,
        false,
      );
      await fetchActions();
      scheduleRefresh();
    } catch (error) {
      setMiniStatus(el.haSensorStatus, String(error.message || error), true);
    }
  }

  async function createTestSensors() {
    try {
      const result = await requestAction("create_test_sensors", {
        rooms: el.haTestRoomsInput?.value || "bedroom,kitchen,living",
        include_occupancy: el.haTestOccupancyInput?.value !== "false",
        initial_state: el.haTestInitialStateInput?.value || "off",
      });
      setMiniStatus(
        el.haSensorStatus,
        `creación solicitada a Home Assistant | ${result.request_id || ""}`,
        false,
      );
      await fetchActions();
      scheduleRefresh();
    } catch (error) {
      setMiniStatus(el.haSensorStatus, String(error.message || error), true);
    }
  }

  async function removeTestResources(removeAreas) {
    try {
      const action = removeAreas
        ? "remove_test_resources"
        : "remove_test_sensors";
      const result = await requestAction(action);
      setMiniStatus(
        el.haSensorStatus,
        `eliminación solicitada a Home Assistant | ${result.request_id || ""}`,
        false,
      );
      await fetchActions();
      scheduleRefresh();
    } catch (error) {
      setMiniStatus(el.haSensorStatus, String(error.message || error), true);
    }
  }

  function buildPayload() {
    const config = workingConfig();
    return {
      rooms: configuredRooms(),
      assignments: (config.assignments || [])
        .map((item) => ({
          entity_id: String(item.entity_id || "").trim(),
          room: String(item.room || "").trim(),
          enabled: item.enabled !== false,
          sensor_type: String(item.sensor_type || "auto"),
          training_role: String(item.training_role || "signal"),
        }))
        .filter((item) => item.entity_id && item.room),
      require_explicit_selection:
        el.realSensorRequireSelect.value !== "false",
    };
  }

  async function applyConfig() {
    try {
      const payload = buildPayload();
      const result = await fetchJson("/api/real_sensor_config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      state.realSensorDirty = false;
      if (result?.config) setConfig(result.config);
      if (result?.catalog) state.haEntityCatalog = result.catalog;
      state.realSensorDraft = cloneRealSensorConfig(state.realSensorConfig);
      setMiniStatus(el.realSensorStatus, "sensores reales aplicados", false);
      renderAll();
    } catch (error) {
      setMiniStatus(el.realSensorStatus, String(error.message || error), true);
    }
  }

  function addRoom() {
    const room = el.realSensorNewRoomInput.value
      .trim()
      .toLowerCase()
      .replace(/\s+/g, "_");
    if (!room) return;
    workingConfig().rooms = [
      ...new Set([...(workingConfig().rooms || []), room]),
    ].sort();
    el.realSensorNewRoomInput.value = "";
    markDirty(`habitacion pendiente: ${roomLabel(room)}`);
    renderAll();
  }

  function resetDraft() {
    state.realSensorDraft = cloneRealSensorConfig(state.realSensorConfig);
    state.realSensorDirty = false;
    setMiniStatus(el.realSensorStatus, "cambios descartados", false);
    renderCatalog();
  }

  function registerActions() {
    el.haRefreshCatalogBtn?.addEventListener("click", refreshCatalog);
    el.haCreateTestSensorsBtn?.addEventListener("click", createTestSensors);
    el.haRemoveTestSensorsBtn?.addEventListener("click", () =>
      removeTestResources(false),
    );
    el.haRemoveTestResourcesBtn?.addEventListener("click", () => {
      if (
        windowRef.confirm(
          "¿Eliminar los sensores y las áreas de prueba que pertenecen a la integración?",
        )
      ) {
        removeTestResources(true);
      }
    });
    el.haCheckDiagnosticsBtn?.addEventListener("click", () => {
      refreshDiagnostics().catch((error) =>
        setMiniStatus(
          el.haDiagnosticStatus,
          String(error.message || error),
          true,
        ),
      );
    });
    if (!sensorSelectionController) {
      el.realSensorAddRoomBtn?.addEventListener("click", addRoom);
      el.realSensorSearchInput?.addEventListener("input", () => {
        state.realSensorSearch = el.realSensorSearchInput.value || "";
        renderCatalog();
      });
      el.realSensorRequireSelect?.addEventListener("change", () => {
        workingConfig().require_explicit_selection =
          el.realSensorRequireSelect.value !== "false";
        markDirty("modo pendiente: confirma para aplicar sensores reales");
        renderCatalog();
      });
      el.realSensorResetBtn?.addEventListener("click", resetDraft);
      el.realSensorApplyBtn?.addEventListener("click", applyConfig);
    }
  }

  return {
    fetchActions,
    fetchCatalog,
    fetchConfig,
    refreshDiagnostics,
    registerActions,
    render,
    setConfig,
  };
}
