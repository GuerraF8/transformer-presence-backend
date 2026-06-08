    import { buildReplayPayload } from "./replay.js";

    const queryParams = new URLSearchParams(window.location.search);
    const embeddedMode = queryParams.get("embedded") === "1";
    const devMode = !embeddedMode || queryParams.get("dev") === "1";

    const state = {
      ws: null,
      rooms: [],
      events: [],
      currentRoom: null,
      activeRooms: [],
      occupancyRooms: [],
      liveSensorRooms: [],
      latestEdge: null,
      inferredEdges: new Map(),
      reference: {
        version: 0,
        source: "auto",
        rooms: [],
        adjacency: {},
        edges: [],
        adjacency_text: "",
      },
      metrics: null,
      replay: {
        mode: "listen",
        running: false,
        paused: false,
        step_budget: 0,
        progress: 0,
        processed_events: 0,
        total_events: 0,
        last_error: null,
        last_replay_config: {},
      },
      scenarioTemplates: {},
      layoutTextDirty: false,
      applyingTemplate: false,
      activeMapTab: "fixed",
      eventSort: "time",
      modelInfo: {},
      presenceFilter: {},
      haEntityCatalog: {},
      haActions: {},
      realSensorConfig: { rooms: [], assignments: [], require_explicit_selection: true },
      realSensorDraft: null,
      realSensorDirty: false,
      realSensorSearch: "",
      trainingPollTimer: null,
      history: {
        config: {},
        items: [],
        total: 0,
        page: 1,
        pages: 1,
        pageSize: 50,
        options: { sensors: [], sensor_types: [], rooms: [], input_modes: [] },
        points: [],
        sourceEvents: 0,
        truncated: false,
        newEvents: false,
        refreshTimer: null,
        filters: {
          query: "",
          sensorType: "",
          room: "",
          inputMode: "listen",
          fromTs: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
          toTs: "",
        },
      },
    };

    const el = {
      wsState: document.getElementById("wsState"),
      modelState: document.getElementById("modelState"),
      totalEvents: document.getElementById("totalEvents"),
      totalRooms: document.getElementById("totalRooms"),
      peopleNow: document.getElementById("peopleNow"),
      peopleMax: document.getElementById("peopleMax"),
      mapPrecision: document.getElementById("mapPrecision"),
      mapRecall: document.getElementById("mapRecall"),
      mapF1: document.getElementById("mapF1"),
      mapTpFpFn: document.getElementById("mapTpFpFn"),
      mapSupportSummary: document.getElementById("mapSupportSummary"),
      nonAdjTotal: document.getElementById("nonAdjTotal"),
      nonAdjBreakdown: document.getElementById("nonAdjBreakdown"),
      latInP95: document.getElementById("latInP95"),
      latProcP95: document.getElementById("latProcP95"),
      replaySummary: document.getElementById("replaySummary"),
      modeSummary: document.getElementById("modeSummary"),
      modeStatus: document.getElementById("modeStatus"),
      modeListenBtn: document.getElementById("modeListenBtn"),
      modeReplayBtn: document.getElementById("modeReplayBtn"),
      haSensorSummary: document.getElementById("haSensorSummary"),
      haSensorStatus: document.getElementById("haSensorStatus"),
      haSensorList: document.getElementById("haSensorList"),
      haRefreshCatalogBtn: document.getElementById("haRefreshCatalogBtn"),
      haCreateTestSensorsBtn: document.getElementById("haCreateTestSensorsBtn"),
      haCheckDiagnosticsBtn: document.getElementById("haCheckDiagnosticsBtn"),
      haTestRoomsInput: document.getElementById("haTestRoomsInput"),
      haTestOccupancyInput: document.getElementById("haTestOccupancyInput"),
      haTestInitialStateInput: document.getElementById("haTestInitialStateInput"),
      haDiagReceivedAt: document.getElementById("haDiagReceivedAt"),
      haDiagScannedAt: document.getElementById("haDiagScannedAt"),
      haDiagSource: document.getElementById("haDiagSource"),
      haDiagPending: document.getElementById("haDiagPending"),
      haDiagHeartbeat: document.getElementById("haDiagHeartbeat"),
      haDiagLastResult: document.getElementById("haDiagLastResult"),
      haDiagEntry: document.getElementById("haDiagEntry"),
      haDiagnosticStatus: document.getElementById("haDiagnosticStatus"),
      realSensorSummary: document.getElementById("realSensorSummary"),
      realSensorNewRoomInput: document.getElementById("realSensorNewRoomInput"),
      realSensorSearchInput: document.getElementById("realSensorSearchInput"),
      realSensorRequireSelect: document.getElementById("realSensorRequireSelect"),
      realSensorAddRoomBtn: document.getElementById("realSensorAddRoomBtn"),
      realSensorResetBtn: document.getElementById("realSensorResetBtn"),
      realSensorApplyBtn: document.getElementById("realSensorApplyBtn"),
      realSensorStatus: document.getElementById("realSensorStatus"),
      petFilterEnabled: document.getElementById("petFilterEnabled"),
      petFilterWindowInput: document.getElementById("petFilterWindowInput"),
      petFilterMinEventsInput: document.getElementById("petFilterMinEventsInput"),
      petFilterMinRoomsInput: document.getElementById("petFilterMinRoomsInput"),
      petFilterApplyBtn: document.getElementById("petFilterApplyBtn"),
      petFilterStatus: document.getElementById("petFilterStatus"),
      replayProgress: document.getElementById("replayProgress"),
      replayStatus: document.getElementById("replayStatus"),
      topStatus: document.getElementById("topStatus"),
      realMapMeta: document.getElementById("realMapMeta"),
      inferredMapMeta: document.getElementById("inferredMapMeta"),
      realGraph: document.getElementById("realGraph"),
      inferredGraph: document.getElementById("inferredGraph"),
      alertList: document.getElementById("alertList"),
      layoutMeta: document.getElementById("layoutMeta"),
      layoutText: document.getElementById("layoutText"),
      layoutApplyBtn: document.getElementById("layoutApplyBtn"),
      layoutStatus: document.getElementById("layoutStatus"),
      eventSummary: document.getElementById("eventSummary"),
      eventList: document.getElementById("eventList"),
      csvPath: document.getElementById("csvPath"),
      speedInput: document.getElementById("speedInput"),
      maxEventsInput: document.getElementById("maxEventsInput"),
      debounceInput: document.getElementById("debounceInput"),
      stepSecondsInput: document.getElementById("stepSecondsInput"),
      useScenarioLayout: document.getElementById("useScenarioLayout"),
      scenarioTemplate: document.getElementById("scenarioTemplate"),
      templateHint: document.getElementById("templateHint"),
      replayNewBtn: document.getElementById("replayNewBtn"),
      replayPauseBtn: document.getElementById("replayPauseBtn"),
      replayResumeBtn: document.getElementById("replayResumeBtn"),
      replayStepBtn: document.getElementById("replayStepBtn"),
      replayResetBtn: document.getElementById("replayResetBtn"),
      mapTabFixed: document.getElementById("mapTabFixed"),
      mapTabLive: document.getElementById("mapTabLive"),
      fixedMapPanel: document.getElementById("fixedMapPanel"),
      liveMapPanel: document.getElementById("liveMapPanel"),
      trainPresenceAutoBtn: document.getElementById("trainPresenceAutoBtn"),
      trainPresenceManualBtn: document.getElementById("trainPresenceManualBtn"),
      trainHistoricalBtn: document.getElementById("trainHistoricalBtn"),
      trainScenariosInput: document.getElementById("trainScenariosInput"),
      trainStepsInput: document.getElementById("trainStepsInput"),
      trainMaxPeopleInput: document.getElementById("trainMaxPeopleInput"),
      trainEpochsInput: document.getElementById("trainEpochsInput"),
      trainMaxSamplesInput: document.getElementById("trainMaxSamplesInput"),
      trainSeedInput: document.getElementById("trainSeedInput"),
      trainStatus: document.getElementById("trainStatus"),
      presenceTrainState: document.getElementById("presenceTrainState"),
      presenceTrainSamples: document.getElementById("presenceTrainSamples"),
      presenceTrainCount: document.getElementById("presenceTrainCount"),
      presenceTrainRooms: document.getElementById("presenceTrainRooms"),
      trainingJobStatus: document.getElementById("trainingJobStatus"),
      trainingUpdatedAt: document.getElementById("trainingUpdatedAt"),
      simulatedCsvRow: document.getElementById("simulatedCsvRow"),
      simulatedCsvMeta: document.getElementById("simulatedCsvMeta"),
      simulatedCsvLink: document.getElementById("simulatedCsvLink"),
      apiBaseUrl: document.getElementById("apiBaseUrl"),
      leftWorkspace: document.getElementById("leftWorkspace"),
      rightWorkspace: document.getElementById("rightWorkspace"),
      configPanel: document.getElementById("configPanel"),
      alertsPanel: document.getElementById("alertsPanel"),
      configDialog: document.getElementById("configDialog"),
      configOpenBtn: document.getElementById("configOpenBtn"),
      configCloseBtn: document.getElementById("configCloseBtn"),
      historyEnabled: document.getElementById("historyEnabled"),
      historyRetentionDays: document.getElementById("historyRetentionDays"),
      historyModeListen: document.getElementById("historyModeListen"),
      historyModeReplay: document.getElementById("historyModeReplay"),
      historyModeSimulator: document.getElementById("historyModeSimulator"),
      historyConfigTotal: document.getElementById("historyConfigTotal"),
      historyConfigSize: document.getElementById("historyConfigSize"),
      historyConfigRange: document.getElementById("historyConfigRange"),
      historyConfigPath: document.getElementById("historyConfigPath"),
      historyConfigSaveBtn: document.getElementById("historyConfigSaveBtn"),
      historyConfigStatus: document.getElementById("historyConfigStatus"),
      historyPurgeConfirmation: document.getElementById("historyPurgeConfirmation"),
      historyPurgeBtn: document.getElementById("historyPurgeBtn"),
      historyFilterForm: document.getElementById("historyFilterForm"),
      historyQuery: document.getElementById("historyQuery"),
      historySensorOptions: document.getElementById("historySensorOptions"),
      historySensorType: document.getElementById("historySensorType"),
      historyRoom: document.getElementById("historyRoom"),
      historyInputMode: document.getElementById("historyInputMode"),
      historyFrom: document.getElementById("historyFrom"),
      historyTo: document.getElementById("historyTo"),
      historyClearBtn: document.getElementById("historyClearBtn"),
      historyChart: document.getElementById("historyChart"),
      historyChartDescription: document.getElementById("historyChartDescription"),
      historyChartStatus: document.getElementById("historyChartStatus"),
      historyPrevBtn: document.getElementById("historyPrevBtn"),
      historyNextBtn: document.getElementById("historyNextBtn"),
      historyPageStatus: document.getElementById("historyPageStatus"),
      historyNewEventsBtn: document.getElementById("historyNewEventsBtn"),
    };

    let configDialogReturnFocus = null;

    function applyDevMode() {
      document.querySelectorAll("[data-dev-only]").forEach((node) => {
        node.hidden = !devMode;
      });

      const simulatorLink = document.querySelector('a[href^="/simulator.html"][data-dev-only]');
      if (simulatorLink && embeddedMode && devMode) {
        simulatorLink.href = "/simulator.html?embedded=1&dev=1";
      }
    }

    function openConfigDialog() {
      if (!el.configDialog || el.configDialog.open) return;
      configDialogReturnFocus = document.activeElement;
      el.configDialog.showModal();
      window.setTimeout(() => {
        if (el.configCloseBtn) el.configCloseBtn.focus();
      }, 0);
    }

    function closeConfigDialog() {
      if (el.configDialog && el.configDialog.open) {
        el.configDialog.close();
      }
    }

    function restoreConfigDialogFocus() {
      if (configDialogReturnFocus && typeof configDialogReturnFocus.focus === "function") {
        configDialogReturnFocus.focus();
      }
      configDialogReturnFocus = null;
    }

    function setTopStatus(text, isError) {
      el.topStatus.textContent = text;
      el.topStatus.className = "status-chip " + (isError ? "error" : "ok");
    }

    function setMiniStatus(target, text, isError) {
      target.textContent = text;
      target.className = "mini " + (isError ? "error" : "ok");
    }

    function toPercent(value) {
      const n = Number(value);
      if (!Number.isFinite(n)) return "0.0%";
      return (n * 100).toFixed(1) + "%";
    }

    function toMs(value) {
      const n = Number(value);
      if (!Number.isFinite(n)) return "n/a";
      return n.toFixed(1) + " ms";
    }

    function formatInteger(value) {
      const n = Number(value);
      if (!Number.isFinite(n)) return "0";
      return new Intl.NumberFormat("es-CL").format(Math.round(n));
    }

    function trainingStateLabel(stateValue) {
      const value = String(stateValue || "idle").toLowerCase();
      if (value === "running") return "Entrenando";
      if (value === "completed") return "Completado";
      if (value === "error") return "Error";
      return "En espera";
    }

    function roomLabel(room) {
      return String(room || "-").replace(/_/g, " ");
    }

    function edgeKey(a, b) {
      return [String(a || ""), String(b || "")].sort().join("|");
    }

    function formatTime(iso) {
      if (!iso) return "-";
      const date = new Date(iso);
      if (!Number.isFinite(date.getTime())) return String(iso);
      return date.toLocaleString("es-CL", {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
      });
    }

    function formatBytes(value) {
      const bytes = Number(value || 0);
      if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
      const units = ["B", "KB", "MB", "GB"];
      const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
      return (bytes / Math.pow(1024, index)).toFixed(index === 0 ? 0 : 1) + " " + units[index];
    }

    function isoToLocalInput(iso) {
      if (!iso) return "";
      const date = new Date(iso);
      if (!Number.isFinite(date.getTime())) return "";
      const local = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
      return local.toISOString().slice(0, 16);
    }

    function localInputToIso(value) {
      if (!value) return "";
      const date = new Date(value);
      return Number.isFinite(date.getTime()) ? date.toISOString() : "";
    }

    function adjacencyToText(adjacency) {
      return Object.keys(adjacency || {})
        .sort()
        .map((room) => {
          const neighbors = Array.isArray(adjacency[room]) ? [...adjacency[room]].sort() : [];
          return room + ": " + neighbors.join(", ");
        })
        .join("\n");
    }

    function adjacencyToEdges(adjacency) {
      const edges = [];
      const seen = new Set();
      Object.keys(adjacency || {}).forEach((room) => {
        const neighbors = Array.isArray(adjacency[room]) ? adjacency[room] : [];
        neighbors.forEach((nb) => {
          const key = edgeKey(room, nb);
          if (seen.has(key)) return;
          seen.add(key);
          const pair = key.split("|");
          edges.push({ a: pair[0], b: pair[1], support: 1 });
        });
      });
      return edges;
    }

    function cloneRealSensorConfig(config) {
      const source = config && typeof config === "object" ? config : {};
      const rooms = Array.isArray(source.rooms) ? source.rooms : [];
      const assignments = Array.isArray(source.assignments) ? source.assignments : [];
      return {
        rooms: [...new Set(rooms.map((room) => String(room || "").trim()).filter(Boolean))].sort(),
        assignments: assignments
          .filter((assignment) => assignment && assignment.entity_id)
          .map((assignment) => ({
            entity_id: String(assignment.entity_id || "").trim(),
            room: String(assignment.room || "").trim(),
            enabled: assignment.enabled !== false,
            sensor_type: String(assignment.sensor_type || "auto"),
          })),
        require_explicit_selection: source.require_explicit_selection !== false,
      };
    }

    function realSensorWorkingConfig() {
      if (!state.realSensorDraft) {
        state.realSensorDraft = cloneRealSensorConfig(state.realSensorConfig);
      }
      return state.realSensorDraft;
    }

    function setRealSensorConfig(config) {
      state.realSensorConfig = cloneRealSensorConfig({
        ...state.realSensorConfig,
        ...(config || {}),
      });
      if (!state.realSensorDirty) {
        state.realSensorDraft = cloneRealSensorConfig(state.realSensorConfig);
      }
    }

    function savedRealSensorByEntity() {
      const saved = Array.isArray(state.realSensorConfig.assignments)
        ? state.realSensorConfig.assignments
        : [];
      const map = new Map();
      saved.forEach((assignment) => {
        if (assignment && assignment.entity_id) {
          map.set(String(assignment.entity_id), assignment);
        }
      });
      return map;
    }

    function realSensorAssignmentChanged(draft, saved) {
      const normalizedDraft = {
        room: String((draft && draft.room) || ""),
        enabled: !!(draft && draft.enabled !== false && draft.room),
        sensor_type: String((draft && draft.sensor_type) || "auto"),
      };
      const normalizedSaved = {
        room: String((saved && saved.room) || ""),
        enabled: !!(saved && saved.enabled !== false && saved.room),
        sensor_type: String((saved && saved.sensor_type) || "auto"),
      };
      return normalizedDraft.room !== normalizedSaved.room ||
        normalizedDraft.enabled !== normalizedSaved.enabled ||
        normalizedDraft.sensor_type !== normalizedSaved.sensor_type;
    }

    function markRealSensorDirty(message) {
      state.realSensorDirty = true;
      if (message) {
        setMiniStatus(el.realSensorStatus, message, false);
      }
    }

    function referenceRooms() {
      const bag = new Set();
      if (Array.isArray(state.realSensorConfig.rooms)) {
        state.realSensorConfig.rooms.forEach((room) => {
          if (room) bag.add(String(room));
        });
      }
      if (Array.isArray(state.reference.rooms)) {
        state.reference.rooms.forEach((room) => {
          if (room) bag.add(String(room));
        });
      }
      Object.keys(state.reference.adjacency || {}).forEach((room) => bag.add(room));
      Object.values(state.reference.adjacency || {}).forEach((neighbors) => {
        (neighbors || []).forEach((nb) => bag.add(nb));
      });
      if (!bag.size) {
        (state.rooms || []).forEach((room) => {
          if (room) bag.add(String(room));
        });
      }
      return [...bag].filter(Boolean).sort();
    }

    function collectRooms() {
      return referenceRooms();
    }

    function svgEl(tag) {
      return document.createElementNS("http://www.w3.org/2000/svg", tag);
    }

    function computePositions(rooms, width, height) {
      const cx = width / 2;
      const cy = height / 2;
      const rx = Math.max(120, width * 0.36);
      const ry = Math.max(90, height * 0.33);
      const total = Math.max(1, rooms.length);
      const map = new Map();

      rooms.forEach((room, index) => {
        const angle = ((Math.PI * 2) * index) / total - Math.PI / 2;
        map.set(room, {
          x: cx + Math.cos(angle) * rx,
          y: cy + Math.sin(angle) * ry,
        });
      });
      return map;
    }

    function drawMap(svg, rooms, edges, options) {
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      const width = 900;
      const height = 520;
      const positions = computePositions(rooms, width, height);
      const activeSet = new Set(
        Array.isArray(options.activeRooms)
          ? options.activeRooms.filter(Boolean).map((room) => String(room))
          : []
      );
      if (options.activeRoom) {
        activeSet.add(String(options.activeRoom));
      }
      const occupancySet = new Set(
        Array.isArray(options.occupancyRooms)
          ? options.occupancyRooms.filter(Boolean).map((room) => String(room))
          : []
      );

      const grid = svgEl("g");
      for (let x = 60; x <= 840; x += 60) {
        const line = svgEl("line");
        line.setAttribute("x1", String(x));
        line.setAttribute("y1", "20");
        line.setAttribute("x2", String(x));
        line.setAttribute("y2", "500");
        line.setAttribute("stroke", "rgba(150, 214, 236, 0.07)");
        line.setAttribute("stroke-width", "1");
        grid.appendChild(line);
      }
      for (let y = 40; y <= 480; y += 60) {
        const line = svgEl("line");
        line.setAttribute("x1", "30");
        line.setAttribute("y1", String(y));
        line.setAttribute("x2", "870");
        line.setAttribute("y2", String(y));
        line.setAttribute("stroke", "rgba(150, 214, 236, 0.07)");
        line.setAttribute("stroke-width", "1");
        grid.appendChild(line);
      }
      svg.appendChild(grid);

      edges.forEach((edge) => {
        const pa = positions.get(edge.a);
        const pb = positions.get(edge.b);
        if (!pa || !pb) return;

        const key = edgeKey(edge.a, edge.b);
        const line = svgEl("line");
        line.setAttribute("x1", String(pa.x));
        line.setAttribute("y1", String(pa.y));
        line.setAttribute("x2", String(pb.x));
        line.setAttribute("y2", String(pb.y));
        line.setAttribute("stroke-linecap", "round");

        const isLatest = options.latestEdge && key === options.latestEdge;
        if (options.mode === "reference") {
          line.setAttribute("stroke", "rgba(116, 189, 220, 0.55)");
          line.setAttribute("stroke-width", "3");
          svg.appendChild(line);
        } else {
          const support = Number(edge.support || 0);
          line.setAttribute("stroke", isLatest ? "#ffbd7a" : "rgba(82, 217, 177, 0.68)");
          line.setAttribute("stroke-width", String(2 + Math.log1p(support) * 2));
          svg.appendChild(line);

          if (support > 0) {
            const text = svgEl("text");
            text.setAttribute("x", String((pa.x + pb.x) / 2));
            text.setAttribute("y", String(((pa.y + pb.y) / 2) - 6));
            text.setAttribute("fill", isLatest ? "#ffd7a5" : "#95ddc9");
            text.setAttribute("text-anchor", "middle");
            text.setAttribute("font-size", "12");
            text.setAttribute("font-family", "Segoe UI, Noto Sans, sans-serif");
            text.textContent = String(Math.round(support));
            svg.appendChild(text);
          }
        }
      });

      rooms.forEach((room) => {
        const pos = positions.get(room);
        if (!pos) return;
        const active = activeSet.has(room);
        const primary = options.activeRoom === room;

        const node = svgEl("circle");
        node.setAttribute("cx", String(pos.x));
        node.setAttribute("cy", String(pos.y));
        node.setAttribute("r", primary ? "30" : (active ? "27" : "24"));
        node.setAttribute("fill", primary ? "#48d9be" : (active ? "#43a9c3" : "#2c5c73"));
        node.setAttribute("stroke", active ? "#f2fff9" : "#b4d9e9");
        node.setAttribute("stroke-width", active ? "2.6" : "1.4");
        svg.appendChild(node);

        if (active) {
          const badge = svgEl("circle");
          badge.setAttribute("cx", String(pos.x + 22));
          badge.setAttribute("cy", String(pos.y - 22));
          badge.setAttribute("r", "12");
          badge.setAttribute("fill", occupancySet.has(room) ? "#48d9be" : "#ffbd7a");
          badge.setAttribute("stroke", "#101010");
          badge.setAttribute("stroke-width", "2");
          svg.appendChild(badge);

          const badgeText = svgEl("text");
          badgeText.setAttribute("x", String(pos.x + 22));
          badgeText.setAttribute("y", String(pos.y - 18));
          badgeText.setAttribute("fill", "#101010");
          badgeText.setAttribute("text-anchor", "middle");
          badgeText.setAttribute("font-size", "12");
          badgeText.setAttribute("font-weight", "800");
          badgeText.setAttribute("font-family", "Segoe UI, Noto Sans, sans-serif");
          badgeText.textContent = "1";
          svg.appendChild(badgeText);
        }

        const label = svgEl("text");
        label.setAttribute("x", String(pos.x));
        label.setAttribute("y", String(pos.y + 44));
        label.setAttribute("fill", "#d8edf7");
        label.setAttribute("text-anchor", "middle");
        label.setAttribute("font-size", "14");
        label.setAttribute("font-family", "Segoe UI, Noto Sans, sans-serif");
        label.textContent = roomLabel(room);
        svg.appendChild(label);
      });
    }

    function renderKpis() {
      el.totalEvents.textContent = String(state.events.length);
      el.totalRooms.textContent = String(collectRooms().length);

      const people = state.metrics && state.metrics.people ? state.metrics.people : {};
      el.peopleNow.textContent = String(Number(people.current_estimate || 0));
      el.peopleMax.textContent = String(Number(people.max_observed || 0));
    }

    function renderMetrics() {
      const metrics = state.metrics || {};
      const map = metrics.map || {};
      const quality = map.live_confirmed_quality || {};
      const nonAdj = metrics.non_adjacent || {};
      const latency = metrics.latency || {};

      el.mapPrecision.textContent = toPercent(quality.precision);
      el.mapRecall.textContent = toPercent(quality.recall);
      el.mapF1.textContent = toPercent(quality.f1);
      el.mapTpFpFn.textContent =
        "TP " + String(quality.tp || 0) +
        " | FP " + String(quality.fp || 0) +
        " | FN " + String(quality.fn || 0);

      el.mapSupportSummary.textContent =
        String(map.live_edges_confirmed || 0) + " confirmadas / " + String(map.reference_edges || 0) + " reales";

      el.nonAdjTotal.textContent = String(nonAdj.total || 0);
      el.nonAdjBreakdown.textContent =
        "m" + String(nonAdj.multi_person_probable || 0) +
        " p" + String(nonAdj.pet_or_noise || 0) +
        " e" + String(nonAdj.sensor_or_data_error || 0);

      el.latInP95.textContent = toMs(latency.ingestion ? latency.ingestion.p95_ms : null);
      el.latProcP95.textContent = toMs(latency.processing ? latency.processing.p95_ms : null);
    }

    function renderTrainingInfo() {
      const modelInfo = state.modelInfo || {};
      const presenceInfo = modelInfo.presence_training_info || {};
      const status = modelInfo.training_status || {};
      const presenceStatus = status.presence || {};
      const historicalStatus = status.historical || {};
      const isRunning = presenceStatus.state === "running" || historicalStatus.state === "running";
      const enabled = !!presenceInfo.enabled;

      if (presenceStatus.state === "running") {
        el.presenceTrainState.textContent = "Entrenando";
      } else if (presenceStatus.state === "error") {
        el.presenceTrainState.textContent = "Error";
      } else {
        el.presenceTrainState.textContent = enabled ? "Activo" : "No entrenado";
      }

      el.presenceTrainSamples.textContent = formatInteger(presenceInfo.samples || 0);
      el.presenceTrainCount.textContent = toPercent(presenceInfo.count_accuracy);
      el.presenceTrainRooms.textContent = formatInteger(presenceInfo.rooms_total || (modelInfo.presence_rooms || []).length || 0);

      const candidates = [presenceStatus, historicalStatus].filter((item) => item && Object.keys(item).length);
      const runningStatus = candidates.find((item) => item.state === "running");
      const activeStatus = runningStatus || candidates.sort((a, b) => {
        const aTime = Date.parse(a.finished_at || a.started_at || "") || 0;
        const bTime = Date.parse(b.finished_at || b.started_at || "") || 0;
        return bTime - aTime;
      })[0] || {};
      const label = activeStatus.label || (presenceStatus.state === "running" ? "Presencia simulador" : "Historico CSV");
      const message = activeStatus.message || "sin entrenamiento activo";
      el.trainingJobStatus.textContent = label + " | " + trainingStateLabel(activeStatus.state) + " | " + message;
      el.trainingJobStatus.className = isRunning ? "is-running" : "";

      const updatedAt = activeStatus.finished_at || activeStatus.started_at || presenceStatus.finished_at || historicalStatus.finished_at;
      el.trainingUpdatedAt.textContent = updatedAt ? formatTime(updatedAt) : "-";

      const csvSummary = presenceStatus.result_summary || {};
      const csvUrl = csvSummary.simulated_csv_url;
      if (csvUrl) {
        el.simulatedCsvRow.hidden = false;
        el.simulatedCsvLink.href = csvUrl;
        el.simulatedCsvMeta.textContent =
          formatInteger(csvSummary.simulated_csv_rows || 0) + " filas | " + csvUrl;
      }
    }

    function selectOption(select, value) {
      if (!select) return;
      const stringValue = String(value);
      const exists = Array.from(select.options).some((option) => option.value === stringValue);
      if (exists) {
        select.value = stringValue;
      }
    }

    function renderPresenceFilter() {
      const config = state.presenceFilter || {};
      if (el.petFilterEnabled) {
        el.petFilterEnabled.value = config.enabled === false ? "false" : "true";
      }
      selectOption(el.petFilterWindowInput, Number(config.window_seconds || 20));
      selectOption(el.petFilterMinEventsInput, Number(config.min_motion_events || 2));
      selectOption(el.petFilterMinRoomsInput, Number(config.min_distinct_rooms || 1));
      if (el.petFilterStatus) {
        const suppressed = Number(config.suppressed_total || 0);
        const pending = Number(config.pending_motion_events || 0);
        el.petFilterStatus.textContent =
          "filtrados: " + String(suppressed) + " | ventana activa: " + String(pending);
        el.petFilterStatus.className = "mini ok";
      }
    }

    function renderHaEntityCatalog() {
      const catalog = state.haEntityCatalog || {};
      const realConfig = realSensorWorkingConfig();
      const entities = Array.isArray(catalog.entities) ? catalog.entities : [];
      const supported = entities.filter((entity) => entity && entity.supported !== false);
      const assignments = Array.isArray(realConfig.assignments) ? realConfig.assignments : [];
      const assignmentByEntity = new Map();
      const savedByEntity = savedRealSensorByEntity();
      assignments.forEach((assignment) => {
        if (assignment && assignment.entity_id) {
          assignmentByEntity.set(String(assignment.entity_id), assignment);
        }
      });
      const configuredRooms = [...new Set([...(realConfig.rooms || []), ...referenceRooms()])]
        .map((room) => String(room || "").trim())
        .filter(Boolean)
        .sort();
      const enabledAssignments = assignments.filter((assignment) => assignment && assignment.enabled !== false && assignment.room);

      if (el.haSensorSummary) {
        el.haSensorSummary.textContent =
          String(supported.length) + " compatibles / " + String(entities.length) + " detectadas";
      }
      if (el.realSensorSummary) {
        const pendingTotal = assignments.filter((assignment) => {
          const id = assignment && assignment.entity_id ? String(assignment.entity_id) : "";
          return id && realSensorAssignmentChanged(assignment, savedByEntity.get(id));
        }).length;
        el.realSensorSummary.textContent =
          String(enabledAssignments.length) + " sensores activos / " +
          String(configuredRooms.length) + " habitaciones reales" +
          (state.realSensorDirty ? " | " + String(pendingTotal) + " cambios pendientes" : "");
      }
      if (el.realSensorRequireSelect) {
        const draftMode = realConfig.require_explicit_selection === false ? "false" : "true";
        if (el.realSensorRequireSelect.value !== draftMode) {
          el.realSensorRequireSelect.value = draftMode;
        }
      }
      if (el.realSensorSearchInput && el.realSensorSearchInput.value !== state.realSensorSearch) {
        el.realSensorSearchInput.value = state.realSensorSearch;
      }
      if (el.realSensorApplyBtn) {
        el.realSensorApplyBtn.disabled = !state.realSensorDirty;
      }
      if (el.realSensorResetBtn) {
        el.realSensorResetBtn.disabled = !state.realSensorDirty;
      }
      if (el.haSensorStatus) {
        const scannedAt = catalog.scanned_at ? formatTime(catalog.scanned_at) : "sin escaneo";
        const mode = catalog.auto_discovery === false ? "lista explicita" : "auto discovery";
        el.haSensorStatus.textContent = mode + " | " + scannedAt;
        el.haSensorStatus.className = "mini " + (entities.length ? "ok" : "");
      }
      if (!el.haSensorList) return;
      el.haSensorList.innerHTML = "";
      if (!entities.length) {
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = 5;
        td.textContent = "Sin catalogo recibido desde Home Assistant";
        tr.appendChild(td);
        el.haSensorList.appendChild(tr);
        return;
      }
      const query = state.realSensorSearch.trim().toLowerCase();
      const searchableEntities = query
        ? entities.filter((entity) => {
          const haystack = [
            entity.entity_id,
            entity.name,
            entity.domain,
            entity.state,
            entity.sensor_type,
            entity.room,
            entity.device_class,
          ].map((value) => String(value || "").toLowerCase()).join(" ");
          return haystack.includes(query);
        })
        : entities;
      if (!searchableEntities.length) {
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = 5;
        td.textContent = "Sin entidades para la busqueda actual";
        tr.appendChild(td);
        el.haSensorList.appendChild(tr);
        return;
      }
      searchableEntities.slice(0, 160).forEach((entity) => {
        const entityId = String(entity.entity_id || "");
        const assignment = assignmentByEntity.get(entityId) || {};
        const selected = assignment.enabled !== false && !!assignment.room;
        const pending = realSensorAssignmentChanged(assignment, savedByEntity.get(entityId));
        const tr = document.createElement("tr");
        tr.classList.toggle("real-sensor-selected", selected);
        tr.classList.toggle("real-sensor-pending", pending);

        const useCell = document.createElement("td");
        useCell.className = "real-sensor-cell";
        const useInput = document.createElement("input");
        useInput.type = "checkbox";
        useInput.checked = selected;
        useInput.dataset.realSensorEnabled = entityId;
        useInput.setAttribute("aria-label", "Usar " + entityId);
        useCell.appendChild(useInput);
        if (selected) {
          const selectedLabel = document.createElement("span");
          selectedLabel.className = "selection-check";
          selectedLabel.textContent = "OK";
          useCell.appendChild(selectedLabel);
        }
        if (pending) {
          const pendingLabel = document.createElement("span");
          pendingLabel.className = "pending-check";
          pendingLabel.textContent = "pendiente";
          useCell.appendChild(pendingLabel);
        }
        tr.appendChild(useCell);
        const entityCell = document.createElement("th");
        entityCell.scope = "row";
        entityCell.textContent = entityId || "-";
        tr.appendChild(entityCell);

        const typeCell = document.createElement("td");
        const typeSelect = document.createElement("select");
        typeSelect.dataset.realSensorType = entityId;
        typeSelect.setAttribute("aria-label", "Tipo de sensor para " + entityId);
        ["auto", "motion", "door", "occupancy", "other"].forEach((sensorType) => {
          const option = document.createElement("option");
          option.value = sensorType;
          option.textContent = sensorType;
          typeSelect.appendChild(option);
        });
        typeSelect.value = String(assignment.sensor_type || entity.sensor_type || "auto");
        typeCell.appendChild(typeSelect);
        tr.appendChild(typeCell);

        const roomCell = document.createElement("td");
        const roomSelect = document.createElement("select");
        roomSelect.dataset.realSensorRoom = entityId;
        roomSelect.setAttribute("aria-label", "Habitación asignada a " + entityId);
        const emptyOption = document.createElement("option");
        emptyOption.value = "";
        emptyOption.textContent = "Sin asignar";
        roomSelect.appendChild(emptyOption);
        configuredRooms.forEach((room) => {
          const option = document.createElement("option");
          option.value = room;
          option.textContent = roomLabel(room);
          roomSelect.appendChild(option);
        });
        roomSelect.value = String(assignment.room || "");
        roomCell.appendChild(roomSelect);
        tr.appendChild(roomCell);

        appendBadgeCell(
          tr,
          String(entity.state || "-"),
          String(entity.state || "").toLowerCase() === "on" ? "on" : "off"
        );
        el.haSensorList.appendChild(tr);
      });
      wireRealSensorCatalogControls();
    }

    function upsertRealSensorAssignment(entityId) {
      const id = String(entityId || "").trim();
      if (!id) return null;
      const realConfig = realSensorWorkingConfig();
      if (!Array.isArray(realConfig.assignments)) {
        realConfig.assignments = [];
      }
      let assignment = realConfig.assignments.find((item) => item && item.entity_id === id);
      if (!assignment) {
        assignment = {
          entity_id: id,
          room: "",
          enabled: false,
          sensor_type: "auto",
        };
        realConfig.assignments.push(assignment);
      }
      return assignment;
    }

    function wireRealSensorCatalogControls() {
      if (!el.haSensorList) return;
      el.haSensorList.querySelectorAll("[data-real-sensor-enabled]").forEach((input) => {
        input.addEventListener("change", () => {
          const entityId = input.dataset.realSensorEnabled || "";
          const assignment = upsertRealSensorAssignment(entityId);
          if (!assignment) return;
          const roomSelect = el.haSensorList.querySelector("[data-real-sensor-room='" + CSS.escape(entityId) + "']");
          const typeSelect = el.haSensorList.querySelector("[data-real-sensor-type='" + CSS.escape(entityId) + "']");
          assignment.enabled = input.checked;
          assignment.room = roomSelect ? roomSelect.value : assignment.room;
          assignment.sensor_type = typeSelect ? typeSelect.value : assignment.sensor_type || "auto";
          if (assignment.enabled && !assignment.room) {
            const rooms = [...new Set([...(realSensorWorkingConfig().rooms || []), ...referenceRooms()])];
            assignment.room = rooms[0] || "";
            if (roomSelect) roomSelect.value = assignment.room;
          }
          markRealSensorDirty("cambio pendiente: confirma para aplicar sensores reales");
          renderHaEntityCatalog();
        });
      });
      el.haSensorList.querySelectorAll("[data-real-sensor-room]").forEach((select) => {
        select.addEventListener("change", () => {
          const entityId = select.dataset.realSensorRoom || "";
          const assignment = upsertRealSensorAssignment(entityId);
          if (!assignment) return;
          const checkbox = el.haSensorList.querySelector("[data-real-sensor-enabled='" + CSS.escape(entityId) + "']");
          assignment.room = select.value;
          assignment.enabled = !!select.value;
          if (checkbox) checkbox.checked = assignment.enabled;
          markRealSensorDirty("cambio pendiente: confirma para aplicar sensores reales");
          renderHaEntityCatalog();
        });
      });
      el.haSensorList.querySelectorAll("[data-real-sensor-type]").forEach((select) => {
        select.addEventListener("change", () => {
          const assignment = upsertRealSensorAssignment(select.dataset.realSensorType || "");
          if (!assignment) return;
          assignment.sensor_type = select.value || "auto";
          markRealSensorDirty("cambio pendiente: confirma para aplicar sensores reales");
          renderHaEntityCatalog();
        });
      });
    }

    function buildRealSensorPayload() {
      const realConfig = realSensorWorkingConfig();
      const rooms = [...new Set([...(realConfig.rooms || []), ...referenceRooms()])]
        .map((room) => String(room || "").trim())
        .filter(Boolean)
        .sort();
      const assignments = (realConfig.assignments || [])
        .map((assignment) => ({
          entity_id: String(assignment.entity_id || "").trim(),
          room: String(assignment.room || "").trim(),
          enabled: assignment.enabled !== false,
          sensor_type: String(assignment.sensor_type || "auto"),
        }))
        .filter((assignment) => assignment.entity_id && assignment.room);
      return {
        rooms,
        assignments,
        require_explicit_selection: el.realSensorRequireSelect
          ? el.realSensorRequireSelect.value !== "false"
          : realConfig.require_explicit_selection !== false,
      };
    }

    function addRealSensorRoom() {
      if (!el.realSensorNewRoomInput) return;
      const room = el.realSensorNewRoomInput.value.trim().toLowerCase().replace(/\s+/g, "_");
      if (!room) return;
      const realConfig = realSensorWorkingConfig();
      const rooms = new Set([...(realConfig.rooms || []), ...referenceRooms()]);
      rooms.add(room);
      realConfig.rooms = [...rooms].sort();
      el.realSensorNewRoomInput.value = "";
      markRealSensorDirty("habitacion pendiente: " + roomLabel(room));
      renderAll();
    }

    function resetRealSensorDraft() {
      state.realSensorDraft = cloneRealSensorConfig(state.realSensorConfig);
      state.realSensorDirty = false;
      setMiniStatus(el.realSensorStatus, "cambios descartados", false);
      renderHaEntityCatalog();
    }

    async function fetchRealSensorConfig() {
      const payload = await fetchJson("/api/real_sensor_config", { cache: "no-store" });
      if (payload && payload.config) {
        setRealSensorConfig(payload.config);
      }
      if (payload && payload.catalog) {
        state.haEntityCatalog = payload.catalog;
      }
      renderHaEntityCatalog();
      renderHaDiagnostics();
      return payload;
    }

    async function applyRealSensorConfig() {
      try {
        const payload = buildRealSensorPayload();
        const result = await fetchJson("/api/real_sensor_config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (result && result.config) {
          state.realSensorDirty = false;
          setRealSensorConfig(result.config);
          state.realSensorDraft = cloneRealSensorConfig(state.realSensorConfig);
        }
        if (result && result.catalog) {
          state.haEntityCatalog = result.catalog;
        }
        const enabledTotal = result && result.config && Array.isArray(result.config.enabled_entities)
          ? result.config.enabled_entities.length
          : payload.assignments.filter((item) => item.enabled).length;
        setMiniStatus(el.realSensorStatus, "sensores reales aplicados: " + String(enabledTotal), false);
        renderAll();
      } catch (err) {
        setMiniStatus(el.realSensorStatus, String(err.message || err), true);
      }
    }

    function renderHaDiagnostics() {
      const catalog = state.haEntityCatalog || {};
      const actions = state.haActions || {};
      const entities = Array.isArray(catalog.entities) ? catalog.entities : [];
      const supported = entities.filter((entity) => entity && entity.supported !== false);
      const pending = Array.isArray(actions.pending) ? actions.pending : [];
      const results = Array.isArray(actions.recent_results) ? actions.recent_results : [];
      const integrationStatus = actions.integration_status || {};
      const latest = results[0] || null;
      const entries = integrationStatus.entries && typeof integrationStatus.entries === "object"
        ? integrationStatus.entries
        : {};
      const entryValues = Object.values(entries);
      const latestHeartbeat = entryValues
        .slice()
        .sort((a, b) => (Date.parse(b.last_seen_at || "") || 0) - (Date.parse(a.last_seen_at || "") || 0))[0] || null;

      if (el.haDiagReceivedAt) {
        el.haDiagReceivedAt.textContent = catalog.received_at ? formatTime(catalog.received_at) : "-";
      }
      if (el.haDiagScannedAt) {
        el.haDiagScannedAt.textContent = catalog.scanned_at ? formatTime(catalog.scanned_at) : "-";
      }
      if (el.haDiagSource) {
        el.haDiagSource.textContent = String(catalog.source || "-");
      }
      if (el.haDiagPending) {
        el.haDiagPending.textContent = String(pending.length || 0);
        el.haDiagPending.className = pending.length ? "is-running" : "";
      }
      if (el.haDiagHeartbeat) {
        el.haDiagHeartbeat.textContent = latestHeartbeat && latestHeartbeat.last_seen_at
          ? formatTime(latestHeartbeat.last_seen_at)
          : "-";
        el.haDiagHeartbeat.className = latestHeartbeat ? "ok" : "error";
      }
      if (el.haDiagLastResult) {
        if (latest) {
          const action = latest.action || latest.result || latest.request_id || "accion HA";
          el.haDiagLastResult.textContent = String(latest.status || "-") + " | " + String(action);
          el.haDiagLastResult.className = latest.status === "error" ? "error" : "";
        } else {
          el.haDiagLastResult.textContent = "-";
          el.haDiagLastResult.className = "";
        }
      }
      if (el.haDiagEntry) {
        const entryText = latestHeartbeat && latestHeartbeat.entry_id
          ? String(latestHeartbeat.entry_id)
          : String(catalog.entry_id || "-");
        el.haDiagEntry.textContent = entryText;
      }
      if (!el.haDiagnosticStatus) return;

      let message = "Diagnóstico listo.";
      let isError = false;
      if (!latestHeartbeat && !catalog.received_at) {
        message = "El backend no ve heartbeat ni catálogo de la integración Home Assistant. Revisa que la integración HACS esté configurada con la URL correcta del backend y reinicia Home Assistant si acabas de instalarla.";
        isError = true;
      } else if (latestHeartbeat && !catalog.received_at) {
        message = "La integración HA está viva, pero todavía no publicó catálogo. Pulsa Refrescar catálogo o revisa el último error de la integración.";
        isError = true;
      } else if (!catalog.received_at) {
        message = "El backend aún no recibió catálogo desde la integración Home Assistant.";
        isError = true;
      } else if (!entities.length) {
        message = "Home Assistant publicó un catálogo vacío. Revisa auto discovery o entidades configuradas.";
        isError = true;
      } else if (!supported.length) {
        message = "Hay entidades detectadas, pero ninguna compatible como motion, door u occupancy.";
        isError = true;
      } else {
        message =
          "Catálogo recibido: " + String(supported.length) +
          " sensores compatibles de " + String(entities.length) + " entidades.";
      }
      if (pending.length) {
        message += " Hay " + String(pending.length) + " acción(es) pendientes; si no bajan, la integración HA no está consultando el backend.";
        isError = true;
      }
      if (latestHeartbeat && latestHeartbeat.last_error) {
        message += " Último error HA: " + String(latestHeartbeat.last_error);
        isError = true;
      }
      if (latest && latest.status === "error") {
        message += " Último resultado con error: " + String(latest.error || "sin detalle");
        isError = true;
      }
      setMiniStatus(el.haDiagnosticStatus, message, isError);
    }

    function renderReplay() {
      const replay = state.replay || {};
      const mode = replay.mode || "listen";
      const running = !!replay.running;
      const paused = !!replay.paused;
      const stepBudget = Number(replay.step_budget || 0);
      const progress = Number(replay.progress || 0);

      el.modeListenBtn.disabled = mode === "listen" && !running;
      if (el.modeReplayBtn) {
        el.modeReplayBtn.disabled = mode === "replay";
      }
      el.modeSummary.textContent =
        mode === "replay" ? "replay historico" : (mode === "simulator" ? "simulador" : "sensores reales");
      el.modeListenBtn.classList.toggle("active", mode === "listen");
      if (el.modeReplayBtn) {
        el.modeReplayBtn.classList.toggle("active", mode === "replay");
      }

      if (running && paused) {
        el.replaySummary.textContent = "pausado";
      } else if (running) {
        el.replaySummary.textContent = "en ejecucion";
      } else if (mode === "listen") {
        el.replaySummary.textContent = "escucha sensores";
      } else {
        el.replaySummary.textContent = "idle";
      }

      el.replayProgress.textContent =
        "progreso: " + (progress * 100).toFixed(1) + "% | " +
        String(replay.processed_events || 0) + "/" + String(replay.total_events || 0) +
        " | steps cola: " + String(stepBudget);

      if (replay.last_error) {
        setMiniStatus(el.replayStatus, "replay detenido: " + replay.last_error, true);
      }

      el.replayStepBtn.disabled = !(running && paused);
    }

    function renderMaps() {
      const rooms = collectRooms();
      const referenceEdges = Array.isArray(state.reference.edges) && state.reference.edges.length
        ? state.reference.edges.map((edge) => ({ a: edge.a, b: edge.b, support: 1 }))
        : adjacencyToEdges(state.reference.adjacency || {});

      const inferredEdges = [];
      for (const [key, support] of state.inferredEdges.entries()) {
        const pair = key.split("|");
        inferredEdges.push({ a: pair[0], b: pair[1], support });
      }

      drawMap(el.realGraph, rooms, referenceEdges, {
        mode: "reference",
        activeRoom: state.currentRoom,
        activeRooms: state.activeRooms,
        occupancyRooms: state.occupancyRooms,
      });
      drawMap(el.inferredGraph, rooms, inferredEdges, {
        mode: "inferred",
        activeRoom: state.currentRoom,
        activeRooms: state.activeRooms,
        occupancyRooms: state.occupancyRooms,
        latestEdge: state.latestEdge,
      });

      const activeLabel = state.activeRooms.length
        ? state.activeRooms.map(roomLabel).join(", ")
        : "sin presencia";

      el.realMapMeta.textContent =
        "version " + String(state.reference.version || 0) +
        " | source " + String(state.reference.source || "-") +
        " | presencia " + activeLabel;

      el.inferredMapMeta.textContent =
        String(inferredEdges.length) + " aristas | personas " +
        String((state.metrics && state.metrics.people && state.metrics.people.current_estimate) || 0) +
        " | presencia " + activeLabel;
      el.layoutMeta.textContent = "version " + String(state.reference.version || 0);
    }

    function appendCell(row, text) {
      const td = document.createElement("td");
      td.textContent = text;
      row.appendChild(td);
      return td;
    }

    function appendBadgeCell(row, text, tone) {
      const td = document.createElement("td");
      const badge = document.createElement("span");
      badge.className = "status-badge " + String(tone || "info");
      badge.textContent = text;
      td.appendChild(badge);
      row.appendChild(td);
      return td;
    }

    function renderAlerts() {
      const metrics = state.metrics || {};
      const nonAdj = metrics.non_adjacent || {};
      const recent = Array.isArray(nonAdj.recent) ? [...nonAdj.recent].reverse() : [];

      el.alertList.innerHTML = "";
      if (!recent.length) {
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = 5;
        td.textContent = "Sin alertas no adyacentes registradas";
        tr.appendChild(td);
        el.alertList.appendChild(tr);
        return;
      }

      recent.slice(0, 40).forEach((alert) => {
        const tr = document.createElement("tr");
        appendCell(tr, roomLabel(alert.from) + " -> " + roomLabel(alert.to));
        appendCell(tr, formatTime(alert.timestamp));
        appendBadgeCell(tr, String(alert.cause || "-"), "alert");
        appendCell(tr, String(alert.estimated_people || 0));
        appendCell(tr, String(alert.gap_seconds || 0) + "s");
        el.alertList.appendChild(tr);
      });
    }

    function renderHistoryConfig() {
      const config = state.history.config || {};
      const modes = new Set(Array.isArray(config.persisted_modes) ? config.persisted_modes : []);
      el.historyEnabled.value = String(config.enabled !== false);
      el.historyRetentionDays.value = String(config.retention_days || 365);
      el.historyModeListen.checked = modes.has("listen");
      el.historyModeReplay.checked = modes.has("replay");
      el.historyModeSimulator.checked = modes.has("simulator");
      el.historyConfigTotal.textContent = formatInteger(config.events_total || 0);
      el.historyConfigSize.textContent = formatBytes(config.database_size_bytes);
      el.historyConfigRange.textContent = config.first_timestamp
        ? formatTime(config.first_timestamp) + " - " + formatTime(config.last_timestamp)
        : "Sin eventos";
      el.historyConfigPath.textContent = String(config.database_path || "-");
      if (config.last_error) {
        setMiniStatus(el.historyConfigStatus, "SQLite: " + config.last_error, true);
      } else {
        setMiniStatus(
          el.historyConfigStatus,
          config.enabled === false ? "persistencia desactivada" : "historial operativo",
          false
        );
      }
    }

    function populateHistorySelect(select, values, emptyLabel, labelFormatter) {
      const current = select.value;
      select.innerHTML = "";
      const empty = document.createElement("option");
      empty.value = "";
      empty.textContent = emptyLabel;
      select.appendChild(empty);
      (values || []).forEach((value) => {
        const option = document.createElement("option");
        option.value = String(value);
        option.textContent = labelFormatter ? labelFormatter(value) : roomLabel(value);
        select.appendChild(option);
      });
      select.value = [...select.options].some((option) => option.value === current) ? current : "";
    }

    function renderHistoryOptions() {
      const options = state.history.options || {};
      populateHistorySelect(el.historySensorType, options.sensor_types, "Todos");
      populateHistorySelect(el.historyRoom, options.rooms, "Todas");
      const selectedMode = el.historyInputMode.value;
      const configuredModes = ["listen", "replay", "simulator"];
      const availableModes = [...new Set([...(options.input_modes || []), ...configuredModes])];
      populateHistorySelect(el.historyInputMode, availableModes, "Todos", (mode) => {
        if (mode === "listen") return "Escucha";
        if (mode === "replay") return "Replay";
        if (mode === "simulator") return "Simulador";
        return String(mode);
      });
      el.historyInputMode.value = availableModes.includes(selectedMode) ? selectedMode : "";

      el.historySensorOptions.innerHTML = "";
      (options.sensors || []).forEach((sensor) => {
        const option = document.createElement("option");
        option.value = String(sensor.entity_id || "");
        option.label = String(sensor.sensor_name || sensor.entity_id || "");
        el.historySensorOptions.appendChild(option);
      });
    }

    function renderHistoryEvents() {
      el.eventList.innerHTML = "";
      el.eventSummary.textContent =
        formatInteger(state.history.total) + " eventos filtrados";

      if (!state.history.items.length) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 9;
        cell.textContent = "No hay eventos para los filtros seleccionados";
        row.appendChild(cell);
        el.eventList.appendChild(row);
      }

      state.history.items.forEach((evt) => {
        const hasLayoutAlert = !!evt.layout_alert;
        const row = document.createElement("tr");
        appendCell(row, "#" + String(evt.id || "-"));
        appendCell(row, formatTime(evt.event_timestamp));
        appendCell(row, roomLabel(evt.room));
        const sensorCell = appendCell(row, String(evt.sensor_name || evt.entity_id || "-"));
        sensorCell.title = String(evt.entity_id || "");
        appendCell(row, String(evt.sensor_type || "other"));
        appendBadgeCell(
          row,
          String(evt.state || "-"),
          String(evt.state || "").toLowerCase() === "on" ? "on" : "off"
        );
        const presenceText = evt.inferred_presence
          ? "Presente: " + roomLabel(evt.inferred_room)
          : "Ausente";
        appendBadgeCell(row, presenceText, evt.inferred_presence ? "on" : "off");
        appendCell(row, String(evt.estimated_people || 0));
        appendBadgeCell(
          row,
          hasLayoutAlert ? String(evt.layout_alert.cause || "no_adyacente") : "ok",
          hasLayoutAlert ? "alert" : "ok"
        );
        el.eventList.appendChild(row);
      });

      el.historyPageStatus.textContent =
        "Página " + String(state.history.page) + " de " + String(state.history.pages);
      el.historyPrevBtn.disabled = state.history.page <= 1;
      el.historyNextBtn.disabled = state.history.page >= state.history.pages;
      el.historyNewEventsBtn.hidden = !state.history.newEvents;
    }

    function svgElement(name, attributes) {
      const node = document.createElementNS("http://www.w3.org/2000/svg", name);
      Object.entries(attributes || {}).forEach(([key, value]) => {
        node.setAttribute(key, String(value));
      });
      return node;
    }

    function renderHistoryChart() {
      const points = state.history.points || [];
      el.historyChart.innerHTML = "";
      const title = svgElement("title", { id: "historyChartTitle" });
      title.textContent = "Gráfico histórico de presencia y personas estimadas";
      el.historyChart.appendChild(title);
      const description = svgElement("desc", { id: "historyChartDescription" });
      el.historyChart.appendChild(description);

      if (!points.length) {
        description.textContent = "Sin datos históricos para los filtros seleccionados.";
        const text = svgElement("text", { x: 450, y: 112, class: "history-chart-empty" });
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
      const end = new Date(points[points.length - 1].timestamp).getTime();
      const duration = Math.max(1, end - start);
      const maxPeople = Math.max(1, ...points.map((point) => Number(point.people || 0)));
      const x = (timestamp) =>
        left + ((new Date(timestamp).getTime() - start) / duration) * (width - left - right);
      const presenceY = (present) => (present ? top + 34 : height - bottom - 34);
      const peopleY = (people) =>
        height - bottom - (Number(people || 0) / maxPeople) * (height - top - bottom);

      el.historyChart.appendChild(
        svgElement("line", {
          x1: left,
          y1: height - bottom,
          x2: width - right,
          y2: height - bottom,
          class: "history-chart-axis",
        })
      );
      el.historyChart.appendChild(
        svgElement("line", {
          x1: left,
          y1: top,
          x2: left,
          y2: height - bottom,
          class: "history-chart-axis",
        })
      );

      let presencePath = "";
      points.forEach((point, index) => {
        const pointX = x(point.timestamp);
        const pointY = presenceY(point.presence);
        if (index === 0) {
          presencePath = "M " + pointX + " " + pointY;
        } else {
          presencePath += " H " + pointX + " V " + pointY;
        }
      });
      const peoplePath = points
        .map((point, index) =>
          (index === 0 ? "M " : " L ") + x(point.timestamp) + " " + peopleY(point.people)
        )
        .join("");
      el.historyChart.appendChild(
        svgElement("path", { d: presencePath, class: "history-presence-line" })
      );
      el.historyChart.appendChild(
        svgElement("path", { d: peoplePath, class: "history-people-line" })
      );

      [
        { x: 10, y: presenceY(true) + 4, text: "ON" },
        { x: 10, y: presenceY(false) + 4, text: "OFF" },
        { x: left, y: height - 10, text: formatTime(points[0].timestamp) },
        { x: width - right, y: height - 10, text: formatTime(points[points.length - 1].timestamp), anchor: "end" },
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
        String(points.length) + " cambios de presencia. Máximo de personas estimadas: " +
        String(maxPeople) + ".";
      setMiniStatus(
        el.historyChartStatus,
        state.history.truncated
          ? "serie truncada a " + String(points.length) + " cambios"
          : String(points.length) + " cambios",
        state.history.truncated
      );
    }

    function renderHistory() {
      renderHistoryOptions();
      renderHistoryEvents();
      renderHistoryChart();
    }

    function renderLayoutEditor() {
      if (!state.layoutTextDirty && state.reference.adjacency_text) {
        el.layoutText.value = state.reference.adjacency_text;
      }
    }

    function setMapTab(tab) {
      state.activeMapTab = tab === "live" ? "live" : "fixed";
      const isLive = state.activeMapTab === "live";

      el.mapTabFixed.classList.toggle("active", !isLive);
      el.mapTabLive.classList.toggle("active", isLive);
      el.mapTabFixed.setAttribute("aria-selected", String(!isLive));
      el.mapTabLive.setAttribute("aria-selected", String(isLive));
      el.fixedMapPanel.hidden = isLive;
      el.liveMapPanel.hidden = !isLive;
      el.fixedMapPanel.classList.toggle("active", !isLive);
      el.liveMapPanel.classList.toggle("active", isLive);
    }

    function renderAll() {
      renderKpis();
      renderMetrics();
      renderTrainingInfo();
      renderPresenceFilter();
      renderHaEntityCatalog();
      renderHaDiagnostics();
      renderReplay();
      renderLayoutEditor();
      renderAlerts();
      renderMaps();
    }

    function applySnapshot(simData) {
      if (!simData || typeof simData !== "object") return;

      if (Array.isArray(simData.rooms)) {
        state.rooms = [...new Set(simData.rooms.map((room) => String(room || "")).filter(Boolean))].sort();
      }

      if (Array.isArray(simData.events)) {
        state.events = simData.events.slice(-30000);
      }

      if (simData.layout_reference) {
        state.reference = {
          ...state.reference,
          ...simData.layout_reference,
        };
      }

      state.inferredEdges.clear();
      const liveEdges =
        simData.inferred_layout_live && Array.isArray(simData.inferred_layout_live.edges)
          ? simData.inferred_layout_live.edges
          : (Array.isArray(simData.final_edges) ? simData.final_edges : []);

      liveEdges.forEach((edge) => {
        if (!edge || !edge.a || !edge.b) return;
        state.inferredEdges.set(edgeKey(edge.a, edge.b), Number(edge.support || 0));
      });

      if (simData.inferred_layout_live && Array.isArray(simData.inferred_layout_live.latest_touched_edge)) {
        const pair = simData.inferred_layout_live.latest_touched_edge;
        if (pair.length === 2) {
          state.latestEdge = edgeKey(pair[0], pair[1]);
        }
      }

      if (simData.evaluation) {
        state.metrics = simData.evaluation;
      }

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

      if (simData.real_sensor_config) {
        setRealSensorConfig(simData.real_sensor_config);
      }

      if (simData.replay) {
        state.replay = {
          ...state.replay,
          ...simData.replay,
        };
      }
      if (simData.meta && simData.meta.input_mode) {
        state.replay.mode = simData.meta.input_mode;
      }

      if (simData.presence && typeof simData.presence === "object") {
        const presence = simData.presence;
        if (presence.current_room) {
          state.currentRoom = String(presence.current_room);
        }
        if (Array.isArray(presence.active_rooms)) {
          state.activeRooms = presence.active_rooms.map((room) => String(room || "")).filter(Boolean);
        }
        if (Array.isArray(presence.occupancy_ground_truth_rooms)) {
          state.occupancyRooms = presence.occupancy_ground_truth_rooms.map((room) => String(room || "")).filter(Boolean);
        }
        if (Array.isArray(presence.live_sensor_rooms)) {
          state.liveSensorRooms = presence.live_sensor_rooms.map((room) => String(room || "")).filter(Boolean);
        }
      }

      const latest = state.events[state.events.length - 1] || null;
      if (!state.currentRoom) {
        state.currentRoom = latest ? (latest.presence_room || latest.room || null) : null;
      }
      if (!state.activeRooms.length && latest && Array.isArray(latest.active_rooms) && latest.active_rooms.length) {
        state.activeRooms = latest.active_rooms.map((room) => String(room || "")).filter(Boolean);
      }
      if (!state.activeRooms.length && state.currentRoom) {
        state.activeRooms = [state.currentRoom];
      }
      if (!state.latestEdge && latest && latest.transition && Array.isArray(latest.transition.edge)) {
        state.latestEdge = edgeKey(latest.transition.edge[0], latest.transition.edge[1]);
      }

      if (simData.model && simData.model.ready) {
        const transformer =
          simData.model.training_info &&
          simData.model.training_info.transformer &&
          simData.model.training_info.transformer.enabled;
        el.modelState.textContent = transformer ? "hf_transformer_markov" : "markov_ai";
      } else {
        el.modelState.textContent = "rule_based";
      }

      renderAll();
    }

    function applyEvent(evt) {
      state.events.push(evt);
      if (state.events.length > 30000) {
        state.events.shift();
      }

      const room = String(evt.room || "");
      if (room && !state.rooms.includes(room)) {
        state.rooms.push(room);
        state.rooms.sort();
      }

      state.currentRoom = evt.presence_room || evt.room || state.currentRoom;
      if (Array.isArray(evt.active_rooms) && evt.active_rooms.length) {
        state.activeRooms = evt.active_rooms.map((room) => String(room || "")).filter(Boolean);
      } else if (state.currentRoom) {
        state.activeRooms = [state.currentRoom];
      }
      if (String(evt.sensor_type || "") === "occupancy" && room) {
        const nextOccupancy = new Set(state.occupancyRooms);
        if (String(evt.state || "").toLowerCase() === "on") {
          nextOccupancy.add(room);
        } else {
          nextOccupancy.delete(room);
        }
        state.occupancyRooms = [...nextOccupancy];
      }

      if (evt.ai_mode) {
        el.modelState.textContent = String(evt.ai_mode);
      }

      if (evt.transition && Array.isArray(evt.transition.edge) && evt.transition.edge.length === 2) {
        const key = edgeKey(evt.transition.edge[0], evt.transition.edge[1]);
        state.inferredEdges.set(key, Number(evt.transition.support || 0));
        state.latestEdge = key;
      }

      renderAll();
      scheduleHistoryRefresh();
    }

    async function fetchJson(url, options) {
      const response = await fetch(url, options || {});
      const raw = await response.text();
      let data = null;

      try {
        data = raw ? JSON.parse(raw) : null;
      } catch (_err) {
        data = null;
      }

      if (!response.ok) {
        const detail = data && data.detail ? String(data.detail) : (raw || (response.status + " " + response.statusText));
        throw new Error(detail);
      }

      return data;
    }

    function historySearchParams() {
      const filters = state.history.filters;
      const params = new URLSearchParams({
        query: filters.query,
        sensor_type: filters.sensorType,
        room: filters.room,
        input_mode: filters.inputMode,
        from_ts: filters.fromTs,
        to_ts: filters.toTs,
      });
      return params;
    }

    async function fetchHistoryConfig() {
      state.history.config = await fetchJson("/api/history/config", { cache: "no-store" });
      renderHistoryConfig();
      return state.history.config;
    }

    async function saveHistoryConfig() {
      const persistedModes = [
        el.historyModeListen,
        el.historyModeReplay,
        el.historyModeSimulator,
      ]
        .filter((input) => input.checked)
        .map((input) => input.value);
      if (!persistedModes.length) {
        setMiniStatus(el.historyConfigStatus, "selecciona al menos un modo", true);
        return;
      }
      try {
        el.historyConfigSaveBtn.disabled = true;
        state.history.config = await fetchJson("/api/history/config", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            enabled: el.historyEnabled.value === "true",
            retention_days: Number(el.historyRetentionDays.value || 365),
            persisted_modes: persistedModes,
          }),
        });
        renderHistoryConfig();
        setMiniStatus(el.historyConfigStatus, "configuración guardada", false);
        await fetchHistory();
      } catch (err) {
        setMiniStatus(el.historyConfigStatus, String(err.message || err), true);
      } finally {
        el.historyConfigSaveBtn.disabled = false;
      }
    }

    async function purgeHistory() {
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
          "historial borrado: " + formatInteger(result.deleted || 0) + " eventos",
          false
        );
        await Promise.all([fetchHistoryConfig(), fetchHistory()]);
      } catch (err) {
        setMiniStatus(el.historyConfigStatus, String(err.message || err), true);
      } finally {
        el.historyPurgeBtn.disabled = el.historyPurgeConfirmation.value !== "BORRAR";
      }
    }

    async function fetchHistory() {
      const params = historySearchParams();
      params.set("page", String(state.history.page));
      params.set("page_size", String(state.history.pageSize));
      const presenceParams = historySearchParams();
      presenceParams.set("max_points", "1000");
      const [eventsResult, presenceResult] = await Promise.all([
        fetchJson("/api/history/events?" + params.toString(), { cache: "no-store" }),
        fetchJson("/api/history/presence?" + presenceParams.toString(), { cache: "no-store" }),
      ]);
      state.history.items = Array.isArray(eventsResult.items) ? eventsResult.items : [];
      state.history.total = Number(eventsResult.total || 0);
      state.history.page = Number(eventsResult.page || 1);
      state.history.pages = Number(eventsResult.pages || 1);
      state.history.options = eventsResult.options || state.history.options;
      state.history.points = Array.isArray(presenceResult.points) ? presenceResult.points : [];
      state.history.sourceEvents = Number(presenceResult.source_events || 0);
      state.history.truncated = !!presenceResult.truncated;
      state.history.newEvents = false;
      renderHistory();
    }

    function applyHistoryFilters() {
      state.history.filters = {
        query: el.historyQuery.value.trim(),
        sensorType: el.historySensorType.value,
        room: el.historyRoom.value,
        inputMode: el.historyInputMode.value,
        fromTs: localInputToIso(el.historyFrom.value),
        toTs: localInputToIso(el.historyTo.value),
      };
      state.history.page = 1;
      fetchHistory().catch((err) => {
        setMiniStatus(el.historyChartStatus, String(err.message || err), true);
      });
    }

    function clearHistoryFilters() {
      state.history.filters = {
        query: "",
        sensorType: "",
        room: "",
        inputMode: "listen",
        fromTs: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
        toTs: "",
      };
      el.historyQuery.value = "";
      el.historySensorType.value = "";
      el.historyRoom.value = "";
      el.historyInputMode.value = "listen";
      el.historyFrom.value = isoToLocalInput(state.history.filters.fromTs);
      el.historyTo.value = "";
      state.history.page = 1;
      fetchHistory().catch((err) => {
        setMiniStatus(el.historyChartStatus, String(err.message || err), true);
      });
    }

    function scheduleHistoryRefresh() {
      if (state.history.page !== 1) {
        state.history.newEvents = true;
        el.historyNewEventsBtn.hidden = false;
        return;
      }
      if (state.history.refreshTimer) {
        window.clearTimeout(state.history.refreshTimer);
      }
      state.history.refreshTimer = window.setTimeout(() => {
        state.history.refreshTimer = null;
        Promise.all([fetchHistory(), fetchHistoryConfig()]).catch(() => {});
      }, 500);
    }

    async function fetchSnapshot() {
      const simData = await fetchJson("/api/sim_data", { cache: "no-store" });
      applySnapshot(simData || {});
    }

    async function fetchModelInfo() {
      const modelInfo = await fetchJson("/api/model_info", { cache: "no-store" });
      state.modelInfo = modelInfo || {};
      renderTrainingInfo();
      return state.modelInfo;
    }

    async function fetchPresenceFilter() {
      const config = await fetchJson("/api/presence_filter", { cache: "no-store" });
      state.presenceFilter = config || {};
      renderPresenceFilter();
      return state.presenceFilter;
    }

    async function fetchHaEntityCatalog() {
      const catalog = await fetchJson("/api/ha_entities", { cache: "no-store" });
      state.haEntityCatalog = catalog || {};
      renderHaEntityCatalog();
      renderHaDiagnostics();
      return state.haEntityCatalog;
    }

    async function fetchHaActions() {
      const actions = await fetchJson("/api/ha_actions", { cache: "no-store" });
      state.haActions = actions || {};
      renderHaDiagnostics();
      return state.haActions;
    }

    async function refreshHaDiagnostics() {
      await Promise.allSettled([fetchHaEntityCatalog(), fetchHaActions(), fetchRealSensorConfig()]);
      renderHaDiagnostics();
    }

    function scheduleHaDiagnosticsRefresh() {
      [1200, 3200, 6200].forEach((delay) => {
        window.setTimeout(() => refreshHaDiagnostics().catch(() => {}), delay);
      });
    }

    async function requestHaAction(action, payload) {
      return fetchJson("/api/ha_actions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action,
          ...(payload || {}),
        }),
      });
    }

    async function refreshHaCatalog() {
      try {
        const result = await requestHaAction("refresh_catalog", {});
        setMiniStatus(
          el.haSensorStatus,
          "refresco solicitado a Home Assistant | " + String(result.request_id || ""),
          false
        );
        await fetchHaActions();
        scheduleHaDiagnosticsRefresh();
      } catch (err) {
        setMiniStatus(el.haSensorStatus, String(err.message || err), true);
      }
    }

    async function createHaTestSensors() {
      const payload = {
        rooms: el.haTestRoomsInput ? el.haTestRoomsInput.value : "bedroom,kitchen,living",
        include_occupancy: el.haTestOccupancyInput ? el.haTestOccupancyInput.value !== "false" : true,
        initial_state: el.haTestInitialStateInput ? el.haTestInitialStateInput.value : "off",
      };
      try {
        const result = await requestHaAction("create_test_sensors", payload);
        setMiniStatus(
          el.haSensorStatus,
          "creación solicitada a Home Assistant | " + String(result.request_id || ""),
          false
        );
        await fetchHaActions();
        scheduleHaDiagnosticsRefresh();
      } catch (err) {
        setMiniStatus(el.haSensorStatus, String(err.message || err), true);
      }
    }

    async function applyPresenceFilter() {
      const payload = {
        enabled: el.petFilterEnabled ? el.petFilterEnabled.value !== "false" : true,
        window_seconds: numberFromSelect(el.petFilterWindowInput, 20),
        min_motion_events: numberFromSelect(el.petFilterMinEventsInput, 2),
        min_distinct_rooms: numberFromSelect(el.petFilterMinRoomsInput, 1),
      };
      try {
        const config = await fetchJson("/api/presence_filter", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        state.presenceFilter = config || payload;
        renderPresenceFilter();
        setMiniStatus(el.petFilterStatus, "filtro aplicado", false);
      } catch (err) {
        setMiniStatus(el.petFilterStatus, String(err.message || err), true);
      }
    }

    function startTrainingPolling() {
      stopTrainingPolling();
      state.trainingPollTimer = window.setInterval(() => {
        fetchModelInfo().catch(() => {});
      }, 1500);
    }

    function stopTrainingPolling() {
      if (state.trainingPollTimer) {
        window.clearInterval(state.trainingPollTimer);
        state.trainingPollTimer = null;
      }
    }

    async function fetchEvaluationMetrics() {
      const payload = await fetchJson("/api/evaluation_metrics?limit=120", { cache: "no-store" });
      if (payload && payload.metrics) {
        state.metrics = payload.metrics;
      }
      if (payload && payload.layout_reference) {
        state.reference = {
          ...state.reference,
          ...payload.layout_reference,
        };
      }
      renderAll();
    }

    async function fetchReplayStatus() {
      const payload = await fetchJson("/api/replay_status", { cache: "no-store" });
      state.replay = {
        ...state.replay,
        ...(payload || {}),
      };
      renderReplay();
    }

    async function setInputMode(mode) {
      try {
        const payload = await fetchJson("/api/input_mode", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mode }),
        });
        state.replay = {
          ...state.replay,
          mode: payload && payload.mode ? payload.mode : mode,
          running: payload && typeof payload.replay_running === "boolean" ? payload.replay_running : state.replay.running,
        };
        setMiniStatus(el.modeStatus, "modo activo: " + state.replay.mode, false);
        await fetchReplayStatus();
      } catch (err) {
        setMiniStatus(el.modeStatus, String(err.message || err), true);
      }
    }

    function updateTemplateHint() {
      const key = el.scenarioTemplate.value;
      const tpl = state.scenarioTemplates[key];
      if (!tpl) {
        el.templateHint.textContent = "sin metadata";
        return;
      }
      const edgeCount = Array.isArray(tpl.edges) ? tpl.edges.length : 0;
      el.templateHint.textContent = String(tpl.description || "template") + " | " + String(edgeCount) + " aristas";
    }

    function populateScenarioTemplates(templates) {
      state.scenarioTemplates = templates || {};
      const keys = Object.keys(state.scenarioTemplates);
      const prev = el.scenarioTemplate.value;
      const orderedKeys = keys.includes("real_home")
        ? ["real_home", ...keys.filter((key) => key !== "real_home")]
        : keys;

      el.scenarioTemplate.innerHTML = "";
      if (!orderedKeys.length) {
        const option = document.createElement("option");
        option.value = "real_home";
        option.textContent = "real_home";
        el.scenarioTemplate.appendChild(option);
      } else {
        orderedKeys.forEach((key) => {
          const option = document.createElement("option");
          option.value = key;
          option.textContent = key;
          el.scenarioTemplate.appendChild(option);
        });
      }

      if (prev && orderedKeys.includes(prev)) {
        el.scenarioTemplate.value = prev;
      } else if (orderedKeys.includes("real_home")) {
        el.scenarioTemplate.value = "real_home";
      }
      updateTemplateHint();
    }

    async function fetchScenarioTemplates() {
      const payload = await fetchJson("/api/scenario_templates", { cache: "no-store" });
      populateScenarioTemplates(payload ? payload.templates : {});
    }

    async function applyScenarioTemplate() {
      if (state.applyingTemplate) return;

      const key = el.scenarioTemplate.value || "real_home";
      const tpl = state.scenarioTemplates[key];
      if (!tpl || !tpl.adjacency) {
        updateTemplateHint();
        return;
      }

      try {
        state.applyingTemplate = true;
        el.layoutText.value = adjacencyToText(tpl.adjacency);
        state.layoutTextDirty = true;

        const response = await fetchJson("/api/layout_reference", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            adjacency: tpl.adjacency,
            rooms: collectRooms(),
          }),
        });

        if (response && response.layout_reference) {
          state.reference = {
            ...state.reference,
            ...response.layout_reference,
          };
        }
        if (response && response.metrics) {
          state.metrics = response.metrics;
        }

        state.layoutTextDirty = false;
        updateTemplateHint();
        setMiniStatus(el.layoutStatus, "plantilla aplicada: " + key, false);
        renderAll();
      } catch (err) {
        setMiniStatus(el.layoutStatus, String(err.message || err), true);
      } finally {
        state.applyingTemplate = false;
      }
    }

    async function startNewReplay() {
      try {
        const payload = buildReplayPayload(el);

        await fetchJson("/api/replay_csv", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        setMiniStatus(el.replayStatus, "replay iniciado", false);
        setMiniStatus(el.modeStatus, "modo activo: replay", false);
        await fetchReplayStatus();
      } catch (err) {
        setMiniStatus(el.replayStatus, String(err.message || err), true);
      }
    }

    async function replayControl(action) {
      try {
        const status = await fetchJson("/api/replay_control", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action }),
        });

        state.replay = {
          ...state.replay,
          ...(status || {}),
        };
        setMiniStatus(el.replayStatus, "accion aplicada: " + action, false);
        renderReplay();
      } catch (err) {
        setMiniStatus(el.replayStatus, String(err.message || err), true);
      }
    }

    function setTrainingBusy(isBusy) {
      [el.trainPresenceAutoBtn, el.trainPresenceManualBtn, el.trainHistoricalBtn].forEach((button) => {
        if (button) button.disabled = isBusy;
      });
    }

    function numberFromSelect(select, fallback) {
      const value = Number(select && select.value);
      return Number.isFinite(value) ? value : fallback;
    }

    async function trainPresenceFromSimulator(useManual) {
      const payload = useManual
        ? {
            template: "real_home",
            scenarios: numberFromSelect(el.trainScenariosInput, 240),
            steps_per_scenario: numberFromSelect(el.trainStepsInput, 90),
            max_people: numberFromSelect(el.trainMaxPeopleInput, 2),
            event_interval_seconds: 4,
            movement_probability: 0.7,
            occupancy_refresh_probability: 0.25,
            epochs: numberFromSelect(el.trainEpochsInput, 5),
            max_samples: numberFromSelect(el.trainMaxSamplesInput, 15000),
            seed: numberFromSelect(el.trainSeedInput, 42),
          }
        : {
            template: "real_home",
            scenarios: 240,
            steps_per_scenario: 90,
            max_people: 2,
            event_interval_seconds: 4,
            movement_probability: 0.7,
            occupancy_refresh_probability: 0.25,
            epochs: 5,
            max_samples: 15000,
            seed: 42,
          };

      try {
        setTrainingBusy(true);
        state.modelInfo.training_status = {
          ...(state.modelInfo.training_status || {}),
          presence: {
            state: "running",
            label: "Presencia simulador",
            message: "entrenando ocupacion desde simulador",
            started_at: new Date().toISOString(),
          },
        };
        renderTrainingInfo();
        startTrainingPolling();
        setMiniStatus(el.trainStatus, "entrenando presencia desde simulador...", false);
        const result = await fetchJson("/api/train_presence_simulator", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const info = result && result.training_info ? result.training_info : {};
        setMiniStatus(
          el.trainStatus,
          "presencia entrenada | muestras " + String(info.samples || 0) +
            " | conteo " + toPercent(info.count_accuracy),
          false
        );
        await fetchModelInfo();
        await fetchSnapshot();
      } catch (err) {
        setMiniStatus(el.trainStatus, String(err.message || err), true);
      } finally {
        stopTrainingPolling();
        setTrainingBusy(false);
      }
    }

    async function trainHistoricalCsv() {
      const payload = {
        csv_path: el.csvPath.value,
        debounce_seconds: numberFromSelect(el.debounceInput, 1),
        include_all_state_transitions: true,
        min_gap_seconds: 0,
        max_gap_seconds: 900,
        epochs: numberFromSelect(el.trainEpochsInput, 5),
        max_samples: numberFromSelect(el.trainMaxSamplesInput, 15000),
        degree_limit: 4,
        use_ollama_validation: false,
      };

      try {
        setTrainingBusy(true);
        setMiniStatus(el.trainStatus, "entrenando mapa desde CSV histórico...", false);
        state.modelInfo.training_status = {
          ...(state.modelInfo.training_status || {}),
          historical: {
            state: "running",
            label: "Historico CSV",
            message: "entrenando mapa desde CSV historico",
            started_at: new Date().toISOString(),
          },
        };
        renderTrainingInfo();
        startTrainingPolling();
        const result = await fetchJson("/api/train_model_full", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const info = result && result.training_info ? result.training_info : {};
        const transformer = info.transformer || {};
        setMiniStatus(
          el.trainStatus,
          "histórico entrenado | eventos " + String(info.events_total || 0) +
            " | transformer " + (transformer.enabled ? "activo" : "inactivo"),
          false
        );
        await fetchModelInfo();
        await fetchSnapshot();
      } catch (err) {
        setMiniStatus(el.trainStatus, String(err.message || err), true);
      } finally {
        stopTrainingPolling();
        setTrainingBusy(false);
      }
    }

    async function applyReferenceLayout() {
      try {
        const payload = {
          adjacency_text: el.layoutText.value,
          rooms: collectRooms(),
        };
        const response = await fetchJson("/api/layout_reference", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (response && response.layout_reference) {
          state.reference = {
            ...state.reference,
            ...response.layout_reference,
          };
        }
        if (response && response.metrics) {
          state.metrics = response.metrics;
        }

        state.layoutTextDirty = false;
        setMiniStatus(el.layoutStatus, "mapa real actualizado", false);
        renderAll();
      } catch (err) {
        setMiniStatus(el.layoutStatus, String(err.message || err), true);
      }
    }

    function connectWebSocket() {
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      const wsUrl = protocol + "://" + window.location.host + "/presencia";
      state.ws = new WebSocket(wsUrl);

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
        window.setTimeout(connectWebSocket, 2200);
      };

      state.ws.onmessage = (messageEvent) => {
        try {
          const payload = JSON.parse(messageEvent.data);
          if (payload && payload.kind === "snapshot" && payload.sim_data) {
            applySnapshot(payload.sim_data);
            setTopStatus("snapshot live", false);
            return;
          }
          applyEvent(payload);
        } catch (_err) {
          setTopStatus("mensaje websocket invalido", true);
        }
      };
    }

    function registerActions() {
      el.modeListenBtn.addEventListener("click", () => setInputMode("listen"));
      if (el.modeReplayBtn) {
        el.modeReplayBtn.addEventListener("click", () => setInputMode("replay"));
      }
      if (el.configOpenBtn) {
        el.configOpenBtn.addEventListener("click", openConfigDialog);
      }
      if (el.configCloseBtn) {
        el.configCloseBtn.addEventListener("click", closeConfigDialog);
      }
      if (el.configDialog) {
        el.configDialog.addEventListener("close", restoreConfigDialogFocus);
        el.configDialog.addEventListener("cancel", (event) => {
          event.preventDefault();
          closeConfigDialog();
        });
        el.configDialog.addEventListener("keydown", (event) => {
          if (event.key === "Escape") {
            event.preventDefault();
            closeConfigDialog();
          }
        });
        el.configDialog.addEventListener("click", (event) => {
          if (event.target !== el.configDialog) return;
          const rect = el.configDialog.getBoundingClientRect();
          const inside =
            event.clientX >= rect.left &&
            event.clientX <= rect.right &&
            event.clientY >= rect.top &&
            event.clientY <= rect.bottom;
          if (!inside) closeConfigDialog();
        });
      }
      if (el.haRefreshCatalogBtn) {
        el.haRefreshCatalogBtn.addEventListener("click", refreshHaCatalog);
      }
      if (el.haCreateTestSensorsBtn) {
        el.haCreateTestSensorsBtn.addEventListener("click", createHaTestSensors);
      }
      if (el.haCheckDiagnosticsBtn) {
        el.haCheckDiagnosticsBtn.addEventListener("click", () => {
          refreshHaDiagnostics().catch((err) => {
            setMiniStatus(el.haDiagnosticStatus, String(err.message || err), true);
          });
        });
      }
      if (el.realSensorAddRoomBtn) {
        el.realSensorAddRoomBtn.addEventListener("click", addRealSensorRoom);
      }
      if (el.realSensorSearchInput) {
        el.realSensorSearchInput.addEventListener("input", () => {
          state.realSensorSearch = el.realSensorSearchInput.value || "";
          renderHaEntityCatalog();
        });
      }
      if (el.realSensorRequireSelect) {
        el.realSensorRequireSelect.addEventListener("change", () => {
          realSensorWorkingConfig().require_explicit_selection = el.realSensorRequireSelect.value !== "false";
          markRealSensorDirty("modo pendiente: confirma para aplicar sensores reales");
          renderHaEntityCatalog();
        });
      }
      if (el.realSensorResetBtn) {
        el.realSensorResetBtn.addEventListener("click", resetRealSensorDraft);
      }
      if (el.realSensorApplyBtn) {
        el.realSensorApplyBtn.addEventListener("click", applyRealSensorConfig);
      }
      el.petFilterApplyBtn.addEventListener("click", applyPresenceFilter);
      el.layoutApplyBtn.addEventListener("click", applyReferenceLayout);
      el.layoutText.addEventListener("input", () => {
        state.layoutTextDirty = true;
      });

      el.replayNewBtn.addEventListener("click", startNewReplay);
      el.replayPauseBtn.addEventListener("click", () => replayControl("pause"));
      el.replayResumeBtn.addEventListener("click", () => replayControl("start"));
      el.replayStepBtn.addEventListener("click", () => replayControl("step"));
      el.replayResetBtn.addEventListener("click", () => replayControl("reset"));
      el.trainPresenceAutoBtn.addEventListener("click", () => trainPresenceFromSimulator(false));
      el.trainPresenceManualBtn.addEventListener("click", () => trainPresenceFromSimulator(true));
      el.trainHistoricalBtn.addEventListener("click", trainHistoricalCsv);
      el.scenarioTemplate.addEventListener("change", applyScenarioTemplate);
      el.mapTabFixed.addEventListener("click", () => setMapTab("fixed"));
      el.mapTabLive.addEventListener("click", () => setMapTab("live"));
      el.historyConfigSaveBtn.addEventListener("click", saveHistoryConfig);
      el.historyPurgeConfirmation.addEventListener("input", () => {
        el.historyPurgeBtn.disabled = el.historyPurgeConfirmation.value !== "BORRAR";
      });
      el.historyPurgeBtn.addEventListener("click", purgeHistory);
      el.historyFilterForm.addEventListener("submit", (event) => {
        event.preventDefault();
        applyHistoryFilters();
      });
      el.historyClearBtn.addEventListener("click", clearHistoryFilters);
      el.historyPrevBtn.addEventListener("click", () => {
        if (state.history.page <= 1) return;
        state.history.page -= 1;
        fetchHistory().catch((err) => setMiniStatus(el.historyChartStatus, String(err.message || err), true));
      });
      el.historyNextBtn.addEventListener("click", () => {
        if (state.history.page >= state.history.pages) return;
        state.history.page += 1;
        fetchHistory().catch((err) => setMiniStatus(el.historyChartStatus, String(err.message || err), true));
      });
      el.historyNewEventsBtn.addEventListener("click", () => {
        state.history.page = 1;
        fetchHistory().catch((err) => setMiniStatus(el.historyChartStatus, String(err.message || err), true));
      });
    }

    async function init() {
      applyDevMode();
      registerActions();
      setMapTab("fixed");
      el.apiBaseUrl.textContent = window.location.origin;
      el.historyFrom.value = isoToLocalInput(state.history.filters.fromTs);

      try {
        await fetchSnapshot();
        setTopStatus("snapshot inicial cargado", false);
      } catch (err) {
        setTopStatus(String(err.message || err), true);
      }

      try {
        await fetchModelInfo();
      } catch (err) {
        setMiniStatus(el.trainStatus, "estado modelo: " + String(err.message || err), true);
      }

      try {
        await fetchPresenceFilter();
      } catch (err) {
        setMiniStatus(el.petFilterStatus, "filtro: " + String(err.message || err), true);
      }

      try {
        await fetchHaEntityCatalog();
      } catch (err) {
        setMiniStatus(el.haSensorStatus, "catálogo HA: " + String(err.message || err), true);
      }

      try {
        await fetchHaActions();
      } catch (err) {
        setMiniStatus(el.haDiagnosticStatus, "acciones HA: " + String(err.message || err), true);
      }

      try {
        await fetchRealSensorConfig();
      } catch (err) {
        setMiniStatus(el.realSensorStatus, "sensores reales: " + String(err.message || err), true);
      }

      try {
        await fetchEvaluationMetrics();
      } catch (err) {
        setMiniStatus(el.layoutStatus, "metricas: " + String(err.message || err), true);
      }

      try {
        await fetchReplayStatus();
      } catch (_err) {
        // no-op
      }

      try {
        await fetchScenarioTemplates();
      } catch (err) {
        setMiniStatus(el.replayStatus, "plantillas: " + String(err.message || err), true);
      }

      try {
        await Promise.all([fetchHistoryConfig(), fetchHistory()]);
      } catch (err) {
        setMiniStatus(el.historyChartStatus, "historial: " + String(err.message || err), true);
      }

      connectWebSocket();

      window.setInterval(() => {
        fetchEvaluationMetrics().catch(() => {});
        fetchReplayStatus().catch(() => {});
        fetchModelInfo().catch(() => {});
        fetchPresenceFilter().catch(() => {});
        fetchHaEntityCatalog().catch(() => {});
        fetchHaActions().catch(() => {});
        fetchRealSensorConfig().catch(() => {});
      }, 5000);
    }

    init();
