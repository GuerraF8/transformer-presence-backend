import assert from "node:assert/strict";
import test from "node:test";

import { buildReplayPayload } from "../replay.js";
import {
  edgeKey,
  formatBytes,
  roomLabel,
  toPercent,
} from "../panel/format.js";
import { createSimulatorState } from "../simulator/state.js";
import { createPanelState } from "../panel/state.js";
import { adjacencyToEdges, adjacencyToText } from "../panel/map.js";
import {
  cloneRealSensorConfig,
  realSensorAssignmentChanged,
} from "../panel/real-sensors.js";
import {
  HISTORY_FILTER_STORAGE_KEY,
  buildHistorySearchParams,
  defaultHistoryFilters,
  hasActiveHistoryFilters,
  loadHistoryFilters,
  saveHistoryFilters,
} from "../panel/history.js";
import { collectRooms } from "../panel/rooms.js";
import { numberFromSelect } from "../panel/replay-training.js";
import {
  applyEventState,
  applySnapshotState,
} from "../panel/realtime.js";
import {
  resolveBackendUrl as resolvePanelBackendUrl,
  resolveBackendWebSocketUrl,
} from "../panel/api.js";
import {
  resolveBackendUrl as resolveSimulatorBackendUrl,
} from "../simulator/api.js";

test("format helpers preserve panel output contracts", () => {
  assert.equal(toPercent(0.125), "12.5%");
  assert.equal(formatBytes(1024), "1.0 KB");
  assert.equal(roomLabel("entertainment_room"), "entertainment room");
  assert.equal(edgeKey("kitchen", "foyer"), "foyer|kitchen");
});

test("replay payload uses current controls", () => {
  const payload = buildReplayPayload({
    csvPath: { value: " /data/history.csv " },
    speedInput: { value: "25" },
    debounceInput: { value: "2" },
    maxEventsInput: { value: "100" },
    useScenarioLayout: { checked: true },
    scenarioTemplate: { value: "anillo" },
    stepSecondsInput: { value: "4" },
  });
  assert.equal(payload.csv_path, "/data/history.csv");
  assert.equal(payload.speed_events_per_second, 25);
  assert.equal(payload.use_scenario_layout, true);
  assert.equal(payload.template, "anillo");
});

test("simulator state is isolated per instance", () => {
  const first = createSimulatorState();
  const second = createSimulatorState();
  first.rooms.push("kitchen");
  first.switches.set("kitchen:motion", true);
  assert.deepEqual(second.rooms, []);
  assert.equal(second.switches.size, 0);
});

test("panel state is isolated and history defaults include all events", () => {
  const first = createPanelState();
  const second = createPanelState();
  first.rooms.push("kitchen");
  assert.deepEqual(second.rooms, []);
  assert.deepEqual(first.history.filters, defaultHistoryFilters());
  assert.equal(first.history.filters.inputMode, "");
  assert.equal(first.history.filters.fromTs, "");
  assert.equal(first.history.alerts.pageSize, 25);
});

test("backend URLs preserve the Home Assistant proxy prefix", () => {
  const panelBase =
    "https://example.ui/api/inferencia_presencia/panel/token/?embedded=1";
  const simulatorBase =
    "https://example.ui/api/inferencia_presencia/panel/token/simulator.html";
  assert.equal(
    resolvePanelBackendUrl("/api/history/alerts", panelBase),
    "https://example.ui/api/inferencia_presencia/panel/token/api/history/alerts",
  );
  assert.equal(
    resolveSimulatorBackendUrl("/api/sim_data", simulatorBase),
    "https://example.ui/api/inferencia_presencia/panel/token/api/sim_data",
  );
  assert.equal(
    resolveBackendWebSocketUrl("presencia", panelBase),
    "wss://example.ui/api/inferencia_presencia/panel/token/presencia",
  );
});

test("map helpers normalize adjacency without duplicate edges", () => {
  const adjacency = { kitchen: ["living"], living: ["kitchen"] };
  assert.deepEqual(adjacencyToEdges(adjacency), [
    { a: "kitchen", b: "living", support: 1 },
  ]);
  assert.equal(adjacencyToText(adjacency), "kitchen: living\nliving: kitchen");
});

test("real sensor helpers normalize and compare assignments", () => {
  const config = cloneRealSensorConfig({
    rooms: [" kitchen ", "kitchen"],
    assignments: [{ entity_id: "binary_sensor.kitchen", room: "kitchen" }],
  });
  assert.deepEqual(config.rooms, ["kitchen"]);
  assert.equal(config.assignments[0].sensor_type, "auto");
  assert.equal(
    realSensorAssignmentChanged(
      config.assignments[0],
      { ...config.assignments[0], room: "living" },
    ),
    true,
  );
});

test("history filters map to the backend query contract", () => {
  const params = buildHistorySearchParams({
    query: "kitchen",
    sensorType: "motion",
    room: "kitchen",
    inputMode: "listen",
    fromTs: "2026-06-09T00:00:00Z",
    toTs: "",
  });
  assert.equal(params.get("sensor_type"), "motion");
  assert.equal(params.get("input_mode"), "listen");
  assert.equal(params.get("query"), "kitchen");
  assert.equal(hasActiveHistoryFilters(defaultHistoryFilters()), false);
  assert.equal(hasActiveHistoryFilters({ room: "kitchen" }), true);
});

test("history filters persist and restore from browser storage", () => {
  const values = new Map();
  const storage = {
    getItem(key) {
      return values.get(key) ?? null;
    },
    setItem(key, value) {
      values.set(key, value);
    },
  };
  const filters = {
    query: "motion",
    sensorType: "motion",
    room: "kitchen",
    inputMode: "simulator",
    fromTs: "2026-06-09T00:00:00Z",
    toTs: "",
  };

  saveHistoryFilters(storage, filters);

  assert.ok(values.has(HISTORY_FILTER_STORAGE_KEY));
  assert.deepEqual(loadHistoryFilters(storage), filters);
});

test("invalid persisted history filters fall back to all events", () => {
  const storage = {
    getItem() {
      return "{invalid";
    },
  };
  assert.deepEqual(loadHistoryFilters(storage), defaultHistoryFilters());
});

test("room collection combines configured and layout rooms", () => {
  assert.deepEqual(
    collectRooms({
      realSensorConfig: { rooms: ["office"] },
      reference: {
        rooms: ["kitchen"],
        adjacency: { kitchen: ["living"] },
      },
      rooms: [],
    }),
    ["kitchen", "living", "office"],
  );
});

test("numeric replay controls use fallbacks for invalid values", () => {
  assert.equal(numberFromSelect({ value: "12" }, 4), 12);
  assert.equal(numberFromSelect({ value: "invalid" }, 4), 4);
});

test("realtime reducers update snapshots and events without DOM", () => {
  const state = createPanelState();
  applySnapshotState(state, {
    rooms: ["kitchen"],
    events: [],
    presence: { current_room: "kitchen", active_rooms: ["kitchen"] },
    inferred_layout_live: {
      edges: [{ a: "kitchen", b: "living", support: 2 }],
    },
  });
  assert.equal(state.currentRoom, "kitchen");
  assert.equal(state.inferredEdges.get("kitchen|living"), 2);

  applyEventState(state, {
    room: "living",
    state: "on",
    sensor_type: "occupancy",
    active_rooms: ["living"],
  });
  assert.deepEqual(state.activeRooms, ["living"]);
  assert.deepEqual(state.occupancyRooms, ["living"]);
});
