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
  normalizePresenceFilterDraft,
} from "../panel/dashboard.js";
import {
  HISTORY_FILTER_STORAGE_KEY,
  buildHistorySearchParams,
  defaultHistoryFilters,
  hasActiveHistoryFilters,
  loadHistoryFilters,
  normalizeHistoryConfigDraft,
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
import {
  availableRoomSlug,
  profileSelectionChanged,
  profileSlug,
  profileUpdatePayload,
  validProfileRoomSelection,
} from "../panel/profile-draft.js";
import {
  appendProfileEdge,
  appendProfileRoom,
  removeProfileRoom,
  setProfileEntity,
} from "../panel/profile-mutations.js";
import { previewLabelLines } from "../panel/profile-preview.js";

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
    stepSecondsInput: { value: "4" },
  });
  assert.equal(payload.csv_path, "/data/history.csv");
  assert.equal(payload.speed_events_per_second, 25);
  assert.equal(payload.use_scenario_layout, false);
  assert.equal(payload.template, "real_home");
});

test("profile preview wraps long room labels inside the canvas", () => {
  assert.deepEqual(
    previewLabelLines("Sala de entretenimiento principal"),
    ["Sala de", "entretenimiento"],
  );
  assert.deepEqual(previewLabelLines("Cocina"), ["Cocina"]);
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
  assert.equal(first.presenceFilterDirty, false);
  assert.equal(first.history.configDirty, false);
});

test("configuration drafts normalize editable controls", () => {
  assert.deepEqual(
    normalizePresenceFilterDraft({
      enabled: false,
      window_seconds: 90,
      min_motion_events: 5,
      min_distinct_rooms: 3,
    }),
    {
      enabled: false,
      window_seconds: 90,
      min_motion_events: 5,
      min_distinct_rooms: 3,
    },
  );
  assert.deepEqual(
    normalizeHistoryConfigDraft({
      enabled: true,
      retention_days: 30,
      persisted_modes: ["listen", "invalid", "simulator"],
    }),
    {
      enabled: true,
      retention_days: 30,
      persisted_modes: ["listen", "simulator"],
    },
  );
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
    profile: { room_labels: { kitchen: "Cocina principal" } },
    events: [],
    meta: { backend_version: "0.7.0" },
    presence: {
      current_room: "kitchen",
      active_rooms: ["kitchen"],
      people_estimate: 2,
    },
    inferred_layout_live: {
      edges: [{ a: "kitchen", b: "living", support: 2 }],
    },
  });
  assert.equal(state.currentRoom, "kitchen");
  assert.equal(state.peopleEstimate, 2);
  assert.equal(state.backendVersion, "0.7.0");
  assert.equal(state.roomLabels.kitchen, "Cocina principal");
  assert.equal(state.inferredEdges.get("kitchen|living"), 2);

  applyEventState(state, {
    room: "living",
    state: "on",
    sensor_type: "occupancy",
    active_rooms: ["living"],
    estimated_people: 1,
  });
  assert.deepEqual(state.activeRooms, ["living"]);
  assert.deepEqual(state.occupancyRooms, ["living"]);
  assert.equal(state.peopleEstimate, 1);
});

test("profile draft helpers preserve stable slugs and update contracts", () => {
  assert.equal(profileSlug("Dormitorio Niños"), "dormitorio_ninos");
  assert.equal(
    availableRoomSlug("kitchen", [{ slug: "kitchen" }]),
    "kitchen_2",
  );
  assert.deepEqual(
    profileUpdatePayload({
      revision: 4,
      name: "Casa",
      rooms: [],
      areas: [],
      assignments: [],
      edges: [],
      model: { compatible: true },
    }),
    {
      revision: 4,
      name: "Casa",
      rooms: [],
      areas: [],
      assignments: [],
      edges: [],
    },
  );
  const rooms = [{ slug: "kitchen" }, { slug: "living" }];
  assert.equal(
    validProfileRoomSelection("kitchen", rooms),
    "kitchen",
  );
  assert.equal(validProfileRoomSelection("bedroom", rooms), "");
  assert.equal(
    profileSelectionChanged("profile-1", { id: "profile-1" }),
    false,
  );
  assert.equal(
    profileSelectionChanged("profile-1", { id: "profile-2" }),
    true,
  );
});

test("profile mutations keep rooms, assignments and edges consistent", () => {
  const profile = { rooms: [], areas: [], assignments: [], edges: [] };
  const room = appendProfileRoom(profile, "Cocina", {
    area_id: "kitchen",
    name: "Cocina",
  });
  setProfileEntity(profile, {
    entity_id: "light.kitchen",
    area_id: "kitchen",
    area_name: "Cocina",
    sensor_type: "other",
    unique_id: "light-1",
    platform: "test",
  }, true);
  appendProfileRoom(profile, "Pasillo");
  assert.equal(appendProfileEdge(profile, room, "pasillo"), true);
  assert.equal(profile.assignments[0].room_slug, "cocina");
  assert.equal(profile.assignments[0].training_role, "signal");

  removeProfileRoom(profile, "cocina");
  assert.equal(profile.assignments.length, 0);
  assert.equal(profile.edges.length, 0);
  assert.equal(profile.areas.length, 0);
});
