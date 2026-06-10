import { fetchJson } from "./api.js";
import { createSimulatorState } from "./state.js";

const state = createSimulatorState();

function byId(...ids) {
  for (const id of ids) {
    const node = document.getElementById(id);
    if (node) return node;
  }
  return null;
}

const el = {
  modeSummary: byId("simModeSummary", "modeSummary"),
  listenModeBtn: byId("simListenModeBtn", "listenModeBtn"),
  refreshBtn: byId("simRefreshBtn", "refreshBtn"),
  sendDataBtn: byId("simSendDataBtn", "sendDataBtn"),
  status: byId("simStatus", "status"),
  sensorSummary: byId("simSensorSummary", "sensorSummary"),
  sensorGrid: byId("simSensorGrid", "sensorGrid"),
  layoutSelect: byId("simLayoutSelect", "layoutSelect"),
  occupantCount: byId("simOccupantCount", "occupantCount"),
  layoutSummary: byId("simLayoutSummary", "layoutSummary"),
  homeSim: byId("simHomeSim", "homeSim"),
  occupantStatus: byId("simOccupantStatus", "occupantStatus"),
  scenarioStatus: byId("simScenarioStatus", "scenarioStatus"),
  scenarioOneBtn: byId("simScenarioOneBtn", "scenarioOneBtn"),
  scenarioTwoBtn: byId("simScenarioTwoBtn", "scenarioTwoBtn"),
  scenarioAnchorBtn: byId("simScenarioAnchorBtn", "scenarioAnchorBtn"),
  scenarioClearBtn: byId("simScenarioClearBtn", "scenarioClearBtn"),
};

const DEFAULT_REAL_HOME_ROOMS = [
  "bedroom",
  "sittingroom",
  "entertainment_room",
  "foyer",
  "kitchen",
  "living",
];

const REAL_HOME_COORDS = {
  bedroom: { col: 0, row: 0 },
  sittingroom: { col: 1, row: 0 },
  entertainment_room: { col: 2, row: 0 },
  foyer: { col: 3, row: 0 },
  kitchen: { col: 4, row: 0 },
  living: { col: 3, row: 1 },
};

const MOVEMENT_SPEED = 0.95;

function setStatus(text, isError) {
  el.status.textContent = text;
  el.status.className = "mini " + (isError ? "error" : "ok");
}

function setScenarioStatus(text, isError) {
  el.scenarioStatus.textContent = text;
  el.scenarioStatus.className = isError ? "error" : "ok";
}

function roomLabel(room) {
  return String(room || "-").replace(/_/g, " ");
}

function sensorKey(room, sensorType) {
  return room + "|" + sensorType;
}

function collectTemplateRooms(template) {
  const roomBag = new Set();
  const adjacency = template && template.adjacency ? template.adjacency : {};

  Object.keys(adjacency).forEach((room) => roomBag.add(String(room)));
  Object.values(adjacency).forEach((neighbors) => {
    (neighbors || []).forEach((room) => roomBag.add(String(room)));
  });

  return [...roomBag].filter(Boolean).sort();
}

async function legacyFetchJson(url, options) {
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

async function setInputMode(mode) {
  const payload = await fetchJson("/api/input_mode", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
  state.mode = payload && payload.mode ? payload.mode : mode;
  renderMode();
}

async function loadRooms() {
  const snapshot = await fetchJson("/api/sim_data", { cache: "no-store" });
  const templatePayload = await fetchJson("/api/scenario_templates", { cache: "no-store" });
  const roomBag = new Set();

  if (snapshot && Array.isArray(snapshot.rooms)) {
    snapshot.rooms.forEach((room) => roomBag.add(String(room)));
  }

  const realHome = templatePayload && templatePayload.templates ? templatePayload.templates.real_home : null;
  if (realHome && realHome.adjacency) {
    Object.keys(realHome.adjacency).forEach((room) => roomBag.add(room));
    Object.values(realHome.adjacency).forEach((neighbors) => {
      (neighbors || []).forEach((room) => roomBag.add(String(room)));
    });
  }
  DEFAULT_REAL_HOME_ROOMS.forEach((room) => roomBag.add(room));

  if (snapshot && snapshot.replay && snapshot.replay.mode) {
    state.mode = snapshot.replay.mode;
  } else if (snapshot && snapshot.meta && snapshot.meta.input_mode) {
    state.mode = snapshot.meta.input_mode;
  }

  state.scenarioTemplates = templatePayload && templatePayload.templates ? templatePayload.templates : {};
  state.rooms = [...roomBag].filter(Boolean).sort();
  populateLayoutSelect();
  buildLayout();
  ensureOccupantsInLayout();
}

function renderMode() {
  el.modeSummary.textContent = state.mode === "simulator" ? "simulador activo" : String(state.mode || "listen");
  el.listenModeBtn.disabled = state.mode === "simulator";
}

function populateLayoutSelect() {
  const keys = Object.keys(state.scenarioTemplates);
  const orderedKeys = keys.includes("real_home")
    ? ["real_home", ...keys.filter((key) => key !== "real_home")]
    : ["real_home", ...keys];
  const uniqueKeys = [...new Set(orderedKeys)];
  const previous = state.layoutKey || el.layoutSelect.value || "real_home";

  el.layoutSelect.innerHTML = "";
  uniqueKeys.forEach((key) => {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = key;
    el.layoutSelect.appendChild(option);
  });

  state.layoutKey = uniqueKeys.includes(previous) ? previous : "real_home";
  el.layoutSelect.value = state.layoutKey;
}

function buildFallbackLayout(rooms) {
  const cols = Math.max(1, Math.ceil(Math.sqrt(rooms.length || 1)));
  const rects = new Map();
  rooms.forEach((room, index) => {
    rects.set(room, {
      x: index % cols,
      y: Math.floor(index / cols),
      w: 1,
      h: 1,
    });
  });
  return {
    rects,
    bounds: {
      cols,
      rows: Math.max(1, Math.ceil((rooms.length || 1) / cols)),
    },
  };
}

function buildLayout() {
  const template = state.scenarioTemplates[state.layoutKey];
  const templateRooms = collectTemplateRooms(template);
  let rooms = templateRooms.length ? templateRooms : [...state.rooms];

  if (state.layoutKey === "real_home") {
    const roomSet = new Set([...DEFAULT_REAL_HOME_ROOMS, ...rooms]);
    rooms = DEFAULT_REAL_HOME_ROOMS.filter((room) => roomSet.has(room));

    const rects = new Map();
    rooms.forEach((room) => {
      const coord = REAL_HOME_COORDS[room];
      if (!coord) return;
      rects.set(room, { x: coord.col, y: coord.row, w: 1, h: 1 });
    });

    state.layoutRooms = [...rects.keys()];
    state.roomRects = rects;
    state.layoutBounds = { cols: 5, rows: 2 };
  } else {
    rooms = [...new Set(rooms)].filter(Boolean).sort();
    const fallback = buildFallbackLayout(rooms);
    state.layoutRooms = rooms;
    state.roomRects = fallback.rects;
    state.layoutBounds = fallback.bounds;
  }

  el.layoutSummary.textContent =
    state.layoutKey + " | " + String(state.layoutRooms.length) + " habitaciones";
}

function roomCenter(room) {
  const rect = state.roomRects.get(room);
  if (!rect) return { x: 0.5, y: 0.5, room: null };
  return {
    x: rect.x + rect.w / 2,
    y: rect.y + rect.h / 2,
    room,
  };
}

function findRoomAt(x, y) {
  for (const [room, rect] of state.roomRects.entries()) {
    if (
      x >= rect.x &&
      x < rect.x + rect.w &&
      y >= rect.y &&
      y < rect.y + rect.h
    ) {
      return room;
    }
  }
  return null;
}

function ensureOccupantsInLayout() {
  const fallbackA = state.layoutRooms.includes("sittingroom") ? "sittingroom" : state.layoutRooms[0];
  const fallbackB = state.layoutRooms.includes("foyer")
    ? "foyer"
    : (state.layoutRooms[state.layoutRooms.length - 1] || fallbackA);

  state.occupants.forEach((occupant, index) => {
    const fallback = index === 0 ? fallbackA : fallbackB;
    if (!fallback) return;

    const currentRoom = findRoomAt(occupant.x, occupant.y);
    if (!currentRoom || !state.layoutRooms.includes(occupant.room)) {
      const center = roomCenter(fallback);
      occupant.x = center.x;
      occupant.y = center.y;
      occupant.room = center.room;
    } else {
      occupant.room = currentRoom;
    }
    occupant.enabled = occupant.id <= state.occupantCount;
  });
}

function currentOccupantRooms() {
  return state.occupants
    .filter((occupant) => occupant.enabled)
    .map((occupant) => occupant.room)
    .filter(Boolean);
}

function currentOccupantRoomSet() {
  return new Set(currentOccupantRooms());
}

function renderOccupantStatus() {
  const active = state.occupants.filter((occupant) => occupant.enabled);
  el.occupantStatus.textContent = active
    .map((occupant) => "P" + String(occupant.id) + ": " + roomLabel(occupant.room))
    .join(" | ");
}

function renderHomeSim() {
  el.homeSim.innerHTML = "";
  el.homeSim.style.setProperty("--layout-cols", String(state.layoutBounds.cols));
  el.homeSim.style.setProperty("--layout-rows", String(state.layoutBounds.rows));

  const activeRooms = new Set(currentOccupantRooms());
  state.layoutRooms.forEach((room) => {
    const rect = state.roomRects.get(room);
    if (!rect) return;

    const tile = document.createElement("button");
    tile.type = "button";
    tile.className = "home-room";
    tile.setAttribute("aria-label", "Alternar movimiento en " + roomLabel(room));
    if (state.switches.get(sensorKey(room, "motion"))) {
      tile.classList.add("motion-on");
    }
    tile.setAttribute("aria-pressed", String(!!state.switches.get(sensorKey(room, "motion"))));
    if (activeRooms.has(room)) {
      tile.classList.add("occupied");
    }
    tile.style.left = ((rect.x / state.layoutBounds.cols) * 100).toFixed(4) + "%";
    tile.style.top = ((rect.y / state.layoutBounds.rows) * 100).toFixed(4) + "%";
    tile.style.width = ((rect.w / state.layoutBounds.cols) * 100).toFixed(4) + "%";
    tile.style.height = ((rect.h / state.layoutBounds.rows) * 100).toFixed(4) + "%";
    tile.addEventListener("click", async () => {
      el.homeSim.focus();
      await toggleSensor(room, "motion");
    });

    const name = document.createElement("span");
    name.className = "home-room-name";
    name.textContent = roomLabel(room);

    const motion = document.createElement("span");
    motion.className = "home-motion";
    motion.textContent = state.switches.get(sensorKey(room, "motion")) ? "Movimiento activo" : "Movimiento inactivo";

    tile.appendChild(name);
    tile.appendChild(motion);
    el.homeSim.appendChild(tile);
  });

  state.occupants.forEach((occupant) => {
    if (!occupant.enabled || !occupant.room) return;
    const marker = document.createElement("div");
    marker.className = "occupant occupant-" + String(occupant.id);
    marker.style.left = ((occupant.x / state.layoutBounds.cols) * 100).toFixed(4) + "%";
    marker.style.top = ((occupant.y / state.layoutBounds.rows) * 100).toFixed(4) + "%";
    marker.textContent = String(occupant.id);
    el.homeSim.appendChild(marker);
  });

  renderOccupantStatus();
}

function renderSensors() {
  el.sensorGrid.innerHTML = "";
  const sensorTypes = [
    { key: "motion", label: "Movimiento" },
    { key: "occupancy", label: "Ocupación" },
  ];

  state.rooms.forEach((room) => {
    const card = document.createElement("article");
    card.className = "sensor-card";
    if (state.switches.get(sensorKey(room, "motion"))) {
      card.classList.add("sensor-card-active");
    }

    const title = document.createElement("h2");
    title.textContent = roomLabel(room);
    card.appendChild(title);

    sensorTypes.forEach((sensor) => {
      const key = sensorKey(room, sensor.key);
      if (!state.switches.has(key)) {
        state.switches.set(key, false);
      }

      const row = document.createElement("div");
      row.className = "switch-row";

      const label = document.createElement("span");
      label.textContent = sensor.label;

      const button = document.createElement("button");
      button.type = "button";
      button.className = state.switches.get(key) ? "warning" : "ghost";
      button.textContent = state.switches.get(key) ? "ON" : "OFF";
      button.addEventListener("click", () => toggleSensor(room, sensor.key));

      row.appendChild(label);
      row.appendChild(button);
      card.appendChild(row);
    });

    el.sensorGrid.appendChild(card);
  });

  el.sensorSummary.textContent = String(state.rooms.length * sensorTypes.length) + " sensores";
  renderHomeSim();
}

async function toggleSensor(room, sensorType) {
  const key = sensorKey(room, sensorType);
  const nextOn = !state.switches.get(key);
  await setSensorState(room, sensorType, nextOn, "sensor_simulator");
}

async function postSensorEvent(room, sensorType, nextOn, source) {
  try {
    if (state.mode !== "simulator") {
      await setInputMode("simulator");
    }

    const entityId = "binary_sensor." + room + "_" + sensorType + "_sim";
    const payload = {
      entity_id: entityId,
      state: nextOn ? "on" : "off",
      sensor_type: sensorType,
      room,
      timestamp: new Date().toISOString(),
      source: source || "sensor_simulator",
    };

    const response = await fetchJson("/api/events", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (response && response.status === "ignored") {
      setStatus("evento ignorado: " + String(response.reason || "-"), true);
      return;
    }

    setStatus(entityId + " -> " + payload.state, false);
  } catch (err) {
    setStatus(String(err.message || err), true);
  }
}

async function setSensorState(room, sensorType, nextOn, source) {
  const key = sensorKey(room, sensorType);
  state.switches.set(key, nextOn);
  renderSensors();
  await postSensorEvent(room, sensorType, nextOn, source);
}

async function sendCurrentData() {
  const activeEntries = [...state.switches.entries()].filter((entry) => entry[1]);
  const occupantRooms = currentOccupantRooms();

  try {
    setStatus("enviando datos simulados", false);

    if (activeEntries.length) {
      for (const [key] of activeEntries) {
        const [room, sensorType] = key.split("|");
        await postSensorEvent(room, sensorType, true, "manual_send");
      }
    } else {
      await syncLayoutOccupancyRooms(currentOccupantRoomSet(), "manual_send");
    }

    setStatus("datos enviados", false);
  } catch (err) {
    setStatus(String(err.message || err), true);
  }
}

function syncLayoutMotionRooms(activeRooms) {
  const nextRooms = new Set([...activeRooms].filter(Boolean));
  const changed = [];

  nextRooms.forEach((room) => {
    if (!state.layoutMotionRooms.has(room) || !state.switches.get(sensorKey(room, "motion"))) {
      changed.push({ room, on: true });
    }
  });

  state.layoutMotionRooms.forEach((room) => {
    if (!nextRooms.has(room)) {
      changed.push({ room, on: false });
    }
  });

  if (!changed.length) {
    return;
  }

  state.layoutMotionRooms = nextRooms;
  changed.forEach((change) => {
    state.switches.set(sensorKey(change.room, "motion"), change.on);
  });
  renderSensors();

  changed.forEach((change) => {
    postSensorEvent(change.room, "motion", change.on, "home_layout_simulator").catch((err) => {
      setStatus(String(err.message || err), true);
    });
  });
}

async function syncLayoutOccupancyRooms(activeRooms, source) {
  const nextRooms = new Set([...activeRooms].filter(Boolean));
  const changed = [];

  nextRooms.forEach((room) => {
    if (!state.layoutOccupancyRooms.has(room) || !state.switches.get(sensorKey(room, "occupancy"))) {
      changed.push({ room, on: true });
    }
  });

  state.layoutOccupancyRooms.forEach((room) => {
    if (!nextRooms.has(room)) {
      changed.push({ room, on: false });
    }
  });

  if (!changed.length) {
    return;
  }

  state.layoutOccupancyRooms = nextRooms;
  changed.forEach((change) => {
    state.switches.set(sensorKey(change.room, "occupancy"), change.on);
  });
  renderSensors();

  for (const change of changed) {
    await postSensorEvent(change.room, "occupancy", change.on, source || "home_layout_simulator");
  }
}

function clearLayoutMotionRooms() {
  syncLayoutMotionRooms(new Set());
}

function moveOccupant(occupant, dx, dy) {
  if (!occupant || !occupant.enabled) return;

  const previousRoom = occupant.room;
  const nextX = occupant.x + dx;
  const nextY = occupant.y + dy;
  const nextRoom = findRoomAt(nextX, nextY);
  if (!nextRoom) return false;

  occupant.x = nextX;
  occupant.y = nextY;
  occupant.room = nextRoom;
  return { moved: true, roomChanged: previousRoom !== nextRoom };
}

function movementVectorForOccupant(occupantId) {
  const keys = occupantId === 1
    ? {
        up: "w",
        down: "s",
        left: "a",
        right: "d",
      }
    : {
        up: "arrowup",
        down: "arrowdown",
        left: "arrowleft",
        right: "arrowright",
      };

  const dx = (state.pressedKeys.has(keys.right) ? 1 : 0) - (state.pressedKeys.has(keys.left) ? 1 : 0);
  const dy = (state.pressedKeys.has(keys.down) ? 1 : 0) - (state.pressedKeys.has(keys.up) ? 1 : 0);
  if (!dx && !dy) {
    return { dx: 0, dy: 0 };
  }

  const length = Math.hypot(dx, dy) || 1;
  return {
    dx: dx / length,
    dy: dy / length,
  };
}

function animationTick(now) {
  if (!state.lastFrameAt) {
    state.lastFrameAt = now;
  }

  const deltaSeconds = Math.min(0.05, Math.max(0, (now - state.lastFrameAt) / 1000));
  state.lastFrameAt = now;

  const movingRooms = new Set();
  let moved = false;
  let occupancyChanged = false;

  state.occupants.forEach((occupant) => {
    if (!occupant.enabled) return;

    const vector = movementVectorForOccupant(occupant.id);
    if (!vector.dx && !vector.dy) return;

    const movement = moveOccupant(
      occupant,
      vector.dx * MOVEMENT_SPEED * deltaSeconds,
      vector.dy * MOVEMENT_SPEED * deltaSeconds
    );

    if (movement && movement.moved) {
      movingRooms.add(occupant.room);
      moved = true;
      occupancyChanged = occupancyChanged || movement.roomChanged;
    }
  });

  syncLayoutMotionRooms(movingRooms);
  if (occupancyChanged) {
    syncLayoutOccupancyRooms(currentOccupantRoomSet(), "home_layout_simulator").catch((err) => {
      setStatus(String(err.message || err), true);
    });
  }
  if (moved) {
    renderHomeSim();
  }

  if (state.pressedKeys.size) {
    state.animationFrame = window.requestAnimationFrame(animationTick);
  } else {
    state.animationFrame = null;
    state.lastFrameAt = 0;
    clearLayoutMotionRooms();
  }
}

function startMovementLoop() {
  if (state.animationFrame !== null) return;
  state.lastFrameAt = 0;
  state.animationFrame = window.requestAnimationFrame(animationTick);
}

function isMovementKey(key) {
  return ["w", "a", "s", "d", "arrowup", "arrowleft", "arrowdown", "arrowright"].includes(key);
}

function handleKeyDown(event) {
  const targetTag = event.target && event.target.tagName ? String(event.target.tagName).toLowerCase() : "";
  if (targetTag === "input" || targetTag === "select" || targetTag === "textarea") {
    return;
  }

  const key = String(event.key || "").toLowerCase();
  if (!isMovementKey(key)) {
    return;
  }

  event.preventDefault();
  state.pressedKeys.add(key);
  startMovementLoop();
}

function handleKeyUp(event) {
  const key = String(event.key || "").toLowerCase();
  if (!isMovementKey(key)) {
    return;
  }

  event.preventDefault();
  state.pressedKeys.delete(key);
  if (!state.pressedKeys.size) {
    clearLayoutMotionRooms();
  }
}

function handleWindowBlur() {
  state.pressedKeys.clear();
  clearLayoutMotionRooms();
}

function resetLocalSwitches() {
  state.layoutMotionRooms = new Set();
  state.layoutOccupancyRooms = new Set();
  state.pressedKeys.clear();
  state.switches.clear();
  renderSensors();
}

function summarizePresenceSnapshot(snapshot) {
  const presence = snapshot && snapshot.presence ? snapshot.presence : {};
  const evaluation = snapshot && snapshot.evaluation ? snapshot.evaluation : {};
  const people = evaluation.people || {};
  const activeRooms = Array.isArray(presence.active_rooms) ? presence.active_rooms : [];
  const gtRooms = Array.isArray(presence.occupancy_ground_truth_rooms)
    ? presence.occupancy_ground_truth_rooms
    : [];
  return (
    "personas=" + String(people.current_estimate || 0) +
    " | actual=" + roomLabel(presence.current_room || "-") +
    " | activas=" + (activeRooms.length ? activeRooms.map(roomLabel).join(", ") : "-") +
    " | GT=" + (gtRooms.length ? gtRooms.map(roomLabel).join(", ") : "-")
  );
}

async function runPresenceScenario(name, steps) {
  try {
    setScenarioStatus("ejecutando " + name, false);
    await fetchJson("/api/reset", { method: "POST" });
    await setInputMode("simulator");
    resetLocalSwitches();

    for (const step of steps) {
      await setSensorState(step.room, step.sensorType, step.on, "presence_test:" + name);
    }

    const snapshot = await fetchJson("/api/sim_data", { cache: "no-store" });
    setScenarioStatus(summarizePresenceSnapshot(snapshot), false);
  } catch (err) {
    setScenarioStatus(String(err.message || err), true);
  }
}

function registerPresenceScenarios() {
  el.scenarioOneBtn.addEventListener("click", () => {
    runPresenceScenario("one_bedroom_gt", [
      { room: "bedroom", sensorType: "occupancy", on: true },
    ]);
  });

  el.scenarioTwoBtn.addEventListener("click", () => {
    runPresenceScenario("two_people_gt", [
      { room: "bedroom", sensorType: "occupancy", on: true },
      { room: "kitchen", sensorType: "occupancy", on: true },
    ]);
  });

  el.scenarioAnchorBtn.addEventListener("click", () => {
    runPresenceScenario("bedroom_anchor_external_motion", [
      { room: "bedroom", sensorType: "occupancy", on: true },
      { room: "sittingroom", sensorType: "motion", on: true },
      { room: "sittingroom", sensorType: "motion", on: false },
    ]);
  });

  el.scenarioClearBtn.addEventListener("click", async () => {
    try {
      await fetchJson("/api/reset", { method: "POST" });
      await setInputMode("simulator");
      resetLocalSwitches();
      setScenarioStatus("estado limpio", false);
    } catch (err) {
      setScenarioStatus(String(err.message || err), true);
    }
  });
}

async function init() {
  registerPresenceScenarios();

  el.listenModeBtn.addEventListener("click", async () => {
    try {
      await setInputMode("simulator");
      setStatus("modo simulador activo", false);
    } catch (err) {
      setStatus(String(err.message || err), true);
    }
  });

  el.refreshBtn.addEventListener("click", async () => {
    try {
      clearLayoutMotionRooms();
      await loadRooms();
      renderMode();
      renderSensors();
      renderHomeSim();
      setStatus("sensores recargados", false);
    } catch (err) {
      setStatus(String(err.message || err), true);
    }
  });

  if (el.sendDataBtn) {
    el.sendDataBtn.addEventListener("click", sendCurrentData);
  }

  el.layoutSelect.addEventListener("change", () => {
    clearLayoutMotionRooms();
    state.layoutKey = el.layoutSelect.value || "real_home";
    buildLayout();
    ensureOccupantsInLayout();
    syncLayoutOccupancyRooms(currentOccupantRoomSet(), "home_layout_simulator").catch((err) => {
      setStatus(String(err.message || err), true);
    });
    renderHomeSim();
  });

  el.occupantCount.addEventListener("change", () => {
    clearLayoutMotionRooms();
    state.occupantCount = Number(el.occupantCount.value || 1) === 2 ? 2 : 1;
    ensureOccupantsInLayout();
    syncLayoutOccupancyRooms(currentOccupantRoomSet(), "home_layout_simulator").catch((err) => {
      setStatus(String(err.message || err), true);
    });
    renderHomeSim();
    el.homeSim.focus();
  });

  window.addEventListener("keydown", handleKeyDown);
  window.addEventListener("keyup", handleKeyUp);
  window.addEventListener("blur", handleWindowBlur);

  try {
    await loadRooms();
    renderMode();
    renderSensors();
    renderHomeSim();
    setStatus("listo", false);
    window.setTimeout(() => el.homeSim.focus(), 0);
  } catch (err) {
    setStatus(String(err.message || err), true);
  }
}

init();
