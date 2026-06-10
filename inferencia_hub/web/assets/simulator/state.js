export function createSimulatorState() {
  return {
    rooms: [],
    mode: "listen",
    switches: new Map(),
    scenarioTemplates: {},
    layoutKey: "real_home",
    layoutRooms: [],
    roomRects: new Map(),
    layoutBounds: { cols: 5, rows: 2 },
    occupantCount: 1,
    occupants: [
      { id: 1, x: 1.5, y: 0.5, room: "sittingroom", enabled: true },
      { id: 2, x: 3.5, y: 0.5, room: "foyer", enabled: false },
    ],
    layoutMotionRooms: new Set(),
    layoutOccupancyRooms: new Set(),
    pressedKeys: new Set(),
    animationFrame: null,
    lastFrameAt: 0,
  };
}
