export function collectRooms(state) {
  const rooms = new Set();
  for (const room of state.realSensorConfig?.rooms || []) {
    if (room) rooms.add(String(room));
  }
  for (const room of state.reference?.rooms || []) {
    if (room) rooms.add(String(room));
  }
  for (const [room, neighbors] of Object.entries(
    state.reference?.adjacency || {},
  )) {
    rooms.add(room);
    for (const neighbor of neighbors || []) rooms.add(String(neighbor));
  }
  if (!rooms.size) {
    for (const room of state.rooms || []) {
      if (room) rooms.add(String(room));
    }
  }
  return [...rooms].filter(Boolean).sort();
}
