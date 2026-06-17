export function cloneRealSensorConfig(config) {
  const source = config && typeof config === "object" ? config : {};
  const rooms = Array.isArray(source.rooms) ? source.rooms : [];
  const assignments = Array.isArray(source.assignments)
    ? source.assignments
    : [];
  return {
    rooms: [
      ...new Set(rooms.map((room) => String(room || "").trim()).filter(Boolean)),
    ].sort(),
    assignments: assignments
      .filter((assignment) => assignment && assignment.entity_id)
      .map((assignment) => ({
        entity_id: String(assignment.entity_id || "").trim(),
        room: String(assignment.room || "").trim(),
        enabled: assignment.enabled !== false,
        sensor_type: String(assignment.sensor_type || "auto"),
        training_role: String(assignment.training_role || "signal"),
      })),
    require_explicit_selection: source.require_explicit_selection !== false,
  };
}

export function realSensorAssignmentChanged(draft, saved) {
  const normalize = (assignment) => ({
    room: String((assignment && assignment.room) || ""),
    enabled: !!(
      assignment &&
      assignment.enabled !== false &&
      assignment.room
    ),
    sensor_type: String((assignment && assignment.sensor_type) || "auto"),
    training_role: String((assignment && assignment.training_role) || "signal"),
  });
  const normalizedDraft = normalize(draft);
  const normalizedSaved = normalize(saved);
  return (
    normalizedDraft.room !== normalizedSaved.room ||
    normalizedDraft.enabled !== normalizedSaved.enabled ||
    normalizedDraft.sensor_type !== normalizedSaved.sensor_type ||
    normalizedDraft.training_role !== normalizedSaved.training_role
  );
}
