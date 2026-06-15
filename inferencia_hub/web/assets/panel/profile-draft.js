export const SENSOR_TYPES = ["motion", "door", "occupancy", "other"];
export const TRAINING_ROLES = [
  ["signal", "Señal de inferencia"],
  ["person_confirmation", "Confirmación de persona"],
  ["pet_confirmation", "Confirmación de mascota"],
];

export function cloneProfile(value) {
  return value == null ? value : JSON.parse(JSON.stringify(value));
}

export function profileSlug(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "") || "habitacion";
}

export function availableRoomSlug(base, rooms) {
  const used = new Set((rooms || []).map((room) => room.slug));
  if (!used.has(base)) return base;
  let suffix = 2;
  while (used.has(`${base}_${suffix}`)) suffix += 1;
  return `${base}_${suffix}`;
}

export function validProfileRoomSelection(value, rooms) {
  const selected = String(value || "");
  return (rooms || []).some((room) => room.slug === selected)
    ? selected
    : "";
}

export function profileSelectionChanged(previousId, selectedProfile) {
  return (selectedProfile?.id || null) !== (previousId || null);
}

export function profileUpdatePayload(profile) {
  return {
    revision: profile.revision,
    name: profile.name,
    rooms: profile.rooms || [],
    areas: profile.areas || [],
    assignments: profile.assignments || [],
    edges: profile.edges || [],
  };
}
