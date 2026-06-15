import {
  availableRoomSlug,
  profileSlug,
  SENSOR_TYPES,
} from "./profile-draft.js";

export function appendProfileRoom(profile, name, area = null) {
  const visibleName = String(name || "").trim();
  if (!visibleName) return null;
  const slug = availableRoomSlug(profileSlug(visibleName), profile.rooms);
  profile.rooms.push({
    slug,
    name: visibleName,
    area_id: area?.area_id || "",
    area_name: area?.name || "",
    status: "active",
    warning: "",
  });
  if (area) {
    profile.areas.push({
      area_id: area.area_id,
      room_slug: slug,
      name: area.name,
    });
  }
  return slug;
}

export function removeProfileRoom(profile, slug) {
  profile.rooms = profile.rooms.filter((room) => room.slug !== slug);
  profile.areas = profile.areas.filter((area) => area.room_slug !== slug);
  profile.assignments = profile.assignments.filter(
    (assignment) => assignment.room_slug !== slug,
  );
  profile.edges = profile.edges.filter((edge) => !edge.includes(slug));
}

export function setProfileEntity(profile, entity, enabled) {
  const entityId = entity.entity_id;
  if (!enabled) {
    profile.assignments = profile.assignments.filter(
      (item) => item.entity_id !== entityId,
    );
    return;
  }
  let assignment = profile.assignments.find(
    (item) => item.entity_id === entityId,
  );
  let roomSlug =
    profile.areas.find((area) => area.area_id === entity.area_id)?.room_slug ||
    profile.rooms[0]?.slug ||
    "";
  if (!roomSlug) {
    roomSlug = appendProfileRoom(
      profile,
      entity.area_name || "Habitación",
    );
  }
  if (!assignment) {
    assignment = {
      entity_id: entityId,
      room_slug: roomSlug,
      enabled: true,
      sensor_type: SENSOR_TYPES.includes(entity.sensor_type)
        ? entity.sensor_type
        : "other",
      training_role: "signal",
      area_id: entity.area_id || "",
      area_name: entity.area_name || "",
      status: "active",
      warning: "",
      unique_id: entity.unique_id || "",
      platform: entity.platform || "",
    };
    profile.assignments.push(assignment);
  } else {
    assignment.enabled = true;
  }
}

export function appendProfileEdge(profile, left, right) {
  if (!left || !right || left === right) return false;
  const key = [left, right].sort();
  const exists = profile.edges.some(
    (edge) => [...edge].sort().join("\u0000") === key.join("\u0000"),
  );
  if (exists) return false;
  profile.edges.push(key);
  return true;
}
