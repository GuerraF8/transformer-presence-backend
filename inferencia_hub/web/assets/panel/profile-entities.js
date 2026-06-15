import { SENSOR_TYPES, TRAINING_ROLES } from "./profile-draft.js";

function filteredEntities(state, el, entities) {
  const query = String(el.realSensorSearchInput?.value || "")
    .trim()
    .toLowerCase();
  const areaFilter = state.profiles.areaFilter;
  return entities.filter((entity) => {
    if (areaFilter === "__none__" && entity.area_id) return false;
    if (areaFilter && areaFilter !== "__none__" && entity.area_id !== areaFilter) {
      return false;
    }
    if (!query) return true;
    return [
      entity.entity_id,
      entity.name,
      entity.area_name,
      entity.domain,
      entity.state,
      entity.sensor_type,
      entity.device_class,
    ]
      .map((value) => String(value || "").toLowerCase())
      .join(" ")
      .includes(query);
  });
}

export function renderProfileEntityFilters({ state, el, areas }) {
  const selected = state.profiles.areaFilter;
  el.profileAreaFilter.innerHTML = "";
  el.profileAreaFilter.add(new Option("Todas las áreas", ""));
  el.profileAreaFilter.add(new Option("Sin área", "__none__"));
  for (const area of areas) {
    el.profileAreaFilter.add(new Option(area.name, area.area_id));
  }
  el.profileAreaFilter.value = selected;
}

export function renderProfileEntities({
  state,
  el,
  profile,
  entities,
  areas,
  appendBadgeCell,
  roomOptions,
  documentRef = document,
}) {
  el.haSensorList.innerHTML = "";
  if (!profile) {
    const row = documentRef.createElement("tr");
    const cell = documentRef.createElement("td");
    cell.colSpan = 7;
    cell.textContent = "Selecciona o crea un perfil para asignar entidades.";
    row.appendChild(cell);
    el.haSensorList.appendChild(row);
    return;
  }

  const assignments = new Map(
    (profile.assignments || []).map((item) => [item.entity_id, item]),
  );
  const visible = filteredEntities(state, el, entities);
  el.haSensorSummary.textContent =
    `${areas.length} áreas / ${entities.length} entidades`;
  el.realSensorSummary.textContent =
    `${(profile.assignments || []).filter((item) => item.enabled).length} entidades confirmadas / ` +
    `${(profile.assignments || []).filter((item) => item.enabled && item.training_role === "person_confirmation").length} persona / ` +
    `${(profile.assignments || []).filter((item) => item.enabled && item.training_role === "pet_confirmation").length} mascota / ` +
    `${(profile.rooms || []).length} habitaciones`;
  el.realSensorApplyBtn.disabled = !state.profiles.dirty;
  el.realSensorResetBtn.disabled = !state.profiles.dirty;

  if (!visible.length) {
    const row = documentRef.createElement("tr");
    const cell = documentRef.createElement("td");
    cell.colSpan = 7;
    cell.textContent = "No hay entidades para el filtro seleccionado.";
    row.appendChild(cell);
    el.haSensorList.appendChild(row);
    return;
  }

  for (const entity of visible) {
    const assignment = assignments.get(entity.entity_id);
    const row = documentRef.createElement("tr");
    row.classList.toggle("real-sensor-selected", !!assignment?.enabled);
    row.classList.toggle(
      "real-sensor-warning",
      !!assignment && assignment.status !== "active",
    );

    const useCell = documentRef.createElement("td");
    const use = documentRef.createElement("input");
    use.type = "checkbox";
    use.checked = !!assignment?.enabled;
    use.dataset.profileEntity = entity.entity_id;
    use.setAttribute("aria-label", `Usar ${entity.entity_id}`);
    useCell.appendChild(use);
    row.appendChild(useCell);

    const name = documentRef.createElement("th");
    name.scope = "row";
    name.textContent = entity.name || entity.entity_id;
    name.title = entity.entity_id;
    row.appendChild(name);

    const area = documentRef.createElement("td");
    area.textContent = entity.area_name || "Sin área";
    row.appendChild(area);

    const typeCell = documentRef.createElement("td");
    const type = documentRef.createElement("select");
    type.dataset.profileEntityType = entity.entity_id;
    type.disabled = !assignment?.enabled;
    type.setAttribute("aria-label", `Categoría de ${entity.entity_id}`);
    for (const value of SENSOR_TYPES) type.add(new Option(value, value));
    type.value = assignment?.sensor_type || (
      SENSOR_TYPES.includes(entity.sensor_type) ? entity.sensor_type : "other"
    );
    typeCell.appendChild(type);
    row.appendChild(typeCell);

    const roleCell = documentRef.createElement("td");
    const role = documentRef.createElement("select");
    role.dataset.profileEntityRole = entity.entity_id;
    role.disabled = !assignment?.enabled;
    role.setAttribute("aria-label", `Uso de ${entity.entity_id}`);
    for (const [value, label] of TRAINING_ROLES) {
      role.add(new Option(label, value));
    }
    role.value = assignment?.training_role || "signal";
    roleCell.appendChild(role);
    row.appendChild(roleCell);

    const roomCell = documentRef.createElement("td");
    const room = documentRef.createElement("select");
    room.dataset.profileEntityRoom = entity.entity_id;
    room.disabled = !assignment?.enabled;
    room.setAttribute("aria-label", `Habitación de ${entity.entity_id}`);
    roomOptions(room, assignment?.room_slug || "");
    roomCell.appendChild(room);
    if (assignment?.warning) {
      const warning = documentRef.createElement("small");
      warning.className = "profile-warning";
      warning.textContent = assignment.warning;
      roomCell.appendChild(warning);
    }
    row.appendChild(roomCell);
    appendBadgeCell(
      row,
      String(entity.state || "-"),
      String(entity.state || "").toLowerCase() === "on" ? "on" : "off",
    );
    el.haSensorList.appendChild(row);
  }
}
