export function renderProfileHeader({ state, el, profile }) {
  const profiles = state.profiles.items;
  el.profileSelect.innerHTML = "";
  el.profileSelect.add(new Option("Selecciona un perfil", ""));
  for (const item of profiles) {
    const active = item.id === state.profiles.activeId ? " · activo" : "";
    el.profileSelect.add(new Option(`${item.name}${active}`, item.id));
  }
  el.profileSelect.value = state.profiles.selectedId || "";
  el.profileNameInput.value = profile?.name || "";
  el.profileNameInput.disabled = !profile;
  el.profileSaveBtn.disabled = !profile || !state.profiles.dirty;
  el.profileActivateBtn.disabled =
    !profile || profile.id === state.profiles.activeId;
  el.profileDeleteBtn.disabled = !profile;
  el.profileEditor.hidden = !profile;
  el.profileOnboarding.hidden =
    !!state.profiles.activeId || profiles.length > 0;
  el.profileNoActiveNotice.hidden = !!state.profiles.activeId;
  el.profileRevision.textContent = profile
    ? `Revisión ${profile.revision} · origen ${profile.source}`
    : "Sin perfil seleccionado";
}

export function renderProfileAreas({
  el,
  profile,
  areas,
  documentRef = document,
}) {
  const linked = new Map(
    (profile.areas || []).map((area) => [area.area_id, area]),
  );
  el.profileAreaList.innerHTML = "";
  if (!areas.length) {
    const empty = documentRef.createElement("p");
    empty.className = "hint";
    empty.textContent = "Home Assistant no publicó áreas.";
    el.profileAreaList.appendChild(empty);
    return;
  }
  for (const area of areas) {
    const label = documentRef.createElement("label");
    label.className = "profile-area-option";
    const input = documentRef.createElement("input");
    input.type = "checkbox";
    input.checked = linked.has(area.area_id);
    input.dataset.profileArea = area.area_id;
    input.setAttribute("aria-label", `Vincular área ${area.name}`);
    const text = documentRef.createElement("span");
    text.textContent =
      `${area.name} · ${(area.entity_ids || []).length} entidades`;
    label.append(input, text);
    el.profileAreaList.appendChild(label);
  }
}

export function renderProfileRooms({
  el,
  profile,
  documentRef = document,
}) {
  el.profileRoomList.innerHTML = "";
  for (const room of profile.rooms || []) {
    const row = documentRef.createElement("tr");
    const slug = documentRef.createElement("th");
    slug.scope = "row";
    slug.textContent = room.slug;
    row.appendChild(slug);

    const nameCell = documentRef.createElement("td");
    const name = documentRef.createElement("input");
    name.value = room.name || "";
    name.dataset.profileRoomName = room.slug;
    name.setAttribute("aria-label", `Nombre visible de ${room.slug}`);
    nameCell.appendChild(name);
    row.appendChild(nameCell);

    const areaCell = documentRef.createElement("td");
    areaCell.textContent = room.area_name || "Manual";
    if (room.warning) {
      const warning = documentRef.createElement("small");
      warning.className = "profile-warning";
      warning.textContent = room.warning;
      areaCell.appendChild(warning);
    }
    row.appendChild(areaCell);

    const actionCell = documentRef.createElement("td");
    const remove = documentRef.createElement("button");
    remove.type = "button";
    remove.className = "secondary compact-button";
    remove.textContent = "Eliminar";
    remove.dataset.profileRoomRemove = room.slug;
    remove.setAttribute("aria-label", `Eliminar habitación ${room.name}`);
    actionCell.appendChild(remove);
    row.appendChild(actionCell);
    el.profileRoomList.appendChild(row);
  }
}

export function renderProfileEdges({
  el,
  profile,
  documentRef = document,
}) {
  for (const select of [el.profileEdgeFrom, el.profileEdgeTo]) {
    select.innerHTML = "";
    select.add(new Option("Selecciona", ""));
    for (const room of profile.rooms || []) {
      select.add(new Option(room.name || room.slug, room.slug));
    }
  }
  el.profileEdgeList.innerHTML = "";
  for (const [index, edge] of (profile.edges || []).entries()) {
    const row = documentRef.createElement("tr");
    const from = profile.rooms.find((room) => room.slug === edge[0]);
    const to = profile.rooms.find((room) => room.slug === edge[1]);
    const connection = documentRef.createElement("th");
    connection.scope = "row";
    connection.textContent =
      `${from?.name || edge[0]} ↔ ${to?.name || edge[1]}`;
    row.appendChild(connection);
    const action = documentRef.createElement("td");
    const remove = documentRef.createElement("button");
    remove.type = "button";
    remove.className = "secondary compact-button";
    remove.textContent = "Eliminar";
    remove.dataset.profileEdgeRemove = String(index);
    action.appendChild(remove);
    row.appendChild(action);
    el.profileEdgeList.appendChild(row);
  }
}

export function renderProfileProposals({
  el,
  proposals,
  documentRef = document,
}) {
  el.profileProposalList.innerHTML = "";
  if (!proposals.length) {
    const empty = documentRef.createElement("p");
    empty.className = "hint";
    empty.textContent = "Sin propuestas históricas pendientes.";
    el.profileProposalList.appendChild(empty);
    return;
  }
  for (const proposal of proposals) {
    const label = documentRef.createElement("label");
    label.className = "profile-proposal";
    const input = documentRef.createElement("input");
    input.type = "checkbox";
    input.checked = true;
    input.dataset.profileProposal = `${proposal.a}\u0000${proposal.b}`;
    const text = documentRef.createElement("span");
    text.textContent =
      `${proposal.a} ↔ ${proposal.b} · soporte ${proposal.support} · ` +
      `confianza ${Math.round(Number(proposal.confidence || 0) * 100)}%`;
    label.append(input, text);
    el.profileProposalList.appendChild(label);
  }
}
