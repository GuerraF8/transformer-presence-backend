import { renderProfilePreview } from "./profile-preview.js";
import { renderProfileHeader } from "./profile-view.js";

export function registerProfileEvents({
  state,
  el,
  documentRef,
  windowRef,
  actions,
}) {
  const reportError = (error) =>
    actions.setStatus(String(error.message || error), true);

  el.profileSelect?.addEventListener("change", () => {
    if (
      state.profiles.dirty &&
      !windowRef.confirm("Hay cambios sin guardar. ¿Cambiar de perfil?")
    ) {
      el.profileSelect.value = state.profiles.selectedId || "";
      return;
    }
    actions.selectDraft(actions.profileById(el.profileSelect.value) || null);
    actions.renderAll();
  });
  el.profileNameInput?.addEventListener("input", () => {
    const profile = actions.draft();
    if (!profile) return;
    profile.name = el.profileNameInput.value;
    actions.markDirty();
    renderProfileHeader({ state, el, profile });
  });
  documentRef.querySelectorAll("[data-profile-create]").forEach((button) => {
    button.addEventListener("click", () =>
      actions.createProfile(button.dataset.profileCreate).catch(reportError),
    );
  });
  el.profileSaveBtn?.addEventListener("click", () =>
    actions.saveProfile().catch(reportError),
  );
  el.profileActivateBtn?.addEventListener("click", () =>
    actions.activateProfile().catch(reportError),
  );
  el.profileDeleteBtn?.addEventListener("click", () =>
    actions.deleteProfile().catch(reportError),
  );
  el.profileAreaList?.addEventListener("change", (event) => {
    const input = event.target.closest("[data-profile-area]");
    if (input) actions.toggleArea(input.dataset.profileArea, input.checked);
  });
  el.profileRoomAddBtn?.addEventListener("click", () => {
    const name = el.profileRoomName.value;
    if (actions.addRoom(name)) {
      el.profileRoomName.value = "";
      actions.render();
    }
  });
  el.profileRoomList?.addEventListener("input", (event) => {
    const input = event.target.closest("[data-profile-room-name]");
    const profile = actions.draft();
    if (!input || !profile) return;
    const room = profile.rooms.find(
      (item) => item.slug === input.dataset.profileRoomName,
    );
    if (!room) return;
    room.name = input.value;
    actions.markDirty();
    renderProfilePreview({
      svg: el.profileMapPreview,
      profile,
      documentRef,
    });
  });
  el.profileRoomList?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-profile-room-remove]");
    if (button) actions.removeRoom(button.dataset.profileRoomRemove);
  });
  el.profileAreaFilter?.addEventListener("change", () => {
    state.profiles.areaFilter = el.profileAreaFilter.value;
    actions.renderEntities();
  });
  el.profileAreaSelectBtn?.addEventListener("click", () =>
    actions.bulkAreaSelection(true),
  );
  el.profileAreaClearBtn?.addEventListener("click", () =>
    actions.bulkAreaSelection(false),
  );
  el.realSensorSearchInput?.addEventListener("input", actions.renderEntities);
  el.haSensorList?.addEventListener("change", (event) => {
    const enabled = event.target.closest("[data-profile-entity]");
    if (enabled) {
      actions.upsertEntity(enabled.dataset.profileEntity, enabled.checked);
      return;
    }
    const room = event.target.closest("[data-profile-entity-room]");
    const type = event.target.closest("[data-profile-entity-type]");
    const entityId =
      room?.dataset.profileEntityRoom || type?.dataset.profileEntityType;
    const profile = actions.draft();
    const assignment = profile?.assignments.find(
      (item) => item.entity_id === entityId,
    );
    if (!assignment) return;
    if (room) assignment.room_slug = room.value;
    if (type) assignment.sensor_type = type.value;
    actions.markDirty();
    renderProfileHeader({ state, el, profile });
  });
  el.profileEdgeAddBtn?.addEventListener("click", () => {
    actions.addEdge(el.profileEdgeFrom.value, el.profileEdgeTo.value);
    actions.render();
  });
  el.profileEdgeList?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-profile-edge-remove]");
    const profile = actions.draft();
    if (!button || !profile) return;
    profile.edges.splice(Number(button.dataset.profileEdgeRemove), 1);
    actions.markDirty("Conexión eliminada");
    actions.render();
  });
  el.profileInferBtn?.addEventListener("click", () =>
    actions.inferLayout().catch(reportError),
  );
  el.profileProposalAcceptBtn?.addEventListener(
    "click",
    actions.acceptProposals,
  );
  el.realSensorApplyBtn?.addEventListener("click", () =>
    actions.saveProfile().catch(reportError),
  );
  el.realSensorResetBtn?.addEventListener("click", actions.resetDraft);
}
