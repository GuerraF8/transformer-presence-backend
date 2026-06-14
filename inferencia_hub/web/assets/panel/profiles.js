import { roomLabel } from "./format.js";
import {
  activateProfileApi,
  createProfileApi,
  deleteProfileApi,
  fetchProfilesApi,
  inferProfileLayoutApi,
  updateProfileApi,
} from "./profile-api.js";
import {
  cloneProfile,
  profileSelectionChanged,
  profileUpdatePayload,
  validProfileRoomSelection,
} from "./profile-draft.js";
import {
  renderProfileEntities,
  renderProfileEntityFilters,
} from "./profile-entities.js";
import { registerProfileEvents } from "./profile-events.js";
import {
  appendProfileEdge,
  appendProfileRoom,
  removeProfileRoom,
  setProfileEntity,
} from "./profile-mutations.js";
import { renderProfilePreview } from "./profile-preview.js";
import {
  renderProfileAreas,
  renderProfileEdges,
  renderProfileHeader,
  renderProfileProposals,
  renderProfileRooms,
} from "./profile-view.js";

export function createProfilesController({
  state,
  el,
  setMiniStatus,
  appendBadgeCell,
  renderAll,
  documentRef = document,
  windowRef = window,
}) {
  function catalogAreas() {
    return Array.isArray(state.haEntityCatalog?.areas)
      ? state.haEntityCatalog.areas
      : [];
  }

  function catalogEntities() {
    return Array.isArray(state.haEntityCatalog?.entities)
      ? state.haEntityCatalog.entities
      : [];
  }

  function draft() {
    return state.profiles.draft;
  }

  function setStatus(message, isError = false) {
    setMiniStatus(el.profileStatus, message, isError);
  }

  function selectDraft(profile, { resetConnections = true } = {}) {
    state.profiles.selectedId = profile?.id || null;
    state.profiles.draft = cloneProfile(profile);
    state.profiles.dirty = false;
    state.profiles.proposals = [];
    if (resetConnections) {
      state.profiles.edgeFrom = "";
      state.profiles.edgeTo = "";
    }
  }

  function markDirty(message = "Hay cambios pendientes en el perfil") {
    state.profiles.dirty = true;
    setStatus(message, false);
  }

  function profileById(profileId) {
    return state.profiles.items.find((profile) => profile.id === profileId);
  }

  function roomOptions(select, selected = "") {
    select.add(new Option("Sin asignar", ""));
    for (const room of draft()?.rooms || []) {
      select.add(new Option(room.name || roomLabel(room.slug), room.slug));
    }
    select.value = selected;
  }

  function renderAreas() {
    const current = draft();
    if (!current) return;
    renderProfileAreas({
      el,
      profile: current,
      areas: catalogAreas(),
      documentRef,
    });
  }

  function renderRooms() {
    const current = draft();
    if (!current) return;
    renderProfileRooms({ el, profile: current, documentRef });
  }

  function renderEntityFilters() {
    renderProfileEntityFilters({
      state,
      el,
      areas: catalogAreas(),
    });
  }

  function renderEntities() {
    renderProfileEntities({
      state,
      el,
      profile: draft(),
      entities: catalogEntities(),
      areas: catalogAreas(),
      appendBadgeCell,
      roomOptions,
      documentRef,
    });
  }

  function renderEdges() {
    const current = draft();
    if (!current) return;
    state.profiles.edgeFrom = validProfileRoomSelection(
      state.profiles.edgeFrom,
      current.rooms,
    );
    state.profiles.edgeTo = validProfileRoomSelection(
      state.profiles.edgeTo,
      current.rooms,
    );
    renderProfileEdges({
      el,
      profile: current,
      selectedFrom: state.profiles.edgeFrom,
      selectedTo: state.profiles.edgeTo,
      documentRef,
    });
  }

  function renderProposals() {
    renderProfileProposals({
      el,
      proposals: state.profiles.proposals,
      documentRef,
    });
  }

  function render() {
    renderProfileHeader({ state, el, profile: draft() });
    renderEntityFilters();
    if (!draft()) {
      renderEntities();
      return;
    }
    renderAreas();
    renderRooms();
    renderEntities();
    renderEdges();
    renderProposals();
    renderProfilePreview({
      svg: el.profileMapPreview,
      profile: draft(),
      documentRef,
    });
  }

  async function fetchProfiles({ preserveDraft = true } = {}) {
    const payload = await fetchProfilesApi();
    state.profiles.items = payload.profiles || [];
    state.profiles.activeId = payload.active_profile_id || null;
    state.roomLabels = Object.fromEntries(
      (payload.active_profile?.rooms || []).map((room) => [
        room.slug,
        room.name || room.slug,
      ]),
    );
    if (!preserveDraft || !state.profiles.dirty) {
      const previousSelectedId = state.profiles.selectedId;
      const selected =
        profileById(state.profiles.selectedId) ||
        payload.active_profile ||
        state.profiles.items[0] ||
        null;
      selectDraft(selected, {
        resetConnections: profileSelectionChanged(
          previousSelectedId,
          selected,
        ),
      });
    }
    render();
    return payload;
  }

  async function createProfile(source) {
    const defaultNames = {
      real_home: "Real home",
      manual: "Nuevo perfil",
      detected: "Perfil detectado",
    };
    const payload = {
      name: defaultNames[source],
      source,
      area_ids: source === "detected"
        ? catalogAreas().map((area) => area.area_id)
        : [],
      entity_ids: [],
    };
    const profile = await createProfileApi(payload);
    await fetchProfiles({ preserveDraft: false });
    selectDraft(profile);
    renderAll();
    setStatus(`Perfil ${profile.name} creado`, false);
  }

  async function saveProfile() {
    const current = draft();
    if (!current) return null;
    const saved = await updateProfileApi(
      current.id,
      profileUpdatePayload(current),
    );
    state.profiles.dirty = false;
    await fetchProfiles({ preserveDraft: false });
    selectDraft(saved);
    renderAll();
    setStatus("Perfil guardado", false);
    return saved;
  }

  async function activateProfile() {
    let current = draft();
    if (!current) return;
    if (state.profiles.dirty) current = await saveProfile();
    const active = await activateProfileApi(current.id);
    await fetchProfiles({ preserveDraft: false });
    selectDraft(active);
    renderAll();
    setStatus(`Perfil ${active.name} activado`, false);
  }

  async function deleteProfile() {
    const current = draft();
    if (!current) return;
    if (!windowRef.confirm(`Eliminar el perfil "${current.name}"?`)) return;
    await deleteProfileApi(current.id);
    await fetchProfiles({ preserveDraft: false });
    renderAll();
    setStatus("Perfil eliminado", false);
  }

  function addRoom(name, area = null) {
    const current = draft();
    if (!current) return null;
    const slug = appendProfileRoom(current, name, area);
    if (!slug) return null;
    markDirty(`Habitación ${name} añadida`);
    return slug;
  }

  function toggleArea(areaId, checked) {
    const current = draft();
    const area = catalogAreas().find((item) => item.area_id === areaId);
    if (!current || !area) return;
    const linked = current.areas.find((item) => item.area_id === areaId);
    if (checked && !linked) {
      addRoom(area.name, area);
    } else if (!checked && linked) {
      removeRoom(linked.room_slug);
    }
    render();
  }

  function removeRoom(slug) {
    const current = draft();
    if (!current) return;
    removeProfileRoom(current, slug);
    markDirty(`Habitación ${slug} eliminada`);
    render();
  }

  function upsertEntity(entityId, enabled, renderAfter = true) {
    const current = draft();
    const entity = catalogEntities().find((item) => item.entity_id === entityId);
    if (!current || !entity) return;
    setProfileEntity(current, entity, enabled);
    markDirty(
      enabled
        ? `Entidad ${entityId} seleccionada`
        : `Entidad ${entityId} retirada`,
    );
    if (renderAfter) render();
  }

  function bulkAreaSelection(enabled) {
    const areaId = state.profiles.areaFilter;
    if (!areaId || areaId === "__none__") {
      setStatus("Selecciona un área concreta para aplicar esta acción", true);
      return;
    }
    for (const entity of catalogEntities().filter(
      (item) => item.area_id === areaId,
    )) {
      upsertEntity(entity.entity_id, enabled, false);
    }
    render();
  }

  function addEdge(left, right) {
    const current = draft();
    if (!current) return;
    if (appendProfileEdge(current, left, right)) {
      markDirty("Conexión añadida");
    }
  }

  async function inferLayout() {
    const current = draft();
    if (!current) return;
    if (state.profiles.dirty) await saveProfile();
    const result = await inferProfileLayoutApi(
      state.profiles.selectedId,
      {
        min_support: Number(el.profileMinSupport.value || 2),
        max_gap_seconds: Number(el.profileMaxGap.value || 600),
      },
    );
    state.profiles.proposals = result.proposals || [];
    renderProposals();
    setStatus(
      `${state.profiles.proposals.length} propuestas históricas encontradas`,
      false,
    );
  }

  function acceptProposals() {
    for (const input of el.profileProposalList.querySelectorAll(
      "[data-profile-proposal]:checked",
    )) {
      const [left, right] = input.dataset.profileProposal.split("\u0000");
      addEdge(left, right);
    }
    state.profiles.proposals = [];
    render();
  }

  function resetDraft() {
    selectDraft(profileById(state.profiles.selectedId) || null);
    setStatus("Cambios descartados", false);
    renderAll();
  }

  function registerActions() {
    registerProfileEvents({
      state,
      el,
      documentRef,
      windowRef,
      actions: {
        acceptProposals,
        activateProfile,
        addEdge,
        addRoom,
        bulkAreaSelection,
        createProfile,
        deleteProfile,
        draft,
        inferLayout,
        markDirty,
        profileById,
        removeRoom,
        render,
        renderAll,
        renderEntities,
        resetDraft,
        saveProfile,
        selectDraft,
        setStatus,
        toggleArea,
        upsertEntity,
      },
    });
  }

  return {
    fetchProfiles,
    registerActions,
    render,
    renderEntities,
  };
}
