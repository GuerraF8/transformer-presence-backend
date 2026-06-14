import { bindPanelDom } from "./dom.js";
import { createDashboardController } from "./dashboard.js";
import { createHistoryController } from "./history.js";
import { createHomeAssistantController } from "./home-assistant.js";
import { applyDevVisibility, createDialogController } from "./modal.js";
import { createProfilesController } from "./profiles.js";
import { createRealtimeController } from "./realtime.js";
import { createReplayTrainingController } from "./replay-training.js";
import { createPanelState } from "./state.js";
import {
  appendBadgeCell,
  appendCell,
  setMiniStatus,
  setTopStatus,
} from "./ui.js";

const query = new URLSearchParams(window.location.search);
const embeddedMode = query.get("embedded") === "1";
const devMode = !embeddedMode || query.get("dev") === "1";
const state = createPanelState();
const el = bindPanelDom();

function renderAll() {
  dashboard.render();
  replayTraining.render();
  profiles.render();
  homeAssistant.render();
  history.render();
}

const history = createHistoryController({
  state,
  el,
  setMiniStatus,
  appendCell,
  appendBadgeCell,
});
const profiles = createProfilesController({
  state,
  el,
  setMiniStatus,
  appendBadgeCell,
  renderAll,
});
const homeAssistant = createHomeAssistantController({
  state,
  el,
  appendBadgeCell,
  setMiniStatus,
  renderAll,
  sensorSelectionController: profiles,
});
const dashboard = createDashboardController({
  state,
  el,
  renderAll,
});
let realtime;
const replayTraining = createReplayTrainingController({
  state,
  el,
  setMiniStatus,
  renderAll,
  refreshSnapshot: () => realtime.fetchSnapshot(),
});
realtime = createRealtimeController({
  state,
  el,
  renderAll,
  setRealSensorConfig: homeAssistant.setConfig,
  scheduleHistoryRefresh: history.scheduleRefresh,
  setTopStatus: (text, isError) =>
    setTopStatus(el.topStatus, text, isError),
});
const configDialog = createDialogController({
  dialog: el.configDialog,
  openButton: el.configOpenBtn,
  closeButton: el.configCloseBtn,
});

function registerActions() {
  configDialog.register();
  profiles.registerActions();
  homeAssistant.registerActions();
  replayTraining.registerActions();
  dashboard.registerActions();
  history.registerActions();
}

async function load(task, onError) {
  try {
    return await task();
  } catch (error) {
    onError(error);
    return null;
  }
}

async function init() {
  applyDevVisibility({ embeddedMode, devMode });
  registerActions();
  dashboard.setMapTab("fixed");
  el.apiBaseUrl.textContent = window.location.origin;
  history.initializeFilters();

  await load(
    async () => {
      await realtime.fetchSnapshot();
      setTopStatus(el.topStatus, "snapshot inicial cargado", false);
    },
    (error) =>
      setTopStatus(el.topStatus, String(error.message || error), true),
  );
  await load(replayTraining.fetchModelInfo, (error) =>
    setMiniStatus(
      el.trainStatus,
      `estado modelo: ${String(error.message || error)}`,
      true,
    ),
  );
  await load(dashboard.fetchPresenceFilter, (error) =>
    setMiniStatus(
      el.petFilterStatus,
      `filtro: ${String(error.message || error)}`,
      true,
    ),
  );
  await load(homeAssistant.fetchCatalog, (error) =>
    setMiniStatus(
      el.haSensorStatus,
      `catálogo HA: ${String(error.message || error)}`,
      true,
    ),
  );
  await load(() => profiles.fetchProfiles({ preserveDraft: false }), (error) =>
    setMiniStatus(
      el.profileStatus,
      `perfiles: ${String(error.message || error)}`,
      true,
    ),
  );
  await load(homeAssistant.fetchActions, (error) =>
    setMiniStatus(
      el.haDiagnosticStatus,
      `acciones HA: ${String(error.message || error)}`,
      true,
    ),
  );
  await load(dashboard.fetchEvaluationMetrics, (error) =>
    setMiniStatus(
      el.layoutStatus,
      `metricas: ${String(error.message || error)}`,
      true,
    ),
  );
  await load(replayTraining.fetchReplayStatus, () => {});
  await load(replayTraining.fetchScenarioTemplates, (error) =>
    setMiniStatus(
      el.replayStatus,
      `plantillas: ${String(error.message || error)}`,
      true,
    ),
  );
  await load(
    () => Promise.all([history.fetchConfig(), history.fetch()]),
    (error) =>
      setMiniStatus(
        el.historyChartStatus,
        `historial: ${String(error.message || error)}`,
        true,
      ),
  );

  realtime.connect();
  window.setInterval(() => {
    dashboard.fetchEvaluationMetrics().catch(() => {});
    dashboard.fetchPresenceFilter().catch(() => {});
    replayTraining.fetchReplayStatus().catch(() => {});
    replayTraining.fetchModelInfo().catch(() => {});
    homeAssistant.fetchCatalog().catch(() => {});
    homeAssistant.fetchActions().catch(() => {});
    profiles.fetchProfiles().catch(() => {});
  }, 5000);
}

init();
