import { buildReplayPayload } from "../replay.js";
import { fetchJson } from "./api.js";
import {
  formatInteger,
  formatTime,
  toPercent,
  trainingStateLabel,
} from "./format.js";
import { adjacencyToText } from "./map.js";
import { collectRooms } from "./rooms.js";

export function numberFromSelect(select, fallback) {
  const value = Number(select?.value);
  return Number.isFinite(value) ? value : fallback;
}

export function createReplayTrainingController({
  state,
  el,
  setMiniStatus,
  renderAll,
  refreshSnapshot,
  documentRef = document,
  windowRef = window,
}) {
  function renderTraining() {
    const model = state.modelInfo || {};
    const info = model.presence_training_info || {};
    const presence = model.training_status?.presence || {};
    const historical = model.training_status?.historical || {};
    const running =
      presence.state === "running" || historical.state === "running";
    el.presenceTrainState.textContent =
      presence.state === "running"
        ? "Entrenando"
        : presence.state === "error"
          ? "Error"
          : info.enabled
            ? "Activo"
            : "No entrenado";
    el.presenceTrainSamples.textContent = formatInteger(info.samples || 0);
    el.presenceTrainCount.textContent = toPercent(info.count_accuracy);
    el.presenceTrainRooms.textContent = formatInteger(
      info.rooms_total || model.presence_rooms?.length || 0,
    );
    const candidates = [presence, historical].filter(
      (item) => Object.keys(item).length,
    );
    const active =
      candidates.find((item) => item.state === "running") ||
      candidates.sort(
        (a, b) =>
          (Date.parse(b.finished_at || b.started_at || "") || 0) -
          (Date.parse(a.finished_at || a.started_at || "") || 0),
      )[0] ||
      {};
    el.trainingJobStatus.textContent =
      `${active.label || "Historico CSV"} | ` +
      `${trainingStateLabel(active.state)} | ` +
      `${active.message || "sin entrenamiento activo"}`;
    el.trainingJobStatus.className = running ? "is-running" : "";
    const updated =
      active.finished_at ||
      active.started_at ||
      presence.finished_at ||
      historical.finished_at;
    el.trainingUpdatedAt.textContent = updated ? formatTime(updated) : "-";
    const csv = presence.result_summary || {};
    if (csv.simulated_csv_url) {
      el.simulatedCsvRow.hidden = false;
      el.simulatedCsvLink.href = csv.simulated_csv_url;
      el.simulatedCsvMeta.textContent =
        `${formatInteger(csv.simulated_csv_rows || 0)} filas | ` +
        csv.simulated_csv_url;
    }
  }

  function renderReplay() {
    const replay = state.replay || {};
    const mode = replay.mode || "listen";
    const running = !!replay.running;
    const paused = !!replay.paused;
    el.modeListenBtn.disabled = mode === "listen" && !running;
    if (el.modeReplayBtn) el.modeReplayBtn.disabled = mode === "replay";
    el.modeSummary.textContent =
      mode === "replay"
        ? "replay historico"
        : mode === "simulator"
          ? "simulador"
          : "sensores reales";
    el.modeListenBtn.classList.toggle("active", mode === "listen");
    el.modeReplayBtn?.classList.toggle("active", mode === "replay");
    el.replaySummary.textContent =
      running && paused
        ? "pausado"
        : running
          ? "en ejecucion"
          : mode === "listen"
            ? "escucha sensores"
            : "idle";
    el.replayProgress.textContent =
      `progreso: ${(Number(replay.progress || 0) * 100).toFixed(1)}% | ` +
      `${replay.processed_events || 0}/${replay.total_events || 0} | ` +
      `steps cola: ${replay.step_budget || 0}`;
    if (replay.last_error) {
      setMiniStatus(
        el.replayStatus,
        `replay detenido: ${replay.last_error}`,
        true,
      );
    }
    el.replayStepBtn.disabled = !(running && paused);
  }

  function render() {
    renderTraining();
    renderReplay();
  }

  async function fetchModelInfo() {
    state.modelInfo =
      (await fetchJson("/api/model_info", { cache: "no-store" })) || {};
    renderTraining();
    return state.modelInfo;
  }

  async function fetchReplayStatus() {
    const payload = await fetchJson("/api/replay_status", {
      cache: "no-store",
    });
    state.replay = { ...state.replay, ...(payload || {}) };
    renderReplay();
    return state.replay;
  }

  async function setInputMode(mode) {
    try {
      const payload = await fetchJson("/api/input_mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode }),
      });
      state.replay = {
        ...state.replay,
        mode: payload?.mode || mode,
        running:
          typeof payload?.replay_running === "boolean"
            ? payload.replay_running
            : state.replay.running,
      };
      setMiniStatus(el.modeStatus, `modo activo: ${state.replay.mode}`, false);
      await fetchReplayStatus();
    } catch (error) {
      setMiniStatus(el.modeStatus, String(error.message || error), true);
    }
  }

  function updateTemplateHint() {
    const template = state.scenarioTemplates[el.scenarioTemplate.value];
    el.templateHint.textContent = template
      ? `${template.description || "template"} | ${template.edges?.length || 0} aristas`
      : "sin metadata";
  }

  function populateTemplates(templates) {
    state.scenarioTemplates = templates || {};
    const keys = Object.keys(state.scenarioTemplates);
    const ordered = keys.includes("real_home")
      ? ["real_home", ...keys.filter((key) => key !== "real_home")]
      : keys;
    const previous = el.scenarioTemplate.value;
    el.scenarioTemplate.innerHTML = "";
    for (const key of ordered.length ? ordered : ["real_home"]) {
      const option = documentRef.createElement("option");
      option.value = key;
      option.textContent = key;
      el.scenarioTemplate.appendChild(option);
    }
    el.scenarioTemplate.value = ordered.includes(previous)
      ? previous
      : ordered.includes("real_home")
        ? "real_home"
        : el.scenarioTemplate.value;
    updateTemplateHint();
  }

  async function fetchScenarioTemplates() {
    const payload = await fetchJson("/api/scenario_templates", {
      cache: "no-store",
    });
    populateTemplates(payload?.templates || {});
  }

  async function applyScenarioTemplate() {
    if (state.applyingTemplate) return;
    const key = el.scenarioTemplate.value || "real_home";
    const template = state.scenarioTemplates[key];
    if (!template?.adjacency) {
      updateTemplateHint();
      return;
    }
    try {
      state.applyingTemplate = true;
      el.layoutText.value = adjacencyToText(template.adjacency);
      state.layoutTextDirty = true;
      const response = await fetchJson("/api/layout_reference", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          adjacency: template.adjacency,
          rooms: collectRooms(state),
        }),
      });
      if (response?.layout_reference) {
        state.reference = {
          ...state.reference,
          ...response.layout_reference,
        };
      }
      if (response?.metrics) state.metrics = response.metrics;
      state.layoutTextDirty = false;
      updateTemplateHint();
      setMiniStatus(el.layoutStatus, `plantilla aplicada: ${key}`, false);
      renderAll();
    } catch (error) {
      setMiniStatus(el.layoutStatus, String(error.message || error), true);
    } finally {
      state.applyingTemplate = false;
    }
  }

  async function startReplay() {
    try {
      await fetchJson("/api/replay_csv", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildReplayPayload(el)),
      });
      setMiniStatus(el.replayStatus, "replay iniciado", false);
      setMiniStatus(el.modeStatus, "modo activo: replay", false);
      await fetchReplayStatus();
    } catch (error) {
      setMiniStatus(el.replayStatus, String(error.message || error), true);
    }
  }

  async function replayControl(action) {
    try {
      const status = await fetchJson("/api/replay_control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      });
      state.replay = { ...state.replay, ...(status || {}) };
      setMiniStatus(el.replayStatus, `accion aplicada: ${action}`, false);
      renderReplay();
    } catch (error) {
      setMiniStatus(el.replayStatus, String(error.message || error), true);
    }
  }

  function setTrainingBusy(busy) {
    for (const button of [
      el.trainPresenceAutoBtn,
      el.trainPresenceManualBtn,
      el.trainHistoricalBtn,
    ]) {
      if (button) button.disabled = busy;
    }
  }

  function startPolling() {
    stopPolling();
    state.trainingPollTimer = windowRef.setInterval(
      () => fetchModelInfo().catch(() => {}),
      1500,
    );
  }

  function stopPolling() {
    if (!state.trainingPollTimer) return;
    windowRef.clearInterval(state.trainingPollTimer);
    state.trainingPollTimer = null;
  }

  async function runTraining(kind, payload, endpoint) {
    try {
      setTrainingBusy(true);
      state.modelInfo.training_status = {
        ...(state.modelInfo.training_status || {}),
        [kind]: {
          state: "running",
          label: kind === "presence" ? "Presencia simulador" : "Historico CSV",
          message:
            kind === "presence"
              ? "entrenando ocupacion desde simulador"
              : "entrenando mapa desde CSV historico",
          started_at: new Date().toISOString(),
        },
      };
      renderTraining();
      startPolling();
      setMiniStatus(el.trainStatus, "entrenamiento en curso...", false);
      const result = await fetchJson(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      setMiniStatus(
        el.trainStatus,
        kind === "presence"
          ? `presencia entrenada | muestras ${result?.training_info?.samples || 0}`
          : `histórico entrenado | eventos ${result?.training_info?.events_total || 0}`,
        false,
      );
      await fetchModelInfo();
      await refreshSnapshot();
    } catch (error) {
      setMiniStatus(el.trainStatus, String(error.message || error), true);
    } finally {
      stopPolling();
      setTrainingBusy(false);
    }
  }

  function trainPresence(manual) {
    return runTraining(
      "presence",
      manual
        ? {
            template: "real_home",
            scenarios: numberFromSelect(el.trainScenariosInput, 240),
            steps_per_scenario: numberFromSelect(el.trainStepsInput, 90),
            max_people: numberFromSelect(el.trainMaxPeopleInput, 2),
            event_interval_seconds: 4,
            movement_probability: 0.7,
            occupancy_refresh_probability: 0.25,
            epochs: numberFromSelect(el.trainEpochsInput, 5),
            max_samples: numberFromSelect(el.trainMaxSamplesInput, 15000),
            seed: numberFromSelect(el.trainSeedInput, 42),
          }
        : {
            template: "real_home",
            scenarios: 240,
            steps_per_scenario: 90,
            max_people: 2,
            event_interval_seconds: 4,
            movement_probability: 0.7,
            occupancy_refresh_probability: 0.25,
            epochs: 5,
            max_samples: 15000,
            seed: 42,
          },
      "/api/train_presence_simulator",
    );
  }

  function trainHistorical() {
    return runTraining(
      "historical",
      {
        csv_path: el.csvPath.value,
        debounce_seconds: numberFromSelect(el.debounceInput, 1),
        include_all_state_transitions: true,
        min_gap_seconds: 0,
        max_gap_seconds: 900,
        epochs: numberFromSelect(el.trainEpochsInput, 5),
        max_samples: numberFromSelect(el.trainMaxSamplesInput, 15000),
        degree_limit: 4,
        use_ollama_validation: false,
      },
      "/api/train_model_full",
    );
  }

  function registerActions() {
    el.modeListenBtn.addEventListener("click", () => setInputMode("listen"));
    el.modeReplayBtn?.addEventListener("click", () => setInputMode("replay"));
    el.replayNewBtn.addEventListener("click", startReplay);
    el.replayPauseBtn.addEventListener("click", () => replayControl("pause"));
    el.replayResumeBtn.addEventListener("click", () => replayControl("start"));
    el.replayStepBtn.addEventListener("click", () => replayControl("step"));
    el.replayResetBtn.addEventListener("click", () => replayControl("reset"));
    el.trainPresenceAutoBtn.addEventListener("click", () =>
      trainPresence(false),
    );
    el.trainPresenceManualBtn.addEventListener("click", () =>
      trainPresence(true),
    );
    el.trainHistoricalBtn.addEventListener("click", trainHistorical);
    el.scenarioTemplate.addEventListener("change", applyScenarioTemplate);
  }

  return {
    fetchModelInfo,
    fetchReplayStatus,
    fetchScenarioTemplates,
    registerActions,
    render,
  };
}
