import { buildReplayPayload } from "../replay.js";
import { fetchJson } from "./api.js";
import {
  formatInteger,
  formatTime,
  toPercent,
  trainingStateLabel,
} from "./format.js";

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
  windowRef = window,
}) {
  function renderTraining() {
    const model = state.modelInfo || {};
    const info = model.presence_training_info || {};
    const presence = model.training_status?.presence || {};
    const historical = model.training_status?.historical || {};
    const supervised = model.training_status?.supervised || {};
    const supervisedFilter = model.pet_filter || {};
    const running =
      presence.state === "running" ||
      historical.state === "running" ||
      supervised.state === "running";
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
    el.supervisedTrainState.textContent =
      supervised.state === "running"
        ? "Entrenando"
        : supervised.state === "error"
          ? "Error"
          : supervisedFilter.enabled
            ? supervisedFilter.suppression_enabled
              ? supervisedFilter.source === "bundled"
                ? "Activo incluido"
                : "Activo"
              : "Reglas temporales"
            : "No entrenado";
    el.supervisedHumanRecall.textContent = Number.isFinite(
      Number(supervisedFilter.test?.recall),
    )
      ? toPercent(supervisedFilter.test.recall)
      : "-";
    el.supervisedPetSuppression.textContent = Number.isFinite(
      Number(supervisedFilter.test?.pet_suppression_rate),
    )
      ? toPercent(supervisedFilter.test.pet_suppression_rate)
      : "-";
    el.supervisedDataset.textContent =
      supervisedFilter.manifest_id || "Sin dataset supervisado";
    const candidates = [presence, historical, supervised].filter(
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

  async function fetchTrainingManifests() {
    const payload = await fetchJson("/api/training/manifests", {
      cache: "no-store",
    });
    const selected = el.supervisedManifestSelect.value;
    const items = Array.isArray(payload?.items) ? payload.items : [];
    el.supervisedManifestSelect.replaceChildren();
    for (const item of items) {
      const option = document.createElement("option");
      option.value = item.id;
      option.textContent =
        item.valid === false ? `${item.name} (inválido)` : item.name;
      option.disabled = item.valid === false;
      el.supervisedManifestSelect.append(option);
    }
    if (items.some((item) => item.id === selected && item.valid !== false)) {
      el.supervisedManifestSelect.value = selected;
    }
    el.trainSupervisedBtn.disabled = !el.supervisedManifestSelect.value;
    el.validateSupervisedManifestBtn.disabled =
      !el.supervisedManifestSelect.value;
    return items;
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
      el.trainSupervisedBtn,
      el.validateSupervisedManifestBtn,
      el.rollbackModelBtn,
    ]) {
      if (button) button.disabled = busy;
    }
    if (!busy && !el.supervisedManifestSelect.value) {
      el.trainSupervisedBtn.disabled = true;
      el.validateSupervisedManifestBtn.disabled = true;
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
          label:
            kind === "presence"
              ? "Presencia simulador"
              : kind === "supervised"
                ? "Presencia supervisada"
                : "Historico CSV",
          message:
            kind === "presence"
              ? "entrenando ocupacion desde simulador"
              : kind === "supervised"
                ? "entrenando con confirmaciones de persona y mascota"
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
          : kind === "supervised"
            ? `modelo supervisado activo | ejecución ${result?.run_id || "-"}`
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

  async function validateSupervisedManifest() {
    try {
      const result = await fetchJson("/api/training/manifests/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          manifest_id: el.supervisedManifestSelect.value,
        }),
      });
      setMiniStatus(
        el.supervisedManifestStatus,
        result.valid
          ? `${result.files.length} archivos válidos y listos para entrenar`
          : result.errors.join(" | "),
        !result.valid,
      );
    } catch (error) {
      setMiniStatus(
        el.supervisedManifestStatus,
        String(error.message || error),
        true,
      );
    }
  }

  function trainSupervised() {
    return runTraining(
      "supervised",
      {
        manifest_id: el.supervisedManifestSelect.value,
        epochs: numberFromSelect(el.trainEpochsInput, 5),
        seed: numberFromSelect(el.trainSeedInput, 42),
        min_human_recall: 0.98,
        synthetic_scenarios: numberFromSelect(el.trainScenariosInput, 120),
        synthetic_steps: numberFromSelect(el.trainStepsInput, 60),
        max_samples: numberFromSelect(el.trainMaxSamplesInput, 15000),
      },
      "/api/train_presence_supervised",
    );
  }

  async function rollbackModel() {
    try {
      setTrainingBusy(true);
      await fetchJson("/api/model/rollback", { method: "POST" });
      setMiniStatus(
        el.supervisedManifestStatus,
        "Modelo anterior restaurado y activado",
        false,
      );
      await fetchModelInfo();
      await refreshSnapshot();
    } catch (error) {
      setMiniStatus(
        el.supervisedManifestStatus,
        String(error.message || error),
        true,
      );
    } finally {
      setTrainingBusy(false);
    }
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
    el.validateSupervisedManifestBtn.addEventListener(
      "click",
      validateSupervisedManifest,
    );
    el.trainSupervisedBtn.addEventListener("click", trainSupervised);
    el.rollbackModelBtn.addEventListener("click", rollbackModel);
    el.supervisedManifestSelect.addEventListener("change", () => {
      el.trainSupervisedBtn.disabled = !el.supervisedManifestSelect.value;
      el.validateSupervisedManifestBtn.disabled =
        !el.supervisedManifestSelect.value;
    });
  }

  return {
    fetchModelInfo,
    fetchReplayStatus,
    fetchTrainingManifests,
    registerActions,
    render,
  };
}
