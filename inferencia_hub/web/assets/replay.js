export function buildReplayPayload(el) {
  return {
    csv_path: el.csvPath.value.trim(),
    speed_events_per_second: Number(el.speedInput.value || 30),
    debounce_seconds: Number(el.debounceInput.value || 1),
    include_all_state_transitions: true,
    max_events: Number(el.maxEventsInput.value || 0),
    use_scenario_layout: !!el.useScenarioLayout.checked,
    template: el.scenarioTemplate.value || "real_home",
    layout_edges: [],
    room_mapping: {},
    step_seconds: Number(el.stepSecondsInput.value || 3),
  };
}
