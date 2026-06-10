"""Controladores utilizados por las rutas FastAPI."""

from .lifecycle import data_dir, model_state_dir, training_status_path, real_sensor_config_path, history_sensor_name, persist_history_event, resolve_training_csv, store_training_artifact, persist_real_sensor_config, load_real_sensor_config, activate_listen_mode, persist_training_status, load_training_status, persist_model_state, export_simulated_sensor_csv, mark_training_status, startup_train_model, shutdown_history_store
from .presence import health, catalog_has_entity, ingest_event, get_sim_data, get_presence_filter, set_presence_filter, model_info, get_input_mode, set_input_mode, reset_state, presencia_socket
from .history import normalize_history_timestamp, get_history_config, update_history_config, get_history_events, get_history_presence, purge_history
from .home_assistant import get_ha_entities, update_ha_entities, get_real_sensor_config, set_real_sensor_config, list_ha_actions, update_ha_integration_status, request_ha_action, claim_ha_action, complete_ha_action
from .layout import scenario_templates, get_layout_reference, set_layout_reference, evaluation_metrics
from .training import download_training_export, train_model, train_model_full, train_presence_simulator
from .replay import replay_csv, replay_control, replay_status

HANDLERS = {
    "health": health,
    "catalog_has_entity": catalog_has_entity,
    "ingest_event": ingest_event,
    "get_sim_data": get_sim_data,
    "get_presence_filter": get_presence_filter,
    "set_presence_filter": set_presence_filter,
    "model_info": model_info,
    "get_input_mode": get_input_mode,
    "set_input_mode": set_input_mode,
    "reset_state": reset_state,
    "presencia_socket": presencia_socket,
    "normalize_history_timestamp": normalize_history_timestamp,
    "get_history_config": get_history_config,
    "update_history_config": update_history_config,
    "get_history_events": get_history_events,
    "get_history_presence": get_history_presence,
    "purge_history": purge_history,
    "get_ha_entities": get_ha_entities,
    "update_ha_entities": update_ha_entities,
    "get_real_sensor_config": get_real_sensor_config,
    "set_real_sensor_config": set_real_sensor_config,
    "list_ha_actions": list_ha_actions,
    "update_ha_integration_status": update_ha_integration_status,
    "request_ha_action": request_ha_action,
    "claim_ha_action": claim_ha_action,
    "complete_ha_action": complete_ha_action,
    "scenario_templates": scenario_templates,
    "get_layout_reference": get_layout_reference,
    "set_layout_reference": set_layout_reference,
    "evaluation_metrics": evaluation_metrics,
    "download_training_export": download_training_export,
    "train_model": train_model,
    "train_model_full": train_model_full,
    "train_presence_simulator": train_presence_simulator,
    "replay_csv": replay_csv,
    "replay_control": replay_control,
    "replay_status": replay_status,
}

__all__ = ['data_dir', 'model_state_dir', 'training_status_path', 'real_sensor_config_path', 'history_sensor_name', 'persist_history_event', 'resolve_training_csv', 'store_training_artifact', 'persist_real_sensor_config', 'load_real_sensor_config', 'activate_listen_mode', 'persist_training_status', 'load_training_status', 'persist_model_state', 'export_simulated_sensor_csv', 'mark_training_status', 'startup_train_model', 'shutdown_history_store', 'health', 'catalog_has_entity', 'ingest_event', 'get_sim_data', 'get_presence_filter', 'set_presence_filter', 'model_info', 'get_input_mode', 'set_input_mode', 'reset_state', 'presencia_socket', 'normalize_history_timestamp', 'get_history_config', 'update_history_config', 'get_history_events', 'get_history_presence', 'purge_history', 'get_ha_entities', 'update_ha_entities', 'get_real_sensor_config', 'set_real_sensor_config', 'list_ha_actions', 'update_ha_integration_status', 'request_ha_action', 'claim_ha_action', 'complete_ha_action', 'scenario_templates', 'get_layout_reference', 'set_layout_reference', 'evaluation_metrics', 'download_training_export', 'train_model', 'train_model_full', 'train_presence_simulator', 'replay_csv', 'replay_control', 'replay_status'] + ['HANDLERS']
