# inferencia_hub

Servicio externo para inferencia de presencia + visualizacion en vivo.

## Que incluye

- API de ingesta de eventos: `POST /api/events`
- Entrenamiento de mapa desde historial CSV: `POST /api/train_model`
- Inspeccion de modelo activo: `GET /api/model_info`
- Plantillas de escenarios para replay: `GET /api/scenario_templates`
- Layout de referencia editable (mapa real): `GET/POST /api/layout_reference`
- Metricas de evaluacion + trazas recientes: `GET /api/evaluation_metrics`
- Simulacion desde CSV: `POST /api/replay_csv`
- Control de replay (start/pause/reset): `POST /api/replay_control`
- Estado de replay: `GET /api/replay_status`
- Reset de estado: `POST /api/reset`
- Snapshot compatible con `sim_data.json`: `GET /api/sim_data`
- Configuracion del historial SQLite: `GET/PUT /api/history/config`
- Consulta paginada y filtrada: `GET /api/history/events`
- Serie de presencia y personas: `GET /api/history/presence`
- Borrado total confirmado: `POST /api/history/purge`
- WebSocket para panel live: `ws://<host>:8080/presencia`
- Web UI servida desde `/` (usa `inferencia_hub/web/index.html`)

## Inferencia inteligente (IA)

`inferencia_hub` aprende la topologia del hogar directamente desde `history-1mes*.csv`:

- Construye transiciones dirigidas por secuencia temporal de sensores (sin hardcodear adyacencias).
- Entrena un `TimeSeriesTransformer` (Hugging Face) para probabilidad de siguiente habitacion.
- Combina Transformer + transiciones historicas (modelo hibrido) para generar aristas validas.
- Mantiene un mapa real de referencia editable en texto de adyacencia y compara contra el mapa inferido.
- Detecta transiciones no adyacentes y clasifica causa probable (multiples personas, mascota/ruido o error).
- Estima presencia por habitacion con un filtro probabilistico (creencia + evidencia de sensores).
- Estima numero de personas presentes a partir de habitaciones activas y conectividad.

Opcionalmente puede ejecutar una validacion semantica del mapa con Ollama (`qwen2.5:0.5b-instruct` por defecto).

## Formato de evento de ingesta

```json
{
  "entity_id": "binary_sensor.kitchen_sensor_motion",
  "state": "on",
  "sensor_type": "motion",
  "room": "kitchen",
  "timestamp": "2026-04-15T18:30:00Z",
  "source": "home_assistant"
}
```

Campos `sensor_type`, `room` y `timestamp` son opcionales; el servicio los infiere cuando faltan.

## Historial persistente

La version 0.3.0 guarda en `/app/data/presence_history.sqlite3` el evento original,
la inferencia resultante y el modo de entrada. SQLite usa WAL, migraciones con
`PRAGMA user_version`, indices de consulta y limpieza por retencion.

La vista inicial del panel consulta las ultimas 24 horas del modo `listen`, con
50 filas por pagina. Los filtros por sensor, tipo, habitacion, modo y fechas se
aplican tanto a la tabla como al grafico escalonado.

Variables iniciales:

```env
HISTORY_DB_PATH=/app/data/presence_history.sqlite3
HISTORY_ENABLED=1
HISTORY_RETENTION_DAYS=365
HISTORY_PERSIST_MODES=listen,replay,simulator
```

Los valores editables quedan guardados en SQLite. Cambiar `HISTORY_DB_PATH`
requiere reiniciar. `POST /api/reset` conserva el historial.

## Entrenar modelo desde historial

### Opción 1: Entrenamiento Estándar (Rápido)

```bash
curl -X POST http://localhost:8080/api/train_model \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "/data/history-1mes_sorted.csv",
    "debounce_seconds": 2,
    "min_gap_seconds": 2,
    "max_gap_seconds": 600,
    "epochs": 3,
    "max_samples": 7000,
    "degree_limit": 3,
    "use_ollama_validation": true
  }'
```

### Opción 2: Entrenamiento Completo (Recomendado para Historial Largo)

Procesa 10x más transiciones sin descartes excesivos:

```bash
curl -X POST http://localhost:8080/api/train_model_full \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "/data/history-1mes_sorted.csv",
    "debounce_seconds": 1,
    "min_gap_seconds": 0,
    "max_gap_seconds": 900,
    "epochs": 5,
    "max_samples": 15000,
    "degree_limit": 4,
    "use_ollama_validation": false
  }'
```

El endpoint devuelve habitaciones, aristas inferidas y metadatos de entrenamiento completo.

## Levantar en local

Desde la raiz del repo:

```bash
cp .env.example .env
docker compose up -d
```

Abrir:

- Panel: `http://localhost:8081/`
- Health: `http://localhost:8081/api/health`

## Simular desde CSV

```bash
curl -X POST http://localhost:8080/api/replay_csv \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "/data/history-1mes.csv",
    "speed_events_per_second": 20,
    "debounce_seconds": 2,
    "max_events": 1200,
    "use_scenario_layout": true,
    "template": "lineal",
    "room_mapping": {"living_room": "sala"},
    "layout_edges": [["sala", "pasillo"], ["pasillo", "cocina"]],
    "step_seconds": 3
  }'
```

## Metricas y validacion

Obtener metricas acumuladas (calidad de mapa, no adyacencias, latencia y personas):

```bash
curl http://localhost:8080/api/evaluation_metrics
```

Obtener o actualizar mapa real de referencia desde texto de adyacencia:

```bash
curl -X POST http://localhost:8080/api/layout_reference \
  -H "Content-Type: application/json" \
  -d '{
    "adjacency_text": "bedroom: sittingroom\nsittingroom: bedroom, entertainment_room\nentertainment_room: sittingroom, foyer\nfoyer: entertainment_room, kitchen, living\nkitchen: foyer\nliving: foyer"
  }'
```

Controlar replay desde API (pause/start/reset):

```bash
curl -X POST http://localhost:8080/api/replay_control \
  -H "Content-Type: application/json" \
  -d '{
    "action": "pause"
  }'
```
