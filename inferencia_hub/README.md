# inferencia_hub

Servicio externo para inferencia de presencia + visualizacion en vivo.

## Que incluye

- API de ingesta de eventos: `POST /api/events`
- Entrenamiento de mapa desde historial CSV: `POST /api/train_model`
- Entrenamiento supervisado de presencia: `POST /api/train_presence_supervised`
- Manifiestos y reportes: `GET /api/training/manifests` y
  `GET /api/training/reports/{run_id}`
- Restauracion del modelo anterior: `POST /api/model/rollback`
- Inspeccion de modelo activo: `GET /api/model_info`
- Plantillas de escenarios para replay: `GET /api/scenario_templates`
- Layout de referencia editable (mapa real): `GET/POST /api/layout_reference`
- Perfiles editables: `GET/POST /api/profiles`
- Activacion y propuestas historicas:
  `POST /api/profiles/{profile_id}/activate` y
  `POST /api/profiles/{profile_id}/infer-layout`
- Metricas de evaluacion + trazas recientes: `GET /api/evaluation_metrics`
- Simulacion desde CSV: `POST /api/replay_csv`
- Control de replay (start/pause/reset): `POST /api/replay_control`
- Estado de replay: `GET /api/replay_status`
- Reset de estado: `POST /api/reset`
- Snapshot compatible con `sim_data.json`: `GET /api/sim_data`
- Configuracion del historial SQLite: `GET/PUT /api/history/config`
- Consulta paginada y filtrada: `GET /api/history/events`
- Alertas no adyacentes paginadas: `GET /api/history/alerts`
- Serie de presencia y personas: `GET /api/history/presence`
- Borrado total confirmado: `POST /api/history/purge`
- WebSocket para panel live: `ws://<host>:8081/presencia`
- Web UI servida desde `/` (usa `inferencia_hub/web/index.html`)

Para navegar el flujo completo de un evento de sensor, consulta
[EVENT_FLOW.md](EVENT_FLOW.md). Esa guia sigue la ruta
`api/presence.py` -> `runtime/presence.py` -> `hub/events.py` y enlaza los
modulos de filtro, inferencia, metricas, snapshots e historial.

## Inferencia inteligente (IA)

`inferencia_hub` aprende la topologia del hogar desde historiales CSV montados
por el usuario:

- Construye transiciones dirigidas por secuencia temporal de sensores (sin hardcodear adyacencias).
- Entrena un `TimeSeriesTransformer` (Hugging Face) para probabilidad de siguiente habitacion.
- Combina Transformer + transiciones historicas (modelo hibrido) para generar aristas validas.
- Mantiene un mapa real de referencia editable en texto de adyacencia y compara contra el mapa inferido.
- Detecta transiciones no adyacentes y clasifica causa probable (multiples personas, mascota/ruido o error).
- Estima presencia por habitacion con un filtro probabilistico (creencia + evidencia de sensores).
- Estima numero de personas presentes a partir de habitaciones activas y conectividad.
- Corrige inferencias con ground truth fresco de persona, mascota, `occupancy` y conteo.

Opcionalmente puede ejecutar una validacion semantica del mapa con Ollama (`qwen2.5:0.5b-instruct` por defecto).

## Perfiles persistentes

Los perfiles se guardan en `/app/data/presence_profiles.json` mediante
escritura atomica. Un perfil define
habitaciones, areas de Home Assistant, entidades seleccionadas, categoria de
sensor y conexiones. Solo existe un perfil activo; sin el, los eventos de
escucha se ignoran y el snapshot declara la inferencia no disponible.

`real_home` es una plantilla copiable. Los modelos entrenados se separan por
perfil y se validan con una huella estructural antes de cargarse. Las APIs
`/api/real_sensor_config` y `/api/layout_reference` siguen disponibles como
adaptadores para clientes anteriores.

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

### Procesamiento despues de la ingesta

El handler `runtime/presence.py::ingest_event` primero valida perfil activo,
modo de entrada, fuente y catalogo de Home Assistant. Luego separa los eventos
por rol:

- `signal`: entra a `InferenceHubState.process_event`.
- `person_confirmation`: se registra como ground truth de persona.
- `pet_confirmation`: se registra como ground truth de mascota.
- `people_count_confirmation`: exige estado entero no negativo y corrige el
  conteo global o por habitacion.

Las confirmaciones no se procesan como movimiento normal. Las senales aceptadas
actualizan filtro de presencia, transiciones, `edge_support`, estado de
presencia, conteo, metricas, historial SQLite y WebSocket.

## Historial persistente

El historial guarda en `/app/data/presence_history.sqlite3` el evento original,
la inferencia resultante y el modo de entrada. SQLite usa WAL, migraciones con
`PRAGMA user_version`, indices de consulta y limpieza por retencion.

La vista inicial del panel consulta todos los eventos, con 50 filas por pagina.
Los filtros por sensor, tipo, habitacion, modo y fechas se aplican a la tabla,
al grafico escalonado y a las alertas no adyacentes. Estas ultimas usan una
paginacion independiente de 25 filas y permanecen disponibles tras reinicios.

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

### Confirmaciones de persona y mascota

El entrenamiento supervisado usa manifiestos JSON para declarar los CSV,
periodos, habitaciones, roles, exclusiones y hashes esperados. Las detecciones
de Frigate pueden generar etiquetas, pero sus entidades no forman parte de las
features ni se aceptan durante la inferencia.

```bash
curl -X POST http://localhost:8081/api/training/manifests/validate \
  -H "Content-Type: application/json" \
  -d '{"manifest_id":"person_pet_foyer"}'

curl -X POST http://localhost:8081/api/train_presence_supervised \
  -H "Content-Type: application/json" \
  -d '{
    "manifest_id": "person_pet_foyer",
    "epochs": 5,
    "seed": 42,
    "min_human_recall": 0.98,
    "synthetic_scenarios": 120,
    "synthetic_steps": 60,
    "max_samples": 15000
  }'
```

Cada ejecucion guarda un reporte por periodo y global con precision, recall,
F1, supresion de eventos solo-mascota, falsos descartes humanos y metricas de
ocupacion. El artefacto se activa al finalizar y deja disponible el artefacto
previo para rollback.

La imagen estandar tambien distribuye un Transformer relativo de ocupacion.
Este modelo puntua habitaciones candidatas mediante tipo y estado de sensor,
gaps temporales y relacion de adyacencia, sin codificar nombres absolutos. Se
carga automaticamente al activar cualquier perfil y no requiere entrenamiento
inicial del usuario.

Las entidades Frigate pueden asignarse como `person_confirmation` o
`pet_confirmation`. Sus cambios se guardan en SQLite como etiquetas, no
modifican presencia directamente y habilitan una adaptacion automatica
validada cuando se alcanzan los minimos configurados.

El backend distribuido carga automaticamente un artefacto incluido al activar un
perfil que todavia no tenga un filtro personalizado. En el panel aparece como
`Activo incluido`, junto con sus metricas de recall y supresion. Los CSV usados
para generar ese artefacto no se distribuyen ni son necesarios en la instalacion
del usuario.

La imagen publicada instala las dependencias ML de forma predeterminada. Para
construir una variante liviana que use solo reglas temporales:

```bash
docker build -f inferencia_hub/Dockerfile \
  --build-arg INSTALL_ML=0 \
  -t transformer-presence-backend:slim .
```

### Opcion 1: entrenamiento rapido

```bash
curl -X POST http://localhost:8081/api/train_model \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "/data/historial.csv",
    "debounce_seconds": 2,
    "min_gap_seconds": 2,
    "max_gap_seconds": 600,
    "epochs": 3,
    "max_samples": 7000,
    "degree_limit": 3,
    "use_ollama_validation": true
  }'
```

### Opcion 2: entrenamiento completo

Usa parametros menos restrictivos para historiales largos:

```bash
curl -X POST http://localhost:8081/api/train_model_full \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "/data/historial.csv",
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
curl -X POST http://localhost:8081/api/replay_csv \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "/data/historial.csv",
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
curl http://localhost:8081/api/evaluation_metrics
```

Obtener o actualizar mapa real de referencia desde texto de adyacencia:

```bash
curl -X POST http://localhost:8081/api/layout_reference \
  -H "Content-Type: application/json" \
  -d '{
    "adjacency_text": "bedroom: sittingroom\nsittingroom: bedroom, entertainment_room\nentertainment_room: sittingroom, foyer\nfoyer: entertainment_room, kitchen, living\nkitchen: foyer\nliving: foyer"
  }'
```

Controlar replay desde API (pause/start/reset):

```bash
curl -X POST http://localhost:8081/api/replay_control \
  -H "Content-Type: application/json" \
  -d '{
    "action": "pause"
  }'
```
