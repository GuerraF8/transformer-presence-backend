# Flujo operativo despues de un evento de sensor

Este documento describe el recorrido real de un evento desde que llega al
backend hasta que actualiza presencia, mapa, metricas, historial y WebSocket.

## Mapa de archivos

| Capa | Archivo | Responsabilidad |
| --- | --- | --- |
| Rutas HTTP/WebSocket | `inferencia_hub/api/presence.py` | Declara `/api/events`, `/api/sim_data`, `/api/reset` y `/presencia`. |
| Runtime de presencia | `inferencia_hub/runtime/presence.py` | Valida modo/fuente/perfil, separa confirmaciones de senales y llama al hub. |
| Persistencia runtime | `inferencia_hub/runtime/lifecycle.py` | Conecta `event_sink` con SQLite y carga perfiles/modelos al inicio. |
| Estado principal | `inferencia_hub/hub/state.py` | Define buffers, mapa vivo, ground truth, sensores reales y modelo activo. |
| Aplicacion de perfil | `inferencia_hub/hub/profiles.py` | Aplica habitaciones, sensores, layout y configuracion del perfil activo. |
| Procesamiento de eventos | `inferencia_hub/hub/events.py` | Ejecuta la inferencia completa de un evento aceptado. |
| Filtro de presencia | `inferencia_hub/hub/filtering.py` | Filtra movimiento aislado o mascota y decide desplazamientos validos. |
| Inferencia | `inferencia_hub/hub/inference.py` | Actualiza creencia probabilistica, transiciones y `edge_support`. |
| Layout | `inferencia_hub/hub/layout.py` | Mantiene mapa real, mapa activo y validacion de adyacencia. |
| Metricas | `inferencia_hub/hub/metrics.py` | Calcula mapa, personas, ground truth, alertas y latencias. |
| Snapshot | `inferencia_hub/hub/snapshot.py` | Construye `/api/sim_data`, `presence`, `final_edges` e `inferred_layout_live`. |
| Modelo AI | `inferencia_hub/ai/` | Transiciones, persistencia, transformers y entrenamiento. |
| Modelos incluidos | `inferencia_hub/relative_occupancy.py`, `inferencia_hub/supervised/artifact.py` | Cargan artefactos empaquetados cuando un perfil no tiene modelo propio. |
| Historial | `inferencia_hub/history_store.py` | Persiste eventos e inferencias en SQLite. |
| Panel | `inferencia_hub/web/assets/panel/` | Consume snapshots, eventos, historial, perfiles y metricas. |

## Flujo resumido

1. Home Assistant o replay envia un evento.
2. La API valida la entrada.
3. El runtime verifica perfil, modo, fuente y catalogo.
4. El backend separa confirmaciones de senales inferidas.
5. El hub filtra el evento, infiere presencia, actualiza metricas y publica el
   snapshot.
6. El evento aceptado se persiste en SQLite y se emite por WebSocket.

## 1. Entrada por API

`inferencia_hub/api/presence.py` registra `POST /api/events` y lo conecta con
el handler `ingest_event`.

`inferencia_hub/runtime/presence.py` recibe `SensorEventInput` y aplica las
primeras compuertas:

- debe existir perfil activo;
- la fuente debe coincidir con el modo actual (`listen`, `replay` o
  `simulator`);
- los eventos reales de Home Assistant deben existir en el catalogo vigente;
- los eventos de confirmacion solo se aceptan desde Home Assistant en modo
  `listen`.

Si una compuerta falla, la respuesta es `status: ignored` y no se actualiza la
inferencia.

## 2. Roles de sensores

Cada entidad del perfil activo tiene `training_role`.

| Rol | Ruta |
| --- | --- |
| `signal` | Se registra como senal para aprendizaje en vivo y entra a `process_event`. |
| `person_confirmation` | Se guarda como ground truth de persona y no entra como senal inferida. |
| `pet_confirmation` | Se guarda como ground truth de mascota y no entra como senal inferida. |
| `people_count_confirmation` | El `state` debe ser entero no negativo; corrige el conteo global o por habitacion. |

Las confirmaciones actualizan `LiveTrainingStore`, `ground_truth_samples` y el
snapshot, pero no se tratan como movimiento normal. Esto evita que el sistema se
evalua a si mismo con inferencias duplicadas.

## 3. Normalizacion y seguridad de sensores reales

`InferenceHubState.process_event` normaliza:

- timestamp en UTC;
- `entity_id` en minusculas;
- fuente;
- habitacion desde perfil o desde nombre de entidad;
- tipo de sensor;
- estado activo mediante reglas del dominio.

Para eventos reales de Home Assistant, el backend vuelve a validar que la
entidad este asignada y habilitada. Si no lo esta, incrementa
`real_sensor_rejected_events` y devuelve `sensor_no_asignado` o
`sensor_deshabilitado`.

## 4. Filtro de presencia

Antes de aceptar una activacion se ejecuta
`FilteringMixin._evaluate_presence_filter_locked`.

La estrategia puede ser:

- `supervised_transformer`, si el filtro de mascota incluido o adaptado esta
  cargado y habilitado;
- `temporal_rules`, si no hay clasificador supervisado disponible.

Solo los eventos aceptados actualizan presencia y transiciones. Los eventos
filtrados conservan trazas en `presence_filter` y pueden dejar la presencia
anclada en la habitacion anterior.

## 5. Transiciones y mapa inferido en vivo

Cuando una activacion aceptada cambia de habitacion,
`InferenceMixin._build_transition` compara la habitacion anterior con la actual.

- Si son iguales, no aumenta `edge_support`.
- Si el mapa activo permite el movimiento, aumenta
  `edge_support[(from, to)]`.
- Si el mapa aprendido o real no permite el movimiento, se marca como
  `rejected_by_ai` o como alerta contra el layout real.

## 6. Inferencia de presencia

`InferenceMixin._infer_presence_with_ai` combina:

- matriz de transicion Markov;
- Transformer de siguiente habitacion si existe;
- Transformer relativo de ocupacion incluido;
- confiabilidad por tipo de sensor;
- vecinos del mapa activo.

Despues, `events.py` aplica reglas de correccion:

- `occupancy` confirma presencia en su habitacion;
- una ocupacion activa puede anclar la habitacion principal;
- movimiento valido puede forzar la habitacion observada sobre una prediccion
  pegada a la habitacion anterior;
- movimiento no valido conserva la presencia previa si no hay evidencia
  adyacente.

## 7. Ground truth y conteo

El ground truth tiene prioridad sobre la inferencia.

- `person_confirmation` confirma presencia humana en la habitacion.
- `pet_confirmation` registra muestras para medir falsos positivos de persona.
- `occupancy` se registra como confirmacion directa de presencia.
- `people_count_confirmation` global corrige `current_people_estimate` al valor
  del sensor mientras este fresco.
- `people_count_confirmation` por habitacion actua como evidencia absoluta de
  esa habitacion y como piso minimo para el total.

Antes de corregir, las muestras guardan la prediccion previa. Asi las metricas
comparan inferencia contra ground truth, no inferencia contra inferencia.

## 8. Estimacion de personas

`MetricsMixin._estimate_people_locked` calcula personas a partir de:

- habitaciones activas recientes;
- componentes conexas en el mapa activo;
- habitaciones con `occupancy` fresco;
- habitaciones con sensores activos;
- conteos por habitacion;
- conteo global fresco, que domina el total.

El resultado queda en `current_people_estimate` y `max_people_estimate`.

## 9. Alertas contra el mapa real

Despues de estimar personas, el evento compara la transicion observada contra
`reference_layout`.

Si la transicion no existe en el mapa real, se registra `layout_alert` con causa:

- `multiples_personas_probable`;
- `mascota_o_ruido`;
- `error_sensor_o_datos`.

Estas alertas aparecen en el evento, en metricas y en el historial.

## 10. Persistencia, metricas y publicacion

Cada evento aceptado construye un registro con:

- habitacion observada;
- habitacion resuelta;
- confianza;
- habitaciones activas;
- transicion;
- personas estimadas;
- alerta de layout;
- detalles del filtro e inferencia;
- latencia de ingesta y procesamiento.

Despues:

1. `MetricsMixin._evaluation_metrics_locked` calcula mapa, personas, inferencia,
   ground truth, alertas y latencia.
2. `runtime.lifecycle.persist_history_event` guarda el evento y la respuesta en
   SQLite mediante `HistoryStore`.
3. `SnapshotMixin.broadcast_event` publica la respuesta por WebSocket.
4. `GET /api/sim_data` expone el snapshot completo para el panel y la
   integracion.

## 11. Lectura recomendada por objetivo

- Para seguir una solicitud HTTP: `api/presence.py` -> `runtime/presence.py`.
- Para entender por que un evento fue ignorado: `runtime/presence.py` y
  `hub/events.py`.
- Para depurar sensores reales: `hub/profiles.py`, `hub/sensors.py` y
  `runtime/home_assistant.py`.
- Para revisar el mapa inferido: `hub/inference.py`, `hub/layout.py` y
  `hub/snapshot.py`.
- Para revisar conteo y ground truth: `hub/metrics.py` y
  `runtime/presence.py`.
- Para revisar persistencia: `runtime/lifecycle.py` y `history_store.py`.
- Para revisar modelos incluidos: `relative_occupancy.py`,
  `supervised/artifact.py` y `defaults/models/*/metadata.json`.
