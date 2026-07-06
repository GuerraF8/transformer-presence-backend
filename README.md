# Transformer Presence Backend

Backend FastAPI para Transformer Presence. Recibe eventos desde Home Assistant,
mantiene el estado de presencia, entrena o adapta modelos desde historicos CSV
y sirve el panel web que la integracion HACS abre como iframe.

La imagen estandar incluye un Transformer relativo de ocupacion y un
clasificador de movimiento humano/mascota. Ambos funcionan con layouts
definidos por perfil, areas de Home Assistant y entidades confirmadas por el
usuario. Consulta [ARCHITECTURE.md](ARCHITECTURE.md) para la estructura interna.

`GET /api/sim_data` publica en `presence` los campos estables usados por las entidades de Home Assistant: `inferred_presence`, `people_estimate`, `confidence` y `updated_at`, ademas de `current_room` y `active_rooms`.

El recorrido completo de un evento de sensor esta documentado en
[inferencia_hub/EVENT_FLOW.md](inferencia_hub/EVENT_FLOW.md). Esa guia enlaza
las capas de API, runtime, hub, modelos, historial, metricas y panel.

## Despliegue rapido

Requisitos:

- Docker Engine o Docker Desktop.
- El archivo `docker-compose.yml`.
- Un archivo `.env` creado desde `.env.example`.

Pasos:

```bash
cp .env.example .env
docker compose up -d
```

En Windows PowerShell:

```powershell
Copy-Item .env.example .env
docker compose up -d
```

URLs por defecto:

- Panel: `http://localhost:8081/`
- Swagger UI: `http://localhost:8081/docs`
- Health: `http://localhost:8081/api/health`

## Configuracion de red

Edita `.env` si necesitas otro puerto o interfaz:

```env
TRANSFORMER_PRESENCE_BIND_HOST=0.0.0.0
TRANSFORMER_PRESENCE_PORT=8081
```

La integracion HACS necesita una URL alcanzable desde Home Assistant y desde el navegador del usuario. Si el backend corre en otro equipo, usa la IP LAN o Tailscale de ese equipo:

```text
http://<ip-del-equipo-docker>:8081
```

No uses `127.0.0.1` en la integracion si Home Assistant o el navegador estan en otra maquina.

## Datos y persistencia

El compose monta:

- `inferencia_hub_data:/app/data` para estado runtime, modelos, metricas, configuracion e historial SQLite.
- `./data:/data:ro` para CSV historicos que quieras usar en replay o entrenamiento.

La imagen ya incluye checkpoints preentrenados. Los CSV usados para generar
esos artefactos no forman parte del paquete instalado. Monta `./data` solo si
quieres ejecutar replays, entrenar con tus propios historiales o regenerar
modelos.

Para entrenar o reproducir historicos, deja tus CSV en `data/` y usa rutas del contenedor como:

```text
/data/historial_ordenado.csv
/data/historial.csv
```

## Variables principales

Las variables disponibles estan documentadas en `.env.example`. Las mas usadas son:

- `TRANSFORMER_PRESENCE_IMAGE`: imagen Docker a ejecutar.
- `TRANSFORMER_PRESENCE_PORT`: puerto publicado en el host.
- `PET_FILTER_ENABLED`: activa/desactiva el filtro de mascotas.
- `CORS_ALLOW_ORIGINS`: origenes permitidos para llamadas desde navegador.
- `HISTORY_DB_PATH`: ruta fisica de SQLite; requiere reinicio para cambiarse.
- `HISTORY_ENABLED`: valor inicial para activar persistencia.
- `HISTORY_RETENTION_DAYS`: retencion inicial, 365 dias por defecto.
- `HISTORY_PERSIST_MODES`: modos iniciales separados por coma.
- `PRESENCE_PROFILES_PATH`: repositorio JSON de perfiles; por defecto
  `/app/data/presence_profiles.json`.

## Modelo preentrenado y aprendizaje en vivo

La imagen estandar incluye un Transformer relativo de ocupacion, independiente
de nombres y cantidad de habitaciones, y un clasificador de movimiento humano
o mascota. Al activar un perfil sin modelo propio, el backend carga ambos
artefactos y guarda una copia compatible en
`/app/data/model_state/profiles/{profile_id}`.

Cada entidad seleccionada puede tener el rol `signal`,
`person_confirmation`, `pet_confirmation` o `people_count_confirmation`. Las
confirmaciones de persona, mascota, sensores `occupancy` y conteo se tratan como
ground truth para correccion y evaluacion; no se procesan como senales inferidas
normales. El scheduler evalua diariamente una adaptacion cuando existen al
menos 500 etiquetas nuevas, con 100 de persona y 100 de mascota, y han
transcurrido siete dias desde la ultima activacion.

Endpoints:

- `GET /api/live_training/status`
- `GET/PUT /api/live_training/config`
- `POST /api/live_training/run`

Solo se activa un candidato que mejora las metricas del modelo actual. La
ocupacion y el filtro de mascotas se validan de forma independiente, con
reemplazo atomico y rollback.

## Historial de presencia

El backend persiste eventos brutos e inferencias de `listen`, `replay` y
`simulator` en SQLite. La configuracion de retencion y modos se administra desde
el modal del panel y queda guardada en la misma base de datos.

Endpoints principales:

- `GET/PUT /api/history/config`
- `GET /api/history/events`
- `GET /api/history/alerts`
- `GET /api/history/presence`
- `POST /api/history/purge`

`/api/reset` solo reinicia el estado operativo y no elimina el historial. El
borrado total exige enviar `{"confirmation":"BORRAR"}`. El historial de Recorder
en Home Assistant sigue funcionando de forma independiente y complementaria.
Las alertas no adyacentes se consultan desde el mismo historial SQLite, con los
mismos filtros y una paginacion independiente en el panel.

## Analisis reproducible de resultados historicos

El repositorio incluye `generar_resultados_presencia.py` para calcular metricas
temporales desde CSV exportados por Home Assistant u otra fuente equivalente.
El script no requiere los historiales privados del proyecto: cada usuario debe
entregar sus propios CSV con columnas de entidad, estado y timestamp, o un CSV
ancho con una columna temporal y una columna por entidad.

Ejemplo minimo:

```bash
python generar_resultados_presencia.py \
  --input data/mi_historial.csv \
  --count-inferred sensor.inferencia_de_presencia_2 \
  --count-reference sensor.num_in_house \
  --room binary_sensor.inferencia_de_presencia_occupancy_6=Kitchen \
  --confirmation-reference input_boolean.kitchen_occupied="Kitchen occ." \
  --output-dir outputs/presencia
```

El resultado principal es `metricas_presencia.json`, junto con
`rendimiento_metricas_presencia.json` y figuras PNG cuando Pillow esta
disponible. Para medir tiempo de ejecucion y emisiones estimadas con
CodeCarbon:

```bash
pip install codecarbon
python generar_resultados_presencia.py \
  --input data/mi_historial.csv \
  --track-emissions \
  --offline-emissions-country CHL
```

Consulta [RESULTADOS_PRESENCIA.md](RESULTADOS_PRESENCIA.md) para el contrato de
entrada, ejemplos de mapeo de habitaciones, diferencias entre ocupacion,
camara, silla y movimiento, y recomendaciones para no publicar historiales con
datos sensibles.

## Analisis de sostenibilidad de inferencias ML

El repositorio incluye `analizar_sostenibilidad_inferencias.py` para medir con
CodeCarbon el costo de inferencias PyTorch reales usando los modelos
distribuidos con el backend. La carga es sintetica y deterministica, por lo que
no requiere historiales privados ni modifica endpoints de la API.

Instala las dependencias ML y CodeCarbon:

```bash
pip install -r inferencia_hub/requirements-ml.txt
pip install codecarbon
```

Ejecuta el benchmark con medicion offline para Chile:

```bash
python analizar_sostenibilidad_inferencias.py \
  --offline-emissions-country CHL \
  --output-dir outputs/sostenibilidad_inferencias
```

El script genera `sostenibilidad_inferencias.json`,
`sostenibilidad_inferencias.md` y un CSV de CodeCarbon por escenario:
`relative_occupancy`, `pet_filter` y `combined_pipeline`. Los resultados
incluyen latencia, throughput, dispositivo PyTorch, llamadas totales, emisiones
estimadas y emisiones normalizadas por 1.000 y 1.000.000 de inferencias.

CodeCarbon entrega una estimacion dependiente del hardware, carga de fondo,
modo energetico y sensores disponibles del sistema. Para comparar ejecuciones,
usa la misma maquina, la misma configuracion energetica y la misma version de
dependencias. Para validar el flujo sin medir emisiones:

```bash
python analizar_sostenibilidad_inferencias.py \
  --iterations 10 \
  --warmup 1 \
  --min-duration-seconds 0 \
  --no-codecarbon
```

## Home Assistant

Instala la integracion desde:

```text
https://github.com/GuerraF8/transformer-presence-hacs
```

Luego configura la URL base del backend con la URL publicada por este compose.

## Perfiles, areas y entidades

Al iniciar una instalacion se debe activar un perfil. Desde el modal del panel
se puede copiar `real_home`, crear un perfil manual o detectar las areas
publicadas por Home Assistant. Cada perfil define habitaciones con slug estable,
areas vinculadas, entidades confirmadas, categorias de sensor y conexiones del
mapa.

Los endpoints principales son:

- `GET/POST /api/profiles`
- `GET/PUT/DELETE /api/profiles/{profile_id}`
- `POST /api/profiles/{profile_id}/activate`
- `POST /api/profiles/{profile_id}/infer-layout`

Las actualizaciones usan revision optimista y devuelven `409` ante un borrador
obsoleto. Sin perfil activo, `/api/events` no procesa inferencia y las entidades
de Home Assistant permanecen no disponibles. Los modelos se guardan por perfil
en `/app/data/model_state/profiles/{profile_id}`.

## Seguridad de entidades reales

El backend mantiene una lista de entidades confirmadas dentro del perfil activo.
Los eventos enviados desde Home Assistant solo se procesan si el `entity_id`
esta asignado, habilitado y presente en el catalogo HA vigente, sin restringir
el dominio. `/api/real_sensor_config` se conserva como adaptador del perfil.

Esto protege instalaciones donde los sensores reales tienen nombres iguales a los de un historico CSV: el replay CSV usa una fuente separada y los eventos reales no afectan presencia ni alertas hasta que el usuario los confirme explicitamente desde el panel.

## Desarrollo local de la imagen

El despliegue de clientes usa la imagen publicada. Para construir localmente desde este repo:

```bash
docker build -t transformer-presence-backend:local -f inferencia_hub/Dockerfile .
```

La imagen liviana no instala PyTorch ni Transformers. Para entrenar y cargar
los modelos Transformer:

```bash
docker compose -f docker-compose.yml -f docker-compose.ml.yml up -d --build
```

Los CSV permanecen localmente en `data/`, montados como solo lectura en `/data`
cuando se desea reproducir el entrenamiento. El manifiesto `person_pet_foyer`
valida sus hashes SHA-256, ordena los eventos, normaliza UTC y genera divisiones
cronologicas independientes para noviembre de 2025 y mayo de 2026. El usuario
final recibe solo los checkpoints y metadatos en `inferencia_hub/defaults/models/`.

Endpoints del entrenamiento supervisado:

- `GET /api/training/manifests`
- `POST /api/training/manifests/validate`
- `POST /api/train_presence_supervised`
- `GET /api/training/reports/{run_id}`
- `POST /api/model/rollback`

Las entidades de confirmacion de Frigate se registran como etiquetas y se
excluyen de las entradas de inferencia. La imagen estandar incluye modelos
supervisados listos para usar, por lo que el usuario no necesita ejecutar un
entrenamiento inicial. El filtro selecciona su umbral con objetivo de recall
humano y el modelo relativo selecciona el umbral de ocupacion sobre una reserva
cronologica.
La imagen liviana se puede construir explicitamente con `INSTALL_ML=0`; en ese
caso se mantienen las reglas temporales.
La validacion del artefacto supervisado distribuido esta documentada en
[SUPERVISED_TRAINING_REPORT.md](SUPERVISED_TRAINING_REPORT.md). Los historiales
privados usados para esa validacion no forman parte del repositorio.

Luego cambia en `.env`:

```env
TRANSFORMER_PRESENCE_IMAGE=transformer-presence-backend:local
```
