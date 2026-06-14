# Transformer Presence Backend

Backend FastAPI para Transformer Presence. Recibe eventos desde Home Assistant, mantiene el estado de presencia, entrena modelos desde historicos CSV y sirve el panel web que la integracion HACS abre como iframe.

La version `0.5.0` incorpora perfiles editables con habitaciones, areas de Home
Assistant, entidades confirmadas y layouts arbitrarios. Consulta
[ARCHITECTURE.md](ARCHITECTURE.md).

`GET /api/sim_data` publica en `presence` los campos estables usados por las entidades de Home Assistant: `inferred_presence`, `people_estimate`, `confidence` y `updated_at`, ademas de `current_room` y `active_rooms`.

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

Para entrenar o reproducir historicos, deja tus CSV en `data/` y usa rutas del contenedor como:

```text
/data/history-1mes_sorted.csv
/data/history-1mes.csv
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

## Historial de presencia

El backend 0.4.0 persiste eventos brutos e inferencias de `listen`, `replay` y
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

## Home Assistant

Instala la integracion desde:

```text
https://github.com/GuerraF8/transformer-presence-hacs
```

Luego configura la URL base del backend con la URL publicada por este compose.

## Perfiles, areas y entidades

Las instalaciones nuevas comienzan sin perfil activo. Desde el modal del panel
se puede copiar `real_home`, crear un perfil manual o detectar las areas
publicadas por Home Assistant. Cada perfil conserva habitaciones con slug
estable, areas vinculadas, entidades confirmadas, categorias de sensor y
conexiones del mapa.

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

Luego cambia en `.env`:

```env
TRANSFORMER_PRESENCE_IMAGE=transformer-presence-backend:local
```
