# Funcionamiento del simulador de sensores

Este documento describe la vista `Simulador de Sensores`, implementada en `simulator.html`,
`assets/simulator.js` y `assets/styles.css`.

## Objetivo

El simulador permite probar eventos de sensores sin depender de Home Assistant. La vista muestra una
casa simulada basada en layouts como `real_home`, permite mover 1 o 2 ocupantes y traduce ese
movimiento a eventos `Movimiento` enviados al backend de `inferencia_hub`.

## Archivos principales

- `simulator.html`: define la estructura de la vista, los selectores de layout y ocupantes, el
  contenedor del mapa y la grilla de sensores.
- `assets/simulator.js`: contiene toda la lógica de carga de datos, renderizado, captura de teclado,
  movimiento, detección de sensores activos y envío de eventos.
- `assets/styles.css`: define el aspecto visual del mapa, habitaciones, ocupantes, tarjetas de
  sensores y estados activos.

## Librerías y APIs utilizadas

En el frontend no se usa una librería externa de UI o simulación. El simulador usa APIs nativas del
navegador:

- `fetch`: para llamar a los endpoints del backend.
- `addEventListener`: para capturar teclado, cambios de selectores y clicks.
- `requestAnimationFrame`: para mover ocupantes de forma fluida mientras una tecla está presionada.
- DOM API (`document.createElement`, `classList`, estilos inline): para renderizar dinámicamente mapa,
  habitaciones, ocupantes y sensores.
- CSS normal: para layout, estados visuales y transición suave de ocupantes.

En el backend, la vista depende del servicio FastAPI definido en `server.py`. Las dependencias del
servicio están en `requirements.txt`, principalmente:

- `fastapi`: API HTTP y WebSocket.
- `uvicorn`: servidor ASGI.
- `numpy`, `requests`, `torch`, `transformers`: usadas por otras partes del hub de inferencia y del
  modelo, no por la lógica visual directa del simulador.

## Carga inicial de datos

Al iniciar, `init()` registra eventos de UI y llama a `loadRooms()`.

`loadRooms()` consulta:

- `GET /api/sim_data`: obtiene snapshot actual, rooms conocidas y modo operativo.
- `GET /api/scenario_templates`: obtiene layouts disponibles, incluyendo `real_home`.

Luego el simulador:

1. Junta habitaciones del snapshot, del template `real_home` y de la lista base local.
2. Guarda los templates en `state.scenarioTemplates`.
3. Pobla el selector de layouts.
4. Construye el layout visual.
5. Ubica los ocupantes en habitaciones válidas.

## Layout de la casa

Para `real_home`, las habitaciones tienen coordenadas fijas y todas usan el mismo tamaño lógico
`1 x 1`:

```text
bedroom | sittingroom | entertainment_room | foyer | kitchen
                                      living
```

Internamente se representa con `REAL_HOME_COORDS`:

```js
bedroom: { col: 0, row: 0 }
sittingroom: { col: 1, row: 0 }
entertainment_room: { col: 2, row: 0 }
foyer: { col: 3, row: 0 }
kitchen: { col: 4, row: 0 }
living: { col: 3, row: 1 }
```

Para otros layouts, `buildFallbackLayout()` distribuye las habitaciones en una grilla regular. En
ambos casos, cada habitación ocupa el mismo tamaño visual.

## Estado interno

El estado principal vive en el objeto `state`:

- `rooms`: habitaciones conocidas por el backend y el simulador.
- `mode`: modo operativo actual (`listen` o `replay`).
- `switches`: mapa local de sensores, con llave `room|sensorType`.
- `scenarioTemplates`: templates recibidos desde `/api/scenario_templates`.
- `layoutKey`: layout seleccionado.
- `roomRects`: rectángulos de habitaciones usados para detección de posición.
- `occupants`: ocupantes disponibles, con posición `x`, `y`, room actual y flag `enabled`.
- `layoutMotionRooms`: rooms cuyo sensor `motion` fue activado por movimiento del simulador.
- `pressedKeys`: teclas de movimiento actualmente presionadas.
- `animationFrame` y `lastFrameAt`: control del loop de animación.

## Captura de teclas

El simulador registra:

```js
window.addEventListener("keydown", handleKeyDown);
window.addEventListener("keyup", handleKeyUp);
window.addEventListener("blur", handleWindowBlur);
```

Las teclas activas se guardan en `state.pressedKeys`.

Controles:

- Ocupante 1: `W`, `A`, `S`, `D`.
- Ocupante 2: flechas del teclado.

`handleKeyDown()` ignora eventos que ocurren dentro de `input`, `select` o `textarea`, para no
interferir con formularios. Si la tecla es de movimiento, se evita el comportamiento por defecto del
navegador y se inicia el loop con `startMovementLoop()`.

`handleKeyUp()` remueve la tecla desde `pressedKeys`. Si no quedan teclas presionadas, se apagan los
sensores de movimiento activados por el layout.

`handleWindowBlur()` limpia las teclas y apaga movimiento si el usuario cambia de ventana o la vista
pierde foco.

## Movimiento suave

El movimiento no ocurre directamente en cada `keydown`. En vez de eso se usa `requestAnimationFrame`.

Cada frame ejecuta `animationTick(now)`:

1. Calcula `deltaSeconds`, el tiempo transcurrido desde el frame anterior.
2. Calcula un vector de movimiento por ocupante con `movementVectorForOccupant()`.
3. Multiplica ese vector por `MOVEMENT_SPEED` y `deltaSeconds`.
4. Llama a `moveOccupant()` para intentar desplazar al ocupante.
5. Si al menos un ocupante se movió, re-renderiza el mapa.
6. Si ya no hay teclas presionadas, detiene el loop y apaga movimiento.

Esto produce desplazamiento continuo mientras la tecla está presionada, en vez de saltos discretos.

## Detección de habitación

La posición de cada ocupante es continua dentro de la grilla del layout. Para saber en qué habitación
está, `findRoomAt(x, y)` revisa si la posición cae dentro de algún rectángulo de `roomRects`.

`moveOccupant()` solo acepta el movimiento si la nueva posición cae dentro de una habitación válida.
Si el ocupante intenta moverse hacia un espacio vacío del layout, el movimiento se bloquea.

## Detección de movimiento

La presencia de un ocupante en una habitación no activa por sí sola el sensor `Movimiento`.

El sensor se activa solo cuando un ocupante se está desplazando realmente dentro de una habitación.
En cada frame, `animationTick()` crea un set `movingRooms` con las habitaciones donde hubo movimiento
efectivo. Luego llama a:

```js
syncLayoutMotionRooms(movingRooms);
```

`syncLayoutMotionRooms()` compara:

- rooms que se están moviendo ahora;
- rooms que el simulador había marcado como movimiento activo antes.

Con esa comparación:

- envía `ON` cuando empieza movimiento en una room;
- envía `OFF` cuando ya no hay movimiento en una room previamente activada por el layout;
- no mantiene `Movimiento ON` solo porque un ocupante siga quieto en la habitación.

El set `layoutMotionRooms` evita apagar sensores que no fueron activados por el simulador de layout.
Los botones manuales de sensores siguen existiendo como controles independientes.

## Envío de datos al backend

Los eventos se envían a:

```http
POST /api/events
```

Antes de enviar, `postSensorEvent()` revisa el modo operativo. Si el hub no está en modo `listen`,
llama a:

```http
POST /api/input_mode
```

con:

```json
{ "mode": "listen" }
```

El payload de sensor tiene esta forma:

```json
{
  "entity_id": "binary_sensor.kitchen_motion_sim",
  "state": "on",
  "sensor_type": "motion",
  "room": "kitchen",
  "timestamp": "2026-04-30T12:00:00.000Z",
  "source": "home_layout_simulator"
}
```

Para apagar movimiento, el mismo payload se envía con:

```json
{
  "state": "off"
}
```

Los switches manuales usan el mismo endpoint, pero con `source: "sensor_simulator"`.

## Pruebas rápidas de presencia

La sección `Pruebas rápidas` del simulador genera secuencias controladas para validar la respuesta
del modelo y del backend:

- `1 persona GT bedroom`: reinicia el estado y envía `bedroom occupancy ON`. Resultado esperado:
  `personas=1`, habitación actual `bedroom`, GT `bedroom`.
- `2 personas GT`: reinicia el estado y envía `bedroom occupancy ON` + `kitchen occupancy ON`.
  Resultado esperado: `personas=2`, habitaciones activas `bedroom` y `kitchen`.
- `Bedroom + motion externo`: reinicia el estado, envía `bedroom occupancy ON`, luego
  `sittingroom motion ON` y `sittingroom motion OFF`. Resultado esperado: mientras el motion externo
  está activo se observan 2 habitaciones; al apagarlo, `bedroom` queda como ground truth y la
  estimación vuelve a 1 persona.
- `Limpiar prueba`: reinicia el estado local y el backend.

Estas pruebas existen porque `occupancy` se trata como ground truth. Si una habitación tiene
`occupancy ON`, movimiento simultáneo en otra habitación se interpreta como evidencia de una segunda
persona, no como desplazamiento automático de la presencia principal.

## Renderizado visual

`renderHomeSim()` dibuja:

- botones absolutos para cada habitación;
- nombre de habitación;
- estado `Movimiento on/off`;
- marcadores de ocupantes.

La habitación recibe clases visuales:

- `occupied`: hay un ocupante dentro.
- `motion-on`: el sensor `Movimiento` está activo.

La grilla de sensores se renderiza con `renderSensors()`. Cada tarjeta muestra botones para:

- `Movimiento`.
- `Occupancy`.

Cuando `Movimiento` está activo, la tarjeta recibe `sensor-card-active`.

## Flujo resumido

1. El usuario abre `/simulator.html`.
2. El JS carga rooms y templates desde el backend.
3. Se construye el layout visual.
4. El usuario selecciona 1 o 2 ocupantes.
5. El usuario mantiene presionadas teclas de movimiento.
6. `keydown` agrega teclas a `pressedKeys`.
7. `requestAnimationFrame` mueve ocupantes suavemente.
8. Cada frame calcula habitaciones con movimiento efectivo.
9. `syncLayoutMotionRooms()` prende o apaga `Movimiento`.
10. `postSensorEvent()` envía eventos `on/off` al backend.
11. Al soltar teclas o perder foco, se apaga el movimiento activado por el layout.

## Consideraciones

- El simulador no estima presencia por sí mismo; solo envía eventos al hub.
- La inferencia de presencia, transición entre habitaciones y métricas ocurren en el backend.
- El layout `real_home` tiene coordenadas visuales explícitas. Otros layouts se muestran en una grilla
  automática.
- El estado visual de `Movimiento` representa actividad reciente de movimiento, no ocupación estática.
