# Analisis reproducible de resultados de presencia

Este documento explica como usar `generar_resultados_presencia.py` para evaluar
historiales de presencia sin depender de los CSV privados usados en la memoria.
El objetivo es que un tercero pueda calcular las mismas familias de metricas
con sus propios datos, conservar trazabilidad del procesamiento y, si lo desea,
medir el costo computacional del calculo con CodeCarbon.

## Alcance del script

`generar_resultados_presencia.py` calcula metricas historicas a partir de cambios
de estado. Cada estado se interpreta como vigente hasta el siguiente cambio de
la misma serie. Las metricas se ponderan por duracion, por lo que un error que
dura una hora pesa mas que un cambio incorrecto de pocos segundos.

El script puede evaluar:

- Conteo inferido frente a una referencia de conteo.
- Habitaciones inferidas frente a referencias de ocupacion, camara, silla o
  movimiento.
- Matrices F1 entre habitaciones inferidas y referencias disponibles.
- Comparaciones directas por habitacion cuando existe una referencia esperada.
- Tiempo de ejecucion, filas procesadas y emisiones estimadas si se activa
  CodeCarbon.

Los historiales reales de una vivienda pueden revelar rutinas, horarios y
ubicaciones. Por esa razon no se incluyen CSV de ejemplo con datos reales en el
repositorio. Para reproducir el procedimiento se debe usar un CSV propio,
anonimizado o sintetico.

## Requisitos

El calculo basico usa solo la biblioteca estandar de Python:

```bash
python generar_resultados_presencia.py --help
```

Dependencias opcionales:

```bash
pip install pillow codecarbon
```

- `pillow`: permite generar figuras PNG.
- `codecarbon`: permite estimar consumo energetico y emisiones del calculo.

Si no se instala Pillow, se puede ejecutar con `--no-figures`. Si no se instala
CodeCarbon, no se debe usar `--track-emissions`.

## Formatos CSV aceptados

### Formato de eventos

El formato recomendado es una fila por cambio de estado:

```csv
last_changed,entity_id,state
2026-06-18T10:00:00Z,sensor.inferencia_de_presencia_2,1
2026-06-18T10:00:05Z,sensor.num_in_house,1
2026-06-18T10:01:20Z,binary_sensor.inferencia_de_presencia_occupancy_6,on
2026-06-18T10:01:25Z,input_boolean.kitchen_occupied,on
```

Columnas equivalentes aceptadas:

- Tiempo: `last_changed`, `last_updated`, `time`, `timestamp`, `datetime` o
  `date`.
- Entidad: `entity_id`, `entity`, `entity_name`, `friendly_name` o `name`.
- Estado: `state`, `value`, `new_state` o `status`.

El campo de entidad puede ser un `entity_id` de Home Assistant, un
`friendly_name`, un nombre normalizado o cualquier identificador estable. Lo
importante es usar el mismo identificador en el CSV y en los argumentos del
script.

### Formato ancho

Tambien se acepta una tabla con una columna temporal y una columna por entidad:

```csv
timestamp,sensor.inferencia_de_presencia_2,sensor.num_in_house,binary_sensor.inferencia_de_presencia_occupancy_6,input_boolean.kitchen_occupied
2026-06-18T10:00:00Z,1,1,off,off
2026-06-18T10:01:20Z,1,1,on,on
```

Cada celda no vacia se interpreta como el estado vigente de esa entidad en el
timestamp de la fila.

## Normalizacion temporal

Por defecto, los timestamps con zona horaria se respetan y los reportes diarios
se generan en `America/Santiago`. Se puede cambiar con:

```bash
python generar_resultados_presencia.py \
  --input data/mi_historial.csv \
  --timezone America/Santiago
```

Si el CSV contiene timestamps sin offset, se debe indicar la zona horaria que se
debe asumir:

```bash
python generar_resultados_presencia.py \
  --input data/mi_historial.csv \
  --naive-timezone America/Santiago
```

Estados nulos como `unknown`, `unavailable`, `null`, `none`, `nan` o celdas
vacias se excluyen hasta la primera observacion valida de cada serie.

## Ejecucion minima

```bash
python generar_resultados_presencia.py \
  --input data/mi_historial.csv \
  --count-inferred sensor.inferencia_de_presencia_2 \
  --count-reference sensor.num_in_house \
  --output-dir outputs/presencia
```

Cuando se entrega mas de un CSV, se puede usar una clave estable:

```bash
python generar_resultados_presencia.py \
  --input antes=data/historial_antes.csv \
  --input despues=data/historial_despues.csv \
  --count-inferred sensor.inferencia_de_presencia_2 \
  --count-reference sensor.num_in_house \
  --output-dir outputs/presencia
```

La clave aparece en `metricas_presencia.json` y permite comparar periodos sin
exponer nombres de archivo sensibles.

## Mapeo de habitaciones y referencias

Las habitaciones inferidas se declaran con `--room`:

```bash
--room binary_sensor.inferencia_de_presencia_occupancy_6=Kitchen
--room binary_sensor.inferencia_de_presencia_occupancy_8=Office
```

Las referencias de ocupacion, camara o silla se declaran con
`--confirmation-reference`:

```bash
--confirmation-reference input_boolean.kitchen_occupied="Kitchen occ."
--confirmation-reference binary_sensor.hall_person_occupancy="Hall camera"
--confirmation-reference binary_sensor.chair_occupied="Chair"
```

Los sensores de movimiento se declaran aparte con `--motion-reference`:

```bash
--motion-reference binary_sensor.kitchen_sensor_motion="Kitchen motion"
--motion-reference binary_sensor.office_sensor_motion="Office motion"
```

Esta separacion es importante. Una camara detecta persona dentro de su campo
visual, un sensor PIR detecta movimiento, una silla detecta permanencia local y
un booleano de ocupacion puede representar una confirmacion agregada. Todas son
referencias utiles, pero no tienen el mismo significado fisico.

Para indicar comparaciones directas esperadas por habitacion:

```bash
--direct-comparison Kitchen=input_boolean.kitchen_occupied,binary_sensor.kitchen_sensor_motion
--direct-comparison Office=input_boolean.office_occupied,binary_sensor.office_sensor_motion
```

## Metricas calculadas

### Conteo

Para cada intervalo temporal `i`, el script calcula el error:

```text
e_i = conteo_inferido_i - conteo_referencia_i
```

Las metricas principales son:

- `mae`: error absoluto medio ponderado por duracion.
- `rmse`: raiz del error cuadratico medio ponderado por duracion.
- `bias`: sesgo promedio. Un valor negativo indica subestimacion.
- `max_error`: mayor error absoluto observado.
- `exact_match_pct`: porcentaje de tiempo con coincidencia exacta.
- `within_1_pct`: porcentaje de tiempo con error dentro de mas/menos una
  persona.
- `overestimation_pct`: porcentaje de tiempo con sobreestimacion.
- `underestimation_pct`: porcentaje de tiempo con subestimacion.

### Ubicacion

Para cada habitacion inferida y cada referencia binaria se acumulan segundos en:

- `TP`: inferencia activa y referencia activa.
- `TN`: inferencia inactiva y referencia inactiva.
- `FP`: inferencia activa y referencia inactiva.
- `FN`: inferencia inactiva y referencia activa.

Desde esos acumulados se calculan exactitud, precision, recall, especificidad,
F1 e IoU. F1 e IoU son utiles cuando hay desbalance entre mucho tiempo inactivo
y pocos intervalos activos.

## Medicion de rendimiento y emisiones

Para medir el costo del calculo:

```bash
pip install codecarbon
python generar_resultados_presencia.py \
  --input data/mi_historial.csv \
  --count-inferred sensor.inferencia_de_presencia_2 \
  --count-reference sensor.num_in_house \
  --track-emissions \
  --offline-emissions-country CHL \
  --output-dir outputs/presencia
```

Salidas relacionadas:

- `rendimiento_metricas_presencia.json`: tiempo de pared, filas procesadas,
  cantidad de archivos de entrada y emisiones si estan disponibles.
- `codecarbon_metricas_presencia.csv`: archivo generado por CodeCarbon.
- Bloque `runtime` dentro de `metricas_presencia.json`.

La medicion de CodeCarbon es una estimacion. Para reportarla, documenta equipo,
sistema operativo, version de Python, numero de filas, si se generaron figuras y
pais usado para el factor de emision.

## Salidas

El directorio indicado por `--output-dir` contiene:

- `metricas_presencia.json`: resultados completos y metadatos de entrada.
- `rendimiento_metricas_presencia.json`: resumen operativo del calculo.
- `conteo_*.png`: graficos escalonados de conteo, si las figuras estan
  habilitadas.
- `f1_*.png`: mapas de calor F1, si existen referencias binarias suficientes.
- `codecarbon_metricas_presencia.csv`: medicion de CodeCarbon, si se activo.

Para ejecutar solo el calculo numerico:

```bash
python generar_resultados_presencia.py \
  --input data/mi_historial.csv \
  --no-figures \
  --output-dir outputs/presencia
```

## Plantilla de ejecucion

El script puede escribir una plantilla JSON de configuracion:

```bash
python generar_resultados_presencia.py --write-template outputs/plantilla_metricas.json
```

La plantilla no ejecuta el analisis. Sirve para documentar los argumentos que se
usaran en una corrida reproducible.

## Buenas practicas para datos sensibles

- No publiques CSV reales de Home Assistant si contienen horarios, rutinas,
  nombres de entidades o ubicaciones identificables.
- Usa claves anonimas en `--input`, por ejemplo `periodo_1=...`.
- Reemplaza nombres de entidades por identificadores estables si vas a compartir
  el archivo.
- Conserva por separado el diccionario privado que relaciona entidades anonimas
  con habitaciones reales.
- Reporta version del script, fecha de ejecucion, zona horaria, numero de filas
  y argumentos usados.

## Ejemplo completo

```bash
python generar_resultados_presencia.py \
  --input periodo_largo=data/historial_anonimizado.csv \
  --timezone America/Santiago \
  --naive-timezone America/Santiago \
  --count-inferred sensor.inferencia_de_presencia_2 \
  --count-reference sensor.num_in_house \
  --room binary_sensor.inferencia_de_presencia_occupancy_6=Kitchen \
  --room binary_sensor.inferencia_de_presencia_occupancy_8=Office \
  --confirmation-reference input_boolean.kitchen_occupied="Kitchen occ." \
  --confirmation-reference input_boolean.office_occupied="Office occ." \
  --confirmation-reference binary_sensor.hall_person_occupancy="Hall camera" \
  --motion-reference binary_sensor.kitchen_sensor_motion="Kitchen motion" \
  --motion-reference binary_sensor.office_sensor_motion="Office motion" \
  --direct-comparison Kitchen=input_boolean.kitchen_occupied,binary_sensor.kitchen_sensor_motion \
  --direct-comparison Office=input_boolean.office_occupied,binary_sensor.office_sensor_motion \
  --track-emissions \
  --offline-emissions-country CHL \
  --output-dir outputs/presencia
```

