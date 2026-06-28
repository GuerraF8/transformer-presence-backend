# Guia de entrenamiento para `inferencia_hub`

Este documento describe las rutas de entrenamiento disponibles en el backend y
como ejecutarlas con datos propios. No requiere los historiales privados usados
para construir los artefactos incluidos en la imagen.

## Tipos de entrenamiento

El backend puede trabajar con tres fuentes de aprendizaje:

- **Modelo incluido**: se carga automaticamente cuando un perfil activo no tiene
  artefactos personalizados.
- **Entrenamiento supervisado**: usa CSV declarados en un manifiesto para
  entrenar el filtro humano/mascota y el modelo relativo de ocupacion.
- **Adaptacion en vivo**: acumula confirmaciones reales de Home Assistant y
  activa un candidato solo si mejora al modelo vigente.

## Requisitos de datos

Los CSV deben montarse dentro del contenedor, normalmente en `/data`. Cada
archivo debe conservar timestamps comparables y entidades estables. Para
entrenamiento supervisado se recomienda declarar los datos mediante un
manifiesto JSON con:

- archivos esperados y hashes SHA-256;
- periodos de entrenamiento, validacion y prueba;
- mapeo de habitaciones;
- roles de entidades;
- exclusiones de sensores o intervalos no confiables.

Las entidades usadas como confirmacion (`person_confirmation` o
`pet_confirmation`) se usan como etiquetas. No se usan como features de
inferencia normal.

## Entrenamiento supervisado

Levanta la variante con dependencias ML:

```bash
docker compose -f docker-compose.yml -f docker-compose.ml.yml up -d --build
```

Valida el manifiesto:

```bash
curl -X POST http://localhost:8081/api/training/manifests/validate \
  -H "Content-Type: application/json" \
  -d '{"manifest_id":"person_pet_foyer"}'
```

Ejecuta el entrenamiento:

```bash
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

El reporte queda disponible en `/app/data/training_reports` y tambien se puede
consultar desde:

```http
GET /api/training/reports/{run_id}
```

El umbral se selecciona en validacion. El conjunto de prueba se informa por
separado para estimar generalizacion.

## Entrenamiento de mapa desde CSV

Para aprender transiciones y relaciones de adyacencia desde un historial:

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

Parametros principales:

| Parametro | Uso |
| --- | --- |
| `csv_path` | Ruta del CSV dentro del contenedor. |
| `debounce_seconds` | Evita recontar activaciones repetidas del mismo sensor. |
| `min_gap_seconds` | Gap minimo entre activaciones de habitaciones distintas. |
| `max_gap_seconds` | Gap maximo para aceptar una transicion. |
| `epochs` | Iteraciones de entrenamiento del modelo. |
| `max_samples` | Limite de muestras usadas para entrenar. |
| `degree_limit` | Maximo de conexiones propuestas por habitacion. |
| `use_ollama_validation` | Activa validacion semantica externa del mapa si Ollama esta disponible. |

Usa `train_model` para una corrida mas rapida y `train_model_full` cuando el
historial es mas largo o se busca una propuesta de mapa mas completa.

## Adaptacion con confirmaciones en vivo

Cada entidad del perfil activo puede tener uno de estos roles:

- `signal`: entrada normal de inferencia.
- `person_confirmation`: etiqueta de presencia humana.
- `pet_confirmation`: etiqueta de mascota.
- `people_count_confirmation`: referencia de conteo de personas.

Las confirmaciones se guardan en SQLite con perfil, revision y fingerprint. La
evaluacion automatica usa una reserva cronologica y solo activa componentes que
mejoran al modelo vigente.

Endpoints:

```http
GET /api/live_training/status
GET/PUT /api/live_training/config
POST /api/live_training/run
```

## Rollback y artefactos

Cada entrenamiento guarda artefactos por perfil en:

```text
/app/data/model_state/profiles/{profile_id}
```

Si un artefacto entrenado degrada el comportamiento, se puede restaurar el
anterior:

```bash
curl -X POST http://localhost:8081/api/model/rollback
```

## Interpretacion de metricas

Las metricas principales son:

- **Precision**: proporcion de predicciones positivas que eran correctas.
- **Recall**: proporcion de casos positivos reales que fueron detectados.
- **F1**: media armonica entre precision y recall.
- **Supresion de mascota**: eventos de mascota descartados por el filtro.
- **Falsos descartes humanos**: eventos humanos descartados por error.

Para automatizaciones de presencia, un recall humano alto suele ser prioritario
porque evita perder ocupacion real. La precision sigue siendo relevante para no
activar habitaciones o personas inexistentes.

## Diagnostico rapido

| Sintoma | Revision recomendada |
| --- | --- |
| F1 bajo | Revisar mapa de referencia, roles de entidades y parametros de gap. |
| Precision baja | Buscar sensores ruidosos, mascotas o habitaciones mal asignadas. |
| Recall bajo | Revisar exclusiones, cobertura de sensores y umbral seleccionado. |
| Sin entrenamiento | Confirmar que los CSV existen dentro del contenedor y que el manifiesto valida. |
| Sin mejora en adaptacion | Revisar cantidad de confirmaciones nuevas y reporte de validacion. |

Comandos utiles:

```bash
curl -s http://localhost:8081/api/evaluation_metrics | jq '.metrics'
curl -s http://localhost:8081/api/training/manifests | jq
curl -s http://localhost:8081/api/live_training/status | jq
```

