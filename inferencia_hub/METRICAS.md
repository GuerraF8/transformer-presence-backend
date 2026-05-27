# Métricas de Inferencia Presencia

## Resumen Ejecutivo

El sistema calcula métricas en tiempo real en tres categorías principales:
1. **Mapa**: Calidad de la topología inferida vs. mapa real de referencia
2. **Personas**: Estimación del número de ocupantes presentes
3. **Anomalías**: Detección y clasificación de transiciones no adyacentes

---

## 1. Métricas de Mapa (Map Quality)

### 1.1 Aristas (Edges)

#### `reference_edges`
- **Definición**: Número de conexiones bidireccionales en el mapa real de referencia.
- **Tipo**: Contador entero
- **Rango**: 0 a N (N = número máximo posible de aristas entre todas las habitaciones)
- **Ejemplo**: Un hogar con 6 habitaciones como máximo puede tener hasta 15 aristas (grafo completo)
- **Importancia**: Línea base contra la que se mide el desempeño del modelo

#### `live_edges_total`
- **Definición**: Número de transiciones habitación→habitación observadas en vivo con al menos 1 ocurrencia.
- **Cálculo**: Aristas extraídas de `edge_support` con support ≥ 1
- **Tipo**: Contador entero
- **Indica**: Exploración del espacio de transiciones por el usuario

#### `live_edges_confirmed`
- **Definición**: Aristas en vivo con evidencia suficiente (support ≥ CONFIRMED_EDGE_SUPPORT, por defecto 2).
- **Cálculo**: Aristas con support ≥ 2
- **Tipo**: Contador entero
- **Importancia**: Filtra transiciones ocasionales o errores de sensor único

#### `model_edges`
- **Definición**: Número de aristas inferidas por el modelo AI entrenado.
- **Dependencia**: Solo disponible si `model_ready = true`
- **Tipo**: Contador entero
- **Indica**: Topología aprendida desde historial CSV

### 1.2 Métricas de Calidad (Precision/Recall/F1)

Se calculan dos conjuntos de métricas:

#### `live_confirmed_quality`
Compara aristas confirmadas en vivo vs. aristas de referencia:

- **TP (True Positives)**: Aristas en vivo confirmadas que existen en referencia
  - Indica transiciones correctamente observadas
  - `TP = |live_confirmed_edges ∩ reference_edges|`

- **FP (False Positives)**: Aristas en vivo confirmadas que NO existen en referencia
  - Indica transiciones erróneas detectadas (ruido de sensor, múltiples personas, etc.)
  - `FP = |live_confirmed_edges - reference_edges|`

- **FN (False Negatives)**: Aristas de referencia que no se han observado en vivo
  - Indica zonas del hogar aún no exploradas
  - `FN = |reference_edges - live_confirmed_edges|`

- **Precision**: `TP / (TP + FP)` — De las transiciones observadas, ¿cuántas son correctas?
  - Rango: 0.0 a 1.0
  - 1.0 = todas las transiciones observadas son válidas
  - 0.0 = ninguna transición es válida

- **Recall**: `TP / (TP + FN)` — De las transiciones válidas, ¿cuántas se han observado?
  - Rango: 0.0 a 1.0
  - 1.0 = se han observado todas las transiciones posibles
  - 0.0 = no se ha observado ninguna transición válida

- **F1**: Media armónica de precision y recall
  - Rango: 0.0 a 1.0
  - Equilibra ambas métricas
  - Útil para evaluación global

**Interpretación**:
- `precision=0.8, recall=0.6, f1=0.67`: El sistema detecta bien (80% precisión) pero le falta explorar más zonas (60% recall)

#### `model_quality`
Compara aristas del modelo AI vs. aristas de referencia:

- Misma fórmula que `live_confirmed_quality`
- **Ventaja**: Evita depender solo de exploración en vivo
- **Desventaja**: Solo disponible después de entrenamiento

---

## 2. Métricas de Personas (People)

### `current_estimate`
- **Definición**: Número estimado de personas presentes en este instante
- **Tipo**: Entero ≥ 1
- **Cálculo**: 
  1. Identifica habitaciones activas en los últimos `PRESENCE_HOLD_SECONDS` (por defecto 180s)
  2. Calcula componentes conexas en el grafo de habitaciones
  3. Estima personas = máximo(1, num_componentes + bonus)
  4. Bonus aplicado si hay ≥ 2 componentes y ≥ 3 habitaciones activas

**Ejemplo**:
- Habitaciones activas: `[bedroom, sittingroom, kitchen, foyer]`
- Grafo: `bedroom ↔ sittingroom ↔ entertainment_room ↔ foyer ↔ kitchen`
- Componentes conexas: 1 (todas conectadas)
- Estimación: 1 persona (solo movimiento)

**Contrareje**:
- Habitaciones activas: `[bedroom, foyer]` (no conectadas directamente en referencia)
- Componentes conexas: 2
- Estimación: 2-3 personas (probable separación física)

### `max_observed`
- **Definición**: Máximo número de personas estimadas registrado en sesión
- **Tipo**: Entero ≥ 0
- **Propósito**: Histórico de ocupación máxima

---

## 3. Métricas de Anomalías (Non-Adjacent)

### `total`
- **Definición**: Número acumulado de transiciones que violan el mapa de referencia
- **Tipo**: Contador entero
- **Rango**: 0 a infinito
- **Importancia**: Detecta eventos inesperados o configuración incorrecta del mapa

### `multi_person_probable`
- **Definición**: Transiciones no adyacentes clasificadas como "múltiples personas"
- **Criterio**:
  - Personas estimadas ≥ 2, O
  - Habitaciones activas simultáneas ≥ 2
- **Ejemplo**: Sensor bedroom + sensor living en simultáneo (no hay camino adyacente)
- **Acción recomendada**: Ignorar o revisar mapa de referencia

### `pet_or_noise`
- **Definición**: Transiciones no adyacentes clasificadas como "mascota o ruido"
- **Criterio**:
  - Sensor type = motion o occupancy, Y
  - Gap < 12 segundos (muy rápido para cruzar habitaciones)
- **Ejemplo**: Sensor sittingroom_motion + sensor kitchen_motion en 3 segundos
- **Causa probable**: Sensor falso positivo o mascota pequeña

### `sensor_or_data_error`
- **Definición**: Transiciones no adyacentes clasificadas como "error de sensor o datos"
- **Criterio**: Todo lo que no entra en las dos categorías anteriores
- **Ejemplo**: Tiempo muy largo entre sensores (> 600s) o datos históricos corruptos
- **Acción recomendada**: Revisar configuración de sensor o calibración de reloj

### `recent`
- **Definición**: Lista de últimas 25 anomalías registradas
- **Estructura por registro**:
  ```json
  {
    "timestamp": "2026-04-23T15:30:45Z",
    "from": "bedroom",
    "to": "kitchen",
    "gap_seconds": 45.2,
    "sensor_type": "motion",
    "estimated_people": 1,
    "active_rooms": ["bedroom", "kitchen"],
    "cause": "mascota_o_ruido"
  }
  ```

---

## 4. Métricas de Latencia (Latency)

### 4.1 Ingestion Latency
- **Definición**: Tiempo entre timestamp del evento en CSV y recepción en servidor
- **Unidad**: Milisegundos (ms)
- **Uso**: Detecta retrasos en transmisión desde Home Assistant o replay

#### Submétricas:
- `count`: Número de eventos con latencia medible
- `avg_ms`: Promedio de latencia
- `p50_ms`: Percentil 50 (mediana)
- `p95_ms`: Percentil 95 (cola de retrasos)
- `max_ms`: Máximo retraso observado

### 4.2 Processing Latency
- **Definición**: Tiempo de procesamiento interno desde ingesta hasta respuesta
- **Unidad**: Milisegundos (ms)
- **Incluye**: 
  - Clasificación de sensor
  - Inferencia de habitación
  - Detección de transición
  - Inferencia AI de presencia
  - Serialización de respuesta
- **Objetivo**: < 50ms por evento

#### Submétricas: (igual a ingestion)

---

## 5. Flujo de Cálculo Completo

```
Evento ingresa
    ↓
Clasificar sensor (motion/door/occupancy/other)
    ↓
Inferir habitación desde entity_id
    ↓
¿Es activación? (on/detected/motion/active/true)
    ├─ SÍ → Procesar como transición
    │   ├─ ¿Hay última activación? → Calcular gap
    │   ├─ ¿Gap en [min_gap, max_gap]? → Validar adyacencia
    │   │   ├─ Referencia adjacent? → TP/FP
    │   │   ├─ AI adjacent? → Aceptar/rechazar
    │   │   └─ Registrar anomalía si no adyacente
    │   └─ Actualizar edge_support
    │
    └─ NO → Solo actualizar presencia
       └─ Calcular active_rooms desde presencia_belief

Calcular personas → max(1, componentes_conexas)
    ↓
Emitir métricas (map, people, non_adjacent, latency)
    ↓
Broadcast vía WebSocket
```

---

## 6. Configuración de Parámetros

### Variables de Entorno (Con Valores Predeterminados)

```bash
PRESENCE_HOLD_SECONDS=180        # Ventana de tiempo para considerar habitación activa
MIN_GAP_SECONDS=2                # Mínimo de segundos entre activaciones de diferentes sensores
MAX_GAP_SECONDS=600              # Máximo de segundos para considerar una transición válida
CONFIRMED_EDGE_SUPPORT=2         # Número de observaciones para confirmar una arista
MAX_EVENTS_BUFFER=30000          # Máximo de eventos almacenados en memoria
```

### Request Parameters (Entrenamiento)

```json
{
  "debounce_seconds": 2,         // Descartar reactivaciones del mismo sensor < 2s
  "min_gap_seconds": 2,          // Mínimo gap entre habitaciones
  "max_gap_seconds": 600,        // Máximo gap (10 minutos)
  "epochs": 3,                   // Iteraciones de entrenamiento del Transformer
  "max_samples": 7000,           // Máximo de muestras para Transformer
  "degree_limit": 3,             // Máximo de vecinos por habitación
  "use_ollama_validation": true  // Validar topología con LLM
}
```

---

## 7. Interpretación de Escenarios

### Escenario A: Sistema Recién Inicializado
```json
{
  "map": {
    "reference_edges": 5,
    "live_edges_total": 0,
    "live_edges_confirmed": 0,
    "model_edges": 0,
    "live_confirmed_quality": {
      "precision": 0.0, "recall": 0.0, "f1": 0.0
    }
  },
  "people": {
    "current_estimate": 0,
    "max_observed": 0
  }
}
```
**Interpretación**: Esperando eventos. No hay exploración aún.

### Escenario B: Exploración Activa (Sin Entrenamiento AI)
```json
{
  "map": {
    "reference_edges": 5,
    "live_edges_total": 3,
    "live_edges_confirmed": 2,
    "live_confirmed_quality": {
      "tp": 2, "fp": 0, "fn": 3,
      "precision": 1.0, "recall": 0.4, "f1": 0.57
    }
  },
  "people": {
    "current_estimate": 1,
    "max_observed": 1
  },
  "non_adjacent": {
    "total": 0
  }
}
```
**Interpretación**: 
- Usuario ha transitado de forma válida (precision 100%)
- Solo ha explorado 40% del hogar (recall 40%)
- Sin anomalías (mapa de referencia bien configurado)

### Escenario C: Con Entrenamiento AI + Múltiples Personas
```json
{
  "map": {
    "reference_edges": 5,
    "live_edges_confirmed": 2,
    "model_edges": 5,
    "live_confirmed_quality": {
      "tp": 2, "fp": 0, "fn": 3,
      "precision": 1.0, "recall": 0.4, "f1": 0.57
    },
    "model_quality": {
      "tp": 4, "fp": 1, "fn": 1,
      "precision": 0.8, "recall": 0.8, "f1": 0.8
    }
  },
  "people": {
    "current_estimate": 2,
    "max_observed": 3
  },
  "non_adjacent": {
    "total": 4,
    "multi_person_probable": 3,
    "pet_or_noise": 1
  }
}
```
**Interpretación**:
- Modelo AI ha inferido topología con 80% F1 (muy bien)
- Modelo es más confiable que solo observación en vivo
- Múltiples personas detectadas (componentes conexas)
- 3 anomalías probablemente reales (personas en diferentes zonas)

---

## 8. Decisiones de Diseño

### Por qué estos filtros de transición

1. **Debounce**: Evita sobreconteo de sensor flaky (ej. motion que vibra)
2. **Min Gap (2s)**: Tiempo mínimo para que una persona camine entre habitaciones
3. **Max Gap (600s)**: Descarta transiciones muy lejanas en tiempo (probablemente nuevas personas)
4. **Confirmed Support (2)**: Requiere dos observaciones para evitar falsos positivos únicos

### Por qué el modelo AI

- **Regla simple**: Cadena de observaciones → transiciones
- **Limitación**: Captura solo lo observado, no estructura inmutable del hogar
- **Modelo AI**: Aprende patrones temporales + topología implícita
- **Ventaja**: Generaliza a habitaciones nunca exploradas

### Por qué estimación de personas

- **Métrica global**: Sirve para evaluar ocupación general del hogar
- **No es perfecto**: Depende de cobertura de sensores
- **Mejor que ocupancy_bolean**: No requiere input manual

---

## 9. Limitaciones Conocidas

1. **Sensor Coverage**: Si una habitación no tiene sensores, no puede ser detectada
2. **Sensor Reliability**: Motion sensors tienen ~64% reliability, doors ~48%
3. **Network Latency**: Replay desde CSV no captura retrasos reales de HA
4. **Layout Inference**: Asume que las observaciones son confiables
5. **Max Buffer**: Solo últimos 30k eventos en memoria (configurable)

---

## 10. Recomendaciones de Configuración

### Para Hogar Pequeño (1-2 personas, 4-5 habitaciones)
```json
{
  "debounce_seconds": 2,
  "min_gap_seconds": 1,
  "max_gap_seconds": 300,
  "confirmed_edge_support": 1,
  "epochs": 2
}
```

### Para Hogar Mediano (2-3 personas, 6-8 habitaciones)
```json
{
  "debounce_seconds": 2,
  "min_gap_seconds": 2,
  "max_gap_seconds": 600,
  "confirmed_edge_support": 2,
  "epochs": 3
}
```

### Para Hogar Grande o Multiuso (3+ personas, 8+ habitaciones)
```json
{
  "debounce_seconds": 3,
  "min_gap_seconds": 3,
  "max_gap_seconds": 900,
  "confirmed_edge_support": 3,
  "epochs": 5,
  "max_samples": 10000
}
```

---

## 11. API Endpoints para Obtener Métricas

### GET `/api/evaluation_metrics`
Retorna todas las métricas actuales en formato descrito arriba.

### GET `/api/sim_data`
Retorna snapshot completo incluyendo:
- Todos los eventos procesados
- Mapa de referencia
- Mapa inferido en vivo
- Métricas
- Estado del modelo

### POST `/api/layout_reference`
Permite actualizar el mapa de referencia y recalcular métricas.

---

## 12. Monitoreo Recomendado

Para verificar salud del sistema:

1. **Precision = 1.0 y Recall < 0.5**: Sistema correcto pero no explorado
2. **Precision = 0.5 y Recall = 1.0**: Configuración del mapa incorrecta
3. **Precision < 0.8**: Revisar confiabilidad de sensores
4. **F1 < 0.5**: Revisar todo (mapa, sensores, gaps)
5. **non_adjacent.total >> live_edges_confirmed**: Mapa probablemente incorrecto

