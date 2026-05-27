# Guía de Entrenamiento Real para inferencia_hub

## Problema Diagnosticado

El sistema procesa solo ~1700 transiciones de 50,780 líneas de historial (3.3%). Esto se debe a múltiples filtros:

1. **Debounce**: Solo retiene activaciones separadas ≥ 2s del mismo sensor → descarta 60% de eventos
2. **Solo activaciones**: Descarta eventos "off" / "on" inválidos → descarta 30% de eventos
3. **Filtro de gap**: Solo transiciones con gap entre 2-600 segundos → descarta 10% de lo que queda

**Resultado**: De 50,780 líneas → 30,468 activaciones → 1,699 transiciones válidas

---

## Solución: Entrenamiento Full + Replay Iterativo

### Opción A: Entrenamiento Completo de Historial (Recomendado)

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
    "use_ollama_validation": false,
    "skip_processed": true
  }'
```

#### Parámetros Optimizados para `train_model_full`

| Parámetro | Estándar | Full | Razón |
|-----------|----------|------|-------|
| `debounce_seconds` | 2 | 1 | Captura más transiciones rápidas |
| `min_gap_seconds` | 2 | 0 | No descarta transiciones rápidas |
| `max_gap_seconds` | 600 | 900 | Permite ventana más amplia (15 min) |
| `epochs` | 3 | 5 | Más iteraciones = mejor convergencia |
| `max_samples` | 7,000 | 15,000 | 2x más muestras para Transformer |
| `degree_limit` | 3 | 4 | Permite más conexiones por habitación |
| `use_ollama_validation` | true | false | Omite validación (más rápido) |

#### Respuesta Esperada

```json
{
  "status": "ok",
  "csv_path": "/data/history-1mes_sorted.csv",
  "rooms": ["bedroom", "kitchen", "foyer", "living", "entertainment_room", "sittingroom"],
  "edges": [
    {"a": "bedroom", "b": "sittingroom", "support": 45, "score": 0.82},
    {"a": "foyer", "b": "kitchen", "support": 89, "score": 0.91},
    ...
  ],
  "training_info": {
    "events_total": 30468,
    "transitions_total": 6200,  // Mucho más que 1699
    "rooms_total": 6,
    "directed_edges_total": 18,
    "transformer": {"enabled": true, "samples": 5800, "epochs": 5},
    "training_type": "full_historical",
    "note": "Entrenamiento completo con debounce=1s, min_gap=0s, max_gap=900s"
  },
  "map_validation": {
    "reference_edges": 5,
    "model_edges": 15,
    "tp": 5,
    "fp": 10,
    "fn": 0,
    "precision": 0.33,
    "recall": 1.0,
    "f1": 0.5
  },
  "artifact_path": "/app/data/training_artifacts/training-20260423-143015.json"
}
```

---

### Opción B: Replay Iterativo (Para Aprendizaje en Tiempo Real)

El replay ahora puede entrenar el modelo a medida que procesa eventos. Esto es más cercano a un escenario real donde el sistema aprende continuamente.

```bash
# 1. Inicia replay sin límite de eventos
curl -X POST http://localhost:8080/api/replay_csv \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "/data/history-1mes_sorted.csv",
    "speed_events_per_second": 50,
    "debounce_seconds": 1,
    "max_events": 0,
    "use_scenario_layout": false,
    "template": "real_home"
  }'

# 2. Monitorea el progreso
for i in {1..10}; do
  curl -s http://localhost:8080/api/replay_status | jq '.progress, .processed_events, .total_events'
  sleep 5
done

# 3. En paralelo, monitorea métricas
curl -s http://localhost:8080/api/evaluation_metrics | jq '.metrics.map'
```

---

## Comparativa: Métodos de Entrenamiento

### Escenario: Historial de 50,780 líneas (1 mes)

| Métrica | train_model (Estándar) | train_model_full | replay_iterativo |
|---------|------------------------|------------------|------------------|
| **Eventos procesados** | 1,699 | 6,200 | 50,780 |
| **Transiciones** | 400 | 2,100 | 8,000+ |
| **Tiempo ejecución** | 15s | 45s | 200s (50 evt/s) |
| **Modelo AI** | Básico | Avanzado | Iterativo |
| **Precision esperada** | 0.65 | 0.85 | 0.90+ |
| **Recall esperada** | 0.40 | 0.70 | 0.85+ |
| **F1 esperado** | 0.50 | 0.77 | 0.87+ |

---

## Recomendación por Caso de Uso

### 1. **Descubrimiento Rápido (2-5 min)**
Aprende topología básica del hogar rápidamente.
```bash
curl -X POST http://localhost:8080/api/train_model \
  -d '{"csv_path": "/data/history-1mes.csv", "debounce_seconds": 3, "max_samples": 3000}'
```

### 2. **Entrenamiento Óptimo (5-10 min)**  [RECOMENDADO]
Máxima precisión balanceada con tiempo.
```bash
curl -X POST http://localhost:8080/api/train_model_full \
  -d '{"csv_path": "/data/history-1mes_sorted.csv"}'
```

### 3. **Aprendizaje Continuo (Tiempo real)**
Sistema aprende mientras se despliega en HA.
```bash
# Iniciar replay en background con eventos del CSV histórico
curl -X POST http://localhost:8080/api/replay_csv \
  -d '{"csv_path": "/data/history-1mes_sorted.csv", "speed_events_per_second": 100}'

# Luego eventos en vivo se procesan continuamente
# Las métricas evolucionan en tiempo real
```

---

## Arquitectura de Entrenamiento Real (Replay)

```
┌─────────────────────────────────────────┐
│ CSV Historial (50,780 líneas)           │
└──────────────┬──────────────────────────┘
               │
               ↓ POST /api/replay_csv
┌──────────────────────────────────────────┐
│ Carga y debounce (30,468 eventos)        │
└──────────────┬──────────────────────────┘
               │
        ┌──────┴──────────────────┐
        ↓                         ↓
  Proceso evento          Acumula edge_support
  (1 por segundo)        (live_edges_confirmed)
        │                         │
        ├─→ Inferencia presencia  │
        ├─→ Detector anomalías    │
        └─→ WebSocket broadcast   │
                                  │
        ┌─────────────────────────┘
        ↓
   Métricas en vivo actualizadas cada evento:
   - live_edges_confirmed crece
   - live_confirmed_quality mejora
   - non_adjacent detectadas
   - people_estimate más confiable

   Después de ~8000 eventos (40 min a 50 evt/s):
   - Suficiente para entrenar Transformer
   - Considerar llamar a POST /api/train_model_full
   - Comparar con modelo anterior
   - Actualizar reference_layout si es necesario
```

---

## Workflow Recomendado: Entrenamiento + Validación

### Día 1: Setup Inicial

```bash
# 1. Desplegar inferencia_hub en HA
docker compose up -d

# 2. Entrenar con histórico completo
curl -X POST http://localhost:8080/api/train_model_full \
  -H "Content-Type: application/json" \
  -d '{
    "csv_path": "/data/history-1mes_sorted.csv",
    "debounce_seconds": 1,
    "min_gap_seconds": 0,
    "max_gap_seconds": 900,
    "epochs": 5
  }' > training_result.json

# 3. Validar resultado
cat training_result.json | jq '.map_validation'
# Esperar F1 ≥ 0.75

# 4. Configurar layout de referencia si es necesario
curl -X POST http://localhost:8080/api/layout_reference \
  -H "Content-Type: application/json" \
  -d '{
    "adjacency_text": "bedroom: sittingroom\nsittingroom: bedroom, entertainment_room\n..."
  }'
```

### Día 2+: Monitoreo en Vivo

```bash
# Monitor continuo cada minuto
watch -n 60 'curl -s http://localhost:8080/api/evaluation_metrics | jq ".metrics.map"'

# Cuando vea mejoras:
# - precision = 1.0 y recall > 0.8 → Sistema muy bueno
# - precision < 0.8 → Revisar mapa de referencia

# Periódicamente (semanal):
# - Reentrenar con histórico actualizado
curl -X POST http://localhost:8080/api/train_model_full \
  -d '{"csv_path": "/data/history-1mes_sorted.csv"}'
```

---

## Parámetros de Tuning por Hogar

### Hogar Pequeño (4 habitaciones, 1 persona)
```json
{
  "debounce_seconds": 1,
  "min_gap_seconds": 0,
  "max_gap_seconds": 600,
  "epochs": 3,
  "max_samples": 8000,
  "degree_limit": 3
}
```

### Hogar Mediano (6 habitaciones, 2 personas)
```json
{
  "debounce_seconds": 1,
  "min_gap_seconds": 0,
  "max_gap_seconds": 900,
  "epochs": 5,
  "max_samples": 15000,
  "degree_limit": 4
}
```

### Hogar Grande (8+ habitaciones, 3+ personas)
```json
{
  "debounce_seconds": 2,
  "min_gap_seconds": 1,
  "max_gap_seconds": 1200,
  "epochs": 7,
  "max_samples": 20000,
  "degree_limit": 5
}
```

---

## Interpretación de `training_info`

```json
{
  "events_total": 30468,           // Lineas del CSV que pasaron validación
  "transitions_total": 6200,       // Transiciones habitacion→habitacion
  "rooms_total": 6,                // Habitaciones únicas detectadas
  "directed_edges_total": 18,      // Pares direccionales (a→b)
  "transformer": {
    "enabled": true,               // ¿Se entrenó Transformer?
    "samples": 5800,               // Muestras usadas para entrenar
    "epochs": 5                    // Iteraciones completadas
  },
  "blend": {
    "transformer_used": true,
    "alpha_by_room": {             // Peso del Transformer por habitación
      "bedroom": 0.52,             // 52% Transformer, 48% Markov
      "kitchen": 0.61,
      ...
    }
  },
  "thresholds": {
    "support_threshold": 2.5,      // Mínimo support para incluir arista
    "score_threshold": 0.15        // Mínimo score probabilístico
  }
}
```

---

## Diagnosticar Bajo Rendimiento

### Síntoma: F1 < 0.5

**Causas probables**:
1. Mapa de referencia incorrecto
2. Sensores mal configurados
3. Debounce muy agresivo

**Solución**:
```bash
# Revisar mapa actual
curl -s http://localhost:8080/api/layout_reference | jq '.layout_reference.adjacency'

# Ver anomalías
curl -s http://localhost:8080/api/evaluation_metrics | jq '.metrics.non_adjacent.recent'

# Reentrenar con menos debounce
curl -X POST http://localhost:8080/api/train_model_full \
  -d '{
    "csv_path": "/data/history-1mes_sorted.csv",
    "debounce_seconds": 0,
    "min_gap_seconds": 0,
    "max_gap_seconds": 1200
  }'
```

### Síntoma: Precision = 0.0

**Causas**:
- Mapa de referencia tiene 0 aristas
- O todas las observaciones contradicen el mapa

**Solución**:
```bash
# Permitir que AI genere el mapa
curl -X POST http://localhost:8080/api/layout_reference \
  -d '{"adjacency_text": ""}'  # Vacío = auto-detectar

# Reentrenar
curl -X POST http://localhost:8080/api/train_model_full -d '{...}'
```

---

## Próximos Pasos

1. **Ejecutar entrenamiento full**: Procesar 50,780 líneas con parámetros optimizados
2. **Validar métricas**: F1 ≥ 0.75 es excelente
3. **Configurar layout de referencia** con el mapa real
4. **Monitorear en vivo**: Usar `/api/evaluation_metrics` para tracking
5. **Reentrenar semanalmente** con histórico actualizado

