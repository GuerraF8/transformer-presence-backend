# Validación del entrenamiento supervisado

Fecha de ejecución: 14 de junio de 2026.

## Entorno

- Imagen: `transformer-presence-backend:ml-test`
- Construcción: `INSTALL_ML=1`
- Python: 3.11, Linux, CPU
- PyTorch: 2.6.0 CPU
- Transformers: 4.48.3
- Semilla: 42
- Épocas: 5
- Escenarios sintéticos: 120 de 60 pasos
- Muestras sintéticas máximas: 15.000
- Recall humano mínimo: 98%

## Datos

Los archivos listados corresponden a los insumos privados usados para validar
el artefacto distribuido. No forman parte del repositorio; los hashes permiten
verificar trazabilidad cuando se dispone de esos datos en un entorno autorizado.

| Archivo | SHA-256 |
|---|---|
| `history-1mes.csv` | `2f0b1cc3e9d99b00dfe70c83aca72e2172d1d9ee9e616193ff86d8b33f391b13` |
| `hall-cat+person-occupancy-nov2025.csv` | `88894c74386d7126ce9a5f7cede89726eef0d72e0005b2f59528140d6cb251a5` |
| `occupancymovementsincemay.csv` | `00dfab9bff4c0b0408c64e053e5310b8daebdc60a2e605958d312fd68a4acb12` |

El dataset preparado produjo 21.930 muestras de entrenamiento, 5.134 de
validación y 4.800 de prueba. Para el clasificador de movimiento se utilizaron
10.002, 2.508 y 2.325 activaciones respectivamente.

## Resultado

Ejecución: `20260614T224850Z-bcbf33a2`.

| Métrica global de prueba | Resultado |
|---|---:|
| Precisión humana | 38,84% |
| Recall humano | 97,04% |
| F1 humano | 55,47% |
| Supresión de eventos solo-mascota | 21,90% |
| Falsos descartes humanos | 2,96% |
| Umbral seleccionado en validación | 0,20 |

Resultados por período:

| Período | Precisión | Recall | F1 | Supresión mascota |
|---|---:|---:|---:|---:|
| Noviembre 2025 | 51,14% | 99,60% | 67,58% | 0,00% |
| Mayo 2026 | 34,79% | 95,86% | 51,05% | 22,22% |

El filtro temporal de referencia obtuvo recall global de 29,82% y F1 de
37,21%. El clasificador mejora ambas métricas. Aunque el recall de prueba queda
0,96 puntos porcentuales bajo el objetivo del 98%, la política operativa
prioriza reducir las activaciones frecuentes de mascotas y mantiene activa la
supresión aprendida. El panel informa tanto esta desviación como los falsos
descartes humanos observados.

El ajuste supervisado de ocupación de `foyer` obtuvo recall de 89,54% y F1 de
63,38% en prueba. El head de conteo no se modificó con etiquetas binarias.

## Persistencia

- El reporte quedó disponible mediante
  `GET /api/training/reports/20260614T224850Z-bcbf33a2`.
- Se valido que el artefacto anterior queda disponible para rollback.
- `POST /api/model/rollback` restauro el artefacto anterior.
- Una segunda llamada restauro el artefacto validado en este reporte.
- El checkpoint reproducible
  `person_pet_foyer-c293bc752c48-seed42` se distribuye dentro de la imagen y se
  carga automáticamente al activar un perfil sin modelo personalizado.
- El SHA-256 del checkpoint es
  `f081e9c90fe4346ab2f6be024c44df5ab6f1dd97fd5cf0b2252a83d25825616b`.

## Reproducción

```bash
docker compose -f docker-compose.yml -f docker-compose.ml.yml up -d --build

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
