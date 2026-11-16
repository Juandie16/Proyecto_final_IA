# Evaluación predeclarada

> **Fecha de declaración: 2026-11-16.** Este archivo se versiona (commit) **antes** de entrenar cualquier modelo
> distinto de la línea base. Cambiar algo exige agregar una entrada en la sección «Cambios», con fecha y justificación.

## 1. Pregunta que responde el proyecto
¿Puede un modelo de aprendizaje automático, usando solo la información del informe inicial de un siniestro
(cuándo, dónde, clase de accidente y con qué colisionó), estimar si el siniestro dejará víctimas (heridos o muertos)
mejor que un clasificador trivial y que una regresión logística sin ajuste, y con qué diferencias entre zonas y franjas horarias?

## 2. Tarea formal (Ruta C: aprendizaje supervisado)
- **Entrada x**: campos del informe inicial del siniestro (conjunto de features `reporte`, ver `src/config.py`).
- **Salida y**: `victimas` ∈ {0, 1}; 1 si `gravedad` = «Con heridos» o «Con muertos», 0 si «Solo daños materiales».
- **Fuente de la etiqueta**: columna `gravedad` del DATT. Se verifica su coherencia con `muertos`/`heridos`, pero estas
  dos columnas **nunca** entran como features (definen la etiqueta: fuga de datos).
- **Conjuntos de features**: `contexto` (cuándo/dónde), `reporte` (+ clase de accidente, choque con, objeto fijo; **primario**),
  `hipotesis` (+ hipótesis 1 del agente). `hipotesis` solo se usa en ablación, porque es un juicio causal que puede formarse
  después del hecho.

## 3. Partición (temporal, nunca aleatoria)
| Conjunto | Años | Uso |
|---|---|---|
| Entrenamiento | 2020–2023 | Ajustar modelos |
| Validación | 2024 | Elegir modelo e hiperparámetros |
| Prueba | 2025 | **Una sola evaluación final** (`python -m src.baseline --final`, y análogo para los demás modelos) |

## 4. Métricas
- **Primarias**: F1 de la clase «con víctimas» y F1 macro (el desbalance hace engañosa la exactitud).
- **Secundarias**: ROC-AUC, PR-AUC, precisión, recall, matriz de confusión.
- **Por subgrupo** (n ≥ 30): recall y F1 por franja horaria y por localidad.
- Umbral de decisión 0,5 salvo que se declare lo contrario en «Cambios».

## 5. Comparación mínima exigida (Ruta C)
1. Línea base: clasificador mayoritario y regresión logística sin ajuste (`src/baseline.py`).
2. Al menos dos modelos adicionales (propuestos: *random forest* y *gradient boosting*) con ajuste de hiperparámetros
   sobre validación temporal.
3. Varias corridas: modelos estocásticos con 5 semillas (media ± desviación); validación de origen móvil (*rolling origin*) por año.
4. Ablación de conjuntos de features (`contexto` vs `reporte` vs `hipotesis`).

## 6. Umbral de éxito (propuesto; el equipo debe confirmarlo antes de entrenar los modelos del hito 3)
El proyecto se considera exitoso si, en la evaluación final sobre **prueba (2025)**, el mejor modelo propuesto:
1. supera a la **mejor línea base** en F1 macro por **≥ 0,03** (diferencia absoluta);
2. alcanza ROC-AUC **≥ 0,70**;
3. mantiene un recall de la clase «con víctimas» **≥ 0,60**;
4. en ningún subgrupo con n ≥ 30 su recall cae más de **0,15** por debajo del recall global (si ocurre, se reporta como limitación y se discute).

> Los valores absolutos (0,70; 0,60; 0,15) son estimaciones iniciales sin haber visto datos reales. Si al ejecutar la línea base
> sobre validación resultan claramente inalcanzables o triviales, pueden ajustarse **una vez, antes de entrenar los modelos del hito 3**,
> dejando constancia abajo.

## Cambios
| Fecha | Qué cambió | Justificación |
|---|---|---|
| — | — | — |
