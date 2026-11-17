# Evaluación predeclarada

> **Fecha de declaración original: 2026-11-16.** Este archivo se versiona (commit) **antes** de entrenar cualquier modelo
> distinto de la línea base. Cambiar algo exige agregar una entrada en la sección «Cambios», con fecha y justificación.
> **Versión vigente: 2** (etiqueta corregida el mismo día, antes de entrenar ningún modelo; ver «Cambios»).

## 1. Pregunta que responde el proyecto
Dado un siniestro vial **con víctimas** en Cartagena, ¿puede un modelo de aprendizaje automático, usando solo la información del
informe inicial (cuándo, dónde, clase de accidente y con qué colisionó), estimar si será **fatal** mejor que un clasificador trivial
y que una regresión logística sin ajuste, y con qué diferencias entre zonas y franjas horarias?

## 2. Tarea formal (Ruta C: aprendizaje supervisado)
- **Entrada x**: campos del informe inicial del siniestro (conjunto de features `reporte`, ver `src/config.py`).
- **Salida y**: `fatal` ∈ {0, 1}; 1 si `gravedad` = «Con muertos», 0 si «Con heridos».
- **Alcance de los datos**: el archivo del DATT contiene **solo siniestros con víctimas** (verificado al ejecutar el preprocesamiento:
  «Con heridos» 6.925, «Con muertos» 568, «Solo daños materiales» 0). Por eso el modelo **no** decide si un siniestro tiene víctimas;
  estima su gravedad una vez se sabe que las hay.
- **Duplicados**: varias filas pueden compartir código IPAT; se conserva la de mayor gravedad y se reporta cuántos casos tenían gravedad distinta.
- **Fuente de la etiqueta**: columna `gravedad` del DATT. Se verifica su coherencia con `muertos`, pero `muertos`, `heridos` y `gravedad`
  **nunca** entran como features (definen la etiqueta: fuga de datos).
- **Conjuntos de features**: `contexto` (cuándo/dónde), `reporte` (+ clase de accidente, choque con, objeto fijo; **primario**),
  `hipotesis` (+ hipótesis 1 del agente). `hipotesis` solo se usa en ablación, porque es un juicio causal que puede formarse después del hecho.

## 3. Partición (temporal, nunca aleatoria)
| Conjunto | Años | Uso |
|---|---|---|
| Entrenamiento | 2020–2023 | Ajustar modelos |
| Validación | 2024 | Elegir modelo e hiperparámetros |
| Prueba | 2025 | **Una sola evaluación final** (`python -m src.baseline --final`, y análogo para los demás modelos) |

## 4. Métricas
- **Primarias**: F1 de la clase «fatal» y PR-AUC. Con ~8 % de casos fatales la exactitud engaña, y la referencia de azar del PR-AUC es la prevalencia.
- **Secundarias**: F1 macro, ROC-AUC, precisión, recall, matriz de confusión.
- **Por subgrupo** (n ≥ 30 **y al menos 10 casos fatales**): recall y F1 por franja horaria y por localidad.
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
3. mantiene un recall de la clase «fatal» **≥ 0,60**;
4. en ningún subgrupo elegible (n ≥ 30 y ≥ 10 fatales) su recall cae más de **0,15** por debajo del recall global (si ocurre, se reporta como limitación y se discute).

> Los valores absolutos (0,70; 0,60; 0,15) son estimaciones iniciales sin haber visto resultados. Si al ejecutar la línea base
> sobre validación resultan claramente inalcanzables o triviales, pueden ajustarse **una vez, antes de entrenar los modelos del hito 3**,
> dejando constancia abajo.

## Cambios
| Fecha | Qué cambió | Justificación |
|---|---|---|
| 2026-11-16 | La etiqueta pasa de «con víctimas (heridos o muertos) vs. solo daños materiales» a **«fatal (Con muertos) vs. no fatal (Con heridos)»**. La métrica primaria incorpora PR-AUC. El criterio 4 exige además ≥ 10 casos fatales por subgrupo. | Al ejecutar el preprocesamiento con los datos reales se comprobó que el dataset solo contiene siniestros con víctimas (6.925 «Con heridos», 568 «Con muertos», 0 «Solo daños materiales»), por lo que la etiqueta original tenía una sola clase y no permitía entrenar ni evaluar. El cambio se hizo **antes de entrenar o evaluar cualquier modelo**; los umbrales numéricos de éxito no se modificaron. PR-AUC y el mínimo de casos por subgrupo se agregan por el desbalance (~8 % fatales): un recall sobre muy pocos casos no es interpretable. |
