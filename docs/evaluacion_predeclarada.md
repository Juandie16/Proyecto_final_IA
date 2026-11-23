# Evaluación predeclarada

> **Fecha de declaración original: 2026-11-16.** Este archivo se versiona (commit) **antes** de entrenar cualquier modelo
> distinto de la línea base. Cambiar algo exige agregar una entrada en la sección «Cambios», con fecha y justificación.
> **Versión vigente: 3.1** (v3: etiqueta y criterios de éxito ajustados el 2026-11-16; v3.1, 2026-11-22: corrección de un error de preprocesamiento
> y nueva referencia de la línea base, **sin cambiar ningún umbral**; ver «Cambios»).

## 1. Pregunta que responde el proyecto
Dado un siniestro vial **con víctimas** en Cartagena, ¿puede un modelo de aprendizaje automático, usando solo la información del
informe inicial (cuándo, dónde, clase de accidente y con qué colisionó), estimar si será **fatal** mejor que un clasificador trivial
y que una regresión logística sin ajuste, y con qué diferencias entre zonas y franjas horarias?

## 2. Tarea formal (Ruta C: aprendizaje supervisado)
- **Entrada x**: campos del informe inicial del siniestro (conjunto de features `reporte`, ver `src/config.py`).
- **Salida y**: `fatal` ∈ {0, 1}; 1 si `gravedad` = «Con muertos», 0 si «Con heridos».
- **Alcance de los datos**: el archivo del DATT contiene **solo siniestros con víctimas** (verificado al ejecutar el preprocesamiento:
  «Con heridos» 6.924, «Con muertos» 569, «Solo daños materiales» 0, tras deduplicar). Por eso el modelo **no** decide si un siniestro tiene víctimas;
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
- **Punto de operación**: la línea base usa umbral 0,5. Para los modelos del hito 3, el umbral se elige **en validación** como el mayor
  que logra recall ≥ 0,60 y se aplica sin cambios a prueba (nunca se ajusta mirando la prueba).

## 5. Comparación mínima exigida (Ruta C)
1. Línea base: clasificador mayoritario y regresión logística sin ajuste (`src/baseline.py`).
2. Al menos dos modelos adicionales (propuestos: *random forest* y *gradient boosting*) con ajuste de hiperparámetros
   sobre validación temporal.
3. Varias corridas: modelos estocásticos con 5 semillas (media ± desviación); validación de origen móvil (*rolling origin*) por año.
4. Ablación de conjuntos de features (`contexto` vs `reporte` vs `hipotesis`).

## 6. Criterios de éxito (versión 3; referencia de la línea base actualizada en 3.1)
El proyecto se considera exitoso si el mejor modelo propuesto cumple **los cuatro** criterios. Los criterios 1 y 2 se evalúan como media de
5 semillas / pliegues de origen móvil en validación; los criterios 2 a 4 se confirman una sola vez en **prueba (2025)**.

| # | Criterio | Qué mide | Referencia de la línea base (validación 2024) |
|---|---|---|---|
| 1 | **PR-AUC** mayor que el de la mejor línea base en **al menos 0,05** (diferencia absoluta) | Qué tan bien concentra los casos fatales entre las primeras alertas | 0,220 (azar = prevalencia ≈ 0,090) |
| 2 | **ROC-AUC ≥ 0,70** | Capacidad general de ordenar fatales por encima de no fatales | 0,666 |
| 3 | **Precisión ≥ 0,20** en el punto de operación con **recall ≥ 0,60** | Utilidad operativa: de cada 5 alertas, al menos 1 es fatal, detectando 6 de cada 10 fatales | 0,136 con recall 0,637 |
| 4 | Ningún subgrupo elegible (n ≥ 30 y ≥ 10 fatales) con recall más de **0,15** por debajo del global | Que el modelo no falle sistemáticamente en una franja o localidad | La línea base **no lo cumple** (mañana: 0,263 frente a 0,637) |

> Si el criterio 4 no se cumple, se reporta como limitación y se analiza; no invalida por sí solo los criterios 1 a 3.
> Los valores 0,05; 0,70; 0,20 y 0,15 son juicios del equipo, fijados tras ver la línea base en validación y antes de entrenar los modelos del hito 3;
> no se podrán modificar sin una nueva entrada en «Cambios».

## Cambios
| Fecha | Qué cambió | Justificación |
|---|---|---|
| 2026-11-16 | La etiqueta pasa de «con víctimas (heridos o muertos) vs. solo daños materiales» a **«fatal (Con muertos) vs. no fatal (Con heridos)»**. La métrica primaria incorpora PR-AUC. El criterio 4 exige además ≥ 10 casos fatales por subgrupo. | Al ejecutar el preprocesamiento con los datos reales se comprobó que el dataset solo contiene siniestros con víctimas (6.925 «Con heridos», 568 «Con muertos», 0 «Solo daños materiales»), por lo que la etiqueta original tenía una sola clase y no permitía entrenar ni evaluar. El cambio se hizo **antes de entrenar o evaluar cualquier modelo**; los umbrales numéricos de éxito no se modificaron. PR-AUC y el mínimo de casos por subgrupo se agregan por el desbalance (~8 % fatales): un recall sobre muy pocos casos no es interpretable. |
| 2026-11-16 | Los criterios de éxito pasan de (F1 macro +0,03; ROC-AUC ≥ 0,70; recall ≥ 0,60; brecha por subgrupo ≤ 0,15) a (PR-AUC +0,05; ROC-AUC ≥ 0,70; precisión ≥ 0,20 con recall ≥ 0,60; brecha por subgrupo ≤ 0,15). Se agrega la regla del punto de operación. | Tras ver la línea base en **validación** (no en prueba): su recall (0,655) ya cumplía el criterio original con precisión de 0,142, de modo que el criterio no distinguía un buen modelo de uno que marca casi todo como fatal. F1 macro dependía del umbral 0,5 y la línea base apenas mejoraba al mayoritario (0,486 frente a 0,476). El ajuste se hizo **antes de entrenar ningún modelo adicional** y sin mirar el conjunto de prueba. |
| 2026-11-16 | Se documenta (sin cambiar la etiqueta) una incoherencia de calidad de datos: en 2025 la columna `muertos` no concuerda con `gravedad` en 282 de 1.390 filas (0 en 2020–2024). | Se mantiene `gravedad` como fuente de la etiqueta porque coincide con las cifras oficiales del DATT en 2024 y es coherente con `muertos` hasta 2024; `muertos` en 2025 implicaría ~361 siniestros mortales, incompatible con las cifras oficiales (33 fallecidos entre enero y abril de 2025). Resultado respaldado en `experiments/calidad_etiqueta_por_anio.json`. |
| 2026-11-22 | **v3.1.** Se corrige el preprocesamiento: (a) `tipo_via` era constante («calle» en las 7.493 filas) porque el vacío de `via1…via6` viene como `"0"`; (b) en `choque_con` y `objeto_fijo` los valores `"0"`, vacío, `"No"` y `"Choque con"` (el nombre de la columna en 8 celdas) se unifican en `sin_dato`. La línea base se re-ejecuta y su resultado pasa a ser la referencia de los criterios: PR-AUC 0,229 → 0,220; ROC-AUC 0,670 → 0,666; precisión 0,142 → 0,136 con recall 0,655 → 0,637. **Los umbrales 0,05; 0,70; 0,20 y 0,15 no cambian.** También se corrigen los conteos de la sección 2 (6.924 / 569, que coinciden con `experiments/data_summary.json`; 6.925 / 568 eran de antes de corregir la deduplicación). | Al explorar los datos antes de entrenar los modelos del hito 3 se vio que `tipo_via` no aportaba información y que la forma de codificar «sin dato» cambiaba por año (`"0"` en 2020–22, vacío en 2023–24, «No» en 2025), lo que permitía al modelo inferir el año, y con él la tasa de fatales (5,9 % en 2020 frente a 9,0 % en 2024), en vez de la gravedad. Se corrigió **antes de entrenar ningún modelo adicional y sin mirar la prueba**. La línea base anterior se conserva en `experiments/historico/`. Aprobado por el equipo el 2026-11-22 [CONFIRMAR integrante]. |
