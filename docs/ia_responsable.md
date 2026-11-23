# IA responsable: riesgos, medición y mitigaciones

> Fuente de cada número: `experiments/subgrupos_final.json` (prueba 2025, predicciones de la evaluación única), salvo que se
> indique otra. Generado por `python -m src.responsible`. Nombres de localidad: **[CONFIRMAR]** (1 = Histórica y del Caribe
> Norte, 2 = De la Virgen y Turística, 3 = Industrial y de la Bahía).

## 1. El modelo no decide si hay víctimas; solo estima la gravedad
- **Riesgo.** El dataset solo trae siniestros con víctimas (el portal omite los de solo daños). Usarlo con cualquier siniestro
  daría probabilidades sin sentido.
- **Medición.** `gravedad` toma solo dos valores («Con heridos», «Con muertos»): `experiments/data_summary.json`.
- **Mitigación.** El prototipo lo dice en la portada y en cada resultado; el informe delimita la población objetivo.

## 2. Recall desigual por franja horaria (criterio 4)
- **Medición (prueba).** Recall global 0,470. Por franja: madrugada 0,583, mañana 0,391, noche 0,564 y **tarde 0,118**
  (IC 95 % 0,00–0,31; solo 17 fatales). La tarde es el único subgrupo que incumple (brecha 0,352 > 0,15). La línea base
  también la incumple, con una brecha menor (0,151). En validación el subgrupo que fallaba era la mañana (0,21).
- **Lectura.** El subgrupo que falla cambia de un año a otro y sus IC son anchos: con 17–23 fatales por franja no se puede
  separar un sesgo estable del ruido. Lo estable es que el modelo asigna probabilidades bajas a los siniestros diurnos y
  pierde los fatales de esas horas (ver `experiments/analisis_errores_resumen.json`, FN por franja).
- **Mitigación propuesta.** (a) Reportar siempre el recall por franja con su IC, no solo el global. (b) Trabajo futuro:
  umbrales por franja elegidos en validación, o reentrenar con variables de la víctima y del vehículo. (c) En el prototipo,
  advertir que en horario diurno el modelo detecta menos casos fatales. [CONFIRMAR que el equipo adopta estas mitigaciones]

## 3. Riesgo de estigmatizar barrios o localidades
- **Medición.** Tasa de alerta frente a tasa real en barrios con n ≥ 30 en prueba. Mamonal recibe alerta en el 47,5 % de sus
  siniestros (1,9 veces la tasa global de alerta) con una tasa real de 11,5 %; El Bosque, 31,7 % de alertas con 5,0 % real.
  Entre localidades, la probabilidad media sigue a la tasa real (calibración en conjunto razonable).
- **Lectura.** Las coordenadas y el barrio pesan en el modelo (`lat` es la 2.ª variable por importancia en validación). Un uso
  ingenuo podría leerse como «este barrio es peligroso» cuando refleja tipo de vía y tráfico (p. ej., zona industrial con
  carga pesada en Mamonal), no a sus habitantes.
- **Mitigación.** El prototipo no muestra mapas de riesgo por barrio ni rankings; la alerta es por siniestro, para priorizar
  la respuesta, nunca para asignar vigilancia o recursos a barrios. El informe lo declara como uso fuera de alcance.

## 4. Calidad de los datos: `muertos` en 2025, sub-registro y cambios de codificación
- **Medición.** 282 incoherencias entre `gravedad` y `muertos` en 2025 (`experiments/calidad_etiqueta_por_anio.json`);
  codificación de «sin dato» distinta por año (`experiments/exploracion.json`).
- **Riesgo.** La etiqueta de 2025 puede tener errores; los muertos posteriores al informe (fallecimiento en el hospital
  días después) pueden no reflejarse en `gravedad` (sub-registro) [VERIFICAR con el DATT cuándo se actualiza `gravedad`].
- **Mitigación.** Etiqueta tomada de `gravedad` (coherente con cifras oficiales hasta 2024); `muertos` nunca se usa;
  codificaciones unificadas (v3.1); consulta al DATT como trabajo futuro.

## 5. Herramienta de apoyo, nunca de reemplazo de la decisión humana
- **Medición.** En prueba, de cada 100 alertas unas 16 son fatales (precisión 0,157) y se escapan 53 de cada 100 fatales
  (recall 0,470): `experiments/resultado_final.json`.
- **Mitigación.** El prototipo muestra la probabilidad y la alerta como apoyo para priorizar; no cancela ni reduce la
  atención de ningún caso marcado como «no alerta». Se recomienda no desplegarlo operativamente sin cumplir los criterios.

## 6. Deriva temporal y calibración (riesgo encontrado en la prueba)
- **Medición.** Con el umbral de validación, la tasa de alerta bajó de 30,3 % (2024) a 24,7 % (2025) y el recall de 0,60 a
  0,47; la tasa real de atropellos bajó de 13,7 % a 9,5 % y la de la tarde de 8,9 % a 4,1 %
  (`experiments/analisis_errores_resumen.json`, `deriva_val_prueba`).
- **Mitigación.** Monitorear recall y tasa de alerta por año; recalibrar el umbral con datos recientes antes de usarlo en un
  año nuevo; reportarlo como limitación.
