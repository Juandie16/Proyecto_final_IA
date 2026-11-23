# Predicción de siniestros viales fatales en Cartagena (Ruta C: aprendizaje automático)

Proyecto final de Inteligencia Artificial. **Pregunta:** dado un siniestro vial **con víctimas** en Cartagena, ¿puede un
modelo estimar con el informe inicial (cuándo, dónde, clase de accidente, con qué chocó) si será **fatal**, mejor que una
línea base, y con qué diferencias por franja horaria y localidad?

**Estado: hito 3.** Datos, línea base, 3 modelos con ajuste de hiperparámetros, evaluación única en prueba, análisis de
errores, IA responsable con medición, prototipo Streamlit e informe IEEE (`informe/`).

## Ejecución (Windows/PowerShell, Python 3.13; 4 comandos)
Requisito previo: crear y activar un entorno virtual (`python -m venv .venv` y `.venv\Scripts\Activate.ps1`).
```powershell
python -m pip install -r requirements.txt   # 1. dependencias con versiones fijas (usar siempre "python -m pip")
python -m src.data_download                 # 2. descarga el dataset del DATT (CC-BY) y registra su SHA-256
python -m src.pipeline                      # 3. limpieza, calidad, línea base, exploración, modelos, errores, IA responsable (~5 min)
python -m streamlit run app/app.py          # 4. prototipo en http://localhost:8501
```
- Si el paso 2 falla (red o portal caído), descargue el XLSX desde la URL que imprime el script y guárdelo en
  `data/raw/accidentes_cartagena_2020_2025.xlsx`; el paso 3 lo lee igual.
- El paso 3 compara el SHA-256 de los datos descargados con el de `data/raw/SOURCE.json` versionado y **avisa si el portal
  actualizó el archivo** (en ese caso los números pueden diferir de los del informe).
- `python -m src.pipeline --rapido` hace una búsqueda reducida (1 min) que **no** reproduce el informe.
- Pruebas (datos sintéticos, sin red): `python -m unittest discover -s tests -v`

### Evaluación en prueba (2025): ya hecha, una sola vez
`python -m src.final --confirmo-evaluacion-unica` entrenó el modelo elegido con 2020–2024 y lo evaluó en 2025 una única vez
(commit `e58aacf`). El script se niega a repetirlo si existe `experiments/resultado_final.json`. Sus predicciones
(`experiments/final_predicciones.csv`) están versionadas y alimentan `src.errors` y `src.responsible`.

## De dónde sale cada resultado del informe
| Resultado | Archivo en `experiments/` | Lo genera |
|---|---|---|
| Datos, limpieza y partición | `data_summary.json` | `src.preprocess` |
| Calidad de la etiqueta por año | `calidad_etiqueta_por_anio.json` | `src.data_quality` |
| Exploración y figuras 1–4 | `exploracion.json`, `figuras/fig1…fig4` | `src.explore` (y `notebooks/01_exploracion.ipynb`) |
| Línea base en validación | `baseline_val.json` | `src.baseline` |
| Búsqueda de hiperparámetros, semillas, ablación, umbral | `modelos_val.json`, `tabla_comparativa.csv`, `modelo_elegido.json` | `src.train` |
| Resultado final en prueba y criterios | `resultado_final.json`, `final_predicciones.csv` | `src.final` (una vez) |
| Análisis de errores | `analisis_errores.csv`, `analisis_errores_resumen.json` | `src.errors` |
| Subgrupos, barrios y figura 5 | `subgrupos_final.json`, `figuras/fig5_recall_subgrupos` | `src.responsible` |
| Tablas del informe en LaTeX | `informe/tablas/*.tex` | `src.make_report_tables` |
| Línea base antes de la corrección v3.1 | `historico/` | (histórico) |

## Estructura
| Ruta | Contenido |
|---|---|
| `src/` | `config.py` (semilla, años, features), `data_download`, `preprocess`, `data_quality`, `baseline`, `explore`, `train`, `evaluacion`, `final`, `errors`, `responsible`, `export_model`, `app_utils`, `pipeline`, `make_report_tables`, `make_hito2` |
| `app/app.py` | Prototipo Streamlit (usa `models/modelo_final.joblib`, que genera `src.export_model`; `models/` no se versiona) |
| `notebooks/` | `01_exploracion.ipynb` (ejecutado) |
| `experiments/` | Resultados que respaldan cada tabla y figura, con semilla, versiones y fecha |
| `docs/` | `evaluacion_predeclarada.md` (métricas y criterios antes de experimentar), `ia_responsable.md`, `bitacora_ia.md` |
| `informe/` | Informe IEEE en LaTeX (IEEEtran), listo para Overleaf |
| `data/` | Procedencia, licencia y codificaciones (`README.md`); los datos se regeneran con los scripts |
| `tests/` | Pruebas con datos sintéticos |

## Reproducibilidad y buenas prácticas
- Semilla única `SEED = 42` en `src/config.py` (las 5 semillas de la búsqueda son 42–46); cada JSON registra semilla, versiones y fecha.
- Partición **temporal**: entrenamiento 2020–2023, validación 2024, prueba 2025. Ajuste con origen móvil (2022, 2023, 2024).
- Etiqueta `fatal` = «Con muertos» frente a «Con heridos» (el dataset solo trae siniestros con víctimas).
- `muertos`, `heridos`, `gravedad`, `gravedad_cod` y `fatal` nunca son features; cada script lo comprueba con un `assert`.
- Criterios de éxito declarados antes de entrenar (`docs/evaluacion_predeclarada.md`, v3.1).

## Datos y licencia
DATT, Alcaldía de Cartagena de Indias, licencia CC-BY. Detalles y codificaciones corregidas en `data/README.md`.

## Uso de IA generativa
Declarado en `docs/bitacora_ia.md`.
