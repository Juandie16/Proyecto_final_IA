"""Fase 2 (hito 3): modelos propuestos, ajuste de hiperparámetros, semillas, ablación y umbral. SOLO validación.

Uso:
  python -m src.train              # búsqueda completa (varios minutos)
  python -m src.train --rapido     # rejillas mínimas y 2 semillas (pruebas)

Protocolo (docs/evaluacion_predeclarada.md):
  * Validación de origen móvil: 2020-21 -> 2022, 2020-22 -> 2023, 2020-23 -> 2024. 2025 (prueba) NUNCA se lee aquí.
  * Cada configuración se entrena con 5 semillas (SEED, SEED+1, ...); se reporta media ± desviación.
  * Dentro de cada familia, la configuración se elige por la media de PR-AUC en los 3 pliegues.
  * El modelo final es la familia con mayor PR-AUC medio en los 3 pliegues; su umbral es el mayor que logra
    recall >= 0,60 en validación 2024 (modelo entrenado con 2020-23 y semilla SEED).
  * Ablación de conjuntos de features (contexto / reporte / hipotesis) con la configuración elegida, en 2024.
Salidas: experiments/modelos_val.json, experiments/tabla_comparativa.csv, experiments/modelo_elegido.json,
         experiments/modelo_val_predicciones.csv
"""
import argparse
import itertools
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src import config as C
from src.baseline import make_pipeline, prep_X, split_columns
from src.evaluacion import brecha_subgrupos, metricas, subgrupos, umbral_por_recall

PLIEGUES = [((2020, 2021), 2022), ((2020, 2021, 2022), 2023), ((2020, 2021, 2022, 2023), 2024)]
VAL_ANIO = 2024
LINEA_BASE = "regresion_logistica_base"

REJILLAS = {
    # Línea base: misma configuración que src/baseline.py (sin ajuste); se corre aquí solo para tener sus pliegues.
    LINEA_BASE: {"C": [1.0]},
    "regresion_logistica_ajustada": {"C": [0.01, 0.1, 1.0, 10.0]},
    "random_forest": {"min_samples_leaf": [5, 20, 50], "max_features": ["sqrt", 0.3],
                      "class_weight": ["balanced_subsample", None]},
    "hist_gradient_boosting": {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [8, 31], "min_samples_leaf": [20, 80],
                               "class_weight": ["balanced", None]},
}
REJILLAS_RAPIDAS = {
    LINEA_BASE: {"C": [1.0]},
    "regresion_logistica_ajustada": {"C": [0.1, 1.0]},
    "random_forest": {"min_samples_leaf": [20], "max_features": ["sqrt"], "class_weight": ["balanced_subsample"]},
    "hist_gradient_boosting": {"learning_rate": [0.1], "max_leaf_nodes": [8], "min_samples_leaf": [20],
                               "class_weight": ["balanced"]},
}
DETERMINISTAS = {LINEA_BASE, "regresion_logistica_ajustada"}  # lbfgs no depende de la semilla


def construir(familia: str, params: dict, seed: int, num_cols, cat_cols) -> Pipeline:
    if familia in (LINEA_BASE, "regresion_logistica_ajustada"):
        m = LogisticRegression(C=params["C"], class_weight="balanced", max_iter=2000, random_state=seed)
        return make_pipeline(m, num_cols, cat_cols)
    if familia == "random_forest":
        pre = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), cat_cols),
        ])
        m = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1, **params)
        return Pipeline([("pre", pre), ("model", m)])
    if familia == "hist_gradient_boosting":
        # Categóricas nativas: códigos ordinales; categorías con < 20 casos se agrupan; las desconocidas -> NaN.
        pre = ColumnTransformer([
            ("num", "passthrough", num_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                   encoded_missing_value=np.nan, min_frequency=20), cat_cols),
        ])
        mask = [False] * len(num_cols) + [True] * len(cat_cols)
        m = HistGradientBoostingClassifier(max_iter=500, early_stopping=True, validation_fraction=0.15,
                                           n_iter_no_change=20, l2_regularization=1.0, categorical_features=mask,
                                           random_state=seed, **params)
        return Pipeline([("pre", pre), ("model", m)])
    raise ValueError(familia)


def configuraciones(rejilla: dict) -> list:
    claves = list(rejilla)
    return [dict(zip(claves, vals)) for vals in itertools.product(*(rejilla[k] for k in claves))]


def ajustar_y_predecir(familia, params, seed, train, evalset, feature_set):
    num_cols, cat_cols = split_columns(feature_set, train)  # contiene el assert de fuga de datos
    pipe = construir(familia, params, seed, num_cols, cat_cols)
    pipe.fit(prep_X(train, num_cols, cat_cols), train["fatal"].to_numpy())
    return pipe, pipe.predict_proba(prep_X(evalset, num_cols, cat_cols))[:, 1]


def media_sd(vals) -> dict:
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    return {"media": round(float(a.mean()), 4), "sd": round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 4), "n": int(len(a))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=C.CLEAN_CSV)
    ap.add_argument("--outdir", type=Path, default=C.EXPERIMENTS)
    ap.add_argument("--rapido", action="store_true", help="Rejillas mínimas y 2 semillas (solo para pruebas).")
    a = ap.parse_args()
    t0 = time.time()

    df = pd.read_csv(a.processed, dtype={"localidad": str})
    df = df[df["split"].isin(["train", "val"])].reset_index(drop=True)   # la prueba no se carga
    assert not (df["anio"].isin(C.TEST_YEARS)).any(), "La prueba no debe usarse en train.py"
    rejillas = REJILLAS_RAPIDAS if a.rapido else REJILLAS
    semillas = [C.SEED + i for i in range(2 if a.rapido else 5)]
    fs = C.PRIMARY_FEATURE_SET

    datos = {}
    for anios_tr, anio_ev in PLIEGUES:
        tr = df[df["anio"].isin(anios_tr)].reset_index(drop=True)
        ev = df[df["anio"] == anio_ev].reset_index(drop=True)
        if tr.empty or ev.empty or ev["fatal"].nunique() < 2:
            continue
        datos[anio_ev] = (tr, ev)
    if VAL_ANIO not in datos:
        raise SystemExit("No hay datos de validación 2024.")

    # ---------------------------------------------------------------- 1. búsqueda con origen móvil
    busqueda = {}
    for familia, rejilla in rejillas.items():
        sems = semillas[:1] if familia in DETERMINISTAS else semillas
        filas = []
        for params in configuraciones(rejilla):
            por_pliegue = {}
            for anio_ev, (tr, ev) in datos.items():
                pr, roc = [], []
                for s in sems:
                    _, p = ajustar_y_predecir(familia, params, s, tr, ev, fs)
                    m = metricas(ev["fatal"], p, 0.5)
                    pr.append(m["pr_auc"]); roc.append(m["roc_auc"])
                por_pliegue[str(anio_ev)] = {"pr_auc": media_sd(pr), "roc_auc": media_sd(roc)}
            filas.append({"params": params, "pliegues": por_pliegue,
                          "pr_auc_media_pliegues": round(float(np.mean([v["pr_auc"]["media"] for v in por_pliegue.values()])), 4),
                          "roc_auc_media_pliegues": round(float(np.mean([v["roc_auc"]["media"] for v in por_pliegue.values()])), 4)})
            print(f"  {familia:30s} {json.dumps(params):70s} PR-AUC medio pliegues={filas[-1]['pr_auc_media_pliegues']:.4f}", flush=True)
        mejor = max(filas, key=lambda r: r["pr_auc_media_pliegues"])
        busqueda[familia] = {"semillas": sems, "configuraciones": filas, "mejor": mejor}

    # ---------------------------------------------------------------- 2. validación 2024 con semillas y umbral
    tr24, ev24 = datos[VAL_ANIO]
    y24 = ev24["fatal"].to_numpy()
    resumen_val, probas_seed42 = {}, {}
    for familia, b in busqueda.items():
        params = b["mejor"]["params"]
        por_semilla = []
        for s in b["semillas"]:
            _, p = ajustar_y_predecir(familia, params, s, tr24, ev24, fs)
            u = umbral_por_recall(y24, p)
            por_semilla.append({"semilla": s, "op": metricas(y24, p, u), "umbral_05": metricas(y24, p, 0.5)})
            if s == C.SEED:
                probas_seed42[familia] = p
        resumen_val[familia] = {
            "params": params,
            "pr_auc": media_sd([r["op"]["pr_auc"] for r in por_semilla]),
            "roc_auc": media_sd([r["op"]["roc_auc"] for r in por_semilla]),
            "f1_fatal_op": media_sd([r["op"]["f1_fatal"] for r in por_semilla]),
            "precision_op": media_sd([r["op"]["precision"] for r in por_semilla]),
            "recall_op": media_sd([r["op"]["recall"] for r in por_semilla]),
            "f1_fatal_umbral_05": media_sd([r["umbral_05"]["f1_fatal"] for r in por_semilla]),
            "pr_auc_media_pliegues": b["mejor"]["pr_auc_media_pliegues"],
            "por_semilla": por_semilla,
        }

    # ---------------------------------------------------------------- 3. ablación de conjuntos de features (2024)
    ablacion = {}
    for familia, b in busqueda.items():
        ablacion[familia] = {}
        for conj in C.FEATURE_SETS:
            pr, roc, prec = [], [], []
            for s in b["semillas"]:
                _, p = ajustar_y_predecir(familia, b["mejor"]["params"], s, tr24, ev24, conj)
                m = metricas(y24, p, umbral_por_recall(y24, p))
                pr.append(m["pr_auc"]); roc.append(m["roc_auc"]); prec.append(m["precision"])
            ablacion[familia][conj] = {"pr_auc": media_sd(pr), "roc_auc": media_sd(roc), "precision_op": media_sd(prec)}

    # ---------------------------------------------------------------- 4. elección del modelo y del umbral
    candidatos = {f: r for f, r in resumen_val.items() if f != LINEA_BASE}
    elegido = max(candidatos, key=lambda f: candidatos[f]["pr_auc_media_pliegues"])
    p_el = probas_seed42[elegido]
    umbral = umbral_por_recall(y24, p_el)
    m_el = metricas(y24, p_el, umbral)
    sub = {col: subgrupos(ev24, y24, p_el, umbral, col) for col in ("franja", "localidad")}
    ref = resumen_val[LINEA_BASE]
    base_json = json.loads((a.outdir / "baseline_val.json").read_text(encoding="utf-8")) if (a.outdir / "baseline_val.json").exists() else None
    ref_pr = base_json["modelos"]["regresion_logistica"]["pr_auc"] if base_json else ref["pr_auc"]["media"]
    criterios_val = {
        "nota": "Vista previa en VALIDACIÓN 2024. Los criterios 2-4 se confirman una sola vez en prueba (Fase 3).",
        "1_pr_auc_mas_0_05": {"modelo_media_semillas": resumen_val[elegido]["pr_auc"]["media"], "linea_base": ref_pr,
                              "diferencia": round(resumen_val[elegido]["pr_auc"]["media"] - ref_pr, 4),
                              "cumple": resumen_val[elegido]["pr_auc"]["media"] - ref_pr >= 0.05},
        "2_roc_auc_0_70": {"modelo_media_semillas": resumen_val[elegido]["roc_auc"]["media"],
                           "cumple": resumen_val[elegido]["roc_auc"]["media"] >= 0.70},
        "3_precision_0_20_con_recall_0_60": {"precision": m_el["precision"], "recall": m_el["recall"],
                                             "precision_media_semillas": resumen_val[elegido]["precision_op"]["media"],
                                             "cumple": m_el["precision"] >= 0.20 and m_el["recall"] >= C.RECALL_OBJETIVO},
        "4_brecha_subgrupos": brecha_subgrupos(sub, m_el["recall"]),
    }
    criterios_val["4_brecha_subgrupos"]["cumple"] = bool(criterios_val["4_brecha_subgrupos"]["cumple"])

    entorno = {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__,
               "numpy": np.__version__}
    out = {
        "fecha_ejecucion_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "semilla_base": C.SEED, "semillas": semillas, "modo_rapido": a.rapido,
        "conjunto_de_features": fs, "pliegues": [{"entrena": list(t), "valida": v} for t, v in PLIEGUES],
        "rejillas": rejillas, "busqueda": busqueda, "validacion_2024": resumen_val, "ablacion_2024": ablacion,
        "modelo_elegido": elegido, "regla_eleccion": "mayor PR-AUC medio en los 3 pliegues de origen móvil (media de semillas)",
        "umbral_elegido": umbral, "metricas_elegido_semilla_base": m_el, "subgrupos_val_elegido": sub,
        "criterios_vista_previa_val": criterios_val, "duracion_s": round(time.time() - t0, 1), "entorno": entorno,
    }
    a.outdir.mkdir(parents=True, exist_ok=True)
    (a.outdir / "modelos_val.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    filas = []
    for familia, r in resumen_val.items():
        for conj, ab in ablacion[familia].items():
            filas.append({
                "modelo": familia, "conjunto_features": conj, "hiperparametros": json.dumps(r["params"]),
                "semillas": len(busqueda[familia]["semillas"]),
                "pr_auc_media": ab["pr_auc"]["media"], "pr_auc_sd": ab["pr_auc"]["sd"],
                "roc_auc_media": ab["roc_auc"]["media"], "roc_auc_sd": ab["roc_auc"]["sd"],
                "precision_op_media": ab["precision_op"]["media"], "precision_op_sd": ab["precision_op"]["sd"],
                "f1_fatal_op_media": r["f1_fatal_op"]["media"] if conj == fs else None,
                "pr_auc_media_pliegues_origen_movil": r["pr_auc_media_pliegues"] if conj == fs else None,
            })
    pd.DataFrame(filas).to_csv(a.outdir / "tabla_comparativa.csv", index=False)

    eleccion = {
        "fecha_utc": out["fecha_ejecucion_utc"], "modelo": elegido, "hiperparametros": resumen_val[elegido]["params"],
        "conjunto_de_features": fs, "semilla": C.SEED, "umbral": umbral, "recall_objetivo": C.RECALL_OBJETIVO,
        "regla_umbral": "mayor umbral con recall >= 0,60 en validación 2024 (modelo entrenado con 2020-2023)",
        "modo_rapido": a.rapido,
    }
    (a.outdir / "modelo_elegido.json").write_text(json.dumps(eleccion, indent=2, ensure_ascii=False), encoding="utf-8")

    pr = ev24[["ipat", "anio", "fatal", "franja", "localidad", "barrio", "clase_accidente", "choque_con"]].copy()
    pr["proba_fatal"] = np.round(p_el, 5)
    pr["pred"] = (p_el >= umbral).astype(int)
    pr["tipo_error"] = np.select([(pr.fatal == 1) & (pr.pred == 0), (pr.fatal == 0) & (pr.pred == 1)], ["FN", "FP"], "ok")
    pr.to_csv(a.outdir / "modelo_val_predicciones.csv", index=False)

    print(f"\nValidación 2024 (media ± sd de {len(semillas)} semillas; features '{fs}'):")
    for familia, r in resumen_val.items():
        print(f"  {familia:30s} PR-AUC={r['pr_auc']['media']:.4f}±{r['pr_auc']['sd']:.4f}  ROC-AUC={r['roc_auc']['media']:.4f}"
              f"  prec@rec0.6={r['precision_op']['media']:.4f}  PR-AUC pliegues={r['pr_auc_media_pliegues']:.4f}")
    print(f"Elegido: {elegido}  umbral={umbral:.4f}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
