"""Línea base (Ruta C): clasificador mayoritario y regresión logística SIN ajuste de hiperparámetros.

Uso:
  python -m src.baseline                 # entrena en train, evalúa en VALIDACIÓN (hito 2)
  python -m src.baseline --final         # entrena en train+val, evalúa en TEST (solo hito 3, UNA vez)
  python -m src.baseline --feature-set contexto|reporte|hipotesis

Las métricas (F1 de la clase "con víctimas", F1 macro, ROC-AUC, PR-AUC, matriz de confusión)
están declaradas en docs/evaluacion_predeclarada.md ANTES de ejecutar este script.
"""
import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as C


def split_columns(feature_set: str, df: pd.DataFrame):
    cols = [c for c in C.FEATURE_SETS[feature_set] if c in df.columns]
    leak = set(cols) & set(C.LEAKY + C.ID_COLS)
    assert not leak, f"FUGA DE DATOS: columnas prohibidas como feature: {leak}"
    cat_all = set(C.CAT_CONTEXTO + C.CAT_REPORTE + C.CAT_HIPOTESIS)
    return [c for c in cols if c not in cat_all], [c for c in cols if c in cat_all]


def make_pipeline(model, num_cols, cat_cols) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="desconocido")),
            ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)),
        ]), cat_cols),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def prep_X(df, num_cols, cat_cols):
    X = df[num_cols + cat_cols].copy()
    X[cat_cols] = X[cat_cols].astype(object)
    return X


def evaluate(y, proba, pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)), "prevalencia_victimas": round(float(np.mean(y)), 4),
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1_victimas": round(f1_score(y, pred, zero_division=0), 4),
        "f1_macro": round(f1_score(y, pred, average="macro", zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4) if both else None,
        "pr_auc": round(average_precision_score(y, proba), 4) if both else None,
        "matriz_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def subgroup_report(df_eval, y, pred, col, min_n=30) -> dict:
    out = {}
    for g, idx in df_eval.groupby(col).groups.items():
        pos = df_eval.index.get_indexer(idx)
        if len(pos) < min_n:
            continue
        yy, pp = y[pos], pred[pos]
        out[str(g)] = {
            "n": int(len(pos)), "prevalencia": round(float(yy.mean()), 3),
            "recall": round(recall_score(yy, pp, zero_division=0), 3),
            "f1_victimas": round(f1_score(yy, pp, zero_division=0), 3),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=C.CLEAN_CSV)
    ap.add_argument("--outdir", type=Path, default=C.EXPERIMENTS)
    ap.add_argument("--feature-set", default=C.PRIMARY_FEATURE_SET, choices=list(C.FEATURE_SETS))
    ap.add_argument("--final", action="store_true", help="Evalúa en TEST. Úselo una sola vez, en el hito 3.")
    a = ap.parse_args()

    df = pd.read_csv(a.processed)
    num_cols, cat_cols = split_columns(a.feature_set, df)

    if a.final:
        train = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
        evalset = df[df["split"] == "test"].reset_index(drop=True)
        tag = "test"
    else:
        train = df[df["split"] == "train"].reset_index(drop=True)
        evalset = df[df["split"] == "val"].reset_index(drop=True)
        tag = "val"
    if train.empty or evalset.empty:
        sys.exit(f"Partición vacía (train={len(train)}, {tag}={len(evalset)}). Revise años en src/config.py.")

    Xtr, ytr = prep_X(train, num_cols, cat_cols), train["victimas"].to_numpy()
    Xev, yev = prep_X(evalset, num_cols, cat_cols), evalset["victimas"].to_numpy()

    models = {
        "mayoritario": DummyClassifier(strategy="most_frequent"),
        "regresion_logistica": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=C.SEED),
    }
    results, preds = {}, {}
    for name, m in models.items():
        pipe = make_pipeline(m, num_cols, cat_cols).fit(Xtr, ytr)
        proba = pipe.predict_proba(Xev)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = evaluate(yev, proba, pred)
        preds[name] = (proba, pred)

    lr_pred = preds["regresion_logistica"][1]
    subgroups = {col: subgroup_report(evalset, yev, lr_pred, col) for col in ("franja", "localidad") if col in evalset}

    out = {
        "fecha_ejecucion_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluado_en": tag, "semilla": C.SEED, "conjunto_de_features": a.feature_set,
        "features_numericas": num_cols, "features_categoricas": cat_cols,
        "n_entrenamiento": int(len(train)), "n_evaluacion": int(len(evalset)),
        "anios_entrenamiento": sorted(map(int, train["anio"].unique())),
        "anios_evaluacion": sorted(map(int, evalset["anio"].unique())),
        "umbral_decision": 0.5,
        "hiperparametros": {"regresion_logistica": {"C": 1.0, "class_weight": "balanced", "ajustados": False}},
        "modelos": results, "subgrupos_regresion_logistica": subgroups,
        "entorno": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                    "pandas": pd.__version__, "numpy": np.__version__},
    }
    a.outdir.mkdir(parents=True, exist_ok=True)
    (a.outdir / f"baseline_{tag}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # Predicciones fila a fila: insumo del análisis de errores (hito 3)
    pr = evalset[[c for c in ("ipat", "fecha_dt", "gravedad_cod", "victimas", "barrio", "clase_accidente", "franja") if c in evalset]].copy()
    pr["proba_victimas"] = np.round(preds["regresion_logistica"][0], 4)
    pr["pred"] = lr_pred
    pr["tipo_error"] = np.select([(pr.victimas == 1) & (pr.pred == 0), (pr.victimas == 0) & (pr.pred == 1)], ["FN", "FP"], "ok")
    pr.to_csv(a.outdir / f"baseline_{tag}_predicciones.csv", index=False)

    print(f"Línea base evaluada en {tag.upper()}  (train n={len(train)}, eval n={len(evalset)}, features='{a.feature_set}')")
    for name, r in results.items():
        print(f"  {name:20s} F1(víct)={r['f1_victimas']:.3f}  F1macro={r['f1_macro']:.3f}  AUC={r['roc_auc']}  PR-AUC={r['pr_auc']}")
    print("Guardado en", a.outdir / f"baseline_{tag}.json")


if __name__ == "__main__":
    main()
