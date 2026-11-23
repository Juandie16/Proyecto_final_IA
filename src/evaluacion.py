"""Utilidades de evaluación compartidas por train.py, final.py y responsible.py.

- umbral_por_recall: el MAYOR umbral cuyo recall es >= objetivo (regla predeclarada del punto de operación).
- metricas: PR-AUC, ROC-AUC, precisión, recall, F1 de la clase fatal, F1 macro y matriz de confusión en un umbral.
- bootstrap_ic: intervalos de confianza del 95 % por remuestreo (percentiles).
- subgrupos: recall y precisión por subgrupo elegible (n >= 30 y >= 10 fatales), con IC por bootstrap.
"""
import math

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)

from src import config as C


def umbral_por_recall(y, proba, objetivo: float = C.RECALL_OBJETIVO) -> float:
    """Mayor t tal que recall(proba >= t) >= objetivo. Se elige SOLO en validación."""
    y, proba = np.asarray(y), np.asarray(proba)
    pos = np.sort(proba[y == 1])[::-1]
    if len(pos) == 0:
        raise ValueError("No hay positivos para fijar el umbral.")
    k = math.ceil(objetivo * len(pos))  # hay que capturar al menos k positivos
    return float(pos[k - 1])


def metricas(y, proba, umbral: float) -> dict:
    y, proba = np.asarray(y), np.asarray(proba)
    pred = (proba >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    ambas = len(np.unique(y)) == 2
    return {
        "n": int(len(y)), "n_fatal": int(y.sum()), "prevalencia": round(float(y.mean()), 4),
        "umbral": round(float(umbral), 6),
        "pr_auc": round(float(average_precision_score(y, proba)), 4) if ambas else None,
        "roc_auc": round(float(roc_auc_score(y, proba)), 4) if ambas else None,
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1_fatal": round(float(f1_score(y, pred, zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
        "tasa_alerta": round(float(pred.mean()), 4),
        "matriz_confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def bootstrap_ic(y, proba, umbral: float, n_boot: int = 1000, seed: int = C.SEED,
                 claves=("pr_auc", "roc_auc", "precision", "recall", "f1_fatal")) -> dict:
    """IC 95 % por percentiles. Remuestreo con reemplazo de filas; se omiten réplicas sin las dos clases."""
    y, proba = np.asarray(y), np.asarray(proba)
    rng = np.random.default_rng(seed)
    acum = {k: [] for k in claves}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        pp = proba[idx]
        pred = (pp >= umbral).astype(int)
        vals = {
            "pr_auc": lambda: average_precision_score(yy, pp),
            "roc_auc": lambda: roc_auc_score(yy, pp),
            "precision": lambda: precision_score(yy, pred, zero_division=0),
            "recall": lambda: recall_score(yy, pred, zero_division=0),
            "f1_fatal": lambda: f1_score(yy, pred, zero_division=0),
        }
        for k in claves:
            acum[k].append(vals[k]())
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in acum.items()}


def subgrupos(df_eval, y, proba, umbral: float, col: str, n_boot: int = 1000, seed: int = C.SEED,
              min_n: int = C.SUBGRUPO_MIN_N, min_pos: int = C.SUBGRUPO_MIN_FATALES) -> dict:
    """Recall y precisión por subgrupo elegible, con IC 95 % por bootstrap dentro del subgrupo."""
    y, proba = np.asarray(y), np.asarray(proba)
    vals = df_eval[col].astype(str).to_numpy()
    out = {}
    for g in sorted(set(vals)):
        m = vals == g
        if m.sum() < min_n or y[m].sum() < min_pos:
            continue
        r = metricas(y[m], proba[m], umbral)
        r["ic95"] = bootstrap_ic(y[m], proba[m], umbral, n_boot=n_boot, seed=seed, claves=("recall", "precision"))
        out[g] = {k: r[k] for k in ("n", "n_fatal", "prevalencia", "recall", "precision", "f1_fatal", "ic95")}
    return out


def brecha_subgrupos(sub: dict, recall_global: float, tolerancia: float = 0.15) -> dict:
    """Criterio 4: ningún subgrupo con recall más de `tolerancia` por debajo del global."""
    peores = {f"{col}={g}": round(recall_global - v["recall"], 4)
              for col, grupos in sub.items() for g, v in grupos.items()}
    fallan = {k: d for k, d in peores.items() if d > tolerancia}
    return {"recall_global": recall_global, "tolerancia": tolerancia, "brecha_por_subgrupo": peores,
            "subgrupos_que_fallan": fallan, "cumple": len(fallan) == 0}
