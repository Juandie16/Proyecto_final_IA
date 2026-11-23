"""Fase 3 (hito 3): evaluación ÚNICA en el conjunto de prueba (2025).

Uso:  python -m src.final --confirmo-evaluacion-unica

Requisitos (regla 5 de CONTEXTO_PROYECTO.md):
  * experiments/modelo_elegido.json existe (lo escribe src/train.py) y está versionado sin cambios pendientes.
  * No existe experiments/resultado_final.json (la prueba se evalúa una sola vez). --forzar solo para repetir
    exactamente la misma corrida, y queda registrado.
Procedimiento:
  * El modelo elegido y la línea base se reentrenan con 2020-2024 y la semilla SEED; se evalúan en 2025.
  * El umbral es el elegido en validación (modelo_elegido.json) y NO se toca. La línea base se reporta con su
    umbral 0,5 (como en baseline.py) y con su propio umbral de validación (misma regla de recall >= 0,60).
  * IC 95 % por bootstrap (1.000 remuestreos) para PR-AUC, ROC-AUC, precisión, recall y F1.
Salidas: experiments/resultado_final.json, experiments/final_predicciones.csv, models/modelo_final.joblib
"""
import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src import config as C
from src.evaluacion import bootstrap_ic, brecha_subgrupos, metricas, subgrupos
from src.train import LINEA_BASE, ajustar_y_predecir

MODELS_DIR = C.ROOT / "models"


def git_limpio(rutas) -> tuple[bool, str]:
    try:
        r = subprocess.run(["git", "status", "--porcelain", "--", *map(str, rutas)], cwd=C.ROOT,
                           capture_output=True, text=True, check=True)
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.ROOT, capture_output=True, text=True, check=True)
        return (r.stdout.strip() == "", h.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return (False, "sin git")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=C.CLEAN_CSV)
    ap.add_argument("--outdir", type=Path, default=C.EXPERIMENTS)
    ap.add_argument("--confirmo-evaluacion-unica", action="store_true")
    ap.add_argument("--forzar", action="store_true", help="Repetir la evaluación (queda registrado).")
    ap.add_argument("--sin-git", action="store_true", help="Solo para pruebas con datos sintéticos.")
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    a = ap.parse_args()

    if not a.confirmo_evaluacion_unica:
        raise SystemExit("La prueba se evalúa UNA sola vez. Ejecute con --confirmo-evaluacion-unica cuando el equipo lo decida.")
    destino = a.outdir / "resultado_final.json"
    if destino.exists() and not a.forzar:
        raise SystemExit(f"{destino} ya existe: la prueba ya se evaluó. No se repite.")
    elegido = json.loads((a.outdir / "modelo_elegido.json").read_text(encoding="utf-8"))
    if elegido.get("modo_rapido"):
        raise SystemExit("modelo_elegido.json viene de una corrida --rapido; ejecute python -m src.train completo.")
    limpio, commit = (True, "sin git") if a.sin_git else git_limpio(["src", "experiments/modelo_elegido.json"])
    if not limpio:
        raise SystemExit("Hay cambios sin commit en src/ o en modelo_elegido.json. Haga commit antes de evaluar en prueba.")

    val = json.loads((a.outdir / "modelos_val.json").read_text(encoding="utf-8"))
    df = pd.read_csv(a.processed, dtype={"localidad": str})
    train = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    y = test["fatal"].to_numpy()
    fs = elegido["conjunto_de_features"]

    pipe, p_mod = ajustar_y_predecir(elegido["modelo"], elegido["hiperparametros"], C.SEED, train, test, fs)
    _, p_base = ajustar_y_predecir(LINEA_BASE, {"C": 1.0}, C.SEED, train, test, fs)
    u_mod = float(elegido["umbral"])
    u_base_val = float(val["validacion_2024"][LINEA_BASE]["por_semilla"][0]["op"]["umbral"])

    res_mod = metricas(y, p_mod, u_mod)
    res_mod["ic95"] = bootstrap_ic(y, p_mod, u_mod)
    res_base05 = metricas(y, p_base, 0.5)
    res_base05["ic95"] = bootstrap_ic(y, p_base, 0.5)
    res_base_op = metricas(y, p_base, u_base_val)
    res_base_op["ic95"] = bootstrap_ic(y, p_base, u_base_val)
    sub = {col: subgrupos(test, y, p_mod, u_mod, col) for col in ("franja", "localidad")}
    brecha = brecha_subgrupos(sub, res_mod["recall"])

    vv = val["validacion_2024"][elegido["modelo"]]
    ref_val = json.loads((a.outdir / "baseline_val.json").read_text(encoding="utf-8"))["modelos"]["regresion_logistica"]["pr_auc"]
    criterios = [
        {"criterio": "1. PR-AUC >= línea base + 0,05 (validación, media de 5 semillas)",
         "valor": vv["pr_auc"]["media"], "referencia": round(ref_val + 0.05, 4),
         "cumple": vv["pr_auc"]["media"] >= ref_val + 0.05, "donde": "validación 2024"},
        {"criterio": "2a. ROC-AUC >= 0,70 (validación, media de 5 semillas)", "valor": vv["roc_auc"]["media"],
         "referencia": 0.70, "cumple": vv["roc_auc"]["media"] >= 0.70, "donde": "validación 2024"},
        {"criterio": "2b. ROC-AUC >= 0,70 (prueba)", "valor": res_mod["roc_auc"], "ic95": res_mod["ic95"]["roc_auc"],
         "referencia": 0.70, "cumple": res_mod["roc_auc"] >= 0.70, "donde": "prueba 2025"},
        {"criterio": "3. Precisión >= 0,20 con recall >= 0,60 (umbral de validación)",
         "valor": {"precision": res_mod["precision"], "recall": res_mod["recall"]},
         "ic95": {"precision": res_mod["ic95"]["precision"], "recall": res_mod["ic95"]["recall"]},
         "referencia": {"precision": 0.20, "recall": 0.60},
         "cumple": res_mod["precision"] >= 0.20 and res_mod["recall"] >= 0.60, "donde": "prueba 2025"},
        {"criterio": "4. Ningún subgrupo elegible con recall > 0,15 por debajo del global",
         "valor": brecha["brecha_por_subgrupo"], "referencia": 0.15, "cumple": bool(brecha["cumple"]),
         "donde": "prueba 2025"},
    ]

    out = {
        "fecha_ejecucion_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit_git": commit, "repetida_con_forzar": bool(a.forzar and destino.exists()),
        "semilla": C.SEED, "entrenado_con": "2020-2024", "evaluado_en": "prueba 2025",
        "modelo": elegido, "n_entrenamiento": int(len(train)), "n_prueba": int(len(test)),
        "resultados": {elegido["modelo"]: res_mod, "linea_base_umbral_0_5": res_base05,
                       "linea_base_umbral_validacion": {**res_base_op, "umbral_validacion": u_base_val}},
        "subgrupos": sub, "brecha_subgrupos": brecha, "criterios": criterios,
        "cumple_todos": all(c["cumple"] for c in criterios),
        "entorno": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__,
                    "numpy": np.__version__},
    }
    destino.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o)),
                       encoding="utf-8")

    pr = test[["ipat", "anio", "fatal", "hora", "franja", "dia_semana", "localidad", "barrio", "tipo_via",
               "clase_accidente", "choque_con", "objeto_fijo", "hipotesis_1"]].copy()
    pr["proba_fatal"] = np.round(p_mod, 5)
    pr["pred"] = (p_mod >= u_mod).astype(int)
    pr["tipo_error"] = np.select([(pr.fatal == 1) & (pr.pred == 0), (pr.fatal == 0) & (pr.pred == 1)], ["FN", "FP"], "ok")
    pr["proba_linea_base"] = np.round(p_base, 5)
    pr.to_csv(a.outdir / "final_predicciones.csv", index=False)

    a.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "eleccion": elegido, "entrenado_con": "2020-2024"}, a.models_dir / "modelo_final.joblib")

    print(f"PRUEBA 2025 (n={len(test)}, fatales={int(y.sum())}) - modelo {elegido['modelo']}, umbral {u_mod:.4f}")
    for nombre, r in out["resultados"].items():
        print(f"  {nombre:32s} PR-AUC={r['pr_auc']}  ROC-AUC={r['roc_auc']}  precisión={r['precision']}  recall={r['recall']}")
    for c in criterios:
        print(f"  [{'OK' if c['cumple'] else 'NO'}] {c['criterio']}")
    print("Guardado en", destino)


if __name__ == "__main__":
    main()
