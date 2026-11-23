"""Genera el modelo que usa el prototipo (app/app.py) SIN tocar la prueba.

Uso:  python -m src.export_model
Entrena la configuración de experiments/modelo_elegido.json con 2020-2024 y la semilla SEED (exactamente lo mismo que
hizo src/final.py antes de la evaluación única) y guarda models/modelo_final.joblib con lo que la app necesita:
pipeline, umbral, categorías vistas, coordenadas medianas por barrio y la curva precisión/recall de VALIDACIÓN 2024
(para predecir el efecto de mover el umbral). No lee 2025.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src import config as C
from src.baseline import prep_X, split_columns
from src.evaluacion import metricas
from src.train import construir

MODELO_PATH = C.ROOT / "models" / "modelo_final.joblib"


def curva_validacion(pred_val: pd.DataFrame, extra=()) -> list:
    umbrales = sorted(set(np.round(np.arange(0.02, 0.62, 0.02), 2).tolist()) | {round(float(u), 4) for u in extra})
    y, p = pred_val["fatal"].to_numpy(), pred_val["proba_fatal"].to_numpy()
    filas = []
    for u in umbrales:
        m = metricas(y, p, float(u))
        filas.append({"umbral": float(u), "precision": m["precision"], "recall": m["recall"], "tasa_alerta": m["tasa_alerta"]})
    return filas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=C.CLEAN_CSV)
    ap.add_argument("--outdir", type=Path, default=C.EXPERIMENTS)
    ap.add_argument("--salida", type=Path, default=MODELO_PATH)
    a = ap.parse_args()

    el = json.loads((a.outdir / "modelo_elegido.json").read_text(encoding="utf-8"))
    df = pd.read_csv(a.processed, dtype={"localidad": str})
    train = df[df["split"].isin(["train", "val"])].reset_index(drop=True)     # 2020-2024; la prueba no se lee
    assert not train["anio"].isin(C.TEST_YEARS).any()
    num, cat = split_columns(el["conjunto_de_features"], train)
    pipe = construir(el["modelo"], el["hiperparametros"], C.SEED, num, cat)
    pipe.fit(prep_X(train, num, cat), train["fatal"].to_numpy())

    coords = train.dropna(subset=["lat", "lon"]).groupby("barrio")[["lat", "lon"]].median().round(6)
    barrios = train["barrio"].value_counts()
    paquete = {
        "pipeline": pipe, "eleccion": el, "entrenado_con": "2020-2024", "semilla": C.SEED,
        "features_numericas": num, "features_categoricas": cat,
        "categorias": {c: sorted(map(str, train[c].dropna().unique())) for c in cat if c != "barrio"},
        "barrios": list(barrios.index),                          # ordenados por frecuencia
        "barrio_coords": {b: (float(r.lat), float(r.lon)) for b, r in coords.iterrows()},
        "coords_mediana": (float(train["lat"].median()), float(train["lon"].median())),
        "localidad_nombres": C.LOCALIDAD_NOMBRES,
        "curva_validacion_2024": curva_validacion(pd.read_csv(a.outdir / "modelo_val_predicciones.csv"), extra=[el["umbral"]]),
        "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versiones": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(paquete, a.salida)
    print(f"Modelo {el['modelo']} entrenado con 2020-2024 (n={len(train)}); umbral {el['umbral']:.4f}. Guardado en {a.salida}")


if __name__ == "__main__":
    main()
