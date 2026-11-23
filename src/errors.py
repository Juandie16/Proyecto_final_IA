"""Fase 4 (hito 3): análisis de errores del modelo elegido en la prueba 2025 (ya evaluada una vez).

Uso:  python -m src.errors [--n-por-tipo 10]
Entradas: experiments/final_predicciones.csv, experiments/modelo_elegido.json, datos procesados.
Salidas:  experiments/analisis_errores.csv          (los N FN y N FP de mayor confianza, con categoría e hipótesis)
          experiments/analisis_errores_resumen.json (conteos por categoría, por clase x franja, deriva val->prueba,
                                                     importancia por permutación en VALIDACIÓN 2024)

Las categorías salen de reglas explícitas (CATEGORIAS) y son reproducibles. Las hipótesis de causa son una PROPUESTA
de la IA que el equipo debe revisar [CONFIRMAR]: no se verifican con los datos, porque las variables que las probarían
(tipo de vehículo, velocidad, edad, uso de casco) no están en el dataset.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src import config as C
from src.baseline import prep_X, split_columns
from src.train import construir

NOCTURNO = {"noche", "madrugada"}

# (código, nombre, condición, hipótesis propuesta). Se asigna la PRIMERA regla que se cumple.
CATEGORIAS = {
    "FN": [
        ("FN1", "Choque entre vehículos de día (mañana/tarde)",
         lambda r: r.clase_accidente == "choque" and r.choque_con == "vehiculo" and r.franja not in NOCTURNO,
         "El informe inicial de un choque diurno entre vehículos es casi idéntico al de miles de choques con solo heridos; "
         "lo que hace fatal el caso (vehículo involucrado, p. ej. moto; velocidad; casco; edad) no está en el dataset."),
        ("FN2", "Choque entre vehículos de noche/madrugada",
         lambda r: r.clase_accidente == "choque" and r.choque_con == "vehiculo",
         "La hora sube la probabilidad, pero un choque vehicular sigue por debajo del umbral; faltan las mismas variables "
         "de la víctima y del vehículo que en FN1."),
        ("FN3", "Atropello o caída de ocupante con baja probabilidad",
         lambda r: r.clase_accidente in ("atropello", "caida ocupante"),
         "Ocurrió en una franja o zona de baja tasa histórica; el modelo pondera más la hora y el lugar que la vulnerabilidad del usuario."),
        ("FN4", "Otros", lambda r: True, "Combinación poco frecuente en entrenamiento."),
    ],
    "FP": [
        ("FP1", "Choque contra objeto fijo",
         lambda r: r.choque_con == "objeto fijo" or r.objeto_fijo not in ("sin_dato",),
         "Choque contra objeto fijo tiene la tasa más alta de los datos (~24 %), pero 3 de cada 4 no son fatales; "
         "el desenlace depende de la velocidad y la protección, que no se registran."),
        ("FP2", "Atropello",
         lambda r: r.clase_accidente == "atropello",
         "Los atropellos tienen ~2 veces la tasa global y el modelo los marca en bloque; en 2025 su tasa observada fue menor "
         "que la histórica. Faltan la edad del peatón y la velocidad."),
        ("FP3", "Choque vehicular de noche/madrugada",
         lambda r: r.clase_accidente == "choque" and r.franja in NOCTURNO,
         "La madrugada tiene la tasa más alta por hora; el modelo convierte ese riesgo de franja en alerta aunque el caso "
         "concreto no lo sea."),
        ("FP4", "Otros (volcamiento, caída de ocupante, choque diurno)", lambda r: True,
         "Clases con tasa alta y pocos casos; el modelo generaliza el riesgo de la clase."),
    ],
}


def categorizar(r, tipo: str):
    for cod, nombre, cond, hip in CATEGORIAS[tipo]:
        if cond(r):
            return cod, nombre, hip
    raise AssertionError("la última regla debe cubrir todo")


def variables_relevantes(r) -> str:
    partes = [f"clase={r.clase_accidente}", f"choque_con={r.choque_con}", f"franja={r.franja} ({int(r.hora)} h)",
              f"localidad={r.localidad}", f"barrio={r.barrio}", f"tipo_via={r.tipo_via}"]
    if r.objeto_fijo != "sin_dato":
        partes.append(f"objeto_fijo={r.objeto_fijo}")
    return "; ".join(partes)


def importancia_validacion(df: pd.DataFrame, elegido: dict, n_repeats: int = 10) -> dict:
    """Importancia por permutación (caída de PR-AUC) del modelo entrenado con 2020-2023, medida en 2024."""
    tr = df[df["split"] == "train"].reset_index(drop=True)
    ev = df[df["split"] == "val"].reset_index(drop=True)
    num, cat = split_columns(elegido["conjunto_de_features"], tr)
    pipe = construir(elegido["modelo"], elegido["hiperparametros"], C.SEED, num, cat)
    pipe.fit(prep_X(tr, num, cat), tr["fatal"].to_numpy())
    X = prep_X(ev, num, cat)
    r = permutation_importance(pipe, X, ev["fatal"].to_numpy(), scoring="average_precision", n_repeats=n_repeats,
                               random_state=C.SEED, n_jobs=1)
    orden = np.argsort(-r.importances_mean)
    return {X.columns[i]: {"media": round(float(r.importances_mean[i]), 4), "sd": round(float(r.importances_std[i]), 4)}
            for i in orden}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=C.CLEAN_CSV)
    ap.add_argument("--outdir", type=Path, default=C.EXPERIMENTS)
    ap.add_argument("--n-por-tipo", type=int, default=10)
    a = ap.parse_args()

    pr = pd.read_csv(a.outdir / "final_predicciones.csv", dtype={"localidad": str, "hipotesis_1": str})
    elegido = json.loads((a.outdir / "modelo_elegido.json").read_text(encoding="utf-8"))
    for col in ("choque_con", "objeto_fijo", "clase_accidente", "franja"):
        pr[col] = pr[col].fillna("sin_dato" if col in ("choque_con", "objeto_fijo") else "desconocido")

    # Categoría para TODOS los errores (para los conteos), y tabla detallada de los de mayor confianza
    err = pr[pr["tipo_error"] != "ok"].copy()
    cats = [categorizar(r, r.tipo_error) for r in err.itertuples()]
    err["categoria_cod"] = [c[0] for c in cats]
    err["categoria"] = [c[1] for c in cats]
    err["hipotesis_causa"] = [c[2] for c in cats]
    fn = err[err.tipo_error == "FN"].nsmallest(a.n_por_tipo, "proba_fatal")   # fatales con la probabilidad más baja
    fp = err[err.tipo_error == "FP"].nlargest(a.n_por_tipo, "proba_fatal")    # no fatales con la probabilidad más alta
    top = pd.concat([fn, fp])
    tabla = pd.DataFrame({
        "ipat": top["ipat"], "real": np.where(top["fatal"] == 1, "fatal", "no fatal"),
        "probabilidad": top["proba_fatal"].round(4), "umbral": round(float(elegido["umbral"]), 4),
        "tipo_error": top["tipo_error"], "categoria_cod": top["categoria_cod"], "categoria": top["categoria"],
        "variables_relevantes": [variables_relevantes(r) for r in top.itertuples()],
        "hipotesis_causa": top["hipotesis_causa"] + " [CONFIRMAR]",
    })
    tabla.to_csv(a.outdir / "analisis_errores.csv", index=False, encoding="utf-8")

    # Deriva entre validación (modelo 2020-23) y prueba (modelo 2020-24): por qué el umbral no se trasladó
    val = pd.read_csv(a.outdir / "modelo_val_predicciones.csv", dtype={"localidad": str})
    u = float(elegido["umbral"])
    deriva = {
        "proba_media": {"val_2024": round(float(val.proba_fatal.mean()), 4), "prueba_2025": round(float(pr.proba_fatal.mean()), 4)},
        "proba_mediana_fatales": {"val_2024": round(float(val[val.fatal == 1].proba_fatal.median()), 4),
                                  "prueba_2025": round(float(pr[pr.fatal == 1].proba_fatal.median()), 4)},
        "tasa_alerta": {"val_2024": round(float((val.proba_fatal >= u).mean()), 4), "prueba_2025": round(float((pr.proba_fatal >= u).mean()), 4)},
        "prevalencia": {"val_2024": round(float(val.fatal.mean()), 4), "prueba_2025": round(float(pr.fatal.mean()), 4)},
        "por_franja": {f: {"proba_media_val": round(float(val[val.franja == f].proba_fatal.mean()), 4),
                           "proba_media_prueba": round(float(pr[pr.franja == f].proba_fatal.mean()), 4),
                           "tasa_real_val": round(float(val[val.franja == f].fatal.mean()), 4),
                           "tasa_real_prueba": round(float(pr[pr.franja == f].fatal.mean()), 4)}
                       for f in ("madrugada", "manana", "tarde", "noche")},
        "por_clase": {c: {"tasa_real_val": round(float(val[val.clase_accidente == c].fatal.mean()), 4),
                          "tasa_real_prueba": round(float(pr[pr.clase_accidente == c].fatal.mean()), 4),
                          "n_prueba": int((pr.clase_accidente == c).sum())}
                      for c in sorted(pr.clase_accidente.unique())},
    }
    df = pd.read_csv(a.processed, dtype={"localidad": str})
    df = df[df["split"].isin(["train", "val"])].reset_index(drop=True)   # importancia solo con validación
    resumen = {
        "fecha_ejecucion_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "semilla": C.SEED,
        "modelo": elegido["modelo"], "umbral": u,
        "n_errores": {"FN": int((err.tipo_error == "FN").sum()), "FP": int((err.tipo_error == "FP").sum())},
        "errores_por_categoria": {f"{t}/{c}": int(n) for (t, c), n in err.groupby(["tipo_error", "categoria_cod"]).size().items()},
        "nombres_categoria": {cod: nombre for tipo in CATEGORIAS.values() for cod, nombre, _, _ in tipo},
        "errores_por_clase_y_tipo": {f"{c}/{t}": int(n) for (c, t), n in pr.groupby(["clase_accidente", "tipo_error"]).size().items()},
        "fn_por_franja": {k: int(v) for k, v in err[err.tipo_error == "FN"].groupby("franja").size().items()},
        "deriva_val_prueba": deriva,
        "importancia_permutacion_val_2024": importancia_validacion(df, elegido),
        "nota": "Categorías por reglas (src/errors.py). Hipótesis = propuesta de la IA [CONFIRMAR]. "
                "La importancia se mide en validación 2024 (caída de PR-AUC al permutar cada variable).",
    }
    (a.outdir / "analisis_errores_resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(tabla)} errores detallados en analisis_errores.csv; categorías:", resumen["errores_por_categoria"])
    print("Importancia (validación):", {k: v["media"] for k, v in list(resumen["importancia_permutacion_val_2024"].items())[:6]})


if __name__ == "__main__":
    main()
