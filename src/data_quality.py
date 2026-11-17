"""Calidad de la etiqueta: coherencia entre `gravedad` (etiqueta) y `muertos` (conteo), por año.

Uso:  python -m src.data_quality [--processed RUTA] [--outdir RUTA]
Salida: experiments/calidad_etiqueta_por_anio.json  (respalda la sección de calidad de datos del informe)

Nota: `muertos` NUNCA se usa como feature (define la etiqueta). Aquí solo se audita.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src import config as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=C.CLEAN_CSV)
    ap.add_argument("--outdir", type=Path, default=C.EXPERIMENTS)
    a = ap.parse_args()

    df = pd.read_csv(a.processed)
    muertos = pd.to_numeric(df["muertos"], errors="coerce").fillna(0)
    heridos = pd.to_numeric(df["heridos"], errors="coerce").fillna(0)
    df["muertos_pos"] = muertos > 0
    df["incoherente"] = df["muertos_pos"] != (df["fatal"] == 1)

    filas = {}
    for anio, g in df.groupby("anio"):
        m, h = muertos[g.index], heridos[g.index]
        filas[int(anio)] = {
            "n": int(len(g)),
            "fatal_por_gravedad": int(g["fatal"].sum()),
            "pct_fatal_por_gravedad": round(float(g["fatal"].mean()) * 100, 1),
            "con_muertos_mayor_0": int(g["muertos_pos"].sum()),
            "pct_con_muertos_mayor_0": round(float(g["muertos_pos"].mean()) * 100, 1),
            "suma_muertos": int(m.sum()),
            "suma_heridos": int(h.sum()),
            "incoherentes": int(g["incoherente"].sum()),
            "gravedad_heridos_pero_muertos_mayor_0": int(((g["fatal"] == 0) & g["muertos_pos"]).sum()),
            "gravedad_muertos_pero_muertos_0": int(((g["fatal"] == 1) & ~g["muertos_pos"]).sum()),
        }
    # En los años incoherentes: ¿se parece `muertos` a `heridos`? (hipótesis de columna mal llenada)
    malos = df[df["incoherente"]]
    out = {"por_anio": filas, "incoherentes_total": int(len(malos))}
    if len(malos):
        out["en_incoherentes_muertos_igual_a_heridos_pct"] = round(float((muertos[malos.index] == heridos[malos.index]).mean()) * 100, 1)

    a.outdir.mkdir(parents=True, exist_ok=True)
    (a.outdir / "calidad_etiqueta_por_anio.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'año':>5} {'n':>6} {'fatal(grav)':>12} {'muertos>0':>10} {'Σmuertos':>9} {'Σheridos':>9} {'incoher.':>9}")
    for y, v in filas.items():
        print(f"{y:>5} {v['n']:>6} {v['fatal_por_gravedad']:>12} {v['con_muertos_mayor_0']:>10} {v['suma_muertos']:>9} {v['suma_heridos']:>9} {v['incoherentes']:>9}")
    if len(malos):
        print(f"\nEn las {len(malos)} filas incoherentes, `muertos` == `heridos` en {out['en_incoherentes_muertos_igual_a_heridos_pct']}% de los casos.")
    print("Guardado en", a.outdir / "calidad_etiqueta_por_anio.json")


if __name__ == "__main__":
    main()
