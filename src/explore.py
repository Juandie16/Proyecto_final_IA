"""Exploración descriptiva (Fase 1 del hito 3). La usa notebooks/01_exploracion.ipynb.

Uso:  python -m src.explore
Salidas: experiments/exploracion.json y experiments/figuras/fig*.{pdf,png}

Las tasas de fatalidad se calculan SOLO con 2020-2024 (entrenamiento + validación): la prueba (2025) no se usa
para ninguna decisión. De 2025 solo se reportan conteos de filas y de calidad de datos, que ya están en data_summary.json.
"""
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C

DESCRIPTIVO_SPLITS = ("train", "val")
DIAS = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
FRANJAS = ["madrugada", "manana", "tarde", "noche"]

# Paleta (validada, ver skill dataviz / references/palette.md): serie 1 azul, serie 2 naranja; tinta de texto.
AZUL, NARANJA = "#2a78d6", "#eb6834"
TINTA, TINTA_2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza de Wilson para una proporción (funciona bien con tasas bajas y n chico)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z**2 / n
    centro = (p + z**2 / (2 * n)) / den
    medio = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (max(0.0, centro - medio), min(1.0, centro + medio))


def tasa_por(df: pd.DataFrame, col: str, orden=None, min_n: int = 0) -> pd.DataFrame:
    g = df.groupby(col, observed=True)["fatal"].agg(n="size", fatales="sum").reset_index()
    g = g[g["n"] >= min_n]
    g["tasa"] = g["fatales"] / g["n"]
    ic = [wilson(int(k), int(n)) for k, n in zip(g["fatales"], g["n"])]
    g["ic95_bajo"], g["ic95_alto"] = [a for a, _ in ic], [b for _, b in ic]
    if orden is not None:
        g[col] = pd.Categorical(g[col], categories=orden, ordered=True)
        g = g.sort_values(col)
        g[col] = g[col].astype(str)
    else:
        g = g.sort_values("n", ascending=False)
    return g.reset_index(drop=True)


def _records(t: pd.DataFrame) -> list:
    out = []
    for r in t.to_dict("records"):
        out.append({k: (round(v, 4) if isinstance(v, float) else (int(v) if isinstance(v, (np.integer,)) else v)) for k, v in r.items()})
    return out


def resumen(df: pd.DataFrame, summary: dict) -> dict:
    d = df[df["split"].isin(DESCRIPTIVO_SPLITS)].copy()
    d["dia_nombre"] = d["dia_semana"].map(dict(enumerate(DIAS)))
    d["localidad_nombre"] = d["localidad"].astype(str).map(C.LOCALIDAD_NOMBRES).fillna(d["localidad"].astype(str))
    tablas = {
        "anio": tasa_por(df, "anio"),  # incluye 2025 solo como conteo/prevalencia ya publicada en data_summary.json
        "franja": tasa_por(d, "franja", FRANJAS),
        "hora": tasa_por(d, "hora", sorted(d["hora"].dropna().unique())),
        "dia_semana": tasa_por(d, "dia_nombre", DIAS),
        "clase_accidente": tasa_por(d, "clase_accidente"),
        "choque_con": tasa_por(d, "choque_con"),
        "objeto_fijo": tasa_por(d, "objeto_fijo"),
        "tipo_via": tasa_por(d, "tipo_via"),
        "localidad": tasa_por(d, "localidad_nombre"),
        "es_finde": tasa_por(d, "es_finde"),
    }
    tablas["anio"] = tablas["anio"].sort_values("anio").reset_index(drop=True)
    # Verificación de los códigos de localidad: barrios más frecuentes de cada código
    barrios_loc = {str(k): g["barrio"].value_counts().head(8).to_dict() for k, g in df.groupby(df["localidad"].astype(str))}
    n_barrio_multi = int((df.groupby("barrio")["localidad"].nunique() > 1).sum())
    # Codificación de "sin dato" por año (muestra por qué se unificó)
    sd = {c: {int(k): round(float(v), 3) for k, v in df.groupby("anio")[c].apply(lambda s: (s == "sin_dato").mean()).items()}
          for c in ("choque_con", "objeto_fijo")}
    tv = {int(k): round(float(v), 3) for k, v in df.groupby("anio")["tipo_via"].apply(lambda s: (s == "desconocido").mean()).items()}
    barrio_n = d["barrio"].value_counts()
    return {
        "fecha_ejecucion_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "semilla": C.SEED,
        "periodo_tasas": "2020-2024 (entrenamiento + validación); 2025 excluido salvo conteos",
        "n_periodo_tasas": int(len(d)), "fatales_periodo_tasas": int(d["fatal"].sum()),
        "tasa_global_periodo": round(float(d["fatal"].mean()), 4),
        "tablas": {k: _records(v) for k, v in tablas.items()},
        "faltantes_pct_features": summary.get("faltantes_pct"),
        "fecha_sin_interpretar": summary.get("fecha_sin_interpretar"),
        "hora_sin_interpretar": summary.get("hora_sin_interpretar"),
        "coordenadas_fuera_de_cartagena_o_ausentes": summary.get("coordenadas_fuera_de_cartagena_o_ausentes"),
        "localidad_codigos_barrios_top": barrios_loc,
        "localidad_barrios_con_mas_de_un_codigo": n_barrio_multi,
        "localidad_nombres_propuestos": C.LOCALIDAD_NOMBRES,
        "sin_dato_por_anio": sd,
        "tipo_via_desconocido_por_anio": tv,
        "barrio": {"n_categorias": int(barrio_n.size), "con_al_menos_20_casos": int((barrio_n >= 20).sum()),
                   "pct_filas_en_barrios_con_20_o_mas": round(float(barrio_n[barrio_n >= 20].sum() / len(d)), 3)},
        "hipotesis_1_n_codigos": int(d["hipotesis_1"].nunique()),
        "entorno": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }


# ------------------------------------------------------------------ figuras
def _estilo(ax, titulo, ylabel=None):
    ax.set_title(titulo, loc="left", fontsize=9, color=TINTA, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=TINTA_2)
    ax.tick_params(colors=TINTA_2, labelsize=7, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(TINTA_2)
    ax.grid(axis="y", color=REJILLA, linewidth=0.6)
    ax.set_axisbelow(True)


def _barras_tasa(ax, t, col, titulo, etiquetas=None, ref=None):
    """Barras de tasa con IC 95 %. El n va en la etiqueta del eje; la línea discontinua es la tasa global."""
    x = np.arange(len(t))
    ax.bar(x, t["tasa"] * 100, color=AZUL, width=0.7, edgecolor="white", linewidth=1)
    err = np.vstack([(t["tasa"] - t["ic95_bajo"]) * 100, (t["ic95_alto"] - t["tasa"]) * 100])
    ax.errorbar(x, t["tasa"] * 100, yerr=err, fmt="none", ecolor=TINTA_2, elinewidth=0.8, capsize=2)
    nombres = etiquetas if etiquetas is not None else list(t[col])
    ax.set_xticks(x, [f"{e}\nn={n}" for e, n in zip(nombres, t["n"])], rotation=0)
    if ref is not None:
        ax.axhline(ref * 100, color=TINTA_2, linewidth=0.8, linestyle="--")
    _estilo(ax, titulo, "% fatales (IC 95 %)")


def figuras(df: pd.DataFrame, res: dict, outdir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    outdir.mkdir(parents=True, exist_ok=True)
    T = {k: pd.DataFrame(v) for k, v in res["tablas"].items()}
    glob = res["tasa_global_periodo"]
    hechos = []

    def guardar(fig, nombre):
        for ext in ("pdf", "png"):
            fig.savefig(outdir / f"{nombre}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        hechos.append(nombre)

    # Fig. 1: tasa por año (2025 se marca aparte: es prueba)
    t = T["anio"]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    colores = [AZUL if a <= 2024 else "#a9c9f0" for a in t["anio"]]
    x = np.arange(len(t))
    ax.bar(x, t["tasa"] * 100, color=colores, width=0.7, edgecolor="white")
    err = np.vstack([(t["tasa"] - t["ic95_bajo"]) * 100, (t["ic95_alto"] - t["tasa"]) * 100])
    ax.errorbar(x, t["tasa"] * 100, yerr=err, fmt="none", ecolor=TINTA_2, elinewidth=0.8, capsize=2)
    ax.set_xticks(x, [f"{a}\n{'prueba' if a == 2025 else ('valid.' if a == 2024 else 'entren.')}\nn={n}"
                      for a, n in zip(t["anio"], t["n"])])
    _estilo(ax, "Siniestros fatales por año", "% fatales (IC 95 %)")
    guardar(fig, "fig1_tasa_por_anio")

    # Fig. 2: tasa por hora del día
    t = T["hora"]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    h = t["hora"].astype(float).astype(int)
    ax.fill_between(h, t["ic95_bajo"] * 100, t["ic95_alto"] * 100, color=AZUL, alpha=0.18, linewidth=0)
    ax.plot(h, t["tasa"] * 100, color=AZUL, linewidth=2, marker="o", markersize=3)
    ax.axhline(glob * 100, color=TINTA_2, linewidth=0.8, linestyle="--")
    ax.text(14.5, glob * 100 + 0.4, f"tasa global {glob*100:.1f} %", ha="center", va="bottom", fontsize=6.5,
            color=TINTA_2, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
    for x0, x1, nom in ((-0.5, 5.5, "madrugada"), (5.5, 11.5, "mañana"), (11.5, 17.5, "tarde"), (17.5, 23.5, "noche")):
        ax.text((x0 + x1) / 2, ax.get_ylim()[1] * 0.97, nom, ha="center", va="top", fontsize=6.5, color=TINTA_2)
        ax.axvline(x1, color=REJILLA, linewidth=0.8) if x1 < 23 else None
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hora del día", fontsize=8, color=TINTA_2)
    _estilo(ax, "Tasa de fatalidad por hora (2020–2024)", "% fatales (IC 95 %)")
    guardar(fig, "fig2_tasa_por_hora")

    # Fig. 3: clase de accidente y choque con (dos paneles, misma escala)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True, gridspec_kw={"width_ratios": [5, 3]})
    t = T["clase_accidente"]
    _barras_tasa(axes[0], t, "clase_accidente", "a) Clase de accidente",
                 etiquetas=[s.replace("caida ocupante", "caída de\nocupante") for s in t["clase_accidente"]], ref=glob)
    t = T["choque_con"]
    t = t[t["n"] >= 30].reset_index(drop=True)   # semoviente (n=4) se omite: su IC cubre 0–70 %
    _barras_tasa(axes[1], t, "choque_con", "b) «Choque con» (n ≥ 30)",
                 etiquetas=[s.replace("sin_dato", "sin dato") for s in t["choque_con"]], ref=glob)
    axes[1].set_ylabel("")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=6.5)
    fig.text(0.01, -0.10, f"Tasa de fatalidad 2020–2024; línea discontinua = tasa global {glob*100:.1f} %",
             ha="left", fontsize=7, color=TINTA_2)
    guardar(fig, "fig3_tasa_por_clase_y_choque")

    # Fig. 4: localidad y día de la semana
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True, gridspec_kw={"width_ratios": [3, 7]})
    t = T["localidad"]
    cortos = {"Histórica y del Caribe Norte": "Histórica y\nCaribe Norte", "De la Virgen y Turística": "Virgen y\nTurística",
              "Industrial y de la Bahía": "Industrial\ny Bahía"}
    _barras_tasa(axes[0], t, "localidad_nombre", "a) Localidad", etiquetas=[cortos.get(s, s) for s in t["localidad_nombre"]], ref=glob)
    axes[0].tick_params(axis="x", labelsize=6)
    t = T["dia_semana"]
    _barras_tasa(axes[1], t, "dia_nombre", "b) Día de la semana", etiquetas=[s[:3] for s in t["dia_nombre"]], ref=glob)
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", labelsize=6.5)
    fig.text(0.01, -0.06, f"Tasa de fatalidad 2020–2024; línea discontinua = tasa global {glob*100:.1f} %. "
             "Nombres de localidad: [CONFIRMAR].", ha="left", fontsize=7, color=TINTA_2)
    guardar(fig, "fig4_tasa_por_localidad_y_dia")
    return hechos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=C.CLEAN_CSV)
    ap.add_argument("--outdir", type=Path, default=C.EXPERIMENTS)
    ap.add_argument("--sin-figuras", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.processed, dtype={"localidad": str})
    sp = a.outdir / "data_summary.json"
    summary = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    res = resumen(df, summary)
    if not a.sin_figuras:
        res["figuras"] = figuras(df, res, a.outdir / "figuras")
    a.outdir.mkdir(parents=True, exist_ok=True)
    (a.outdir / "exploracion.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Periodo de tasas: {res['periodo_tasas']}  n={res['n_periodo_tasas']}  tasa global={res['tasa_global_periodo']}")
    for k in ("franja", "clase_accidente", "localidad"):
        print(f"  {k}:", {r[list(r)[0]]: f"{r['tasa']*100:.1f}% (n={r['n']})" for r in res["tablas"][k]})
    print("Guardado en", a.outdir / "exploracion.json")


if __name__ == "__main__":
    main()
