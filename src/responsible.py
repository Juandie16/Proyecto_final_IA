"""Fase 5 (hito 3): IA responsable con medición, sobre la prueba 2025 ya evaluada (no se elige nada aquí).

Uso:  python -m src.responsible
Entradas: experiments/final_predicciones.csv, experiments/modelo_elegido.json, experiments/modelos_val.json
Salidas:  experiments/subgrupos_final.json, experiments/figuras/fig5_recall_subgrupos.{pdf,png}

Mide, por franja horaria y localidad (solo subgrupos elegibles: n >= 30 y >= 10 fatales):
  recall y precisión con IC 95 % por bootstrap, tasa de alerta, tasa de falsas alarmas y calibración (probabilidad
  media frente a tasa real). Compara la brecha de recall con la de la línea base en su umbral de validación.
Riesgo de estigmatizar barrios: tasa de alerta frente a tasa real por barrio (barrios con n >= 30 en prueba).
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.evaluacion import bootstrap_ic, brecha_subgrupos, metricas

AZUL, NARANJA, TINTA, TINTA_2, REJILLA = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
FRANJAS = ["madrugada", "manana", "tarde", "noche"]


def por_subgrupo(df, proba_col, umbral, col, n_boot=1000):
    y_all, p_all = df["fatal"].to_numpy(), df[proba_col].to_numpy()
    out = {}
    for g in sorted(df[col].astype(str).unique()):
        m = (df[col].astype(str) == g).to_numpy()
        y, p = y_all[m], p_all[m]
        if m.sum() < C.SUBGRUPO_MIN_N or y.sum() < C.SUBGRUPO_MIN_FATALES:
            out[g] = {"n": int(m.sum()), "n_fatal": int(y.sum()), "elegible": False}
            continue
        r = metricas(y, p, umbral)
        pred = p >= umbral
        out[g] = {
            "n": r["n"], "n_fatal": r["n_fatal"], "elegible": True, "prevalencia": r["prevalencia"],
            "recall": r["recall"], "precision": r["precision"], "tasa_alerta": r["tasa_alerta"],
            "tasa_falsas_alarmas": round(float(pred[y == 0].mean()), 4),
            "proba_media": round(float(p.mean()), 4),
            "ic95": bootstrap_ic(y, p, umbral, n_boot=n_boot, claves=("recall", "precision")),
        }
    return out


def solo_elegibles(sub):
    return {col: {g: v for g, v in grupos.items() if v.get("elegible")} for col, grupos in sub.items()}


def figura(sub_mod, recall_global, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    filas = [("franja", g) for g in FRANJAS if sub_mod["franja"].get(g, {}).get("elegible")]
    filas += [("localidad", g) for g in sorted(sub_mod["localidad"]) if sub_mod["localidad"][g].get("elegible")]
    nombres = {"manana": "mañana"}
    etiquetas = [f"{nombres.get(g, g)}" if c == "franja" else f"loc. {g}" for c, g in filas]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    y = np.arange(len(filas))[::-1]
    ax.axvspan(0, recall_global - 0.15, color="#fbe3d6", linewidth=0)
    ax.axvline(recall_global, color=TINTA_2, linestyle="--", linewidth=0.8)
    for yi, (c, g) in zip(y, filas):
        v = sub_mod[c][g]
        lo, hi = v["ic95"]["recall"]
        ax.plot([lo, hi], [yi, yi], color=TINTA_2, linewidth=1)
        ax.plot(v["recall"], yi, "o", color=AZUL, markersize=6, markeredgecolor="white", markeredgewidth=1.5)
        ax.text(1.02, yi, f"{v['n_fatal']} fat.", va="center", fontsize=6, color=TINTA_2, transform=ax.get_yaxis_transform())
    ax.set_yticks(y, [f"{e}" for e in etiquetas], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Recall en prueba 2025 (IC 95 % bootstrap)", fontsize=7, color=TINTA_2)
    ax.axhline(len(filas) - sum(1 for c, _ in filas if c == "franja") - 0.5, color=REJILLA, linewidth=0.8)
    ax.text(recall_global, len(filas) - 0.35, f"global {recall_global:.2f}", fontsize=6, color=TINTA_2, ha="center")
    ax.text((recall_global - 0.15) / 2, -0.9, "zona que incumple\nel criterio 4", fontsize=5.5, color="#a0441c", ha="center", va="top")
    ax.set_ylim(-1.6, len(filas) - 0.1)
    ax.tick_params(colors=TINTA_2, labelsize=7, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("Recall por subgrupo (random forest)", loc="left", fontsize=8.5, color=TINTA, fontweight="bold")
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig5_recall_subgrupos.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=C.EXPERIMENTS)
    a = ap.parse_args()

    pr = pd.read_csv(a.outdir / "final_predicciones.csv", dtype={"localidad": str})
    el = json.loads((a.outdir / "modelo_elegido.json").read_text(encoding="utf-8"))
    val = json.loads((a.outdir / "modelos_val.json").read_text(encoding="utf-8"))
    u_mod = float(el["umbral"])
    u_base = float(val["validacion_2024"]["regresion_logistica_base"]["por_semilla"][0]["op"]["umbral"])

    g_mod = metricas(pr["fatal"], pr["proba_fatal"], u_mod)
    g_base = metricas(pr["fatal"], pr["proba_linea_base"], u_base)
    sub_mod = {c: por_subgrupo(pr, "proba_fatal", u_mod, c) for c in ("franja", "localidad")}
    sub_base = {c: por_subgrupo(pr, "proba_linea_base", u_base, c) for c in ("franja", "localidad")}

    # Estigmatización: ¿el modelo alerta mucho más en ciertos barrios de lo que justifica su tasa real?
    b = pr.assign(alerta=pr["proba_fatal"] >= u_mod).groupby("barrio").agg(
        n=("fatal", "size"), fatales=("fatal", "sum"), tasa_real=("fatal", "mean"),
        tasa_alerta=("alerta", "mean"), proba_media=("proba_fatal", "mean"))
    b = b[b["n"] >= 30].copy()
    b["alerta_sobre_global"] = b["tasa_alerta"] / g_mod["tasa_alerta"]
    barrios = b.sort_values("tasa_alerta", ascending=False).round(4).reset_index().to_dict("records")

    out = {
        "fecha_ejecucion_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "semilla": C.SEED,
        "evaluado_en": "prueba 2025 (predicciones de la evaluación única; aquí no se reentrena ni se elige nada)",
        "criterio_elegibilidad": {"n_min": C.SUBGRUPO_MIN_N, "fatales_min": C.SUBGRUPO_MIN_FATALES},
        "localidad_nombres_CONFIRMAR": C.LOCALIDAD_NOMBRES,
        "modelo": {"nombre": el["modelo"], "umbral": u_mod, "global": g_mod, "subgrupos": sub_mod,
                   "criterio_4": brecha_subgrupos(solo_elegibles(sub_mod), g_mod["recall"])},
        "linea_base_umbral_validacion": {"umbral": u_base, "global": g_base, "subgrupos": sub_base,
                                         "criterio_4": brecha_subgrupos(solo_elegibles(sub_base), g_base["recall"])},
        "barrios_n30_tasa_alerta_vs_real": barrios,
    }
    (a.outdir / "subgrupos_final.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=bool), encoding="utf-8")
    figura(sub_mod, g_mod["recall"], a.outdir / "figuras")

    print(f"Recall global modelo={g_mod['recall']}  línea base={g_base['recall']}")
    for c in ("franja", "localidad"):
        for g, v in sub_mod[c].items():
            if v.get("elegible"):
                print(f"  {c}={g:10s} n={v['n']:4d} fat={v['n_fatal']:3d} recall={v['recall']:.3f} {v['ic95']['recall']} "
                      f"prec={v['precision']:.3f} alerta={v['tasa_alerta']:.3f} p_media={v['proba_media']:.3f} real={v['prevalencia']:.3f}")
    print("Subgrupos que incumplen (modelo):", out["modelo"]["criterio_4"]["subgrupos_que_fallan"])
    print("Subgrupos que incumplen (línea base):", out["linea_base_umbral_validacion"]["criterio_4"]["subgrupos_que_fallan"])


if __name__ == "__main__":
    main()
