"""Fase 8: genera las cifras, tablas y figuras del informe IEEE a partir de experiments/ (ningún número a mano).

Uso:  python -m src.make_report_tables
Salidas: informe/cifras.tex (macros \\Cifra...), informe/tablas/*.tex, informe/figuras/*.pdf
"""
import json
import shutil
from pathlib import Path

import pandas as pd

from src import config as C

INF = C.ROOT / "informe"
E = C.EXPERIMENTS
NOMBRE_MODELO = {"regresion_logistica_base": "Línea base (RL, $C{=}1$)", "regresion_logistica_ajustada": "RL ajustada",
                 "random_forest": "Random forest", "hist_gradient_boosting": "HistGradientBoosting"}
FRANJA = {"madrugada": "Madrugada", "manana": "Mañana", "tarde": "Tarde", "noche": "Noche"}


def j(nombre):
    return json.loads((E / nombre).read_text(encoding="utf-8"))


def f(x, nd=3):
    """Número con coma decimal (español)."""
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x, nd=1):
    return f"{x * 100:.{nd}f}".replace(".", ",") + r"\,\%"


def miles(n):
    return f"{int(n):,}".replace(",", ".")


def ic(par, nd=3):
    return f"[{f(par[0], nd)}; {f(par[1], nd)}]"


def macro(nombre, valor):
    assert nombre.isalpha(), nombre
    return f"\\newcommand{{\\Cifra{nombre}}}{{{valor}}}\n"


def main() -> None:
    (INF / "tablas").mkdir(parents=True, exist_ok=True)
    (INF / "figuras").mkdir(parents=True, exist_ok=True)
    ds, cal, ex, bl = j("data_summary.json"), j("calidad_etiqueta_por_anio.json"), j("exploracion.json"), j("baseline_val.json")
    mv, el, fin = j("modelos_val.json"), j("modelo_elegido.json"), j("resultado_final.json")
    er, sg = j("analisis_errores_resumen.json"), j("subgrupos_final.json")
    hist = json.loads((E / "historico" / "baseline_val_v3_tipo_via_constante.json").read_text(encoding="utf-8"))
    T = {k: pd.DataFrame(v) for k, v in ex["tablas"].items()}

    def tasa(tabla, col, valor):
        return float(T[tabla].set_index(col).loc[valor, "tasa"])

    m = []
    # ---- datos
    sp = ds["split"]
    m += [macro("NRaw", miles(ds["n_raw"])), macro("NDup", miles(ds["duplicados_ipat_eliminados"])),
          macro("N", miles(ds["n_limpio"])), macro("NFatal", miles(ds["gravedad_cod_conteo"]["2"])),
          macro("NHeridos", miles(ds["gravedad_cod_conteo"]["1"])),
          macro("PctFatal", pct(ds["gravedad_cod_conteo"]["2"] / ds["n_limpio"])),
          macro("CoordFuera", ds["coordenadas_fuera_de_cartagena_o_ausentes"]),
          macro("Incoherentes", cal["incoherentes_total"]),
          macro("MuertosDosMilVeinticinco", cal["por_anio"]["2025"]["con_muertos_mayor_0"]),
          macro("TipoViaDesconocido", pct(ds["faltantes_pct"]["tipo_via"] / 100)),
          macro("ObjetoSinDato", pct(ds["faltantes_pct"]["objeto_fijo"] / 100)),
          macro("ChoqueSinDato", pct(ds["faltantes_pct"]["choque_con"] / 100)),
          macro("NBarrios", ex["barrio"]["n_categorias"]), macro("NHipotesis", ex["hipotesis_1_n_codigos"])]
    for s, nom in (("train", "Train"), ("val", "Val"), ("test", "Test")):
        m += [macro(f"N{nom}", miles(sp[s]["n"])), macro(f"Fat{nom}", sp[s]["n_fatal"]),
              macro(f"Pct{nom}", f(sp[s]["pct_fatal"], 2) + r"\,\%")]
    # ---- exploración
    m += [macro("TasaGlobal", pct(ex["tasa_global_periodo"])),
          macro("TasaMadrugada", pct(tasa("franja", "franja", "madrugada"))),
          macro("TasaManana", pct(tasa("franja", "franja", "manana"))),
          macro("TasaTarde", pct(tasa("franja", "franja", "tarde"))),
          macro("TasaNoche", pct(tasa("franja", "franja", "noche"))),
          macro("TasaAtropello", pct(tasa("clase_accidente", "clase_accidente", "atropello"))),
          macro("TasaChoque", pct(tasa("clase_accidente", "clase_accidente", "choque"))),
          macro("TasaCaida", pct(tasa("clase_accidente", "clase_accidente", "caida ocupante"))),
          macro("TasaObjetoFijo", pct(tasa("choque_con", "choque_con", "objeto fijo"))),
          macro("TasaVehiculo", pct(tasa("choque_con", "choque_con", "vehiculo"))),
          macro("TasaDosMilVeinte", pct(tasa("anio", "anio", 2020))),
          macro("TasaDosMilVeinticuatro", pct(tasa("anio", "anio", 2024)))]
    # ---- línea base
    lr = bl["modelos"]["regresion_logistica"]
    m += [macro("BasePR", f(lr["pr_auc"])), macro("BaseROC", f(lr["roc_auc"])), macro("BasePrec", f(lr["precision"])),
          macro("BaseRec", f(lr["recall"])), macro("BaseAzar", f(bl["modelos"]["mayoritario"]["pr_auc"])),
          macro("BaseAcc", f(bl["modelos"]["mayoritario"]["accuracy"])),
          macro("BasePRAntes", f(hist["modelos"]["regresion_logistica"]["pr_auc"]))]
    # ---- validación
    vv = mv["validacion_2024"]
    rf = vv[el["modelo"]]
    me = mv["metricas_elegido_semilla_base"]
    hp = el["hiperparametros"]
    m += [macro("Umbral", f(el["umbral"], 4)), macro("ValPR", f(rf["pr_auc"]["media"])), macro("ValPRsd", f(rf["pr_auc"]["sd"])),
          macro("ValROC", f(rf["roc_auc"]["media"])), macro("ValROCsd", f(rf["roc_auc"]["sd"])),
          macro("ValPrec", f(me["precision"])), macro("ValRec", f(me["recall"])), macro("ValAlerta", pct(me["tasa_alerta"])),
          macro("ValPRMovil", f(rf["pr_auc_media_pliegues"])), macro("ValDifPR", f(rf["pr_auc"]["media"] - lr["pr_auc"])),
          macro("ValPrecMedia", f(rf["precision_op"]["media"])),
          macro("RFHojas", hp["min_samples_leaf"]), macro("RFMaxFeat", str(hp["max_features"]).replace("_", r"\_")),
          macro("DuracionBusqueda", f"{mv['duracion_s']:.0f}"),
          macro("NConfig", sum(len(b["configuraciones"]) for b in mv["busqueda"].values()))]
    ab = mv["ablacion_2024"][el["modelo"]]
    m += [macro("AblContexto", f(ab["contexto"]["pr_auc"]["media"])), macro("AblReporte", f(ab["reporte"]["pr_auc"]["media"])),
          macro("AblHipotesis", f(ab["hipotesis"]["pr_auc"]["media"])),
          macro("AblHipotesisROC", f(ab["hipotesis"]["roc_auc"]["media"])),
          macro("AblHipotesisPrec", f(ab["hipotesis"]["precision_op"]["media"]))]
    cv = mv["criterios_vista_previa_val"]["4_brecha_subgrupos"]
    peor_val = max(cv["brecha_por_subgrupo"], key=cv["brecha_por_subgrupo"].get)
    m += [macro("ValPeorSubgrupo", FRANJA.get(peor_val.split("=")[1], peor_val)),
          macro("ValPeorRecall", f(mv["subgrupos_val_elegido"]["franja"][peor_val.split("=")[1]]["recall"]))]
    # ---- prueba
    R = fin["resultados"]
    rm, b5, bo = R[el["modelo"]], R["linea_base_umbral_0_5"], R["linea_base_umbral_validacion"]
    m += [macro("TestPR", f(rm["pr_auc"])), macro("TestPRic", ic(rm["ic95"]["pr_auc"])),
          macro("TestROC", f(rm["roc_auc"])), macro("TestROCic", ic(rm["ic95"]["roc_auc"])),
          macro("TestPrec", f(rm["precision"])), macro("TestPrecic", ic(rm["ic95"]["precision"])),
          macro("TestRec", f(rm["recall"])), macro("TestRecic", ic(rm["ic95"]["recall"])),
          macro("TestAlerta", pct(rm["tasa_alerta"])), macro("TestFN", rm["matriz_confusion"]["fn"]),
          macro("TestTP", rm["matriz_confusion"]["tp"]), macro("TestFP", rm["matriz_confusion"]["fp"]),
          macro("TestTN", rm["matriz_confusion"]["tn"]),
          macro("TestBasePR", f(b5["pr_auc"])), macro("TestBasePRic", ic(b5["ic95"]["pr_auc"])),
          macro("TestBaseROC", f(b5["roc_auc"])), macro("TestDifPR", f(rm["pr_auc"] - b5["pr_auc"])),
          macro("CommitFinal", fin["commit_git"][:7])]
    sub = sg["modelo"]["subgrupos"]
    tarde = sub["franja"]["tarde"]
    m += [macro("TestTardeRec", f(tarde["recall"])), macro("TestTardeic", ic(tarde["ic95"]["recall"])),
          macro("TestTardeFat", tarde["n_fatal"]),
          macro("TestBrechaTarde", f(sg["modelo"]["criterio_4"]["brecha_por_subgrupo"]["franja=tarde"])),
          macro("TestBrechaTardeBase", f(sg["linea_base_umbral_validacion"]["criterio_4"]["brecha_por_subgrupo"]["franja=tarde"]))]
    for cod, nom in (("1", "Uno"), ("2", "Dos"), ("3", "Tres")):
        m += [macro(f"TasaLoc{nom}", pct(tasa("localidad", "localidad_nombre", C.LOCALIDAD_NOMBRES[cod]))),
              macro(f"TestRecLoc{nom}", f(sub["localidad"][cod]["recall"])),
              macro(f"TestRecLoc{nom}ic", ic(sub["localidad"][cod]["ic95"]["recall"], 2))]
    for fr, nom in (("madrugada", "Madrugada"), ("manana", "Manana"), ("noche", "Noche")):
        m += [macro(f"TestRec{nom}", f(sub["franja"][fr]["recall"]))]
    bar = {r["barrio"]: r for r in sg["barrios_n30_tasa_alerta_vs_real"]}
    if "mamonal" in bar:
        m += [macro("MamonalAlerta", pct(bar["mamonal"]["tasa_alerta"])), macro("MamonalReal", pct(bar["mamonal"]["tasa_real"])),
              macro("MamonalVeces", f(bar["mamonal"]["alerta_sobre_global"], 1))]
    # ---- errores y deriva
    d = er["deriva_val_prueba"]
    cat = er["errores_por_categoria"]
    m += [macro("NFN", er["n_errores"]["FN"]), macro("NFP", er["n_errores"]["FP"]),
          macro("FNChoqueVeh", cat.get("FN/FN1", 0) + cat.get("FN/FN2", 0)), macro("FPAtropello", cat.get("FP/FP2", 0)),
          macro("AlertaVal", pct(d["tasa_alerta"]["val_2024"])), macro("AlertaTest", pct(d["tasa_alerta"]["prueba_2025"])),
          macro("ProbaVal", f(d["proba_media"]["val_2024"])), macro("ProbaTest", f(d["proba_media"]["prueba_2025"])),
          macro("AtropelloVal", pct(d["por_clase"]["atropello"]["tasa_real_val"])),
          macro("AtropelloTest", pct(d["por_clase"]["atropello"]["tasa_real_prueba"])),
          macro("TardeVal", pct(d["por_franja"]["tarde"]["tasa_real_val"])),
          macro("TardeTest", pct(d["por_franja"]["tarde"]["tasa_real_prueba"]))]
    imp = list(er["importancia_permutacion_val_2024"].items())
    for i, (k, v) in enumerate(imp[:3]):
        nom = ["Uno", "Dos", "Tres"][i]
        m += [macro(f"Imp{nom}", k.replace("_", r"\_")), macro(f"Imp{nom}Val", f(v["media"]))]
    (INF / "cifras.tex").write_text("% GENERADO por python -m src.make_report_tables. No editar a mano.\n" + "".join(m),
                                    encoding="utf-8")

    # ---- tabla comparativa (validación)
    filas = []
    for fam in ("regresion_logistica_base", "regresion_logistica_ajustada", "random_forest", "hist_gradient_boosting"):
        r = vv[fam]
        sd = lambda k: "" if r[k]["sd"] == 0 else f" $\\pm$ {f(r[k]['sd'])}"
        neg = r"\textbf{" if fam == el["modelo"] else "{"
        filas.append(f"{neg}{NOMBRE_MODELO[fam]}}} & {f(r['pr_auc_media_pliegues'])} & {f(r['pr_auc']['media'])}{sd('pr_auc')} & "
                     f"{f(r['roc_auc']['media'])}{sd('roc_auc')} & {f(r['precision_op']['media'])}{sd('precision_op')} \\\\")
    tab = ("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nModelo & PR-AUC & PR-AUC & ROC-AUC & Prec. \\\\\n"
           " & mov. & 2024 & 2024 & rec.$\\geq$0,6 \\\\\n\\midrule\n" + "\n".join(filas) + "\n\\bottomrule\n\\end{tabular}\n")
    (INF / "tablas" / "comparativa.tex").write_text(tab, encoding="utf-8")

    # ---- ablación
    fil = []
    for fam in ("regresion_logistica_base", "random_forest", "hist_gradient_boosting"):
        a = mv["ablacion_2024"][fam]
        fil.append(f"{NOMBRE_MODELO[fam]} & {f(a['contexto']['pr_auc']['media'])} & {f(a['reporte']['pr_auc']['media'])} & "
                   f"{f(a['hipotesis']['pr_auc']['media'])} \\\\")
    tab = ("\\begin{tabular}{@{}lccc@{}}\n\\toprule\nModelo & \\texttt{contexto} & \\texttt{reporte} & \\texttt{hipotesis} \\\\\n"
           "\\midrule\n" + "\n".join(fil) + "\n\\bottomrule\n\\end{tabular}\n")
    (INF / "tablas" / "ablacion.tex").write_text(tab, encoding="utf-8")

    # ---- prueba
    def fila(nombre, r, u):
        return (f"{nombre} & {f(u, 3)} & {f(r['pr_auc'])} & {f(r['roc_auc'])} & {f(r['precision'])} & {f(r['recall'])} & "
                f"{f(r['f1_fatal'])} \\\\\n & & {ic(r['ic95']['pr_auc'], 2)} & {ic(r['ic95']['roc_auc'], 2)} & "
                f"{ic(r['ic95']['precision'], 2)} & {ic(r['ic95']['recall'], 2)} & \\\\")
    tab = ("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\nModelo & Umbral & PR-AUC & ROC-AUC & Prec. & Recall & F1 \\\\\n\\midrule\n"
           + fila("Random forest", rm, rm["umbral"]) + "\n\\midrule\n" + fila("Línea base", b5, 0.5) + "\n\\midrule\n"
           + fila("Línea base", bo, bo["umbral"]) + "\n\\bottomrule\n\\end{tabular}\n")
    (INF / "tablas" / "prueba.tex").write_text(tab, encoding="utf-8")

    # ---- criterios
    nombres = {"1": "1. PR-AUC $\\geq$ base + 0,05 (val.)", "2a": "2. ROC-AUC $\\geq$ 0,70 (val.)",
               "2b": "2. ROC-AUC $\\geq$ 0,70 (prueba)", "3": "3. Prec. $\\geq$ 0,20 con rec. $\\geq$ 0,60",
               "4": "4. Brecha de recall $\\leq$ 0,15"}
    fil = []
    for c in fin["criterios"]:
        k = c["criterio"].split(".")[0]
        v = c["valor"]
        if k == "3":
            val = f"P={f(v['precision'])}, R={f(v['recall'])}"
        elif k == "4":
            peor = max(v, key=v.get)
            val = f"{FRANJA.get(peor.split('=')[1], peor)}: {f(v[peor])}"
        else:
            val = f(v)
        fil.append(f"{nombres[k]} & {val} & {'Sí' if c['cumple'] else 'No'} \\\\")
    tab = ("\\begin{tabular}{@{}lcc@{}}\n\\toprule\nCriterio & Valor & Cumple \\\\\n\\midrule\n" + "\n".join(fil)
           + "\n\\bottomrule\n\\end{tabular}\n")
    (INF / "tablas" / "criterios.tex").write_text(tab, encoding="utf-8")

    # ---- errores (categorías)
    nc = er["nombres_categoria"]
    fil = [f"{k.split('/')[1]} & {nc[k.split('/')[1]]} & {n} \\\\" for k, n in sorted(cat.items())]
    tab = ("\\begin{tabular}{@{}llr@{}}\n\\toprule\nCód. & Categoría (regla en \\texttt{src/errors.py}) & $n$ \\\\\n\\midrule\n"
           + "\n".join(fil) + "\n\\bottomrule\n\\end{tabular}\n")
    (INF / "tablas" / "errores.tex").write_text(tab, encoding="utf-8")

    # ---- ejemplos de errores (6 de los 20)
    ae = pd.read_csv(E / "analisis_errores.csv")
    ej = pd.concat([ae[ae.tipo_error == "FN"].head(3), ae[ae.tipo_error == "FP"].head(3)])
    fil = []
    for r in ej.itertuples():
        vr = "; ".join(x.split("=")[1].split(" (")[0] for x in r.variables_relevantes.split("; ")[:3])
        fil.append(f"{r.ipat} & {r.tipo_error} & {f(r.probabilidad, 3)} & {r.categoria_cod} & {vr} \\\\")
    tab = ("\\begin{tabular}{@{}rcccl@{}}\n\\toprule\nIPAT & Tipo & Prob. & Cat. & clase; choque con; franja \\\\\n\\midrule\n"
           + "\n".join(fil) + "\n\\bottomrule\n\\end{tabular}\n")
    (INF / "tablas" / "ejemplos_errores.tex").write_text(tab, encoding="utf-8")

    for fig in sorted((E / "figuras").glob("*.pdf")):
        shutil.copy2(fig, INF / "figuras" / fig.name)
    print("Generado:", INF / "cifras.tex", "+", len(list((INF / "tablas").glob("*.tex"))), "tablas +",
          len(list((INF / "figuras").glob("*.pdf"))), "figuras")


if __name__ == "__main__":
    main()
