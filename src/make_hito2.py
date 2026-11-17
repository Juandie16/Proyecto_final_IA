"""Genera el PDF del hito 2 (1 página de avance + bitácora IA) leyendo resultados reales de experiments/.

Uso:  python -m src.make_hito2 --integrantes "Nombre A, Nombre B" [--out entregas/NOMBRE_hito2.pdf]
Lo que aún no se ha ejecutado aparece como PENDIENTE y el archivo sale con sufijo _BORRADOR.
"""
import argparse
import json
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src import config as C

PEND = '<font backColor="#FFE680"><b>PENDIENTE</b></font>'
NAME = "siniestros_fatales_cartagena_hito2"


def load(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(v, nd=3):
    return PEND if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def md_inline(t: str) -> str:
    """Convierte el markdown mínimo de la bitácora a marcado de reportlab."""
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", t)
    return re.sub(r"\[CONFIRMAR[^\]]*\]", lambda m: f'<font backColor="#FFE680">{m.group(0)}</font>', t)


def parse_bitacora_md(path, nombres=()):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and not line.startswith("|---") and not line.startswith("| Fecha"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            for ph, nm in zip(("[Integrante A]", "[Integrante B]"), nombres):
                cells = [c.replace(ph, nm) for c in cells]
            rows.append([md_inline(c) for c in cells])
    return rows


def build(integrantes: str, out: Path) -> Path:
    summ = load(C.EXPERIMENTS / "data_summary.json")
    base = load(C.EXPERIMENTS / "baseline_val.json")
    pending = summ is None or base is None

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=7.6, leading=9.4, alignment=TA_JUSTIFY)
    small = ParagraphStyle("s", parent=body, fontSize=6.8, leading=8.2, alignment=0)
    h = ParagraphStyle("h", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=8.8, leading=10.5,
                       spaceBefore=4, spaceAfter=1.5, textColor=colors.HexColor("#1F3A5F"))
    title = ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=12.5, leading=14.5, spaceAfter=1)
    sub = ParagraphStyle("sub", parent=body, alignment=1, textColor=colors.HexColor("#444444"))
    P = lambda t, st=body: Paragraph(t, st)

    def grid(data, widths, header=True, font=6.9):
        t = Table([[Paragraph(str(c), small) for c in r] for r in data], colWidths=widths, hAlign="LEFT")
        st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9AA5B1")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
              ("TOPPADDING", (0, 0), (-1, -1), 1.4), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4),
              ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
        if header:
            st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF5")))
        t.setStyle(TableStyle(st))
        return t

    W = A4[0] - 3.0 * cm
    story = []
    if pending:
        story.append(P(f"{PEND} <b>BORRADOR.</b> Faltan resultados reales: ejecute los 4 comandos del README y vuelva a correr "
                       "<font face='Courier'>python -m src.make_hito2</font>. No entregar así.", small))
        story.append(Spacer(1, 2))

    story += [P("Predicción de siniestros viales fatales en Cartagena (Ruta C: aprendizaje automático)", title),
              P(f"Proyecto final de Inteligencia Artificial · Hito 2: avance · {integrantes} · 16 de noviembre de 2026", sub)]

    # 1. Problema
    story += [P("1. Problema, usuario y relevancia", h), P(
        "En Cartagena la accidentalidad vial con motocicletas es el principal foco de víctimas: según el DATT, en lo corrido de 2026 "
        "(corte de agosto) se registraron 1.134 siniestros, 985 con participación de motos, y 95 fallecidos, 86 de ellos asociados a motos "
        "(El Universal, 18-ago-2026). <b>Usuario objetivo:</b> el DATT y los organismos de socorro, que deben decidir con información "
        "incompleta a qué siniestros dar prioridad. Hoy esa priorización depende del criterio del operador (supuesto que validaremos "
        "con una entrevista, pendiente). <b>Qué cambiaría:</b> una estimación de la probabilidad de que un siniestro con víctimas termine con al menos una muerte, "
        "a partir del informe inicial, como apoyo (nunca reemplazo) a la decisión humana. <b>Pregunta:</b> ¿mejora un modelo de aprendizaje "
        "automático a una línea base sencilla al estimar si un siniestro con víctimas será fatal, y cómo varía su desempeño por zona y franja horaria?")]

    # 2. Formulación
    story += [P("2. Formulación como problema de IA", h), grid([
        ["<b>Medida de desempeño</b>", "F1 de la clase «fatal» y PR-AUC (primarias); F1 macro, ROC-AUC, recall, precisión (secundarias); recall por franja y localidad."],
        ["<b>Entorno</b>", "Reportes de siniestros del DATT en Cartagena. Parcialmente observable, estocástico, episódico, estático (evaluación sobre datos históricos), mixto discreto/continuo."],
        ["<b>Actuadores</b>", "Probabilidad estimada y alerta binaria «fatal» para un operador humano."],
        ["<b>Sensores</b>", "Informe inicial: fecha, hora, barrio, localidad, coordenadas, tipo de vía, clase de accidente, con qué chocó, objeto fijo."],
        ["<b>Tarea</b>", "Clasificación binaria supervisada sobre siniestros con víctimas: <i>y</i> = 1 si <i>gravedad</i> es «Con muertos»; 0 si «Con heridos»."],
    ], [2.9 * cm, W - 2.9 * cm], header=False)]

    # 3. Datos
    if summ:
        sp = summ["split"]
        filas = (f"{summ['n_raw']:,} crudas, {summ['n_limpio']:,} tras limpieza".replace(",", "."))
        part = " · ".join(f"{k}: n={sp[k]['n']}, fatales={sp[k]['n_fatal']} ({sp[k]['pct_fatal']}%)" for k in ("train", "val", "test"))
        calidad = (f"fechas no interpretadas: {summ['fecha_sin_interpretar']}; horas: {summ['hora_sin_interpretar']}; "
                   f"coordenadas ausentes o fuera de Cartagena: {summ['coordenadas_fuera_de_cartagena_o_ausentes']}; "
                   f"IPAT duplicados eliminados: {summ.get('duplicados_ipat_eliminados', 0)} (con gravedad distinta: {summ.get('ipat_duplicados_con_gravedad_distinta', 0)}; se conserva la fila más grave)")
    else:
        filas, part, calidad = PEND, PEND, PEND
    story += [P("3. Datos", h), grid([
        ["<b>Fuente y licencia</b>", "DATT, Alcaldía de Cartagena, portal de datos abiertos (también en datos.gov.co, id 5z62-bn99). Licencia CC-BY. 25 columnas, 2020–2025, archivo de 1,1 MiB. Sin nombres ni documentos de identidad. <b>Solo contiene siniestros con víctimas</b> (verificado al ejecutar el pipeline): no hay «Solo daños materiales»."],
        ["<b>Filas</b>", filas],
        ["<b>Partición</b>", f"Temporal: entrenamiento 2020–2023, validación 2024, prueba 2025 (sin tocar). {part}"],
        ["<b>Prevención de fuga</b>", "<i>muertos</i>, <i>heridos</i> y <i>gravedad</i> definen la etiqueta y se excluyen de los features (el código lo verifica con un <i>assert</i>). La hipótesis causal del agente solo entra en ablación."],
        ["<b>Calidad</b>", calidad],
    ], [2.9 * cm, W - 2.9 * cm], header=False)]

    # 4. Línea base y evaluación
    story += [P("4. Línea base y evaluación predeclarada", h), P(
        "Línea base: clasificador mayoritario y regresión logística sin ajuste de hiperparámetros (C=1, pesos balanceados, umbral 0,5), "
        "semilla 42. Métricas y umbral de éxito declarados el 16-nov-2026 (etiqueta corregida el mismo día, antes de entrenar modelos; ver «Cambios») en <font face='Courier'>docs/evaluacion_predeclarada.md</font> antes de "
        "entrenar otros modelos: el mejor modelo debe superar a la mejor línea base en F1 macro por al menos 0,03, con ROC-AUC de al menos 0,70 y recall de la clase "
        "«fatal» de al menos 0,60 en prueba. Comparación mínima de la ruta (hito 3): dos modelos adicionales con ajuste de hiperparámetros, varias corridas y "
        "ablación de features.")]

    # 5. Primer resultado
    story += [P("5. Primer resultado (línea base, evaluada en <u>validación 2024</u>)", h)]
    hdr = ["Modelo", "F1 fatal", "PR-AUC", "F1 macro", "ROC-AUC", "Recall", "Precisión"]
    rows = [hdr]
    for key, nm in (("mayoritario", "Mayoritario"), ("regresion_logistica", "Regresión logística")):
        r = base["modelos"][key] if base else {}
        rows.append([nm] + [fmt(r.get(k)) for k in ("f1_fatal", "pr_auc", "f1_macro", "roc_auc", "recall", "precision")])
    story.append(grid(rows, [3.6 * cm] + [(W - 3.6 * cm) / 6] * 6))
    if base:
        lr, mj = base["modelos"]["regresion_logistica"], base["modelos"]["mayoritario"]
        cm_ = lr["matriz_confusion"]
        txt = (f"Entrenamiento n={base['n_entrenamiento']}, validación n={base['n_evaluacion']}. La regresión logística logra F1 macro "
               f"{lr['f1_macro']:.3f} frente a {mj['f1_macro']:.3f} del mayoritario, y PR-AUC {lr['pr_auc']:.3f} (referencia de azar: la prevalencia, {lr['prevalencia_fatal']:.3f}); matriz de confusión: VN={cm_['tn']}, FP={cm_['fp']}, FN={cm_['fn']}, VP={cm_['tp']}. "
               "<b>Alcance:</b> una sola corrida, solo validación, sin ajuste; no permite concluir sobre el modelo final. Predicciones fila a fila en "
               "<font face='Courier'>experiments/baseline_val_predicciones.csv</font> (insumo del análisis de errores).")
    else:
        txt = f"Resultados {PEND}: ejecutar <font face='Courier'>python -m src.baseline</font>."
    story += [Spacer(1, 2), P(txt)]

    # 6. Riesgos y próximos pasos
    story += [P("6. Riesgos y próximos pasos", h), P(
        "<b>Riesgos:</b> el dataset solo incluye siniestros con víctimas, por lo que el modelo no sirve para decidir si hay víctimas, solo para estimar su gravedad; la clase «fatal» es minoritaria (cerca de 8 %) y habrá pocos casos por subgrupo; "
        "campos categóricos con mucha variación de escritura; riesgo de usar la herramienta para estigmatizar barrios (se medirá desempeño por localidad). "
        "<b>Próximos pasos:</b> (1) exploración y estadísticas descriptivas; (2) random forest y gradient boosting con ajuste y 5 semillas; (3) ablación de features; "
        "(4) análisis de al menos 10 errores; (5) prototipo en Streamlit; (6) entrevista con un actor del DATT para validar el uso.")]
    story.append(PageBreak())

    # Página 2: bitácora
    story += [P("Bitácora humano–IA: primeras entradas", title),
              P("Detalle completo y reflexión en <font face='Courier'>docs/bitacora_ia.md</font> del repositorio.", sub), Spacer(1, 4)]
    nombres = [n.strip() for n in integrantes.split(",") if n.strip()]
    bit = [["Fecha", "Integrante", "Herram.", "Para qué se usó", "Qué se aceptó, modificó o descartó", "Cómo se verificó"]]
    bit += parse_bitacora_md(C.ROOT / "docs" / "bitacora_ia.md", nombres)
    story.append(grid(bit, [1.6 * cm, 1.9 * cm, 1.4 * cm, 2.6 * cm, (W - 7.5 * cm) * 0.5, (W - 7.5 * cm) * 0.5]))

    out = Path(out)
    if pending and "BORRADOR" not in out.stem:
        out = out.with_name(out.stem + "_BORRADOR.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.2 * cm,
                      bottomMargin=1.0 * cm, title="Hito 2 - Gravedad de siniestros viales en Cartagena").build(story)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--integrantes", default="[Integrante A], [Integrante B]")
    ap.add_argument("--out", type=Path, default=C.ROOT / "entregas" / f"{NAME}.pdf")
    a = ap.parse_args()
    print("PDF generado:", build(a.integrantes, a.out))


if __name__ == "__main__":
    main()
