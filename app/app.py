"""Prototipo: probabilidad de que un siniestro CON VÍCTIMAS sea fatal (Cartagena, DATT 2020-2024).

Ejecutar desde la raíz del repositorio:
    python -m src.export_model          # una vez: genera models/modelo_final.joblib
    python -m streamlit run app/app.py
"""
import datetime as dt
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import streamlit as st  # noqa: E402

from src.app_utils import fila_desde_formulario, predecir, punto_curva  # noqa: E402

MODELO = Path(os.environ.get("MODELO_PATH", ROOT / "models" / "modelo_final.joblib"))
NOMBRES = {"manana": "mañana", "caida ocupante": "caída de ocupante", "sin_dato": "sin dato / no aplica",
           "vehiculo": "vehículo", "corredor_carga": "corredor de carga"}


def bonito(x: str) -> str:
    return NOMBRES.get(x, x)


st.set_page_config(page_title="Siniestros fatales · Cartagena", page_icon="🚦", layout="centered")
st.title("Estimación de fatalidad de un siniestro vial")
st.info(
    "**Solo para siniestros con víctimas.** El modelo no decide si hay víctimas: estima, con el informe inicial, "
    "la probabilidad de que el siniestro sea fatal. Es una **herramienta de apoyo** para priorizar; no reemplaza la "
    "decisión humana ni justifica reducir la atención a ningún caso."
)


@st.cache_resource
def cargar(ruta: str):
    return joblib.load(ruta)


if not MODELO.exists():
    st.error(f"No se encontró el modelo en `{MODELO}`. Ejecute primero: `python -m src.export_model`.")
    st.stop()
pq = cargar(str(MODELO))
el = pq["eleccion"]
cats = pq["categorias"]

with st.form("siniestro"):
    c1, c2 = st.columns(2)
    fecha = c1.date_input("Fecha", value=dt.date(2025, 6, 15), key="fecha")
    hora = c2.number_input("Hora (0-23)", min_value=0, max_value=23, value=14, step=1, key="hora")
    barrio = st.selectbox("Barrio", pq["barrios"], index=0, key="barrio",
                          help="Barrios vistos en 2020-2024. Si no aparece, elija el más cercano.")
    locs = cats["localidad"]
    localidad = st.selectbox("Localidad", locs, key="localidad",
                             format_func=lambda c: f"{c} · {pq['localidad_nombres'].get(c, '')} [CONFIRMAR]")
    tipo_via = st.selectbox("Tipo de vía", cats["tipo_via"], format_func=bonito, key="tipo_via")
    clase = st.selectbox("Clase de accidente", cats["clase_accidente"], format_func=bonito, key="clase")
    choque = st.selectbox("Choque con", cats["choque_con"], format_func=bonito, key="choque")
    objeto = st.selectbox("Objeto fijo", cats["objeto_fijo"], format_func=bonito, key="objeto")
    lat0, lon0 = pq["barrio_coords"].get(barrio, pq["coords_mediana"])
    with st.expander("Coordenadas (por defecto, la mediana del barrio elegido)"):
        usar = st.checkbox("Escribir coordenadas exactas", key="usar_coords")
        lat = st.number_input("Latitud", value=lat0, format="%.6f", key="lat")
        lon = st.number_input("Longitud", value=lon0, format="%.6f", key="lon")
    enviado = st.form_submit_button("Estimar")

st.subheader("Umbral de alerta")
umbral = st.slider("Umbral (probabilidad a partir de la cual se emite alerta)", 0.02, 0.60, float(round(el["umbral"], 4)),
                   0.01, key="umbral",
                   help=f"Elegido en validación 2024: {el['umbral']:.4f} (el mayor con recall ≥ 0,60).")
pc = punto_curva(pq, umbral)
st.caption(
    f"Con un umbral ≈ {pc['umbral']:.3f}, en **validación 2024** el modelo detectaba el {pc['recall']*100:.0f} % de los fatales "
    f"(recall), {pc['precision']*100:.0f} % de las alertas eran fatales (precisión) y alertaba en el {pc['tasa_alerta']*100:.0f} % "
    "de los siniestros. Subir el umbral reduce las alertas y el recall y suele subir la precisión. "
    "(Curva medida con el modelo entrenado en 2020-2023; en 2025 el recall con el umbral elegido fue menor.)"
)

if enviado or "proba" in st.session_state:
    if enviado:
        fila = fila_desde_formulario(fecha, int(hora), barrio, localidad, tipo_via, clase, choque, objeto,
                                     lat if usar else lat0, lon if usar else lon0)
        st.session_state["proba"] = predecir(pq, fila)
    p = st.session_state["proba"]
    st.subheader("Resultado")
    col1, col2 = st.columns(2)
    col1.metric("Probabilidad estimada de fatalidad", f"{p*100:.1f} %")
    if p >= umbral:
        col2.error(f"⚠️ ALERTA: supera el umbral ({umbral:.2f})")
    else:
        col2.success(f"Sin alerta: por debajo del umbral ({umbral:.2f})")
    st.caption(
        "Referencia: en 2020-2024 el 7,4 % de los siniestros con víctimas fue fatal. "
        "El modelo detecta menos casos fatales en horario diurno y en choques entre vehículos (ver docs/ia_responsable.md)."
    )

with st.expander("Sobre el modelo"):
    st.write(f"Modelo: **{el['modelo']}** · hiperparámetros `{el['hiperparametros']}` · features `{el['conjunto_de_features']}` · "
             f"entrenado con {pq['entrenado_con']} · semilla {pq['semilla']}.")
    st.write("Datos: DATT, Alcaldía de Cartagena (CC-BY). Resultados y limitaciones: `experiments/` y `docs/`.")
