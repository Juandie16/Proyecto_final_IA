"""Funciones puras del prototipo (probadas en tests/): convierten el formulario en la fila de features del modelo
con las MISMAS reglas que src/preprocess.py."""
import datetime as dt

import numpy as np
import pandas as pd

from src.preprocess import FRANJA_BINS, FRANJA_LABELS


def fila_desde_formulario(fecha: dt.date, hora: int, barrio: str, localidad: str, tipo_via: str, clase_accidente: str,
                          choque_con: str, objeto_fijo: str, lat=None, lon=None) -> pd.DataFrame:
    if not 0 <= int(hora) <= 23:
        raise ValueError("La hora debe estar entre 0 y 23.")
    dia = pd.Timestamp(fecha).dayofweek
    franja = str(pd.cut([float(hora)], bins=FRANJA_BINS, labels=FRANJA_LABELS)[0])
    return pd.DataFrame([{
        "hora": float(hora), "mes": int(pd.Timestamp(fecha).month), "dia_semana": int(dia),
        "es_finde": int(dia >= 5), "es_noche": float(hora >= 18 or hora < 6),
        "lat": np.nan if lat is None else float(lat), "lon": np.nan if lon is None else float(lon),
        "franja": franja, "barrio": barrio, "localidad": str(localidad), "tipo_via": tipo_via,
        "clase_accidente": clase_accidente, "choque_con": choque_con, "objeto_fijo": objeto_fijo,
    }])


def predecir(paquete: dict, fila: pd.DataFrame) -> float:
    cols = paquete["features_numericas"] + paquete["features_categoricas"]
    X = fila[cols].copy()
    X[paquete["features_categoricas"]] = X[paquete["features_categoricas"]].astype(object)
    return float(paquete["pipeline"].predict_proba(X)[0, 1])


def punto_curva(paquete: dict, umbral: float) -> dict:
    """Precisión/recall/tasa de alerta en VALIDACIÓN 2024 para el umbral de la curva más cercano."""
    curva = paquete["curva_validacion_2024"]
    return min(curva, key=lambda r: abs(r["umbral"] - umbral))
