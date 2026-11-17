"""Datos SINTÉTICOS con el mismo esquema (25 columnas de texto, solo siniestros con víctimas) que el dataset del DATT.
Solo para probar que el código funciona. NO son datos reales y sus resultados no significan nada."""
import numpy as np
import pandas as pd


def make_synthetic(n=4000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    clases = ["Choque", "Atropello", "Volcamiento", "Caída de ocupante", "Otro"]
    # Como en el dataset real: SOLO siniestros con víctimas. Probabilidad de muerte según clase (y noche).
    p_fatal = {"Choque": 0.05, "Atropello": 0.55, "Volcamiento": 0.40, "Caída de ocupante": 0.08, "Otro": 0.04}
    barrios = [f"BARRIO {i}" for i in range(25)] + ["Bocagrande", "El Bosque", "Crespo"]
    fechas = pd.to_datetime(rng.integers(pd.Timestamp("2020-01-01").value // 10**9,
                                         pd.Timestamp("2025-09-30").value // 10**9, n), unit="s")
    rows = []
    for i in range(n):
        c = rng.choice(clases, p=[0.55, 0.12, 0.05, 0.1, 0.18])
        h = int(rng.integers(0, 24))
        pr = p_fatal[c] + (0.10 if h >= 20 or h < 5 else 0)
        grav = "Con Muertos" if rng.random() < pr else "Con Heridos"
        muertos = 1 if grav == "Con Muertos" else 0
        heridos = int(rng.integers(1, 4)) if grav == "Con Heridos" else int(rng.integers(0, 3))
        f = fechas[i]
        fecha = f.strftime("%d/%m/%Y") if i % 2 == 0 else f.strftime("%Y-%m-%d 00:00:00")
        hora = f"{h:02d}:{int(rng.integers(0, 60)):02d}" if i % 3 else f"{h % 12 or 12}:{int(rng.integers(0, 60)):02d} {'PM' if h >= 12 else 'AM'}"
        lat, lon = 10.39 + rng.normal(0, 0.02), -75.49 + rng.normal(0, 0.02)
        rows.append({
            "ipat": f"A{100000 + i}", "fecha": fecha, "hora": hora, "gravedad": grav,
            "clase_accidente": c, "choque_con": rng.choice(["Vehículo", "Objeto fijo", "Peatón", None]),
            "objeto_fijo": rng.choice([None, "Poste", "Muro"]),
            "hipotesis_1": rng.choice(["Exceso de velocidad", "No mantener distancia", "Embriaguez", None]),
            "hipotesis_2": None, "hipotesis_3": None, "hipotesis_4": None,
            "BARRIO": rng.choice(barrios), "lugar": "Cra 5 # 3-10",
            "via1": "Calle 30" if i % 2 else None, "via2": None if i % 2 else "Carrera 10",
            "via3": None, "via4": None, "via5": None, "via6": None,
            "localidad": rng.choice(["Histórica y del Caribe Norte", "De la Virgen y Turística", "Industrial de la Bahía"]),
            "muertos": str(muertos), "heridos": str(heridos),
            "coordenadas": f"{lat:.6f},{lon:.6f}",
            "latitud": f"{lat:.6f}".replace(".", ",") if i % 5 == 0 else f"{lat:.6f}",
            "longitud": f"{abs(lon):.6f}" if i % 50 == 0 else f"{lon:.6f}",  # algunas con signo perdido
        })
    df = pd.DataFrame(rows)
    df.loc[rng.choice(n, 20, replace=False), "latitud"] = "0"  # coordenadas basura
    return df
