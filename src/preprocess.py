"""Limpieza, construcción de la etiqueta y partición temporal.

Uso:  python -m src.preprocess [--input RUTA] [--output RUTA] [--summary RUTA]

Decisiones clave (justificadas en docs/evaluacion_predeclarada.md):
  * Etiqueta binaria  fatal = 1 si gravedad es "Con muertos"; 0 si "Con heridos".
    (Se verificó con los datos reales que el dataset SOLO contiene siniestros con víctimas.)
  * Las columnas muertos/heridos/gravedad definen la etiqueta -> se excluyen de los features (fuga).
  * Partición TEMPORAL por año (train 2020-23, val 2024, test 2025), nunca aleatoria.
Cada paso de limpieza cuenta cuántas filas afecta, y todo queda en experiments/data_summary.json.
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C

VIA_COLS = ["via1", "via2", "via3", "via4", "via5"]
VIA_NAMES = ["calle", "carrera", "diagonal", "transversal", "avenida"]
# En el archivo del DATT el campo vacío de via1..via6 viene como "0" (verificado 2026-11-22: 2020-2022 tienen
# via1..via5 = "0" en el 100 % de las filas). via6 no es un nombre de vía: vale "Corredor de Carga", "Otro" o "0".
VIA_VACIO = {"", "0"}
# choque_con / objeto_fijo: la forma de decir "sin dato" cambia según el año ("0" en 2020-22, vacío en 2023-24,
# "No" en 2025; "Choque con" son 8 celdas con el nombre de la columna). Se unifican para que la codificación
# no delate el año del registro.
FRANJA_BINS = [-1, 5, 11, 17, 23]                       # 0-5 madrugada, 6-11 mañana, 12-17 tarde, 18-23 noche
FRANJA_LABELS = ["madrugada", "manana", "tarde", "noche"]
SIN_DATO = "sin_dato"
SIN_DATO_TOKENS = {"", "0", "no", "desconocido", "choque con"}
COLS_SIN_DATO = ["choque_con", "objeto_fijo"]
DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d",
    "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y", "%d-%m-%Y",
]


# ---------------------------------------------------------------- utilidades de texto
def normalize_text(x) -> str:
    """minúsculas, sin tildes, espacios colapsados. NaN/None -> ''."""
    if x is None or (isinstance(x, float) and np.isnan(x)) or pd.isna(x):
        return ""
    s = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip().lower()


def parse_gravedad(x) -> float:
    t = normalize_text(x)
    if not t:
        return np.nan
    if "muert" in t or "fatal" in t:
        return 2
    if "herid" in t or "lesion" in t:
        return 1
    if "dano" in t or "material" in t:
        return 0
    return np.nan


def parse_hour(x) -> float:
    """Devuelve la hora entera 0-23 desde '14:30', '14:30:00', '2:30 PM', '1900-01-01 14:30:00'..."""
    t = normalize_text(x)
    if not t:
        return np.nan
    m = re.search(r"(\d{1,2})[:h.](\d{2})(?:[:.]\d{2})?\s*([ap])?\.?\s*m?", t)
    if not m:
        if re.fullmatch(r"\d{3,4}", t):  # 1430 / 930
            h = int(t) // 100
            return float(h) if 0 <= h <= 23 else np.nan
        return np.nan
    h = int(m.group(1))
    mer = m.group(3)
    if mer == "p" and h < 12:
        h += 12
    if mer == "a" and h == 12:
        h = 0
    return float(h) if 0 <= h <= 23 else np.nan


def parse_fecha(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    result = pd.Series(pd.NaT, index=s.index, dtype="datetime64[us]")
    num = pd.to_numeric(s, errors="coerce")
    serial = num.between(30000, 60000)  # fecha serial de Excel
    if serial.any():
        parsed = pd.to_datetime(num[serial], unit="D", origin="1899-12-30").astype("datetime64[us]")
        result = result.where(~serial, parsed.reindex(result.index))
    for fmt in DATE_FORMATS:
        pending = result.isna() & s.notna()
        if not pending.any():
            break
        parsed = pd.to_datetime(s[pending], format=fmt, errors="coerce").astype("datetime64[us]")
        result = result.where(~pending, parsed.reindex(result.index))
    return result


def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.strip().str.replace(",", ".", regex=False), errors="coerce")


# ---------------------------------------------------------------- lectura
def read_raw(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str)
    else:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = [normalize_text(c).replace(" ", "_") for c in df.columns]
    return df.drop(columns=[c for c in ("_id",) if c in df.columns])


def default_input() -> Path:
    for p in (C.RAW_CSV, C.RAW_XLSX):
        if p.exists():
            return p
    raise FileNotFoundError("No hay datos crudos. Ejecute primero: python -m src.data_download")


# ---------------------------------------------------------------- limpieza
def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log: dict = {"n_raw": int(len(raw)), "columnas_raw": list(raw.columns)}
    df = raw.copy()

    # 1. Etiqueta (antes de deduplicar, para poder conservar la fila más grave de cada IPAT)
    df["gravedad_cod"] = df["gravedad"].map(parse_gravedad)
    log["gravedad_sin_interpretar"] = int(df["gravedad_cod"].isna().sum())
    log["gravedad_valores_originales"] = {
        str(k): int(v) for k, v in df["gravedad"].value_counts(dropna=False).head(10).items()
    }
    df = df[df["gravedad_cod"].notna()].copy()
    df["gravedad_cod"] = df["gravedad_cod"].astype(int)

    # 2. Duplicados por código IPAT: se conserva la fila MÁS GRAVE (no la primera al azar)
    log["duplicados_ipat_eliminados"] = 0
    log["ipat_duplicados_con_gravedad_distinta"] = 0
    if "ipat" in df.columns:
        has_id = df["ipat"].notna() & (df["ipat"].str.strip() != "")
        nun = df[has_id].groupby("ipat")["gravedad_cod"].nunique()
        log["ipat_duplicados_con_gravedad_distinta"] = int((nun > 1).sum())
        df["_has_id"] = has_id
        df = df.sort_values("gravedad_cod", ascending=False, kind="stable")
        dup = df.duplicated(subset="ipat", keep="first") & df["_has_id"]
        log["duplicados_ipat_eliminados"] = int(dup.sum())
        df = df[~dup].drop(columns="_has_id").sort_index()

    df["fatal"] = (df["gravedad_cod"] == 2).astype(int)
    if df["fatal"].nunique() < 2:
        raise ValueError(
            "La etiqueta tiene una sola clase tras la limpieza. Revise gravedad_valores_originales en el resumen: "
            "no se puede entrenar ni evaluar un clasificador así."
        )

    # Coherencia de la etiqueta con el conteo de muertos (calidad de datos; no se usa como feature)
    if "muertos" in df.columns:
        muertos = pd.to_numeric(df["muertos"], errors="coerce").fillna(0)
        incoh = ((df["fatal"] == 1) & (muertos == 0)) | ((df["fatal"] == 0) & (muertos > 0))
        log["filas_gravedad_incoherente_con_conteos"] = int(incoh.sum())

    # 3. Fecha y hora
    df["fecha_dt"] = parse_fecha(df["fecha"])
    log["fecha_sin_interpretar"] = int(df["fecha_dt"].isna().sum())
    df = df[df["fecha_dt"].notna()].copy()
    df["anio"] = df["fecha_dt"].dt.year
    df["mes"] = df["fecha_dt"].dt.month
    df["dia_semana"] = df["fecha_dt"].dt.dayofweek
    df["es_finde"] = (df["dia_semana"] >= 5).astype(int)
    df["hora"] = df["hora"].map(parse_hour) if "hora" in df.columns else np.nan
    log["hora_sin_interpretar"] = int(df["hora"].isna().sum())
    df["es_noche"] = np.where(df["hora"].isna(), np.nan, ((df["hora"] >= 18) | (df["hora"] < 6)).astype(float))
    df["franja"] = pd.cut(df["hora"], bins=FRANJA_BINS, labels=FRANJA_LABELS).astype(object)

    # 4. Coordenadas (comas decimales, signo perdido, fuera de Cartagena)
    lat = to_float(df["latitud"]) if "latitud" in df.columns else pd.Series(np.nan, index=df.index)
    lon = to_float(df["longitud"]) if "longitud" in df.columns else pd.Series(np.nan, index=df.index)
    flip = lon.notna() & (lon > 0) & (-lon).between(*C.BBOX["lon"])
    log["longitud_signo_corregido"] = int(flip.sum())
    lon = lon.where(~flip, -lon)
    inside = lat.between(*C.BBOX["lat"]) & lon.between(*C.BBOX["lon"])
    log["coordenadas_fuera_de_cartagena_o_ausentes"] = int((~inside).sum())
    df["lat"], df["lon"] = lat.where(inside), lon.where(inside)

    # 5. Tipo de vía: primer campo via1..via5 con valor real ("0" = vacío); si no hay, corredor de carga (via6)
    def tipo_via(row):
        for col, name in zip(VIA_COLS, VIA_NAMES):
            if col in row.index and normalize_text(row[col]) not in VIA_VACIO:
                return name
        if "via6" in row.index and normalize_text(row["via6"]) == "corredor de carga":
            return "corredor_carga"
        return "desconocido"

    df["tipo_via"] = df.apply(tipo_via, axis=1)
    log["tipo_via_conteo"] = {str(k): int(v) for k, v in df["tipo_via"].value_counts().items()}

    # 6. Categóricas de texto normalizadas
    for col in ["barrio", "localidad", "clase_accidente", "choque_con", "objeto_fijo", "hipotesis_1"]:
        df[col] = df[col].map(normalize_text).replace("", "desconocido") if col in df.columns else "desconocido"
    for col in COLS_SIN_DATO:
        df[col] = df[col].where(~df[col].isin(SIN_DATO_TOKENS), SIN_DATO)
        log[f"{col}_sin_dato_por_anio"] = {
            int(k): int(v) for k, v in (df[col] == SIN_DATO).groupby(df["anio"]).sum().items()
        }

    # 7. Partición temporal
    def split_of(y):
        if y in C.TRAIN_YEARS:
            return "train"
        if y in C.VAL_YEARS:
            return "val"
        if y in C.TEST_YEARS:
            return "test"
        return "fuera"

    df["split"] = df["anio"].map(split_of)

    keep = (
        ["ipat", "fecha_dt", "anio", "split", "gravedad_cod", "fatal", "muertos", "heridos"]
        + C.NUM_FEATURES + C.CAT_CONTEXTO + C.CAT_REPORTE + C.CAT_HIPOTESIS
    )
    keep = [c for c in dict.fromkeys(keep) if c in df.columns]
    df = df[keep].reset_index(drop=True)

    # 8. Resumen descriptivo (alimenta el informe de avance)
    log["n_limpio"] = int(len(df))
    log["anios"] = {int(k): int(v) for k, v in df["anio"].value_counts().sort_index().items()}
    log["split"] = {}
    for sp in ("train", "val", "test", "fuera"):
        sub = df[df["split"] == sp]
        log["split"][sp] = {
            "n": int(len(sub)),
            "n_fatal": int(sub["fatal"].sum()),
            "pct_fatal": round(float(sub["fatal"].mean()) * 100, 2) if len(sub) else None,
        }
    log["gravedad_cod_conteo"] = {int(k): int(v) for k, v in df["gravedad_cod"].value_counts().sort_index().items()}
    feats = [c for c in C.FEATURE_SETS["hipotesis"] if c in df.columns]
    log["faltantes_pct"] = {
        c: round(float(df[c].isna().mean() * 100 + df[c].isin(["desconocido", SIN_DATO]).mean() * 100), 1) for c in feats
    }
    log["aviso"] = []
    for sp, minimo in (("train", 500), ("val", 150), ("test", 150)):
        v = log["split"][sp]
        if v["n"] < minimo:
            log["aviso"].append(f"El split '{sp}' tiene menos de {minimo} filas: revisar fechas o reconfigurar años.")
        if v["n"] and (v["n_fatal"] < 30 or v["n"] - v["n_fatal"] < 30):
            log["aviso"].append(f"El split '{sp}' tiene menos de 30 casos en alguna clase (fatales={v['n_fatal']}).")
    return df, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=C.CLEAN_CSV)
    ap.add_argument("--summary", type=Path, default=C.EXPERIMENTS / "data_summary.json")
    a = ap.parse_args()

    src = a.input or default_input()
    print("Leyendo", src)
    df, log = clean(read_raw(src))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.summary.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.output, index=False)
    log["archivo_entrada"] = str(Path(src).name)
    a.summary.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Filas crudas: {log['n_raw']}  ->  limpias: {log['n_limpio']}")
    for sp, v in log["split"].items():
        print(f"  {sp:6s} n={v['n']:6d}  fatales={v['n_fatal']:5d} ({v['pct_fatal']}%)")
    for w in log["aviso"]:
        print("  AVISO:", w)
    print("Guardado:", a.output, "y", a.summary)


if __name__ == "__main__":
    main()
