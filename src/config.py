"""Configuración central del proyecto. Todo lo que afecta a los experimentos vive aquí."""
from pathlib import Path

SEED = 42  # semilla única para todo el proyecto

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
EXPERIMENTS = ROOT / "experiments"

# --- Fuente de datos (Portal de Datos Abiertos de la Alcaldía de Cartagena, DATT) ---
RESOURCE_ID = "f77c2eb8-0305-40f8-8c0e-fa667f6847ba"
PACKAGE_ID = "10027406-e043-43a6-9ea8-855cd27a8eb4"
BASE = "https://datosabiertos.cartagena.gov.co"
CSV_URL = f"{BASE}/datastore/dump/{RESOURCE_ID}?bom=True"
XLSX_URL = (
    f"{BASE}/dataset/{PACKAGE_ID}/resource/{RESOURCE_ID}/download/"
    "accidentes-de-transito-municipio-de-cartgaena-2020-2025.xlsx"
)
LICENSE = "Creative Commons Attribution (CC-BY)"
RAW_CSV = DATA_RAW / "accidentes_cartagena_2020_2025.csv"
RAW_XLSX = DATA_RAW / "accidentes_cartagena_2020_2025.xlsx"
CLEAN_CSV = DATA_PROCESSED / "accidentes_clean.csv"

# --- Partición TEMPORAL (evita mezclar futuro y pasado) ---
TRAIN_YEARS = (2020, 2021, 2022, 2023)
VAL_YEARS = (2024,)
TEST_YEARS = (2025,)  # NO se toca hasta la evaluación final (hito 3)

# --- Plausibilidad geográfica de Cartagena de Indias (caja generosa) ---
BBOX = {"lat": (10.20, 10.55), "lon": (-75.65, -75.30)}

# --- Columnas ---
# Definen la etiqueta => NUNCA pueden ser features (fuga de datos).
LEAKY = ["gravedad", "muertos", "heridos", "fatal", "gravedad_cod"]
ID_COLS = ["ipat"]

NUM_FEATURES = ["hora", "mes", "dia_semana", "es_finde", "es_noche", "lat", "lon"]
CAT_CONTEXTO = ["franja", "barrio", "localidad", "tipo_via"]
CAT_REPORTE = ["clase_accidente", "choque_con", "objeto_fijo"]
CAT_HIPOTESIS = ["hipotesis_1"]

FEATURE_SETS = {
    "contexto": NUM_FEATURES + CAT_CONTEXTO,                                  # cuándo y dónde
    "reporte": NUM_FEATURES + CAT_CONTEXTO + CAT_REPORTE,                     # + informe inicial (PRIMARIO)
    "hipotesis": NUM_FEATURES + CAT_CONTEXTO + CAT_REPORTE + CAT_HIPOTESIS,   # + juicio causal del agente
}
PRIMARY_FEATURE_SET = "reporte"

# Nombres de las localidades, SOLO para reportes (el modelo usa el código).
# [CONFIRMAR] El DATT no publica un diccionario de este campo. El orden coincide con la numeración oficial de las
# alcaldías locales y con los barrios de cada código en los datos (1: Centro, Bocagrande, Manga; 2: Olaya, El Pozón,
# La Boquilla; 3: Mamonal, Ceballos, San Fernando). Verificado 2026-11-22; ver experiments/exploracion.json.
LOCALIDAD_NOMBRES = {
    "1": "Histórica y del Caribe Norte",
    "2": "De la Virgen y Turística",
    "3": "Industrial y de la Bahía",
}

# Umbral de recall del punto de operación (docs/evaluacion_predeclarada.md, sección 4)
RECALL_OBJETIVO = 0.60
# Subgrupos elegibles para métricas por subgrupo
SUBGRUPO_MIN_N = 30
SUBGRUPO_MIN_FATALES = 10
