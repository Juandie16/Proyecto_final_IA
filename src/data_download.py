"""Descarga el dataset oficial y registra su procedencia (URL, fecha, hash, licencia).

Uso:  python -m src.data_download
Si la descarga automática falla (red, portal caído), descargue manualmente el XLSX desde
la URL impresa y guárdelo en data/raw/accidentes_cartagena_2020_2025.xlsx.
"""
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone

from src import config as C


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"Descargando CSV desde:\n  {C.CSV_URL}")
    try:
        req = urllib.request.Request(C.CSV_URL, headers={"User-Agent": "proyecto-ia-academico/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(C.RAW_CSV, "wb") as f:
            f.write(r.read())
    except Exception as e:  # noqa: BLE001
        print(f"\nNo se pudo descargar automáticamente: {e}", file=sys.stderr)
        print("Descargue manualmente el XLSX y guárdelo en:", C.RAW_XLSX, file=sys.stderr)
        print("URL:", C.XLSX_URL, file=sys.stderr)
        return 1
    meta = {
        "dataset": "Accidentes de Tránsito Ocurridos en la Ciudad de Cartagena de Indias (2020-2025)",
        "publicador": "Departamento Administrativo de Tránsito y Transporte (DATT) - Alcaldía de Cartagena",
        "portal": C.BASE,
        "url_descargada": C.CSV_URL,
        "licencia": C.LICENSE,
        "fecha_descarga_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": sha256(C.RAW_CSV),
        "bytes": C.RAW_CSV.stat().st_size,
    }
    (C.DATA_RAW / "SOURCE.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Listo:", C.RAW_CSV, f"({meta['bytes']/1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
