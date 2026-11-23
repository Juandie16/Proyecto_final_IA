"""Ejecuta todo el pipeline reproducible en orden (no evalúa la prueba: eso se hizo una sola vez con src.final).

Uso:  python -m src.pipeline [--rapido]
Pasos: preprocess -> data_quality -> baseline -> explore -> train -> export_model -> errors -> responsible
  * errors y responsible leen experiments/final_predicciones.csv (resultado versionado de la evaluación única);
    si no existe, se omiten.
  * Si el SHA-256 de los datos crudos no coincide con data/raw/SOURCE.json versionado, se avisa: el portal pudo
    actualizar el archivo y los números pueden diferir de los del informe.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time

from src import config as C


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_versionado() -> str | None:
    """Hash de SOURCE.json en el último commit (el que respalda el informe), si hay git."""
    try:
        r = subprocess.run(["git", "show", "HEAD:data/raw/SOURCE.json"], cwd=C.ROOT, capture_output=True, text=True, check=True)
        return json.loads(r.stdout)["sha256"]
    except (OSError, subprocess.CalledProcessError, KeyError, json.JSONDecodeError):
        return None


def paso(modulo: str, *args: str) -> None:
    t0 = time.time()
    print(f"\n=== python -m {modulo} {' '.join(args)}", flush=True)
    r = subprocess.run([sys.executable, "-m", modulo, *args], cwd=C.ROOT)
    if r.returncode != 0:
        raise SystemExit(f"Falló {modulo} (código {r.returncode}).")
    print(f"--- {modulo} OK ({time.time() - t0:.0f} s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="Búsqueda de hiperparámetros reducida (no reproduce el informe).")
    a = ap.parse_args()

    if C.RAW_CSV.exists():
        esperado, actual = hash_versionado(), sha256(C.RAW_CSV)
        if esperado and esperado != actual:
            print(f"AVISO: los datos crudos (sha256 {actual[:12]}…) no son los del informe ({esperado[:12]}…). "
                  "El portal pudo actualizarlos; los resultados pueden cambiar.")
        elif esperado:
            print(f"Datos crudos verificados: sha256 {actual[:12]}… coincide con el del informe.")
    paso("src.preprocess")
    paso("src.data_quality")
    paso("src.baseline")
    paso("src.explore")
    paso("src.train", *(["--rapido"] if a.rapido else []))
    paso("src.export_model")
    if (C.EXPERIMENTS / "final_predicciones.csv").exists():
        paso("src.errors")
        paso("src.responsible")
    else:
        print("Sin experiments/final_predicciones.csv: se omiten errors y responsible (requieren la evaluación única).")
    print("\nListo. Lance la app con:  python -m streamlit run app/app.py")


if __name__ == "__main__":
    main()
