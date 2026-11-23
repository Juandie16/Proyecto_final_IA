"""Pruebas del pipeline con datos sintéticos.  Ejecutar:  python -m unittest discover -s tests -v"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.preprocess import clean, normalize_text, parse_fecha, parse_gravedad, parse_hour, read_raw
from tests.synthetic import make_synthetic

ROOT = Path(__file__).resolve().parents[1]


class TestParsers(unittest.TestCase):
    def test_gravedad(self):
        self.assertEqual(parse_gravedad("Solo daños materiales"), 0)
        self.assertEqual(parse_gravedad("  CON HERIDOS "), 1)
        self.assertEqual(parse_gravedad("Con Muertos"), 2)
        self.assertTrue(np.isnan(parse_gravedad(None)))
        self.assertTrue(np.isnan(parse_gravedad("otro")))

    def test_hora(self):
        casos = {"14:30": 14, "14:30:00": 14, "2:30 PM": 14, "12:05 AM": 0, "12:10 PM": 12,
                 "1900-01-01 23:59:00": 23, "0930": 9, "25:00": None, "": None}
        for txt, esperado in casos.items():
            r = parse_hour(txt)
            if esperado is None:
                self.assertTrue(np.isnan(r), txt)
            else:
                self.assertEqual(r, esperado, txt)

    def test_fecha(self):
        s = pd.Series(["05/01/2021", "2022-03-07 00:00:00", "2023-12-31", "44197", "basura", None])
        r = parse_fecha(s)
        self.assertEqual(r[0], pd.Timestamp("2021-01-05"))   # d/m/Y (convención colombiana)
        self.assertEqual(r[1], pd.Timestamp("2022-03-07"))
        self.assertEqual(r[2], pd.Timestamp("2023-12-31"))
        self.assertEqual(r[3], pd.Timestamp("2021-01-01"))   # serial de Excel
        self.assertTrue(pd.isna(r[4]) and pd.isna(r[5]))

    def test_normalize(self):
        self.assertEqual(normalize_text("  Caída   de OCUPANTE "), "caida de ocupante")


class TestClean(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = make_synthetic()
        fat = cls.base[cls.base["gravedad"] == "Con Muertos"].head(30).copy()
        dup = fat.copy()
        dup["gravedad"] = "Con Heridos"            # misma IPAT, versión menos grave, ubicada ANTES
        cls.dup_ids = set(fat["ipat"])
        cls.raw = pd.concat([dup, cls.base], ignore_index=True)
        tmp = Path(tempfile.mkdtemp()) / "raw.csv"
        cls.raw.to_csv(tmp, index=False)
        cls.df, cls.log = clean(read_raw(tmp))   # pasa por read_raw: normaliza nombres de columnas

    def test_no_pierde_filas_validas(self):
        self.assertGreater(self.log["n_limpio"], 0.99 * len(self.base))

    def test_etiqueta_fatal(self):
        self.assertEqual(set(self.df["fatal"].unique()), {0, 1})
        self.assertTrue((self.df.loc[self.df.gravedad_cod == 2, "fatal"] == 1).all())
        self.assertTrue((self.df.loc[self.df.gravedad_cod == 1, "fatal"] == 0).all())

    def test_duplicados_conservan_la_fila_mas_grave(self):
        self.assertEqual(self.log["duplicados_ipat_eliminados"], 30)
        self.assertEqual(self.log["ipat_duplicados_con_gravedad_distinta"], 30)
        quedan = self.df[self.df["ipat"].isin(self.dup_ids)]
        self.assertEqual(len(quedan), 30)
        self.assertTrue((quedan["fatal"] == 1).all())

    def test_error_si_la_etiqueta_tiene_una_sola_clase(self):
        solo_heridos = self.base.copy()
        solo_heridos["gravedad"] = "Con Heridos"
        with self.assertRaises(ValueError):
            clean(solo_heridos.rename(columns={"BARRIO": "barrio"}))

    def test_split_temporal_sin_solape(self):
        r = {sp: self.df[self.df.split == sp]["anio"] for sp in ("train", "val", "test")}
        self.assertLessEqual(r["train"].max(), max(C.TRAIN_YEARS))
        self.assertGreaterEqual(r["train"].min(), min(C.TRAIN_YEARS))
        self.assertLess(r["train"].max(), r["val"].min())
        self.assertLess(r["val"].max(), r["test"].min())

    def test_coordenadas_corregidas(self):
        self.assertGreater(self.log["longitud_signo_corregido"], 0)
        self.assertGreater(self.log["coordenadas_fuera_de_cartagena_o_ausentes"], 0)
        self.assertTrue(self.df["lat"].dropna().between(*C.BBOX["lat"]).all())

    def test_sin_columnas_con_fuga_en_features(self):
        for fs, cols in C.FEATURE_SETS.items():
            self.assertFalse(set(cols) & set(C.LEAKY + C.ID_COLS), fs)


class TestCodificacionesDelDATT(unittest.TestCase):
    """Casos reales del archivo del DATT (verificados 2026-11-22): '0' como vacío y 'sin dato' que cambia por año."""

    @classmethod
    def setUpClass(cls):
        base = make_synthetic(n=400, seed=1).rename(columns={"BARRIO": "barrio"})
        for c in ["via1", "via2", "via3", "via4", "via5"]:
            base[c] = "0"
        base["via6"] = "Otro"
        base.loc[0, "via2"] = "CARRERA 10"                        # via1 = "0" -> debe ser carrera, no calle
        base.loc[1, "via5"] = "Avenida pedro de heredia"
        base.loc[2, "via6"] = "Corredor de Carga"                 # sin via1..5 -> corredor de carga
        # filas 3+ : todo "0"/"Otro" -> desconocido
        base.loc[4, "objeto_fijo"], base.loc[5, "objeto_fijo"], base.loc[6, "objeto_fijo"] = "0", "No", None
        base.loc[7, "objeto_fijo"] = "Muro"
        base.loc[4, "choque_con"], base.loc[5, "choque_con"], base.loc[6, "choque_con"] = "Choque con", "No", None
        base.loc[7, "choque_con"] = "Vehiculo"
        cls.base = base
        cls.df, cls.log = clean(base)
        cls.by_ipat = cls.df.set_index("ipat")

    def row(self, i):
        return self.by_ipat.loc[self.base.loc[i, "ipat"]]

    def test_tipo_via_ignora_ceros(self):
        self.assertEqual(self.row(0)["tipo_via"], "carrera")
        self.assertEqual(self.row(1)["tipo_via"], "avenida")
        self.assertEqual(self.row(2)["tipo_via"], "corredor_carga")
        self.assertEqual(self.row(3)["tipo_via"], "desconocido")
        self.assertNotIn("calle", self.log["tipo_via_conteo"])   # antes del arreglo, todo salía "calle"

    def test_sin_dato_unificado(self):
        for i in (4, 5, 6):
            self.assertEqual(self.row(i)["objeto_fijo"], "sin_dato", i)
            self.assertEqual(self.row(i)["choque_con"], "sin_dato", i)
        self.assertEqual(self.row(7)["objeto_fijo"], "muro")
        self.assertEqual(self.row(7)["choque_con"], "vehiculo")
        for col in ("objeto_fijo", "choque_con"):
            self.assertFalse(self.df[col].isin(["0", "no", "desconocido", "choque con"]).any(), col)


class TestExploracion(unittest.TestCase):
    def test_wilson_y_resumen_sin_prueba(self):
        from src.explore import resumen, wilson
        lo, hi = wilson(10, 100)
        self.assertTrue(0 < lo < 0.10 < hi < 0.2)
        self.assertEqual(wilson(0, 50)[0], 0.0)
        df, _ = clean(make_synthetic(n=1500, seed=3).rename(columns={"BARRIO": "barrio"}))
        res = resumen(df, {})
        # las tasas descriptivas NO usan la prueba (2025)
        self.assertEqual(res["n_periodo_tasas"], int(df["split"].isin(["train", "val"]).sum()))


class TestEvaluacion(unittest.TestCase):
    def test_umbral_por_recall_es_el_mayor_que_cumple(self):
        from src.evaluacion import metricas, umbral_por_recall
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 500)
        p = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
        u = umbral_por_recall(y, p, 0.60)
        self.assertGreaterEqual(metricas(y, p, u)["recall"], 0.60)
        # un umbral apenas mayor ya no cumple
        self.assertLess(metricas(y, p, np.nextafter(u, 2))["recall"], 0.60)

    def test_bootstrap_contiene_el_valor_puntual(self):
        from src.evaluacion import bootstrap_ic, metricas
        rng = np.random.default_rng(1)
        y = (rng.random(800) < 0.1).astype(int)
        p = np.clip(y * 0.2 + rng.random(800) * 0.8, 0, 1)
        ic = bootstrap_ic(y, p, 0.5, n_boot=200)
        m = metricas(y, p, 0.5)
        for k in ("pr_auc", "roc_auc", "recall"):
            self.assertLessEqual(ic[k][0], m[k]); self.assertGreaterEqual(ic[k][1], m[k])


class TestErrores(unittest.TestCase):
    def test_categorias_cubren_todo_y_respetan_el_orden(self):
        from types import SimpleNamespace as R
        from src.errors import categorizar
        dia = R(clase_accidente="choque", choque_con="vehiculo", franja="tarde", objeto_fijo="sin_dato")
        noche = R(clase_accidente="choque", choque_con="vehiculo", franja="noche", objeto_fijo="sin_dato")
        self.assertEqual(categorizar(dia, "FN")[0], "FN1")
        self.assertEqual(categorizar(noche, "FN")[0], "FN2")
        self.assertEqual(categorizar(R(clase_accidente="otro", choque_con="x", franja="tarde", objeto_fijo="sin_dato"), "FN")[0], "FN4")
        muro = R(clase_accidente="choque", choque_con="objeto fijo", franja="madrugada", objeto_fijo="muro")
        self.assertEqual(categorizar(muro, "FP")[0], "FP1")          # objeto fijo gana sobre "nocturno"
        self.assertEqual(categorizar(noche, "FP")[0], "FP3")
        self.assertEqual(categorizar(dia, "FP")[0], "FP4")


class TestResponsable(unittest.TestCase):
    def test_subgrupos_no_elegibles_no_se_evaluan(self):
        from src.responsible import por_subgrupo, solo_elegibles
        rng = np.random.default_rng(2)
        df = pd.DataFrame({"g": ["a"] * 400 + ["b"] * 25, "fatal": np.r_[(rng.random(400) < 0.1).astype(int), np.ones(25, int)],
                           "p": rng.random(425)})
        sub = por_subgrupo(df, "p", 0.5, "g", n_boot=50)
        self.assertTrue(sub["a"]["elegible"])
        self.assertFalse(sub["b"]["elegible"])           # n < 30
        self.assertNotIn("recall", sub["b"])
        self.assertEqual(list(solo_elegibles({"g": sub})["g"]), ["a"])


class TestEndToEnd(unittest.TestCase):
    def test_train_rapido_no_toca_la_prueba(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            make_synthetic(n=3000).to_csv(d / "raw.csv", index=False)
            run = lambda *a: subprocess.run([sys.executable, "-m", *a], cwd=ROOT, capture_output=True, text=True)
            p1 = run("src.preprocess", "--input", str(d / "raw.csv"), "--output", str(d / "c.csv"), "--summary", str(d / "s.json"))
            self.assertEqual(p1.returncode, 0, p1.stderr)
            self.assertEqual(run("src.baseline", "--processed", str(d / "c.csv"), "--outdir", str(d)).returncode, 0)
            p2 = run("src.train", "--rapido", "--processed", str(d / "c.csv"), "--outdir", str(d))
            self.assertEqual(p2.returncode, 0, p2.stderr)
            res = json.loads((d / "modelos_val.json").read_text(encoding="utf-8"))
            self.assertNotIn(2025, [pl["valida"] for pl in res["pliegues"]])
            for pl in res["pliegues"]:
                self.assertTrue(all(a < pl["valida"] for a in pl["entrena"]))   # origen móvil: pasado -> futuro
            preds = pd.read_csv(d / "modelo_val_predicciones.csv")
            self.assertEqual(set(preds["anio"]), {2024})
            el = json.loads((d / "modelo_elegido.json").read_text(encoding="utf-8"))
            self.assertIn(el["modelo"], ("regresion_logistica_ajustada", "random_forest", "hist_gradient_boosting"))
            self.assertGreaterEqual(res["metricas_elegido_semilla_base"]["recall"], 0.60)
            self.assertTrue((d / "tabla_comparativa.csv").exists())

            # Fase 3: final.py se niega sin confirmación, con un modelo de corrida rápida y una segunda vez.
            fin = lambda *x: run("src.final", "--processed", str(d / "c.csv"), "--outdir", str(d), "--sin-git",
                               "--models-dir", str(d / "models"), *x)
            self.assertNotEqual(fin().returncode, 0)
            self.assertNotEqual(fin("--confirmo-evaluacion-unica").returncode, 0)        # modo_rapido
            el["modo_rapido"] = False
            (d / "modelo_elegido.json").write_text(json.dumps(el), encoding="utf-8")
            p3 = fin("--confirmo-evaluacion-unica")
            self.assertEqual(p3.returncode, 0, p3.stderr)
            fr = json.loads((d / "resultado_final.json").read_text(encoding="utf-8"))
            self.assertEqual(fr["evaluado_en"], "prueba 2025")
            self.assertAlmostEqual(fr["resultados"][el["modelo"]]["umbral"], el["umbral"], places=5)  # umbral sin tocar
            self.assertEqual(len(fr["criterios"]), 5)
            self.assertNotEqual(fin("--confirmo-evaluacion-unica").returncode, 0)        # una sola vez

    def test_baseline_corre_y_supera_al_mayoritario_en_auc(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            make_synthetic().to_csv(d / "raw.csv", index=False)
            run = lambda *a: subprocess.run([sys.executable, "-m", *a], cwd=ROOT, capture_output=True, text=True)
            p1 = run("src.preprocess", "--input", str(d / "raw.csv"), "--output", str(d / "c.csv"), "--summary", str(d / "s.json"))
            self.assertEqual(p1.returncode, 0, p1.stderr)
            p2 = run("src.baseline", "--processed", str(d / "c.csv"), "--outdir", str(d))
            self.assertEqual(p2.returncode, 0, p2.stderr)
            res = json.loads((d / "baseline_val.json").read_text())
            self.assertEqual(res["evaluado_en"], "val")           # el test NO se toca sin --final
            self.assertFalse((d / "baseline_test.json").exists())
            m = res["modelos"]
            self.assertAlmostEqual(m["mayoritario"]["roc_auc"], 0.5, places=2)
            self.assertGreater(m["regresion_logistica"]["roc_auc"], 0.6)
            # reproducibilidad: misma semilla => mismo resultado
            p3 = run("src.baseline", "--processed", str(d / "c.csv"), "--outdir", str(d / "r2"))
            res2 = json.loads((d / "r2" / "baseline_val.json").read_text())
            self.assertEqual(res["modelos"], res2["modelos"])


if __name__ == "__main__":
    unittest.main()
