"""Pruebas de caracterizacion de P2: fijan lo que hace hoy el entrenamiento antes de refactorizar.

No buscan decir que el comportamiento sea el ideal, solo que un refactor no lo cambie sin
querer. Usan un panel horario sintetico pequeno, cortes de fecha reducidos y una rejilla corta,
y todo lo que se escribe (panel, manifiesto, figuras, base de MLflow y artefactos) va a tmp_path.
Donde el comportamiento actual es discutible se deja un comentario con la observacion.
"""

import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from mibici import features_p2 as fp
from mibici import mlflow_project as mp
from mibici import registro
from mibici.config import CONFIG_P2, PROJ_ROOT
from mibici.modeling import train_p2 as t2

# Cortes reducidos (mismo orden y mismas reglas de inicio incluido y fin excluido que los reales)
CORTES_CHICOS = {
    "train_inicio": "2026-01-05",
    "val_inicio": "2026-02-02",
    "test_inicio": "2026-02-16",
    "test_fin": "2026-03-02",
}
REJILLA_CORTA = (
    {"learning_rate": 0.1, "max_leaf_nodes": 7, "max_iter": 20, "l2_regularization": 0.0},
    {"learning_rate": 0.05, "max_leaf_nodes": 15, "max_iter": 30, "l2_regularization": 1.0},
)
ETIQUETAS_BASE = {
    "question", "team", "author", "data_range", "seed", "data_version", "git_commit", "git_dirty",
}  # fmt: skip
ETIQUETAS_MODELO = {"modelo", "model_family", "stage_of_work"}
VERSION = "abc123def456"
RF_CHICO = {"n_estimators": 10, "max_samples": 500, "min_samples_leaf": 5}
CONFIG_CHICA = {
    "rejilla": REJILLA_CORTA,
    "params_logistica": {"max_iter": 100},
    "params_random_forest": RF_CHICO,
    "muestra_importancia": 200,
    "repeticiones_importancia": 1,
}
CANDIDATAS_NUEVAS = ("p2_logistica_balanceada", "p2_random_forest_balanceado")
BUSQUEDA_INI50 = {"p2_busqueda_ini50", "p2_busqueda_ini50_c1", "p2_busqueda_ini50_c2"}
RUNS_INI50 = {
    "p2_baseline_mayoritaria_ini50",
    "p2_baseline_semana_previa_ini50",
    "p2_baseline_persistencia_ini50",
    "p2_hgb_balanceado_ini50",
    "p2_hgb_sin_pesos_ini50",
    "p2_logistica_balanceada_ini50",
    "p2_random_forest_balanceado_ini50",
}


class EstimadorAjeno(BaseEstimator):
    """Estimador que no es de sklearn ni de numpy, para probar la lista de tipos confiables."""


def hacer_panel(estaciones=(1, 2, 3), nueva=4) -> pd.DataFrame:
    """Panel estacion-hora sintetico, como el que arma features_p2.construir_panel.

    Cada estacion trae dias de servicio completos (de 5:00 a 4:00) del 2026-01-01 al 2026-03-01.
    Por la manana salen muchas bicicletas y por la tarde llegan muchas (la estacion 2 al reves),
    asi que las tres clases aparecen todos los dias. La estacion nueva empieza el primer dia
    de la prueba, de modo que cuenta como estacion nueva en test.
    """
    rng = np.random.default_rng(29)
    piezas = []
    for estacion in (*estaciones, nueva):
        inicio = "2026-02-16 05:00" if estacion == nueva else "2026-01-01 05:00"
        horas = pd.date_range(inicio, "2026-03-02 05:00", freq="h", inclusive="left")
        h = horas.hour.to_numpy()
        manana, tarde = (h >= 7) & (h <= 9), (h >= 17) & (h <= 19)
        if estacion == 2:
            manana, tarde = tarde, manana
        salidas = 1 + 4 * manana + rng.integers(0, 2, len(horas))
        llegadas = 1 + 4 * tarde + rng.integers(0, 2, len(horas))
        piezas.append(
            pd.DataFrame(
                {
                    "ts": horas,
                    "estacion": estacion,
                    "capacidad": np.float32(10),
                    "salidas": salidas.astype("float32"),
                    "llegadas": llegadas.astype("float32"),
                    "primer_dia": horas[0],
                }
            )
        )
    panel = pd.concat(piezas, ignore_index=True)
    panel["neto"] = panel["llegadas"] - panel["salidas"]
    return panel


def escribir_entradas(carpeta: Path, panel: pd.DataFrame) -> dict[str, Path]:
    """Escribe panel, resumen del panel y manifiesto en carpeta, como los lee train_p2.main."""
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_panel = carpeta / "panel.parquet"
    panel.to_parquet(ruta_panel, index=False)
    resumen = {
        "estaciones_modeladas": int(panel["estacion"].nunique()),
        "estaciones_sin_nomenclatura": 2,
        "estaciones_capacidad_cero": 1,
    }
    ruta_resumen = ruta_panel.with_suffix(".json")
    ruta_resumen.write_text(json.dumps(resumen))
    ruta_manifiesto = carpeta / "manifiesto.json"
    ruta_manifiesto.write_text(json.dumps({"version": VERSION}))
    return {"panel": ruta_panel, "resumen": ruta_resumen, "manifiesto": ruta_manifiesto}


def archivos_de(cliente, run_id: str, carpeta: str | None = None) -> set[str]:
    """Rutas de todos los archivos de los artefactos de un run, entrando a las subcarpetas."""
    rutas = set()
    for a in cliente.list_artifacts(run_id, carpeta):
        if a.is_dir:
            rutas |= archivos_de(cliente, run_id, a.path)
        else:
            rutas.add(a.path)
    return rutas


def foto_del_repo() -> set[str]:
    """Rutas relativas de todos los archivos del repo (sin .git ni caches de herramientas)."""
    omitir = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
    archivos = set()
    for raiz, carpetas, nombres in os.walk(PROJ_ROOT):
        carpetas[:] = [c for c in carpetas if c not in omitir]
        archivos.update(str((Path(raiz) / n).relative_to(PROJ_ROOT)) for n in nombres)
    return archivos


@pytest.fixture(autouse=True)
def cortes_y_configuracion_chicos(monkeypatch, tmp_path):
    """Cortes reducidos y CONFIG_P2 con rejilla corta y MLflow dentro de tmp_path."""
    for clave, valor in CORTES_CHICOS.items():
        monkeypatch.setitem(t2.CORTES, clave, valor)
    config = dataclasses.replace(
        CONFIG_P2,
        **CONFIG_CHICA,
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow_auto.db'}",
        artefactos_dir=tmp_path / "mlartifacts_auto",
        corridas_dir=tmp_path / "corridas_auto",
    )
    monkeypatch.setattr(t2, "CONFIG_P2", config)
    # MLflow guarda la direccion del almacen en estado global: se restaura al terminar
    uri_previa = mlflow.get_tracking_uri()
    variable_previa = os.environ.get("MLFLOW_TRACKING_URI")
    yield
    mlflow.set_tracking_uri(uri_previa)
    if variable_previa is None:
        os.environ.pop("MLFLOW_TRACKING_URI", None)
    else:
        os.environ["MLFLOW_TRACKING_URI"] = variable_previa


def correr_main(carpeta: Path, extra: list[str] | None = None, niveles=("0.5",)) -> dict:
    """Corre train_p2.main con un panel chico y todas las rutas dentro de carpeta."""
    entradas = escribir_entradas(carpeta, hacer_panel())
    figuras = carpeta / "figuras"
    uri = f"sqlite:///{carpeta / 'mlflow.db'}"
    args = [
        "--panel", str(entradas["panel"]), "--manifiesto", str(entradas["manifiesto"]),
        "--figuras", str(figuras), "--tracking-uri", uri, "--niveles", *niveles,
        "--artefactos", str(carpeta / "mlartifacts"),
        *(extra or []),
    ]  # fmt: skip
    t2.main(args)
    cliente = MlflowClient(tracking_uri=uri)
    experimento = cliente.get_experiment_by_name(CONFIG_P2.experimento)
    runs = cliente.search_runs([experimento.experiment_id], max_results=100)
    return {
        "cliente": cliente,
        "experimento": experimento,
        "runs": {r.info.run_name: r for r in runs},
        "uri": uri,
        "figuras": figuras,
        "carpeta": carpeta,
        "resultados": json.loads((figuras / "p2_resultados.json").read_text()),
        "entradas": entradas,
    }


def correr_con_config_chica(fabrica, nombre: str, extra=None, niveles=("0.5",)) -> dict:
    """Corre main con cortes y config reducidos (para fixtures de modulo, fuera del autouse)."""
    with pytest.MonkeyPatch.context() as mpatch:
        for clave, valor in CORTES_CHICOS.items():
            mpatch.setitem(t2.CORTES, clave, valor)
        config = dataclasses.replace(CONFIG_P2, **CONFIG_CHICA)
        mpatch.setattr(t2, "CONFIG_P2", config)
        uri_previa = mlflow.get_tracking_uri()
        try:
            return correr_main(fabrica.mktemp(nombre), extra, niveles)
        finally:
            mlflow.set_tracking_uri(uri_previa)


@pytest.fixture(scope="module")
def corrida_completa(tmp_path_factory) -> dict:
    """main con afinacion y dos niveles iniciales; registra que archivos nuevos hay en el repo."""
    antes = foto_del_repo()
    corrida = correr_con_config_chica(tmp_path_factory, "completa", niveles=("0.5", "0.3"))
    corrida["nuevos_en_repo"] = foto_del_repo() - antes
    return corrida


@pytest.fixture(scope="module")
def corrida_sin_afinar(tmp_path_factory) -> dict:
    """main con --sin-afinar: usa la primera configuracion de la rejilla y no hay rejilla."""
    return correr_con_config_chica(tmp_path_factory, "sin_afinar", ["--sin-afinar"])


@pytest.fixture(scope="module")
def corrida_con_hiperparametros(tmp_path_factory) -> dict:
    """main con --hyperparameters: se salta la afinacion y mezcla con la primera configuracion."""
    return correr_con_config_chica(
        tmp_path_factory,
        "hiperparametros",
        ["--hyperparameters", '{"max_iter": 12, "l2_regularization": 0.5}'],
    )


# particionar y rango_meses


def test_cortes_salen_de_la_configuracion_y_rango_meses_excluye_el_dia_final(monkeypatch):
    monkeypatch.undo()  # sin los cortes reducidos del autouse
    assert t2.CORTES == {
        "train_inicio": CONFIG_P2.fecha_inicio_entrenamiento,
        "val_inicio": CONFIG_P2.fecha_inicio_validacion,
        "test_inicio": CONFIG_P2.fecha_inicio_prueba,
        "test_fin": CONFIG_P2.fecha_fin_prueba,
    }
    assert t2.RIESGO == ["sin_bicicletas", "sin_anclajes"]
    # las carpetas de salida por defecto estan ignoradas por git: correr P2 no ensucia el repo
    for carpeta in (CONFIG_P2.corridas_dir, CONFIG_P2.artefactos_dir):
        ignorada = subprocess.run(
            ["git", "check-ignore", "-q", str(carpeta / "archivo.png")], check=False, cwd=PROJ_ROOT
        )
        assert ignorada.returncode == 0, carpeta
    assert t2.rango_meses("2025-01-01", "2026-05-01") == "2025-01..2026-04"
    assert t2.rango_meses("2026-05-01", "2026-10-01") == "2026-05..2026-09"
    assert t2.rango_meses("2026-03-01", "2026-04-15") == "2026-03..2026-04"  # fin a mitad de mes


def test_particionar_respeta_limites_no_solapa_y_usa_la_fecha_objetivo():
    un_seg = pd.Timedelta(seconds=1)
    c = {k: pd.Timestamp(v) for k, v in CORTES_CHICOS.items()}
    marcas = {
        "antes_train": c["train_inicio"] - un_seg,
        "train_inicio": c["train_inicio"],
        "fin_fit": c["val_inicio"] - un_seg,
        "val_inicio": c["val_inicio"],
        "fin_val": c["test_inicio"] - un_seg,
        "test_inicio": c["test_inicio"],
        "fin_test": c["test_fin"] - un_seg,
        "test_fin": c["test_fin"],
    }
    df = pd.DataFrame({"ts_objetivo": list(marcas.values()), "marca": list(marcas)})
    # la columna ts no cuenta: solo ts_objetivo define el conjunto
    df["ts"] = df["ts_objetivo"] - pd.Timedelta(days=60)
    partes = t2.particionar(df)
    assert set(partes) == {"fit", "val", "train", "test"}
    assert partes["fit"]["marca"].tolist() == ["train_inicio", "fin_fit"]
    assert partes["val"]["marca"].tolist() == ["val_inicio", "fin_val"]
    assert partes["train"]["marca"].tolist() == [
        "train_inicio",
        "fin_fit",
        "val_inicio",
        "fin_val",
    ]
    assert partes["test"]["marca"].tolist() == ["test_inicio", "fin_test"]  # fin excluido


def test_particionar_falla_si_alguna_particion_queda_vacia():
    # igual que train_p1.particionar: sin fechas de prueba (o de otro corte) no se sigue en silencio
    sin_test = pd.DataFrame({"ts_objetivo": pd.date_range("2026-01-05", "2026-02-10", freq="D")})
    with pytest.raises(ValueError, match="test"):
        t2.particionar(sin_test)
    sin_val = pd.DataFrame({"ts_objetivo": pd.date_range("2026-02-16", "2026-02-20", freq="D")})
    with pytest.raises(ValueError, match="fit"):
        t2.particionar(sin_val)


# metricas


def test_metricas_con_prediccion_perfecta_y_prefijo():
    y = np.array(["segura", "sin_bicicletas", "sin_anclajes", "segura"], dtype=object)
    m = t2.metricas(y, y, None, fp.CLASES, "test_nuevas")
    esperadas = {"test_nuevas_macro_f1"} | {
        f"test_nuevas_{k}_{c}" for c in fp.CLASES for k in ("precision", "recall", "f1")
    }
    assert set(m) == esperadas  # sin probas no hay PR AUC
    assert all(v == pytest.approx(1.0) and type(v) is float for v in m.values())


def test_metricas_con_valores_conocidos_y_clase_ausente():
    y_true = np.array(["segura", "segura", "sin_bicicletas", "sin_anclajes"], dtype=object)
    y_pred = np.array(["segura", "sin_bicicletas", "sin_bicicletas", "segura"], dtype=object)
    m = t2.metricas(y_true, y_pred, None, fp.CLASES, "t")
    assert m["t_precision_segura"] == pytest.approx(1 / 2)
    assert m["t_recall_segura"] == pytest.approx(1 / 2)
    assert m["t_precision_sin_bicicletas"] == pytest.approx(1 / 2)
    assert m["t_recall_sin_bicicletas"] == pytest.approx(1.0)
    assert m["t_f1_sin_bicicletas"] == pytest.approx(2 / 3)
    assert m["t_precision_sin_anclajes"] == 0.0 and m["t_recall_sin_anclajes"] == 0.0
    assert m["t_macro_f1"] == pytest.approx((1 / 2 + 2 / 3 + 0) / 3)
    # una clase ausente cuenta como 0 en el macro F1 (aqui 1/3 con todo acertado)
    solo_segura = np.array(["segura", "segura"], dtype=object)
    assert t2.metricas(solo_segura, solo_segura, None, fp.CLASES, "t")["t_macro_f1"] == (
        pytest.approx(1 / 3)
    )


def test_macro_f1_con_clase_predicha_que_no_aparece_en_la_verdad():
    # predecir una clase que no existe en y_true no se ignora: esa clase aporta F1 de 0
    y_true = np.array(["segura", "segura"], dtype=object)
    y_pred = np.array(["segura", "sin_anclajes"], dtype=object)
    assert t2.macro_f1(y_true, y_pred) == pytest.approx((2 / 3 + 0 + 0) / 3)


def test_metricas_pr_auc_solo_con_probas_y_de_las_clases_de_riesgo_presentes():
    y = np.array(["segura", "sin_bicicletas", "sin_anclajes", "sin_bicicletas"], dtype=object)
    # el orden de columnas de proba es el de clases (el de classes_ de sklearn, alfabetico)
    clases = ["segura", "sin_anclajes", "sin_bicicletas"]
    proba = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]], dtype=float
    )
    pred = np.array(clases, dtype=object)[proba.argmax(1)]
    m = t2.metricas(y, pred, proba, clases, "test")
    assert m["test_pr_auc_sin_bicicletas"] == pytest.approx(1.0)
    assert m["test_pr_auc_sin_anclajes"] == pytest.approx(1.0)
    assert "test_pr_auc_segura" not in m
    assert "test_pr_auc_sin_bicicletas" not in t2.metricas(y, pred, None, clases, "test")
    # si una clase de riesgo no aparece en y_true se omite su PR AUC
    sin_anclajes = np.array(["segura", "sin_bicicletas", "segura"], dtype=object)
    p = np.array([[0.7, 0.1, 0.2], [0.1, 0.1, 0.8], [0.6, 0.2, 0.2]])
    m = t2.metricas(sin_anclajes, sin_anclajes, p, clases, "test")
    assert "test_pr_auc_sin_bicicletas" in m and "test_pr_auc_sin_anclajes" not in m


# figura_confusion


def test_figura_confusion_guarda_un_png_con_una_o_varias_clases(tmp_path):
    y = np.array(["segura", "sin_bicicletas", "sin_anclajes", "segura"], dtype=object)
    pred = np.array(["segura", "segura", "sin_anclajes", "sin_bicicletas"], dtype=object)
    ruta = tmp_path / "cm.png"
    assert t2.figura_confusion(y, pred, "Titulo", ruta) == ruta
    assert ruta.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    una = np.array(["segura", "segura"], dtype=object)
    assert t2.figura_confusion(una, una, "Una clase", tmp_path / "una.png").stat().st_size > 0


def test_figura_pr_dibuja_las_dos_clases_de_riesgo_y_omite_las_ausentes(tmp_path):
    y = np.array(["segura", "sin_bicicletas", "sin_anclajes", "sin_bicicletas"], dtype=object)
    clases = ["segura", "sin_anclajes", "sin_bicicletas"]
    proba = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]], dtype=float
    )
    ruta = t2.figura_pr(y, proba, clases, "PR", tmp_path / "pr.png")
    assert ruta.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # sin sin_anclajes en y_true solo queda una curva y no falla
    solo = y[y != "sin_anclajes"]
    assert t2.figura_pr(solo, proba[y != "sin_anclajes"], clases, "PR", tmp_path / "pr2.png")


def test_heatmap_error_toma_las_estaciones_de_mas_volumen_y_guarda_la_matriz(tmp_path):
    test = pd.DataFrame(
        {
            "estacion": [1, 1, 2, 2, 3, 3],
            "hora": [8, 9, 8, 9, 8, 9],
            "objetivo": ["segura"] * 6,
        }
    )
    pred = ["segura", "sin_anclajes", "sin_anclajes", "sin_anclajes", "segura", "segura"]
    volumen = pd.Series({1: 50.0, 2: 10.0, 3: 30.0, 9: 99.0})
    ruta, vista = t2.heatmap_error(test, pred, volumen, tmp_path / "heat.png", top_n=2)
    assert ruta.exists()
    assert list(vista.index) == [1, 3]  # top 2 por volumen, ordenadas por error medio
    assert vista.loc[1].tolist() == [0.0, 1.0]
    assert pd.read_csv(tmp_path / "heat.csv", index_col=0).shape == (3, 2)


def test_tipos_confiables_solo_acepta_tipos_de_sklearn_y_numpy():
    partes = t2.preparar(0.5, hacer_panel())
    train = fp.unir_stats(partes["train"], *fp.stats_estacion(partes["train"]))
    bosque = t2.ajustar_random_forest(train, RF_CHICO, 7)
    tipos = t2.tipos_confiables(bosque)
    assert tipos and all(t.startswith(("sklearn.", "numpy.")) for t in tipos)

    with pytest.raises(ValueError, match="no confiables"):
        t2.tipos_confiables(EstimadorAjeno())


# preparar, ajustar y afinar


def test_preparar_etiqueta_construye_features_y_particiona():
    partes = t2.preparar(0.5, hacer_panel())
    assert set(partes) == {"fit", "val", "train", "test"}
    for parte in partes.values():
        assert set(parte["objetivo"].unique()) <= set(fp.CLASES)
        assert parte["hora"].between(min(CONFIG_P2.horas_etiqueta), 23).all()
    assert set(partes["train"]["objetivo"].unique()) == set(fp.CLASES)
    assert partes["test"]["ts_objetivo"].min() >= pd.Timestamp(CORTES_CHICOS["test_inicio"])
    assert partes["test"]["ts_objetivo"].max() < pd.Timestamp(CORTES_CHICOS["test_fin"])
    assert len(partes["train"]) == len(partes["fit"]) + len(partes["val"])


def test_ajustar_usa_pesos_balanceados_semilla_y_sin_parada_temprana():
    partes = t2.preparar(0.5, hacer_panel())
    fit = fp.unir_stats(partes["fit"], *fp.stats_estacion(partes["fit"]))
    modelo = t2.ajustar(fit, REJILLA_CORTA[0], 7)
    p = modelo.get_params()
    assert p["class_weight"] == "balanced" and p["random_state"] == 7
    assert p["early_stopping"] is False
    assert p["max_iter"] == 20 and p["max_leaf_nodes"] == 7
    # las clases quedan en orden alfabetico, distinto de fp.CLASES
    assert list(modelo.classes_) == ["segura", "sin_anclajes", "sin_bicicletas"]
    # sin pesos solo cambia class_weight; el resto de la configuracion es la misma
    sin_pesos = t2.ajustar(fit, REJILLA_CORTA[0], 7, balanceado=False).get_params()
    assert sin_pesos["class_weight"] is None
    assert {k: v for k, v in sin_pesos.items() if k != "class_weight"} == {
        k: v for k, v in p.items() if k != "class_weight"
    }


def test_candidatas_logistica_y_random_forest_se_ajustan_balanceadas_y_se_miden():
    partes = t2.preparar(0.5, hacer_panel())
    train = fp.unir_stats(partes["train"], *fp.stats_estacion(partes["train"]))
    logistica = t2.ajustar_logistica(train, {"max_iter": 100}, 7)
    assert logistica[-1].get_params()["class_weight"] == "balanced"
    assert logistica[-1].get_params()["random_state"] == 7
    bosque = t2.ajustar_random_forest(train, RF_CHICO, 7)
    rf = bosque[-1]
    assert rf.get_params()["class_weight"] == "balanced_subsample"
    assert rf.n_estimators == 10 and rf.max_samples == 500 and rf.min_samples_leaf == 5
    # max_samples se recorta al tamano de train para no fallar con paneles chicos
    chico = t2.ajustar_random_forest(train.head(50), RF_CHICO, 7)[-1]
    assert chico.max_samples == 50
    for modelo in (logistica, bosque):
        clases, proba, pred = t2.predecir(modelo, train.head(20))
        assert clases == ["segura", "sin_anclajes", "sin_bicicletas"]
        assert proba.shape == (20, 3) and np.allclose(proba.sum(1), 1.0)
        assert set(pred) <= set(fp.CLASES)
    # medir_ajuste regresa el modelo, los segundos de ajuste y su tamano en MB (pickle)
    modelo, segundos, mb = t2.medir_ajuste(lambda df: t2.ajustar_logistica(df, {}, 7), train)
    assert segundos > 0 and 0 < mb < 5 and hasattr(modelo, "predict_proba")
    # el bosque con hojas minimas mas grandes pesa menos que uno de hojas de una muestra
    grande = t2.ajustar_random_forest(train, {**RF_CHICO, "min_samples_leaf": 1}, 7)
    assert (
        t2.medir_ajuste(lambda df: grande, train)[2] > t2.medir_ajuste(lambda df: bosque, train)[2]
    )


def test_afinar_elige_por_macro_f1_de_validacion_y_regresa_la_rejilla():
    partes = t2.preparar(0.5, hacer_panel())
    mejor, rejilla = t2.afinar(partes, 29)
    assert len(rejilla) == len(REJILLA_CORTA)
    for fila, params in zip(rejilla, REJILLA_CORTA, strict=True):
        assert {k: v for k, v in fila.items() if k != "val_macro_f1"} == params
        assert 0.0 <= fila["val_macro_f1"] <= 1.0
    esperado = max(rejilla, key=lambda r: r["val_macro_f1"])
    assert mejor == {k: v for k, v in esperado.items() if k != "val_macro_f1"}
    assert t2.afinar(partes, 29) == (mejor, rejilla)  # misma semilla, mismo resultado


def test_afinar_en_empate_se_queda_con_la_primera_configuracion(monkeypatch):
    partes = t2.preparar(0.5, hacer_panel())

    class Constante:
        classes_ = np.array(["segura", "sin_anclajes", "sin_bicicletas"], dtype=object)

        def predict_proba(self, x):
            return np.tile([1.0, 0.0, 0.0], (len(x), 1))

    monkeypatch.setattr(t2, "ajustar", lambda train, params, semilla: Constante())
    mejor, rejilla = t2.afinar(partes, 29)
    assert rejilla[0]["val_macro_f1"] == rejilla[1]["val_macro_f1"]
    assert mejor == REJILLA_CORTA[0]
    # afinar y metricas comparten el macro F1 con las tres clases fijas: si la validacion solo
    # trae una clase y se acierta, el valor es 1/3 (antes afinar promediaba solo las presentes)
    solo_segura = {**partes, "val": partes["val"][partes["val"]["objetivo"] == "segura"]}
    assert t2.afinar(solo_segura, 29)[1][0]["val_macro_f1"] == pytest.approx(1 / 3)


def test_afinar_ajusta_solo_con_fit(monkeypatch):
    partes = t2.preparar(0.5, hacer_panel())
    vistos = []
    original = t2.ajustar

    def espiar(train, params, semilla):
        vistos.append((train["ts_objetivo"].min(), train["ts_objetivo"].max(), len(train)))
        return original(train, params, semilla)

    monkeypatch.setattr(t2, "ajustar", espiar)
    t2.afinar(partes, 29)
    assert len(vistos) == len(REJILLA_CORTA)
    for minimo, maximo, n in vistos:
        assert minimo >= pd.Timestamp(CORTES_CHICOS["train_inicio"])
        assert maximo < pd.Timestamp(CORTES_CHICOS["val_inicio"])  # nunca ve validacion ni test
        assert n == len(partes["fit"])


# correr_nivel


def test_correr_nivel_regresa_resultados_y_ajusta_el_modelo_final_con_train(tmp_path, monkeypatch):
    entradas = escribir_entradas(tmp_path, hacer_panel())
    panel = pd.read_parquet(entradas["panel"])
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'directo.db'}")
    mlflow.create_experiment("directo", artifact_location=(tmp_path / "arte").as_uri())
    mlflow.set_experiment("directo")
    args = argparse.Namespace(
        figuras=str(tmp_path / "figuras"),
        semilla=29,
        manifiesto=str(entradas["manifiesto"]),
        panel=str(entradas["panel"]),
    )
    vistos = []
    original = t2.ajustar

    def espiar(train, params, semilla, **pesos):
        vistos.append((train["ts_objetivo"].min(), train["ts_objetivo"].max()))
        return original(train, params, semilla, **pesos)

    monkeypatch.setattr(t2, "ajustar", espiar)
    salida = t2.correr_nivel(
        0.5,
        panel,
        args,
        {"version": VERSION},
        json.loads(entradas["resumen"].read_text()),
        REJILLA_CORTA[0],
        [],
        {"question": "P2"},
        sin_pesos=True,
    )
    assert list(salida) == [
        "baseline_mayoritaria",
        "baseline_semana_previa",
        "baseline_persistencia",
        "logistica",
        "hgb",
        "random_forest",
        "hgb_sin_pesos",
    ]
    # cada HGB (con y sin pesos) se ajusta dos veces: con fit para la validacion y con train
    # completo (fit + val) para la prueba; nunca ve la prueba
    assert len(vistos) == 4
    fin_fit = pd.Timestamp(CORTES_CHICOS["val_inicio"])
    fin_train = pd.Timestamp(CORTES_CHICOS["test_inicio"])
    for (_, ultimo_fit), (_, ultimo_train) in (vistos[0:2], vistos[2:4]):
        assert ultimo_fit < fin_fit <= ultimo_train < fin_train
    assert salida["hgb"]["val_macro_f1"] > 0  # la validacion se registra aunque no haya rejilla
    figuras = {p.name for p in (tmp_path / "figuras").iterdir()}
    assert {
        "p2_cm_baseline_semana_ini50.png",
        "p2_cm_baseline_persistencia_ini50.png",
        "p2_cm_logistica_ini50.png",
        "p2_cm_logistica_nuevas_ini50.png",
        "p2_cm_hgb_ini50.png",
        "p2_cm_hgb_nuevas_ini50.png",
        "p2_cm_random_forest_ini50.png",
        "p2_cm_random_forest_nuevas_ini50.png",
        "p2_cm_hgb_sin_pesos_ini50.png",
        "p2_importancia_permutacion_ini50.csv",
        "p2_importancia_permutacion_ini50.png",
        "p2_heatmap_error_ini50.png",
        "p2_heatmap_error_ini50.csv",
    } <= figuras
    assert "p2_rejilla_validacion.json" not in figuras


# main de extremo a extremo


def test_main_crea_los_runs_esperados(corrida_completa):
    # el primer nivel trae los cinco runs y la busqueda (padre mas un hijo por combinacion);
    # los demas niveles no traen ni el HGB sin pesos ni la busqueda
    assert set(corrida_completa["runs"]) == RUNS_INI50 | BUSQUEDA_INI50 | {
        "p2_baseline_mayoritaria_ini30",
        "p2_baseline_semana_previa_ini30",
        "p2_baseline_persistencia_ini30",
        "p2_hgb_balanceado_ini30",
        "p2_logistica_balanceada_ini30",
        "p2_random_forest_balanceado_ini30",
    }
    assert corrida_completa["experimento"].name == "mibici-equipo29"


def test_main_etiqueta_cada_run_con_la_convencion_del_equipo(corrida_completa):
    sha, _ = t2.git_commit()
    modelos = {}
    for nombre, run in corrida_completa["runs"].items():
        tags = run.data.tags
        # etiquetas propias (sin las de sistema de MLflow, salvo la del commit) mas las del modelo
        propias = {k for k in tags if not k.startswith("mlflow.") or k.endswith("git.commit")}
        assert propias == ETIQUETAS_BASE | ETIQUETAS_MODELO | {"mlflow.source.git.commit"}, nombre
        assert tags["question"] == "P2" and tags["team"] == "29"
        assert tags["author"] == "carlosgalvan" and tags["seed"] == "29"
        assert tags["data_range"] == "2025-01..2026-09"
        assert tags["data_version"] == VERSION
        assert tags["git_commit"] == sha == tags["mlflow.source.git.commit"]
        assert tags["git_dirty"] in {"True", "False"}
        modelos[nombre] = tags["modelo"]
    assert modelos["p2_baseline_mayoritaria_ini50"] == "baseline_mayoritaria"
    assert modelos["p2_baseline_semana_previa_ini50"] == "baseline_semana_previa"
    assert modelos["p2_baseline_persistencia_ini50"] == "baseline_persistencia"
    assert modelos["p2_hgb_balanceado_ini50"] == "hist_gradient_boosting"
    assert modelos["p2_hgb_sin_pesos_ini50"] == "hist_gradient_boosting_sin_pesos"
    # model_family y stage_of_work: las baselines son baseline, los modelos entrenados candidate
    familias = {n: r.data.tags["model_family"] for n, r in corrida_completa["runs"].items()}
    etapas = {n: r.data.tags["stage_of_work"] for n, r in corrida_completa["runs"].items()}
    for nivel in ("ini50", "ini30"):
        assert familias[f"p2_baseline_mayoritaria_{nivel}"] == "baseline_mayoritaria"
        assert familias[f"p2_baseline_semana_previa_{nivel}"] == "baseline_semana_previa"
        assert familias[f"p2_baseline_persistencia_{nivel}"] == "baseline_persistencia"
        assert familias[f"p2_hgb_balanceado_{nivel}"] == "hist_gradient_boosting"
        for nombre in (f"p2_baseline_{b}_{nivel}" for b in ("mayoritaria", "semana_previa")):
            assert etapas[nombre] == "baseline"
        assert etapas[f"p2_baseline_persistencia_{nivel}"] == "baseline"
        assert etapas[f"p2_hgb_balanceado_{nivel}"] == "candidate"
    assert familias["p2_hgb_sin_pesos_ini50"] == "hist_gradient_boosting"
    assert etapas["p2_hgb_sin_pesos_ini50"] == "candidate"
    for nivel in ("ini50", "ini30"):
        assert familias[f"p2_logistica_balanceada_{nivel}"] == "logistic_regression"
        assert familias[f"p2_random_forest_balanceado_{nivel}"] == "random_forest"
        assert etapas[f"p2_logistica_balanceada_{nivel}"] == "candidate"
        assert etapas[f"p2_random_forest_balanceado_{nivel}"] == "candidate"
    assert modelos["p2_logistica_balanceada_ini50"] == "regresion_logistica_balanceada"
    assert modelos["p2_random_forest_balanceado_ini50"] == "random_forest_balanceado"


def test_main_registra_metricas_de_test_y_de_estaciones_nuevas(corrida_completa):
    for nombre, run in corrida_completa["runs"].items():
        if run.data.tags["stage_of_work"] == "search":
            continue  # los runs de busqueda solo traen metricas de validacion
        metricas = run.data.metrics
        for prefijo in ("test", "test_nuevas"):
            assert f"{prefijo}_macro_f1" in metricas, nombre
            for c in fp.CLASES:
                assert f"{prefijo}_f1_{c}" in metricas, nombre
        assert sum(metricas[f"test_prop_{c}"] for c in fp.CLASES) == pytest.approx(1.0)
        con_probas = run.data.tags["stage_of_work"] == "candidate"
        assert ("test_pr_auc_sin_bicicletas" in metricas) == con_probas, nombre
        assert ("test_nuevas_pr_auc_sin_anclajes" in metricas) == con_probas, nombre
    # el baseline mayoritario predice una sola clase: la mas frecuente de train
    mayoritaria = t2.preparar(0.5, hacer_panel())["train"]["objetivo"].mode()[0]
    run = corrida_completa["runs"]["p2_baseline_mayoritaria_ini50"]
    assert run.data.params["clase"] == mayoritaria
    for c in fp.CLASES:
        assert run.data.metrics[f"test_recall_{c}"] == (1.0 if c == mayoritaria else 0.0)


def test_main_registra_validacion_costo_y_mejora_de_cada_candidata(corrida_completa):
    runs = corrida_completa["runs"]
    for nivel in ("ini50", "ini30"):
        baselines = {
            b: runs[f"p2_baseline_{b}_{nivel}"].data.metrics
            for b in ("mayoritaria", "semana_previa", "persistencia")
        }
        for nombre, m in baselines.items():
            # las baselines tambien traen validacion, pero no costo (no hay modelo que ajustar)
            assert "val_macro_f1" in m and "val_recall_sin_bicicletas" in m, nombre
            assert "val_pr_auc_sin_bicicletas" not in m and "fit_seconds" not in m
        # la mejor baseline se elige por macro F1 de validacion
        mejor = max(baselines, key=lambda b: baselines[b]["val_macro_f1"])
        candidatas = ("hgb_balanceado", "logistica_balanceada", "random_forest_balanceado")
        for nombre in candidatas:
            run = runs[f"p2_{nombre}_{nivel}"]
            m = run.data.metrics
            assert 0 <= m["val_macro_f1"] <= 1 and 0 <= m["test_macro_f1"] <= 1
            for c in t2.RIESGO:
                assert f"val_recall_{c}" in m and f"val_pr_auc_{c}" in m
            assert m["fit_seconds"] > 0 and m["model_size_mb"] > 0
            assert m["test_mejora_macro_f1_vs_mejor_baseline"] == pytest.approx(
                m["test_macro_f1"] - baselines[mejor]["test_macro_f1"]
            )
            assert run.data.params["mejor_baseline"] == mejor
    # el HGB del primer nivel valida con la mejor combinacion de la busqueda (mismo ajuste)
    elegido = max(corrida_completa["resultados"]["rejilla"], key=lambda r: r["val_macro_f1"])
    hgb = runs["p2_hgb_balanceado_ini50"].data.metrics
    assert hgb["val_macro_f1"] == pytest.approx(elegido["val_macro_f1"])


def test_main_registra_parametros_comunes(corrida_completa):
    runs = corrida_completa["runs"]
    p = runs["p2_hgb_balanceado_ini50"].data.params
    assert p["nivel_inicial"] == "0.5"
    assert (p["umbral_bajo"], p["umbral_alto"]) == ("0.1", "0.9")
    assert p["horizonte_horas"] == "1" and p["horas_objetivo"] == "6-23"
    assert p["hora_inicio_dia_servicio"] == "5"
    assert p["train_rango"] == "2026-01..2026-02"
    assert p["val_rango"] == "2026-02..2026-02"
    assert p["test_rango"] == "2026-02..2026-03"
    assert p["semilla"] == "29" and p["data_version"] == VERSION
    assert p["estaciones_modeladas"] == "4"
    assert p["estaciones_sin_nomenclatura"] == "2" and p["estaciones_capacidad_cero"] == "1"
    assert p["estaciones_nuevas_test"] == "1"
    assert p["modelo"] == "HistGradientBoostingClassifier" and p["class_weight"] == "balanced"
    assert p["features"] == ",".join(fp.FEATURES)
    assert runs["p2_hgb_sin_pesos_ini50"].data.params["class_weight"] == "ninguno"
    # parametros de las candidatas nuevas: salen de CONFIG_P2 (aqui, la configuracion chica)
    logistica = runs["p2_logistica_balanceada_ini50"].data.params
    assert logistica["max_iter"] == "100" and logistica["class_weight"] == "balanced"
    bosque = runs["p2_random_forest_balanceado_ini50"].data.params
    assert bosque["n_estimators"] == "10" and bosque["max_samples"] == "500"
    assert bosque["min_samples_leaf"] == "5" and bosque["class_weight"] == "balanced_subsample"


def test_main_afina_solo_con_el_primer_nivel_y_guarda_el_json_de_resultados(corrida_completa):
    resultados, runs = corrida_completa["resultados"], corrida_completa["runs"]
    assert set(resultados) == {"params_hgb", "rejilla", "resultados", "campeon"}
    assert set(resultados["resultados"]) == {"0.5", "0.3"}
    assert "hgb_sin_pesos" in resultados["resultados"]["0.5"]
    assert "hgb_sin_pesos" not in resultados["resultados"]["0.3"]
    assert len(resultados["rejilla"]) == len(REJILLA_CORTA)
    elegido = max(resultados["rejilla"], key=lambda r: r["val_macro_f1"])
    assert resultados["params_hgb"] == {k: v for k, v in elegido.items() if k != "val_macro_f1"}
    # el run del primer nivel reporta val_macro_f1 igual al mejor de la rejilla
    hgb50 = runs["p2_hgb_balanceado_ini50"]
    assert hgb50.data.metrics["val_macro_f1"] == pytest.approx(elegido["val_macro_f1"])
    assert 0 <= runs["p2_hgb_balanceado_ini30"].data.metrics["val_macro_f1"] <= 1
    assert hgb50.data.metrics["test_macro_f1"] == pytest.approx(
        resultados["resultados"]["0.5"]["hgb"]["test_macro_f1"]
    )
    # los dos niveles usan los mismos hiperparametros elegidos
    for nombre in ("p2_hgb_balanceado_ini50", "p2_hgb_balanceado_ini30"):
        p = runs[nombre].data.params
        assert {k: float(p[k]) for k in resultados["params_hgb"]} == pytest.approx(
            resultados["params_hgb"]
        )


def test_main_registra_la_busqueda_como_runs_anidados(corrida_completa):
    runs, resultados = corrida_completa["runs"], corrida_completa["resultados"]
    padre = runs["p2_busqueda_ini50"]
    hijos = {n: r for n, r in runs.items() if n.startswith("p2_busqueda_ini50_c")}
    assert set(hijos) == BUSQUEDA_INI50 - {"p2_busqueda_ini50"}
    assert padre.data.tags["stage_of_work"] == "search"
    assert "mlflow.parentRunId" not in padre.data.tags  # el padre es de primer nivel
    for i, (nombre, hijo) in enumerate(sorted(hijos.items()), start=1):
        t = hijo.data.tags
        assert t["mlflow.parentRunId"] == padre.info.run_id, nombre
        assert t["stage_of_work"] == "search" and t["model_family"] == "hist_gradient_boosting"
        assert ETIQUETAS_BASE <= set(t)
        # cada hijo registra su combinacion, val_macro_f1, las metricas de riesgo y fit_seconds
        combinacion = REJILLA_CORTA[i - 1]
        assert {k: float(hijo.data.params[k]) for k in combinacion} == pytest.approx(combinacion)
        m = hijo.data.metrics
        assert m["val_macro_f1"] == pytest.approx(resultados["rejilla"][i - 1]["val_macro_f1"])
        for c in t2.RIESGO:
            assert f"val_recall_{c}" in m and f"val_pr_auc_{c}" in m
        assert m["fit_seconds"] > 0
    # el padre guarda la mejor combinacion (por val macro F1) y la rejilla completa
    mejor = resultados["params_hgb"]
    assert {k: float(padre.data.params[f"mejor_{k}"]) for k in mejor} == pytest.approx(mejor)
    mejor_f1 = max(h.data.metrics["val_macro_f1"] for h in hijos.values())
    assert padre.data.metrics["val_macro_f1"] == pytest.approx(mejor_f1)
    cliente = corrida_completa["cliente"]
    assert archivos_de(cliente, padre.info.run_id) == {"config/p2_rejilla_validacion.json"}


def test_main_ordena_los_artefactos_en_plots_data_y_config(corrida_completa):
    cliente, runs = corrida_completa["cliente"], corrida_completa["runs"]

    def artefactos(nombre):
        return archivos_de(cliente, runs[nombre].info.run_id)

    assert artefactos("p2_baseline_mayoritaria_ini50") == set()
    assert artefactos("p2_baseline_semana_previa_ini50") == {
        "plots/p2_cm_baseline_semana_ini50.png"
    }
    assert artefactos("p2_baseline_persistencia_ini50") == {
        "plots/p2_cm_baseline_persistencia_ini50.png"
    }
    # todas las candidatas: matriz de confusion, curva PR de las clases de riesgo, manifiesto
    # y resumen del panel, estadisticas por estacion y hora, y la configuracion del run
    comunes = {
        "data/manifiesto.json",
        "data/panel.json",
        "data/stats_estacion.csv",
        "data/stats_estacion_hora.csv",
        "config/config_p2.json",
    }
    for run, clave in (
        ("p2_logistica_balanceada_ini50", "logistica"),
        ("p2_random_forest_balanceado_ini50", "random_forest"),
    ):
        assert artefactos(run) == comunes | {
            f"plots/p2_cm_{clave}_ini50.png",
            f"plots/p2_cm_{clave}_nuevas_ini50.png",
            f"plots/p2_pr_{clave}_ini50.png",
        }
    assert artefactos("p2_hgb_sin_pesos_ini50") == comunes | {
        "plots/p2_cm_hgb_sin_pesos_ini50.png",
        "plots/p2_pr_hgb_sin_pesos_ini50.png",
    }
    # el HGB suma la importancia por permutacion, el heatmap de error y la rejilla de la busqueda
    assert artefactos("p2_hgb_balanceado_ini50") == comunes | {
        "plots/p2_heatmap_error_ini50.png",
        "data/p2_heatmap_error_ini50.csv",
        "plots/p2_cm_hgb_ini50.png",
        "plots/p2_cm_hgb_nuevas_ini50.png",
        "plots/p2_pr_hgb_ini50.png",
        "plots/p2_importancia_permutacion_ini50.png",
        "data/p2_importancia_permutacion_ini50.csv",
        "config/p2_rejilla_validacion.json",
    }
    assert "config/p2_rejilla_validacion.json" not in artefactos("p2_hgb_balanceado_ini30")


def test_main_loguea_cada_candidata_con_firma_ejemplo_y_carga(corrida_completa):
    mlflow.set_tracking_uri(corrida_completa["uri"])
    runs = corrida_completa["runs"]
    candidatas = {n: r for n, r in runs.items() if r.data.tags["stage_of_work"] == "candidate"}
    modelos = mlflow.search_logged_models(
        experiment_ids=[corrida_completa["experimento"].experiment_id], output_format="list"
    )
    por_run = {m.source_run_id: m for m in modelos}
    assert len(modelos) == len(candidatas) == 7  # solo las candidatas (ni baselines ni busqueda)
    for nombre, run in candidatas.items():
        info = mlflow.models.get_model_info(por_run[run.info.run_id].model_uri)
        # la firma pide las features del entrenamiento, en ese orden, y entrega la clase
        assert info.signature.inputs.input_names() == fp.FEATURES, nombre
        assert info.signature.outputs.input_names() == ["clase_riesgo"], nombre
        assert info.saved_input_example_info is not None, nombre
        assert info.flavors["sklearn"]["serialization_format"] == "skops", nombre
    # carga con pyfunc y predice con el ejemplo guardado (hgb del primer nivel)
    modelo = por_run[runs["p2_hgb_balanceado_ini50"].info.run_id]
    ejemplo = mlflow.models.get_model_info(modelo.model_uri).saved_input_example_info
    assert ejemplo["type"] == "dataframe"
    cargado = mlflow.pyfunc.load_model(modelo.model_uri)
    entrada = pd.DataFrame(
        np.zeros((2, len(fp.FEATURES))), columns=fp.FEATURES, dtype="float64"
    ).assign(capacidad=10.0, hora=8.0)
    assert set(cargado.predict(entrada)) <= set(fp.CLASES)


def test_main_guarda_las_estadisticas_que_un_consumidor_necesita_para_predecir(corrida_completa):
    mlflow.set_tracking_uri(corrida_completa["uri"])
    run = corrida_completa["runs"]["p2_hgb_balanceado_ini50"]
    ruta = f"runs:/{run.info.run_id}/data"
    est = pd.read_csv(mlflow.artifacts.download_artifacts(f"{ruta}/stats_estacion.csv"))
    hora = pd.read_csv(mlflow.artifacts.download_artifacts(f"{ruta}/stats_estacion_hora.csv"))
    esperado_est, esperado_hora = fp.stats_estacion(t2.preparar(0.5, hacer_panel())["train"])
    # son las de train: con ellas el consumidor rearma las mismas features que vio el modelo
    pd.testing.assert_frame_equal(est, esperado_est, check_dtype=False)
    pd.testing.assert_frame_equal(hora, esperado_hora, check_dtype=False)
    assert list(est.columns) == ["estacion", "est_tasa_sin_bicis", "est_tasa_sin_anclajes"]
    assert list(hora.columns) == ["estacion", "hora", "est_neto_medio_hora"]
    config = json.loads(
        Path(
            mlflow.artifacts.download_artifacts(f"runs:/{run.info.run_id}/config/config_p2.json")
        ).read_text()
    )
    assert config["features"] == fp.FEATURES  # columnas y orden de entrada del modelo
    assert config["cortes"] == CORTES_CHICOS and config["semilla"] == 29
    assert config["nivel_inicial"] == 0.5
    elegidos = corrida_completa["resultados"]["params_hgb"]
    assert {k: config["modelo"][k] for k in elegidos} == elegidos


def test_main_registra_los_datasets_de_entrada_sin_subir_los_datos(corrida_completa):
    cliente, runs = corrida_completa["cliente"], corrida_completa["runs"]
    run = cliente.get_run(runs["p2_hgb_balanceado_ini50"].info.run_id)
    contextos = {}
    for entrada in run.inputs.dataset_inputs:
        etiquetas = {t.key: t.value for t in entrada.tags}
        contextos[etiquetas["mlflow.data.context"]] = (entrada.dataset.name, etiquetas)
    assert set(contextos) == {"training", "validation", "testing"}
    assert contextos["training"][0] == "p2_training_ini50"
    for _, etiquetas in contextos.values():
        assert etiquetas["data_version"] == VERSION and int(etiquetas["n_filas"]) > 0
    # son una muestra pequena: no se sube ninguna tabla de datos como artefacto
    assert not [a for a in archivos_de(cliente, run.info.run_id) if a.endswith(".parquet")]


def test_main_no_escribe_archivos_en_el_repo(corrida_completa):
    assert corrida_completa["nuevos_en_repo"] == set()
    # --artefactos manda: los artefactos de MLflow van a la carpeta indicada, no al repo
    esperado = (corrida_completa["carpeta"] / "mlartifacts").resolve().as_uri()
    assert corrida_completa["experimento"].artifact_location == esperado
    assert (corrida_completa["figuras"] / "p2_resultados.json").exists()


def test_main_con_sin_afinar_usa_la_primera_configuracion(corrida_sin_afinar):
    r = corrida_sin_afinar["resultados"]
    assert r["params_hgb"] == REJILLA_CORTA[0]
    assert r["rejilla"] == []
    assert set(corrida_sin_afinar["runs"]) == RUNS_INI50  # sin busqueda no hay runs anidados
    hgb = corrida_sin_afinar["runs"]["p2_hgb_balanceado_ini50"]
    assert "val_macro_f1" in hgb.data.metrics  # la validacion se registra aunque no haya rejilla
    nombres = archivos_de(corrida_sin_afinar["cliente"], hgb.info.run_id)
    assert "config/p2_rejilla_validacion.json" not in nombres


def test_main_con_hiperparametros_no_busca_en_la_rejilla_pero_evalua_esa_configuracion(
    corrida_con_hiperparametros,
):
    r = corrida_con_hiperparametros["resultados"]
    esperado = REJILLA_CORTA[0] | {"max_iter": 12, "l2_regularization": 0.5}
    assert r["params_hgb"] == esperado  # lo no indicado sale de la primera configuracion
    # la rejilla tiene un solo elemento: la configuracion evaluada en validacion
    assert len(r["rejilla"]) == 1
    assert {k: v for k, v in r["rejilla"][0].items() if k != "val_macro_f1"} == esperado
    hgb = corrida_con_hiperparametros["runs"]["p2_hgb_balanceado_ini50"]
    p = hgb.data.params
    assert p["max_iter"] == "12" and p["l2_regularization"] == "0.5"
    assert p["max_leaf_nodes"] == "7"
    assert hgb.data.metrics["val_macro_f1"] == pytest.approx(r["rejilla"][0]["val_macro_f1"])
    # la busqueda de una sola combinacion tambien queda como padre con un hijo
    nombres = set(corrida_con_hiperparametros["runs"])
    assert {"p2_busqueda_ini50", "p2_busqueda_ini50_c1"} <= nombres
    assert "p2_busqueda_ini50_c2" not in nombres
    cliente = corrida_con_hiperparametros["cliente"]
    assert "config/p2_rejilla_validacion.json" in archivos_de(cliente, hgb.info.run_id)


def test_main_registra_solo_al_campeon_en_el_registry_con_el_alias(corrida_completa):
    mlflow.set_tracking_uri(corrida_completa["uri"])
    cliente, runs = corrida_completa["cliente"], corrida_completa["runs"]
    campeon = corrida_completa["resultados"]["campeon"]
    # la regla se evalua con la validacion del primer nivel (el de referencia) y solo ve candidatas
    esperado = registro.seleccionar_campeon(
        corrida_completa["resultados"]["resultados"]["0.5"], CONFIG_P2.candidatas_campeon
    )
    assert campeon["clave"] == esperado["clave"]
    assert campeon["clave"] in {"logistica", "hgb", "random_forest"}
    nombre_run = {
        "logistica": "p2_logistica_balanceada_ini50",
        "hgb": "p2_hgb_balanceado_ini50",
        "random_forest": "p2_random_forest_balanceado_ini50",
    }[campeon["clave"]]
    registrados = cliente.search_registered_models()
    assert [m.name for m in registrados] == ["mibici-equipo29-p2-riesgo"]
    assert registrados[0].aliases == {"champion": 1}
    versiones = cliente.search_model_versions("name='mibici-equipo29-p2-riesgo'")
    assert len(versiones) == 1  # solo el campeon, no las demas candidatas
    version = versiones[0]
    assert version.run_id == runs[nombre_run].info.run_id == campeon["run_id"]
    assert version.tags["nivel_inicial"] == "0.5"
    assert version.tags["experimento"] == CONFIG_P2.experimento
    assert version.tags["git_commit"] == runs[nombre_run].data.tags["git_commit"]
    assert version.tags["data_version"] == VERSION
    assert "recall" in version.tags["criterio_seleccion"]
    # carga por alias y predice con el ejemplo: lo mismo que verifico main
    ejemplo = pd.DataFrame(np.zeros((2, len(fp.FEATURES))), columns=fp.FEATURES)
    clases = registro.verificar_campeon(
        CONFIG_P2.modelo_registrado, CONFIG_P2.alias_campeon, ejemplo
    )
    assert len(clases) == 2 and set(clases) <= set(fp.CLASES)


def test_main_con_no_registrar_no_toca_el_registry(tmp_path):
    corrida = correr_main(tmp_path, ["--sin-afinar", "--no-registrar"])
    assert corrida["cliente"].search_registered_models() == []
    assert "campeon" not in corrida["resultados"]


def test_main_registra_por_omision_y_con_sin_afinar_tambien(tmp_path):
    corrida = correr_main(tmp_path, ["--sin-afinar"])
    assert [m.name for m in corrida["cliente"].search_registered_models()] == [
        CONFIG_P2.modelo_registrado
    ]
    assert corrida["resultados"]["campeon"]["version"] == 1


def test_main_rechaza_argumentos_invalidos_antes_de_tocar_datos(tmp_path):
    invalidos = [
        ["--hyperparameters", "{no es json"],
        ["--hyperparameters", "[1, 2]"],
        ["--hyperparameters", '{"profundidad": 3}'],
        ["--model", "random_forest"],
    ]
    for extra in invalidos:
        with pytest.raises(SystemExit) as salida:
            t2.main(["--panel", str(tmp_path / "no_existe.parquet"), *extra])
        assert salida.value.code == 2, extra


def test_git_commit_regresa_el_sha_del_repo_y_desconocido_si_falla_git(monkeypatch):
    sha, sucio = t2.git_commit()
    esperado = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, text=True)
    assert sha == esperado.strip() and isinstance(sucio, bool)

    def falla(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(t2.subprocess, "check_output", falla)
    assert t2.git_commit() == ("desconocido", True)


# despachador del MLflow Project (mibici.mlflow_project) para P2


def test_hiperparametros_se_leen_como_json_o_como_pares_clave_valor():
    assert mp._parse_hyperparameters("") == {}
    assert mp._parse_hyperparameters('{"max_iter": 10, "learning_rate": 0.1}') == {
        "max_iter": 10,
        "learning_rate": 0.1,
    }
    asignado = mp._parse_hyperparameters("max_iter=10,learning_rate=0.05,nombre=abc,l2=1e-3")
    assert asignado == {"max_iter": 10, "learning_rate": 0.05, "nombre": "abc", "l2": 1e-3}
    assert type(asignado["max_iter"]) is int and type(asignado["learning_rate"]) is float
    for texto in ["[1, 2]", "max_iter", "max_iter=", "=3", "a=1,,b=2"]:
        with pytest.raises(ValueError):
            mp._parse_hyperparameters(texto)


@pytest.fixture
def espias_p2(monkeypatch):
    """Cambia features_p2.main y train_p2.main por espias para ver como los llama el despacho."""
    llamadas = []
    monkeypatch.setattr(fp, "main", lambda *a, **k: llamadas.append(("features_p2", a, k)))
    monkeypatch.setattr(t2, "main", lambda argv=None: llamadas.append(("train_p2", argv)))
    return llamadas


@pytest.mark.parametrize("uri", [None, "sqlite:///evidence/tracking-comun.db"])
def test_despachador_p2_prepara_features_y_llama_a_train_con_model_e_hiperparametros(
    espias_p2, monkeypatch, uri
):
    monkeypatch.delenv("MIBICI_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    if uri is not None:
        monkeypatch.setenv("MIBICI_TRACKING_URI", uri)
    expected_uri = mp._project_tracking_uri()
    mp.run_train("p2", hyperparameters={"max_iter": 10, "learning_rate": 0.05})
    assert [c[0] for c in espias_p2] == ["features_p2", "train_p2"]  # primero las features
    # los hiperparametros viajan como JSON compacto y el modelo por defecto sale de la config
    assert espias_p2[1][1] == [
        "--model", CONFIG_P2.model, "--hyperparameters", '{"max_iter":10,"learning_rate":0.05}',
        "--tracking-uri", expected_uri,
    ]  # fmt: skip
    espias_p2.clear()
    mp.main(["train", "--question", "p2"])  # sin hiperparametros viaja un objeto vacio
    assert espias_p2[1][1] == [
        "--model",
        "hist_gradient_boosting",
        "--hyperparameters",
        "{}",
        "--tracking-uri",
        expected_uri,
    ]
    espias_p2.clear()
    mp.main(["train", "--question", "p2", "--hyperparameters", "max_iter=10,max_leaf_nodes=7"])
    assert json.loads(espias_p2[1][1][3]) == {"max_iter": 10, "max_leaf_nodes": 7}


def test_despachador_rechaza_modelo_pregunta_o_hiperparametros_invalidos_sin_entrenar(espias_p2):
    with pytest.raises(ValueError, match="P2"):
        mp.run_train("p2", model="random_forest")
    with pytest.raises(ValueError, match="question"):
        mp.run_train("p9")
    with pytest.raises(TypeError):
        mp.run_train("p2", hyperparameters=[1, 2])
    with pytest.raises(SystemExit):  # la pregunta va en minusculas (la etiqueta de MLflow es P2)
        mp.main(["train", "--question", "P2"])
    assert espias_p2 == []


def test_despachador_quita_mlflow_run_id_durante_el_entrenamiento_y_lo_restaura(
    monkeypatch, espias_p2
):
    # con mlflow run el proyecto ya abre un run; train_p2 debe crear los suyos sin anidarse
    monkeypatch.setenv("MLFLOW_RUN_ID", "run_del_proyecto")
    monkeypatch.setenv("MIBICI_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    visto = []
    monkeypatch.setattr(
        t2, "main", lambda argv=None: visto.append(os.environ.get("MLFLOW_RUN_ID"))
    )
    mp.run_train("p2")
    assert visto == [None]
    assert os.environ["MLFLOW_RUN_ID"] == "run_del_proyecto"
