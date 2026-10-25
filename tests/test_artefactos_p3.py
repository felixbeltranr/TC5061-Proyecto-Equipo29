"""Pruebas de los artefactos de P3 con datos sintéticos y un MLflow temporal."""

from dataclasses import replace
import json

import mlflow
import numpy as np
import pandas as pd
import pytest

from mibici.config import CONFIG_P3
from mibici.features_p3 import COLUMNAS_PERSONALES, DatasetP3
from mibici.modeling import artefactos_p3 as art
from mibici.modeling.train_p3 import entrenar_candidata, registrar_candidata

VARIABLES = list(CONFIG_P3.prioridad_variables)
CFG = replace(CONFIG_P3, n_init=2)


def _dataset(n: int = 60) -> DatasetP3:
    """Perfiles con las 12 variables, nomenclatura con coordenadas y algunas estaciones nuevas."""
    rng = np.random.default_rng(0)
    grupo = np.repeat([0, 1, 2], n // 3)
    datos = {c: rng.normal(0, 1, len(grupo)) for c in VARIABLES}
    datos["log_trips_day"] = grupo * 3.0 + rng.normal(0, 0.3, len(grupo))
    datos["trips_day"] = np.exp(datos["log_trips_day"])
    idx = pd.Index(range(1, n + 1), name="estacion")
    idx_ev = pd.Index(range(1, n + 6), name="estacion")  # 5 estaciones nuevas en evaluación
    nom = pd.DataFrame(
        {
            "name": [f"est {i}" for i in idx_ev],
            "municipio": "GDL",
            "latitude": 20.6 + rng.random(len(idx_ev)) / 10,
            "longitude": -103.4 + rng.random(len(idx_ev)) / 10,
        },
        index=idx_ev,
    )
    en = pd.DataFrame(datos, index=idx)
    ev = pd.concat([en, en.iloc[:5].set_axis(idx_ev[-5:])])
    return DatasetP3(
        bitacora=pd.DataFrame({"mes": ["2025_01"], "viajes_validos": [10]}),
        nom=nom,
        entrenamiento=en,
        evaluacion=ev,
        variables=["log_trips_day", "weekend_share", "net_am"],
        nuevas=idx_ev[-5:],
        data_version="prueba",
        nomenclatura="nomenclatura_prueba.csv",
    )


def test_asignacion_sin_datos_personales_y_marca_nuevas():
    ds = _dataset()
    tabla = art.asignacion_estaciones(ds, np.zeros(60, int), np.zeros(65, int))
    assert not set(tabla.columns) & set(COLUMNAS_PERSONALES)
    assert set(tabla.columns) == {*art.COLUMNAS_ESTACION, "grupo", "periodo", "nueva"}
    assert tabla.loc[tabla.periodo == "evaluacion", "nueva"].sum() == 5


def test_perfil_por_grupo_y_perfiles_z():
    ds = _dataset()
    et = np.repeat([0, 1, 2], 20)
    perfil = art.perfil_por_grupo(ds.entrenamiento, et, VARIABLES)
    assert perfil["n_estaciones"].tolist() == [20, 20, 20]
    z = art.perfiles_z(ds.entrenamiento, et, ds.variables)
    assert z.loc[0, "log_trips_day"] < 0 < z.loc[2, "log_trips_day"]


def test_tipos_confiables_acepta_solo_sklearn_numpy_y_modelos_p3():
    modelo, _ = entrenar_candidata(_dataset(), "kmeans_uso", {"k": 3}, CFG)
    tipos = art.tipos_confiables(modelo)
    assert tipos and all(t.startswith(art.PREFIJOS_CONFIABLES) for t in tipos)
    with pytest.raises(ValueError, match="no confiables"):
        art.tipos_confiables({"x": pd.Timestamp("2025-01-01")})


def test_registrar_candidata_deja_modelo_datos_graficas_y_config(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("p3-prueba")
    ds = _dataset()
    modelo, m = entrenar_candidata(ds, "kmeans_uso", {"k": 3}, CFG)
    run_id = registrar_candidata(ds, "kmeans_uso", {"k": 3}, modelo, m, CFG)

    cliente = mlflow.MlflowClient()
    archivos = {
        a.path
        for carpeta in ("data", "plots", "config")
        for a in cliente.list_artifacts(run_id, carpeta)
    }
    assert {
        "data/asignacion_estaciones.csv",
        "data/perfil_uso_por_grupo.csv",
        "data/bitacora_limpieza.csv",
        f"data/{CFG.manifiesto_path.name}",
        "plots/mapa_grupos.png",
        "plots/perfiles_grupos.png",
        "config/config_corrida.json",
    } <= archivos

    run = cliente.get_run(run_id)
    assert run.data.tags["stage_of_work"] == "candidate"
    assert {d.dataset.name for d in run.inputs.dataset_inputs} == {
        "p3_perfiles_training",
        "p3_perfiles_evaluation",
    }
    config = json.loads(mlflow.artifacts.load_text(f"runs:/{run_id}/config/config_corrida.json"))
    assert config["hiperparametros"]["k"] == 3 and config["variables_vif"]

    cargado = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    entradas = [c.name for c in cargado.metadata.get_input_schema().inputs]
    assert entradas == VARIABLES
    pred = cargado.predict(ds.entrenamiento[VARIABLES].astype("float64"))
    assert set(np.asarray(pred)) <= {0, 1, 2}
