"""Reproducibilidad del entrenamiento de P3: misma semilla, mismos grupos y mismas métricas.

Usan perfiles sintéticos de estaciones (no necesitan los datos reales). Cada prueba entrena dos
veces desde cero con la misma configuración y compara los resultados exactos.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from mibici.config import CONFIG_P3
from mibici.features_p3 import DatasetP3
from mibici.modeling.busqueda_p3 import evaluar_validacion
from mibici.modeling.train_p3 import entrenar_candidata

VARIABLES = list(CONFIG_P3.prioridad_variables)
CFG = replace(CONFIG_P3, n_init=2)  # menos reinicios para que la prueba sea rápida
CANDIDATAS = [("kmeans_uso", {"k": 3}), ("gmm_uso", {"n_components": 3})]


def _dataset(n: int = 60) -> DatasetP3:
    """Tres grupos de uso con ruido; mismo formato que el dataset de features_p3."""
    rng = np.random.default_rng(0)
    grupo = np.repeat([0, 1, 2], n // 3)
    datos = {c: rng.normal(0, 0.3, len(grupo)) for c in VARIABLES}
    datos["net_am"] = np.array([-0.5, 0.5, 0.0])[grupo] + rng.normal(0, 0.2, len(grupo))
    datos["log_trips_day"] = np.array([2.0, 2.0, 4.0])[grupo] + rng.normal(0, 0.3, len(grupo))
    idx = pd.Index(range(1, len(grupo) + 1), name="estacion")
    perfiles = pd.DataFrame(datos, index=idx)
    perfiles["trips_day"] = np.exp(perfiles["log_trips_day"])
    nom = pd.DataFrame(
        {
            "latitude": 20.6 + rng.random(len(idx)) / 10,
            "longitude": -103.4 + rng.random(len(idx)) / 10,
        },
        index=idx,
    )
    return DatasetP3(
        bitacora=pd.DataFrame(),
        nom=nom,
        entrenamiento=perfiles,
        evaluacion=perfiles.sample(frac=1.0, random_state=1),  # otro orden de estaciones
        variables=["log_trips_day", "net_am"],
        nuevas=pd.Index([], name="estacion"),
        data_version="prueba",
        nomenclatura="nomenclatura_prueba.csv",
    )


@pytest.mark.parametrize("model,params", CANDIDATAS)
def test_misma_semilla_mismos_grupos_y_metricas(model, params):
    ds = _dataset()
    entrada = list(CFG.prioridad_variables)
    modelo_a, metricas_a = entrenar_candidata(ds, model, params, CFG)
    modelo_b, metricas_b = entrenar_candidata(ds, model, params, CFG)
    for periodo in (ds.entrenamiento, ds.evaluacion):
        assert np.array_equal(
            modelo_a.predict(periodo[entrada]), modelo_b.predict(periodo[entrada])
        )
    assert metricas_a == metricas_b  # métricas idénticas, no solo parecidas


@pytest.mark.parametrize("model,params", CANDIDATAS)
def test_validacion_por_estaciones_es_reproducible(model, params):
    ds = _dataset()
    a = evaluar_validacion(ds, model, params, CFG)
    b = evaluar_validacion(ds, model, params, CFG)
    for m in (a, b):
        del m["val_fit_seconds"]  # tiempo de reloj: cambia en cada corrida aunque el modelo no
    assert a == pytest.approx(b, rel=0, abs=0, nan_ok=True)
