"""Pruebas de los pipelines de uso de P3 con datos sintéticos (no necesitan los datos reales)."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from mibici.config import CONFIG_P3
from mibici.modeling.modelos_p3 import (
    GruposOrdenados,
    SeleccionVIF,
    construir_modelo,
    nombre_corrida,
    validar_hiperparametros,
)
from mibici.modeling.train_p3 import combinaciones

VARIABLES = list(CONFIG_P3.prioridad_variables)


def _perfiles(n: int = 120, semilla: int = 0) -> pd.DataFrame:
    """Perfiles con las 12 variables candidatas y tres grupos de uso bien separados."""
    rng = np.random.default_rng(semilla)
    grupo = np.repeat([0, 1, 2], n // 3)
    datos = {c: rng.normal(0, 0.3, len(grupo)) for c in VARIABLES}  # ruido
    datos["log_trips_day"] = grupo * 3.0 + rng.normal(0, 0.3, len(grupo))
    # net_am separa al grupo 1 de los otros dos sin correlacionar con log_trips_day
    datos["net_am"] = np.array([3.0, -3.0, 3.0])[grupo] + rng.normal(0, 0.3, len(grupo))
    # dep_am casi copia de net_am: el VIF debe quitarla (tiene menor prioridad)
    datos["dep_am"] = datos["net_am"] + rng.normal(0, 0.01, len(grupo))
    return pd.DataFrame(datos)[VARIABLES]


def test_seleccion_vif_quita_la_redundante_de_menor_prioridad():
    sel = SeleccionVIF().fit(_perfiles())
    assert "net_am" in sel.variables_
    assert "dep_am" not in sel.variables_
    assert list(sel.transform(_perfiles()).columns) == sel.variables_


def test_seleccion_vif_se_aprende_solo_con_lo_que_recibe_fit():
    # con dep_am independiente de net_am se conserva: la selección depende de los datos de fit
    df = _perfiles()
    df["dep_am"] = np.random.default_rng(1).normal(0, 0.3, len(df))
    assert "dep_am" in SeleccionVIF().fit(df).variables_


@pytest.mark.parametrize(
    "model,params",
    [("kmeans_uso", {"k": 3, "n_init": 3}), ("gmm_uso", {"n_components": 3, "n_init": 2})],
)
def test_pipeline_recibe_las_12_variables_y_agrupa(model, params):
    df = _perfiles()
    modelo = construir_modelo(model, params).fit(df)
    etiquetas = modelo.predict(df)
    assert set(etiquetas) == {0, 1, 2}
    # grupo 0 = menor uso: el orden sigue a log_trips_day
    medias = df.groupby(etiquetas)["log_trips_day"].mean()
    assert medias.is_monotonic_increasing


def test_grupos_ordenados_no_dependen_de_la_semilla():
    # solo variables informativas, para que ambas semillas encuentren los mismos tres grupos
    cfg = replace(CONFIG_P3, prioridad_variables=("log_trips_day", "net_am", "dep_am"))
    df = _perfiles()[list(cfg.prioridad_variables)]
    a = construir_modelo("kmeans_uso", {"k": 3, "n_init": 1}, cfg).fit(df).predict(df)
    otro = construir_modelo("kmeans_uso", {"k": 3, "n_init": 1}, cfg)
    otro.set_params(grupos__estimador__random_state=7)
    b = otro.fit(df).predict(df)
    assert (a == b).all()


def test_grupos_ordenados_expone_labels():
    from sklearn.cluster import KMeans

    X = np.array([[0.0], [0.1], [5.0], [5.1], [10.0], [10.1]])
    g = GruposOrdenados(KMeans(n_clusters=3, n_init=5, random_state=0)).fit(X)
    assert list(g.labels_) == [0, 0, 1, 1, 2, 2]


@pytest.mark.parametrize(
    "model,params,mensaje",
    [
        ("dbscan", {}, "no soportado"),
        ("kmeans_uso", {"n_components": 3}, "no soportados"),
        ("kmeans_uso", {"k": 1}, "k debe"),
        ("gmm_uso", {"n_components": 2.5}, "n_components debe"),
        ("gmm_uso", {"covariance_type": "otra"}, "covariance_type"),
        ("kmeans_geo", {"n_init": True}, "n_init debe"),
    ],
)
def test_validar_hiperparametros_rechaza_lo_invalido(model, params, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        validar_hiperparametros(model, params)


def test_nombres_de_corrida_descriptivos():
    assert nombre_corrida("kmeans_uso", {"k": 6}) == "p3-kmeans-uso-k6"
    gmm = {"n_components": 4, "covariance_type": "diag"}
    assert nombre_corrida("gmm_uso", gmm) == "p3-gmm-uso-n4-diag"


def test_combinaciones_de_la_cli():
    assert combinaciones("kmeans_geo", None, None, None, None) == [
        {"k": k} for k in CONFIG_P3.k_comparacion
    ]
    assert combinaciones("kmeans_uso", None, None, None, 10) == [
        {"k": CONFIG_P3.k_baseline, "n_init": 10}
    ]
    assert combinaciones("gmm_uso", None, [3, 4], "tied", None) == [
        {"n_components": 3, "covariance_type": "tied"},
        {"n_components": 4, "covariance_type": "tied"},
    ]


def test_hiperparametros_efectivos_completan_con_config():
    from mibici.modeling.modelos_p3 import hiperparametros_efectivos

    assert hiperparametros_efectivos("kmeans_uso", {"k": 7}) == {
        "n_init": CONFIG_P3.n_init,
        "k": 7,
    }
    assert hiperparametros_efectivos("gmm_uso", {"covariance_type": "tied"}) == {
        "n_init": CONFIG_P3.n_init,
        "n_components": CONFIG_P3.k_baseline,
        "covariance_type": "tied",
    }
