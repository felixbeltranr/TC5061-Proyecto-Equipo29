"""Pruebas de la validación por estaciones y la búsqueda de P3 con datos sintéticos."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from mibici.config import CONFIG_P3
from mibici.features_p3 import DatasetP3
from mibici.modeling.busqueda_p3 import (
    complejidad,
    evaluar_validacion,
    ordenar,
    particiones,
    rejilla,
)

VARIABLES = list(CONFIG_P3.prioridad_variables)
# pocas variables (dos informativas y una de ruido) para que los grupos sintéticos sean claros
CFG = replace(
    CONFIG_P3, n_init=3, prioridad_variables=("log_trips_day", "weekend_share", "net_am")
)


def _dataset(n: int = 90) -> DatasetP3:
    """Tres grupos de uso separados y ubicaciones al azar; mismo formato que features_p3."""
    rng = np.random.default_rng(0)
    grupo = np.repeat([0, 1, 2], n // 3)
    datos = {c: rng.normal(0, 1, len(grupo)) for c in VARIABLES}
    datos["log_trips_day"] = grupo * 4.0 + rng.normal(0, 0.3, len(grupo))
    datos["net_am"] = np.array([4.0, -4.0, 4.0])[grupo] + rng.normal(0, 0.3, len(grupo))
    idx = pd.Index(range(1, len(grupo) + 1), name="estacion")
    perfiles = pd.DataFrame(datos, index=idx)
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
        evaluacion=perfiles,
        variables=["log_trips_day", "net_am"],
        nuevas=pd.Index([], name="estacion"),
        data_version="prueba",
        nomenclatura="nomenclatura_prueba.csv",
    )


def test_particiones_reproducibles_y_cubren_todas_las_estaciones():
    estaciones = pd.Index(range(100))
    a, b = particiones(estaciones, CFG), particiones(estaciones, CFG)
    assert len(a) == CFG.n_particiones
    reservadas = []
    for (en_a, va_a), (en_b, va_b) in zip(a, b):
        assert en_a.equals(en_b) and va_a.equals(va_b)  # misma semilla, mismas particiones
        assert len(va_a) == 20 and not set(en_a) & set(va_a)  # 20 % reservado, sin traslape
        reservadas += list(va_a)
    assert sorted(reservadas) == list(estaciones)  # cada estación se reserva una sola vez


@pytest.mark.parametrize(
    "model,params",
    [("kmeans_uso", {"k": 3}), ("gmm_uso", {"n_components": 3}), ("kmeans_geo", {"k": 3})],
)
def test_evaluar_validacion_regresa_metricas_y_estabilidad(model, params):
    m = evaluar_validacion(_dataset(), model, params, CFG)
    assert set(m) == {
        "val_silhouette_uso",
        "val_silhouette_uso_std",
        "val_davies_bouldin_uso",
        "val_calinski_harabasz_uso",
        "val_ari_estabilidad",
        "val_fit_seconds",
    }
    assert -1 <= m["val_silhouette_uso"] <= 1
    assert -1 <= m["val_ari_estabilidad"] <= 1


def test_grupos_de_uso_claros_validan_mejor_que_la_ubicacion_al_azar():
    ds = _dataset()
    uso = evaluar_validacion(ds, "kmeans_uso", {"k": 3}, CFG)
    geo = evaluar_validacion(ds, "kmeans_geo", {"k": 3}, CFG)
    assert uso["val_silhouette_uso"] > geo["val_silhouette_uso"]
    assert uso["val_ari_estabilidad"] > 0.9  # grupos claros: casi las mismas asignaciones


def _r(model, params, sil, ari, db):
    return {
        "model": model,
        "params": params,
        "val_silhouette_uso": sil,
        "val_ari_estabilidad": ari,
        "val_davies_bouldin_uso": db,
    }


def test_ordenar_desempata_por_ari_dentro_del_margen():
    rs = [
        _r("kmeans_uso", {"k": 3}, 0.186, 0.70, 1.6),
        _r("kmeans_uso", {"k": 4}, 0.190, 0.60, 1.5),  # mejor silueta, pero empata con k=3
        _r("kmeans_uso", {"k": 5}, 0.170, 0.90, 1.4),  # fuera del margen: no gana por ARI
    ]
    ranking = ordenar(rs, CFG)
    assert [r["params"]["k"] for r in ranking] == [3, 4, 5]
    assert [r["posicion"] for r in ranking] == [1, 2, 3]
    assert [r["empate_silueta"] for r in ranking] == [True, True, False]


def test_ordenar_usa_davies_bouldin_y_luego_complejidad():
    rs = [
        _r("gmm_uso", {"n_components": 3, "covariance_type": "full"}, 0.15, 0.5, 1.6),
        _r("gmm_uso", {"n_components": 3, "covariance_type": "diag"}, 0.15, 0.5, 1.6),
        _r("gmm_uso", {"n_components": 4, "covariance_type": "diag"}, 0.15, 0.5, 1.5),
    ]
    ranking = ordenar(rs, CFG)
    assert [(r["params"]["n_components"], r["params"]["covariance_type"]) for r in ranking] == [
        (4, "diag"),  # menor Davies-Bouldin
        (3, "diag"),  # mismo DB: menor complejidad que full
        (3, "full"),
    ]


def test_ordenar_deja_al_final_las_que_no_tienen_silueta():
    rs = [_r("kmeans_uso", {"k": 8}, np.nan, 0.1, np.nan), _r("kmeans_uso", {"k": 3}, 0.1, 0.5, 2)]
    assert [r["params"]["k"] for r in ordenar(rs, CFG)] == [3, 8]


def test_rejilla_y_complejidad():
    assert len(rejilla("kmeans_uso", CONFIG_P3)) == 6
    assert len(rejilla("gmm_uso", CONFIG_P3)) == 24
    with pytest.raises(ValueError):
        rejilla("kmeans_geo", CONFIG_P3)
    assert complejidad("gmm_uso", {"n_components": 3, "covariance_type": "spherical"}) < (
        complejidad("gmm_uso", {"n_components": 3, "covariance_type": "full"})
    )


def test_mlflow_project_lanza_la_busqueda_con_buscar(monkeypatch):
    from mibici import features_p3, mlflow_project
    from mibici.modeling import train_p3

    llamadas = {}
    monkeypatch.setattr(features_p3, "main", lambda: None)
    monkeypatch.setattr(train_p3, "main", lambda **kw: llamadas.update(kw))
    mlflow_project.run_train("p3", "gmm_uso", {"buscar": 1})
    assert llamadas["buscar_"] is True and llamadas["model"] == "gmm_uso"
    with pytest.raises(ValueError, match="buscar=1"):
        mlflow_project.run_train("p3", "kmeans_uso", {"buscar": 1, "k": 4})
