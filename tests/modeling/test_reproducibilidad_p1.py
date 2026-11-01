"""Repetibilidad de candidatos P1 con datos sinteticos y configuracion fija.

Compara dos ajustes por candidato; no ejecuta train_p1.main ni runs MLflow.
No afirma equivalencia entre plataformas o versiones de dependencias.
"""

import numpy as np
import pandas as pd
import pytest

from mibici import features as fe
from mibici.config import CONFIG_P1
from mibici.modeling import train_p1 as t1


def _datos():
    rng = np.random.default_rng(29)
    days = pd.date_range("2026-01-01", periods=180, freq="D")
    panel = pd.concat(
        [
            pd.DataFrame(
                {
                    "estacion": station,
                    "fecha": days,
                    "salidas": rng.poisson(10 + station * 5, len(days)),
                    "primer_dia": days[0],
                }
            )
            for station in (1, 2, 3)
        ],
        ignore_index=True,
    )
    features = fe.construir_features(panel)
    features = features.loc[features["elegible"]].copy()
    boundary = days[140]
    # Mismo embargo D-2 que el código de P1; no usar el día anterior al primer test.
    train = features[features["fecha"] <= fe.ultimo_dia_conocido(boundary)]
    test = features[features["fecha"] >= boundary]
    assert not train.empty and not test.empty
    assert (test["fecha"].min() - train["fecha"].max()).days == fe.LAG_MINIMO
    return train, test


@pytest.mark.parametrize(
    "candidate,params",
    [
        ("poisson", {"alpha": 1.0}),
        ("hgb_poisson", {"max_iter": 20, "max_leaf_nodes": 7, "learning_rate": 0.1}),
        ("hgb_absoluto", {"max_iter": 20, "max_leaf_nodes": 7, "learning_rate": 0.1}),
    ],
)
def test_predicciones_p1_repetibles_misma_semilla(candidate, params):
    train, test = _datos()
    before = train.copy(deep=True)
    semilla = CONFIG_P1.semilla
    a = t1.ajustar_candidato(candidate, train, params.copy(), semilla)
    b = t1.ajustar_candidato(candidate, train, params.copy(), semilla)
    if candidate != "poisson":
        # La semilla de la configuración llega al modelo y no hay paro temprano.
        assert a.get_params()["random_state"] == b.get_params()["random_state"] == semilla
        assert a.n_iter_ == b.n_iter_ == params["max_iter"]
    pa = np.asarray(a.predict(test[fe.FEATURES]), dtype=float)
    pb = np.asarray(b.predict(test[fe.FEATURES]), dtype=float)
    assert pa.shape == pb.shape == (len(test),)
    assert np.isfinite(pa).all() and np.isfinite(pb).all()
    assert (pa >= 0).all() and (pb >= 0).all()
    np.testing.assert_allclose(pa, pb, rtol=1e-10, atol=1e-10)
    assert t1.mae(test["objetivo"], pa) == pytest.approx(t1.mae(test["objetivo"], pb), abs=1e-10)
    pd.testing.assert_frame_equal(train, before)
