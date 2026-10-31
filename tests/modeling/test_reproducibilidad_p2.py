"""Reproducibilidad del entrenamiento de P2: misma semilla, mismas probabilidades y metricas.

Usan un panel horario sintetico pequeno (no necesitan los datos reales) y cortes de fecha
reducidos. Cada prueba entrena dos veces desde cero con la semilla de P2 y compara los
resultados exactos.
"""

import numpy as np
import pandas as pd
import pytest

from mibici import features_p2 as fp
from mibici.config import CONFIG_P2
from mibici.modeling import train_p2 as t2

SEMILLA = CONFIG_P2.seed  # 29
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
CANDIDATAS = {
    "hgb": lambda train, s: t2.ajustar(train, REJILLA_CORTA[0], s),
    "logistica": lambda train, s: t2.ajustar_logistica(train, {"max_iter": 100}, s),
    "random_forest": lambda train, s: t2.ajustar_random_forest(
        train, {"n_estimators": 10, "max_samples": 500, "min_samples_leaf": 5}, s
    ),
}


@pytest.fixture(autouse=True)
def cortes_chicos(monkeypatch):
    for clave, valor in CORTES_CHICOS.items():
        monkeypatch.setitem(t2.CORTES, clave, valor)


def _panel() -> pd.DataFrame:
    """Tres estaciones con dias de servicio completos y salidas y llegadas con ruido."""
    rng = np.random.default_rng(0)
    horas = pd.date_range("2026-01-01 05:00", "2026-03-02 05:00", freq="h", inclusive="left")
    h = horas.hour.to_numpy()
    manana, tarde = (h >= 7) & (h <= 9), (h >= 17) & (h <= 19)
    piezas = []
    for estacion in (1, 2, 3):
        salidas = 1 + 4 * manana + rng.integers(0, 3, len(horas))
        llegadas = 1 + 4 * tarde + rng.integers(0, 3, len(horas))
        piezas.append(
            pd.DataFrame(
                {
                    "ts": horas,
                    "estacion": estacion,
                    "capacidad": np.float32(10 + 5 * estacion),
                    "salidas": salidas.astype("float32"),
                    "llegadas": llegadas.astype("float32"),
                    "primer_dia": horas[0],
                }
            )
        )
    panel = pd.concat(piezas, ignore_index=True)
    panel["neto"] = panel["llegadas"] - panel["salidas"]
    return panel


def _entrenar_y_evaluar(nombre: str, semilla: int) -> tuple[np.ndarray, dict]:
    """Desde el panel: etiqueta, particiona, entrena con train completo y mide en test."""
    partes = t2.preparar(0.5, _panel())
    por_est, por_hora = fp.stats_estacion(partes["train"])
    train = fp.unir_stats(partes["train"], por_est, por_hora)
    test = fp.unir_stats(partes["test"], por_est, por_hora)
    modelo = CANDIDATAS[nombre](train, semilla)
    clases, proba, pred = t2.predecir(modelo, test)
    return proba, t2.metricas(test["objetivo"], pred, proba, clases, "test")


@pytest.mark.parametrize("nombre", list(CANDIDATAS))
def test_misma_semilla_mismas_probabilidades_y_metricas(nombre):
    proba_a, metricas_a = _entrenar_y_evaluar(nombre, SEMILLA)
    proba_b, metricas_b = _entrenar_y_evaluar(nombre, SEMILLA)
    if nombre == "random_forest":
        # con n_jobs=-1 los arboles suman su probabilidad en hilos y el orden de la suma
        # cambia; los arboles son los mismos, pero el promedio puede diferir en el ultimo bit
        np.testing.assert_allclose(proba_a, proba_b, rtol=0, atol=1e-12)
        assert np.array_equal(proba_a.argmax(1), proba_b.argmax(1))
    else:
        assert np.array_equal(proba_a, proba_b)  # iguales bit a bit, no solo parecidas
    assert metricas_a == metricas_b


def test_afinar_con_la_misma_semilla_elige_igual():
    # la rejilla completa se vuelve a recorrer desde cero: misma eleccion y mismo val_macro_f1
    a = t2.afinar(t2.preparar(0.5, _panel()), SEMILLA, rejilla=REJILLA_CORTA)
    b = t2.afinar(t2.preparar(0.5, _panel()), SEMILLA, rejilla=REJILLA_CORTA)
    assert a == b
