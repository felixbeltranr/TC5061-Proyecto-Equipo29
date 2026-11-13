import numpy as np
import pandas as pd
import pytest

from mibici.monitoring import p3_ab
from mibici.monitoring.p3_datasets import P3_FEATURES


def _perfiles(n: int, seed: int) -> pd.DataFrame:
    """Perfiles sintéticos con tres grupos de uso bien separados."""
    rng = np.random.default_rng(seed)
    centros = rng.normal(0, 3, size=(3, len(P3_FEATURES)))
    grupo = np.arange(n) % 3
    datos = centros[grupo] + rng.normal(0, 1, size=(n, len(P3_FEATURES)))
    return pd.DataFrame(datos, columns=list(P3_FEATURES), index=pd.Index(range(n), name="estacion"))


@pytest.fixture(scope="module")
def modelos():
    entrenamiento = _perfiles(90, seed=1)
    a = p3_ab.entrenar(entrenamiento, {"k": 3, "n_init": 2})
    b = p3_ab.entrenar(_perfiles(90, seed=2), {"k": 3, "n_init": 2})
    return a, b, entrenamiento


def test_comparar_versiones_identicas_no_tiene_diferencias(modelos):
    a, _, perfiles = modelos

    tabla = p3_ab.comparar(a, a, perfiles, perfiles, perfiles, n_bootstrap=20)

    assert set(tabla.metrica) == {
        "silueta",
        "davies_bouldin",
        "calinski_harabasz",
        "ari_entre_quincenas",
        "proporcion_grupo_mas_chico",
    }
    assert np.allclose(tabla.b_menos_a, 0)
    assert np.allclose(tabla.dropna().ic95_bajo, 0) and np.allclose(tabla.dropna().ic95_alto, 0)
    assert p3_ab.decidir(tabla)["decision"] == "mantener A"


def _tabla(silueta_ic, ari_ic, tamano_b):
    return pd.DataFrame(
        [
            ["silueta", "neutral_12_variables", 0.1, 0.2, 0.1, *silueta_ic],
            ["ari_entre_quincenas", "asignaciones", 0.6, 0.6, 0.0, *ari_ic],
            ["proporcion_grupo_mas_chico", "asignaciones", 0.3, tamano_b, 0.0, np.nan, np.nan],
        ],
        columns=["metrica", "espacio", "a", "b", "b_menos_a", "ic95_bajo", "ic95_alto"],
    )


@pytest.mark.parametrize(
    ("silueta_ic", "ari_ic", "tamano_b", "decision"),
    [
        ((0.01, 0.05), (-0.02, 0.03), 0.2, "promover B"),
        ((-0.01, 0.05), (-0.02, 0.03), 0.2, "mantener A"),  # silueta: el IC incluye el cero
        ((0.01, 0.05), (-0.20, -0.01), 0.2, "mantener A"),  # B es menos estable
        ((0.01, 0.05), (-0.02, 0.03), 0.01, "mantener A"),  # un grupo de B es muy chico
    ],
)
def test_regla_de_decision(silueta_ic, ari_ic, tamano_b, decision):
    assert p3_ab.decidir(_tabla(silueta_ic, ari_ic, tamano_b))["decision"] == decision


def test_asignaciones_marcan_estaciones_nuevas_para_a(modelos):
    a, b, entrenamiento = modelos
    actual = _perfiles(6, seed=3).rename(index=lambda e: e + 88)  # 88 y 89 ya estaban

    asig = p3_ab.asignaciones(a, b, actual, entrenamiento)

    assert asig.nueva_para_a.tolist() == [False, False, True, True, True, True]
    assert set(asig.columns) == {"grupo_a", "grupo_b", "nueva_para_a"}
