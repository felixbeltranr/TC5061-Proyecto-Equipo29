import numpy as np
import pytest

from mibici.features_p2 import acumular_ocupacion, etiquetar_ocupacion


def test_etiquetas_por_umbral():
    cap = np.array([20, 20, 20, 20, 20])
    occ = np.array([0, 1.9, 2, 18, 18.5])
    assert list(etiquetar_ocupacion(occ, cap)) == [
        "sin_bicicletas",
        "sin_bicicletas",
        "segura",
        "segura",
        "sin_anclajes",
    ]


def test_estacion_llena_es_sin_anclajes():
    assert etiquetar_ocupacion([15], [15])[0] == "sin_anclajes"


def test_rechaza_ocupacion_fuera_de_rango():
    with pytest.raises(ValueError):
        etiquetar_ocupacion([16], [15])
    with pytest.raises(ValueError):
        etiquetar_ocupacion([-1], [15])


def test_rechaza_capacidad_cero():
    with pytest.raises(ValueError):
        etiquetar_ocupacion([0], [0])


def test_acumulado_respeta_capacidad():
    neto = np.array([[5, 5, -20, 3]])
    occ = acumular_ocupacion(neto, np.array([10]), 0.5)
    assert occ.tolist() == [[10, 10, 0, 3]]
    assert (occ <= 10).all() and (occ >= 0).all()
