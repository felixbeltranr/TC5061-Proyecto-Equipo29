"""El barrido de lotes de P3 sólo calcula; la medición la hace la sonda del orquestador.

Estas pruebas fijan las mediciones para comprobar el cálculo sin levantar nada.
"""

from pathlib import Path
import sys

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts" / "carga"))
import sonda_lotes_p3 as sonda  # noqa: E402

from mibici.operacion import finops  # noqa: E402

TARIFA = finops.TARIFAS["cloud_run_peticion"]
DEMANDA = finops.Demanda()
# Una petición de más estaciones tarda más, pero mucho menos que proporcionalmente.
MEDICIONES = {1: 13.0, 10: 25.0, 100: 29.0}


@pytest.fixture
def filas():
    return sonda.tabla(MEDICIONES, TARIFA, 1.0, 2.0, DEMANDA)


def test_el_costo_por_inferencia_baja_al_agrupar(filas):
    assert [f["estaciones"] for f in filas] == [1, 10, 100]
    por_inferencia = [f["ms_por_inferencia"] for f in filas]
    assert por_inferencia == sorted(por_inferencia, reverse=True)
    assert filas[0]["ms_por_inferencia"] == pytest.approx(13.0)
    assert filas[2]["ms_por_inferencia"] == pytest.approx(0.29)
    assert filas[0]["factor_vs_una"] == 1.0
    assert filas[2]["factor_vs_una"] == pytest.approx(44.8, abs=0.1)


def test_la_facturacion_redondea_a_la_granularidad_de_la_tarifa(filas):
    # Las tres medianas caen por debajo de los 100 ms que cobra Cloud Run por petición, así que
    # el costo por petición es el mismo y todo el ahorro viene de mandar menos peticiones.
    usd_peticion = [f["usd_1000_inferencias"] * f["estaciones"] / 1000 for f in filas]
    assert usd_peticion[0] == pytest.approx(usd_peticion[1])
    assert usd_peticion[0] == pytest.approx(usd_peticion[2])


def test_agrupar_reduce_las_peticiones_del_mes(filas):
    peticiones = [f["peticiones_mes"] for f in filas]
    assert peticiones == sorted(peticiones, reverse=True)
    assert peticiones[0] == DEMANDA.estaciones          # una estación por petición
    assert peticiones[2] == 5                           # 468 estaciones en lotes de 100


def test_el_markdown_trae_un_renglon_por_tamano(filas):
    texto = sonda.a_markdown(filas, TARIFA, 1.0, 2.0, 20, DEMANDA.inferencias()["p3"])
    assert texto.count("\n| ") >= len(filas)
    for f in filas:
        assert f"| {f['estaciones']} |" in texto
    assert TARIFA.nombre in texto
