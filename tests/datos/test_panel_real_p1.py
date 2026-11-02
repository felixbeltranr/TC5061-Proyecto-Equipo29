"""Validación del panel diario real de P1 con el contrato y DeepChecks.

Lee ``data/interim/p1_panel_diario.parquet``, el panel que genera ``python -m mibici.features``
(o ``mlflow run . -e train -P question=p1``), y se salta si todavía no existe. Aplica sobre todas
las filas el contrato y los dos checks de ``scripts/p1/validar_panel_p1.py``, sin corregir ni
modificar el archivo. Los reportes HTML quedan en la carpeta temporal de pytest, fuera del
repositorio. Las pruebas con paneles fabricados están en ``test_integridad_p1.py``.
"""

import os
import sys

import pytest

if sys.version_info[:2] != (3, 11):
    pytest.skip(
        "Validaciones P1 requieren el entorno DeepChecks Python 3.11; "
        "este SKIP no acredita calidad de datos ni ejecución de checks",
        allow_module_level=True,
    )
os.environ["DISABLE_DEEPCHECKS_ANONYMOUS_TELEMETRY"] = "1"
os.environ["DISABLE_LATEST_VERSION_CHECK"] = "1"
pytest.importorskip(
    "deepchecks",
    reason="Falta DeepChecks en este intérprete; ejecutar con el entorno 3.11 separado",
)

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts/p1"))
from validar_panel_p1 import SPEC, contract, deepchecks_integrity, sha256

PANEL = RAIZ / "data" / "interim" / "p1_panel_diario.parquet"
COLUMNAS = SPEC["p1"]["columns"]


@pytest.fixture(scope="module")
def panel() -> dict:
    """El panel real con su huella antes de leerlo; se salta si no existe."""
    if not PANEL.is_file():
        pytest.skip(
            "No existe data/interim/p1_panel_diario.parquet. Genéralo con "
            "'python -m mibici.features' o 'mlflow run . -e train -P question=p1 "
            "--env-manager local' en el entorno del proyecto; este SKIP no valida el panel real."
        )
    huella = sha256(PANEL)
    faltantes = sorted(set(COLUMNAS) - set(pq.read_schema(PANEL).names))
    assert not faltantes, f"Faltan columnas en el panel real: {faltantes}"
    return {"datos": pd.read_parquet(PANEL, columns=COLUMNAS), "sha256": huella}


def test_panel_real_cumple_el_contrato(panel):
    datos = panel["datos"]
    resultado = contract(datos, "p1")
    assert resultado["failures"] == []
    assert resultado["rows"] == len(datos) > 0
    assert resultado["stations"] == datos["estacion"].nunique() > 0
    assert resultado["duplicate_keys"] == 0
    assert pd.Timestamp(resultado["min_timestamp"]) < pd.Timestamp(resultado["max_timestamp"])
    assert datos["salidas"].sum() > 0


def test_panel_real_pasa_los_checks_de_deepchecks(panel, tmp_path):
    datos = panel["datos"]
    resultado = deepchecks_integrity(datos, "p1", tmp_path)
    assert resultado["passed"]
    assert resultado["checks_run"] == 2
    assert resultado["rows_checked"] == len(datos)
    assert [c["status"] for c in resultado["conditions"]] == ["PASS", "PASS"]
    assert (tmp_path / resultado["html"]).is_file()
    assert sha256(PANEL) == panel["sha256"]


def test_el_contrato_detecta_cambios_en_el_panel_real(panel):
    """Control negativo sobre una copia: una clave repetida y un conteo negativo no pasan."""
    copia = panel["datos"].head(1000).copy()
    copia = pd.concat([copia, copia.iloc[[0]]], ignore_index=True)
    copia.loc[1, "salidas"] = -1
    reglas = {f["rule"] for f in contract(copia, "p1")["failures"]}
    assert {"unique_station_timestamp", "nonnegative_integer_count"} <= reglas
    assert sha256(PANEL) == panel["sha256"]
