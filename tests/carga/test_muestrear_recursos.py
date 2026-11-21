"""Lectura de ``docker stats``: las unidades mezclan binario (MiB) y decimal (MB)."""

from datetime import datetime
from pathlib import Path
import sys

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts" / "carga"))
import muestrear_recursos as muestreo  # noqa: E402

LECTURA = {
    "BlockIO": "0B / 12.3kB",
    "CPUPerc": "12.34%",
    "Container": "a1b2c3",
    "ID": "a1b2c3",
    "MemPerc": "1.23%",
    "MemUsage": "123.4MiB / 7.654GiB",
    "Name": "tc5061-api-1",
    "NetIO": "1.2MB / 3.4MB",
    "PIDs": "7",
}


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("0B", 0.0),
        ("123.4MiB", 123.4 * 1024**2),
        ("7.654GiB", 7.654 * 1024**3),
        ("12.3kB", 12_300.0),
        ("1.2MB", 1_200_000.0),
        ("--", None),
        ("", None),
    ],
)
def test_a_bytes_distingue_binario_de_decimal(texto, esperado):
    obtenido = muestreo.a_bytes(texto)
    if esperado is None:
        assert obtenido is None
    else:
        assert obtenido == pytest.approx(esperado)


def test_a_porcentaje():
    assert muestreo.a_porcentaje("12.34%") == pytest.approx(12.34)
    assert muestreo.a_porcentaje("--") is None


def test_par_separa_uso_y_limite():
    usado, limite = muestreo.par("123.4MiB / 7.654GiB")
    assert usado == pytest.approx(123.4 * 1024**2)
    assert limite == pytest.approx(7.654 * 1024**3)


def test_fila_convierte_a_las_unidades_del_csv():
    fila = muestreo.fila(LECTURA, datetime(2026, 11, 22, 10, 30), 4.0)
    assert set(fila) == set(muestreo.COLUMNAS)
    assert fila["contenedor"] == "tc5061-api-1"
    assert fila["cpu_pct"] == pytest.approx(12.34)
    assert fila["mem_mib"] == pytest.approx(123.4)
    assert fila["mem_limite_mib"] == pytest.approx(7.654 * 1024)
    assert fila["net_rx_mb"] == pytest.approx(1.2)
    assert fila["net_tx_mb"] == pytest.approx(3.4)
    assert fila["disco_lectura_mb"] == pytest.approx(0.0)
    assert fila["disco_escritura_mb"] == pytest.approx(0.0123)
    assert fila["segundo"] == 4.0


def test_fila_tolera_lecturas_sin_datos():
    lectura = {"Name": "x", "CPUPerc": "--", "MemUsage": "--"}
    fila = muestreo.fila(lectura, datetime(2026, 11, 22), 0.0)
    assert fila["cpu_pct"] is None
    assert fila["mem_mib"] is None
