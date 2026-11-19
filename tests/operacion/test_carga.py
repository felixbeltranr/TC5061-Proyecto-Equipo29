"""Lectura de los CSV de Locust (mismos encabezados que escribe Locust 2.x)."""

import math

import pytest

from mibici.operacion import carga

STATS = """Type,Name,Request Count,Failure Count,Median Response Time,Average Response Time,Min Response Time,Max Response Time,Average Content Size,Requests/s,Failures/s,50%,66%,75%,80%,90%,95%,98%,99%,99.9%,99.99%,100%
GET,GET /live,48,0,2,4.6,1.3,20.5,18.0,3.2,0.0,2,3,5,6,16,19,21,21,21,21,21
POST,P1 POST /api/v1/predict,150,3,18,25.5,5.3,89.5,123.4,10.3,0.2,18,24,32,38,56,71,83,87,90,90,90
,Aggregated,198,3,27,34.4,1.3,162.1,128.5,29.7,0.2,27,39,45,53,75,100,150,160,160,160,160
"""

ENCABEZADO = (
    "Timestamp,User Count,Type,Name,Requests/s,Failures/s,50%,66%,75%,80%,90%,95%,98%,99%,"
    "99.9%,99.99%,100%,Total Request Count,Total Failure Count,Total Median Response Time,"
    "Total Average Response Time,Total Min Response Time,Total Max Response Time,"
    "Total Average Content Size"
)


def _historial(filas):
    renglones = [ENCABEZADO]
    for segundo, usuarios, rps in filas:
        renglones.append(
            f"{1000 + segundo},{usuarios},,Aggregated,{rps},0.0,20,25,30,32,40,50,60,70,80,80,80,"
            f"{segundo * 10},0,20,22.0,3,90,120"
        )
    renglones.insert(1, "1000,0,,Aggregated,0.0,0.0,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,0,0,0,0.0,0,0,0")
    return "\n".join(renglones) + "\n"


@pytest.fixture
def archivos(tmp_path):
    (tmp_path / "e_stats.csv").write_text(STATS, encoding="utf-8")
    filas = [(s, min(50, s * 10), 5.0 if s < 10 else 40.0) for s in range(1, 31)]
    (tmp_path / "e_stats_history.csv").write_text(_historial(filas), encoding="utf-8")
    return tmp_path / "e_stats.csv", tmp_path / "e_stats_history.csv"


def test_leer_stats_y_fila_agregada(archivos):
    stats = carga.leer_stats(archivos[0])
    total = carga.fila_agregada(stats)
    assert total["Request Count"] == 198
    assert total["95%"] == 100


def test_historial_con_na_y_segundos(archivos):
    historial = carga.leer_historial(archivos[1])
    assert historial["segundo"].min() == 0
    assert math.isnan(historial.iloc[0]["50%"])


def test_rps_sostenido_descarta_la_rampa(archivos):
    historial = carga.leer_historial(archivos[1])
    # Con 50 usuarios desde el segundo 5 y descartando 10 s, sólo quedan filas de 40 RPS.
    assert carga.rps_sostenido(historial, usuarios=50, descartar_s=10) == pytest.approx(40.0)


def test_resumen_etapa(archivos):
    stats, historial = carga.leer_stats(archivos[0]), carga.leer_historial(archivos[1])
    resumen = carga.resumen_etapa(stats, historial, usuarios=50, duracion_s=30)
    assert resumen["peticiones"] == 198
    assert resumen["tasa_error"] == pytest.approx(3 / 198)
    assert resumen["p95_ms"] == 100
    assert resumen["rps_sostenido"] == pytest.approx(40.0)
    nombres = [e["endpoint"] for e in resumen["endpoints"]]
    assert nombres == ["GET /live", "P1 POST /api/v1/predict"]


def test_resumen_etapa_separa_las_sondas_live(archivos):
    stats, historial = carga.leer_stats(archivos[0]), carga.leer_historial(archivos[1])
    resumen = carga.resumen_etapa(stats, historial, usuarios=50, duracion_s=30)
    # 150 de las 198 peticiones son inferencias de P1; las 48 restantes son GET /live.
    assert resumen["rps_inferencias"] == pytest.approx(40.0 * 150 / 198)
    assert resumen["promedio_inferencia_ms"] == pytest.approx(25.5)


def test_promedio_de_inferencias_ponderado(tmp_path):
    texto = STATS.replace(
        ",Aggregated,",
        "POST,P2 POST /api/v1/p2/predict,50,0,40,45.5,5.0,90.0,100.0,3.0,0.0,"
        "40,45,50,55,60,70,80,85,90,90,90\n,Aggregated,",
    )
    ruta = tmp_path / "p_stats.csv"
    ruta.write_text(texto, encoding="utf-8")
    filas = [(s, 50, 40.0) for s in range(1, 31)]
    (tmp_path / "p_stats_history.csv").write_text(_historial(filas), encoding="utf-8")
    resumen = carga.resumen_etapa(
        carga.leer_stats(ruta),
        carga.leer_historial(tmp_path / "p_stats_history.csv"),
        usuarios=50,
        duracion_s=30,
    )
    assert resumen["promedio_inferencia_ms"] == pytest.approx((150 * 25.5 + 50 * 45.5) / 200)
    assert resumen["rps_inferencias"] == pytest.approx(resumen["rps_sostenido"] * 200 / 248)


def test_sin_fila_agregada(tmp_path):
    ruta = tmp_path / "x.csv"
    ruta.write_text(STATS.rsplit("\n", 2)[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        carga.fila_agregada(carga.leer_stats(ruta))


def test_capacidad_sostenida_respeta_objetivos():
    etapas = [
        {"usuarios": 10, "rps_sostenido": 8.0, "p95_ms": 40, "tasa_error": 0.0},
        {"usuarios": 50, "rps_sostenido": 40.0, "p95_ms": 120, "tasa_error": 0.0},
        {"usuarios": 100, "rps_sostenido": 70.0, "p95_ms": 800, "tasa_error": 0.0},
        {"usuarios": 200, "rps_sostenido": 75.0, "p95_ms": 300, "tasa_error": 0.05},
    ]
    assert carga.capacidad_sostenida(etapas)["usuarios"] == 50
    assert carga.capacidad_sostenida(etapas, p95_max_ms=1000)["usuarios"] == 100
    assert carga.capacidad_sostenida(etapas, p95_max_ms=10) is None
