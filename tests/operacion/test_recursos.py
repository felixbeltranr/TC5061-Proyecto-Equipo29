import json
import math
import subprocess
import time

import pytest

from mibici.operacion import recursos

DOCKER = (
    '{"BlockIO":"0B / 0B","CPUPerc":"85.50%","Container":"a1","ID":"a1",'
    '"MemPerc":"30.00%","MemUsage":"307.2MiB / 1GiB","Name":"src-api-1",'
    '"NetIO":"1.5kB / 2.25MB","PIDs":"12"}\n'
    '{"BlockIO":"0B / 0B","CPUPerc":"0.40%","Container":"b2","ID":"b2",'
    '"MemPerc":"20.00%","MemUsage":"204.8MiB / 1GiB","Name":"src-mlflow-1",'
    '"NetIO":"--","PIDs":"5"}\n'
)


@pytest.mark.parametrize(
    "texto,esperado",
    [
        ("0B", 0),
        ("512B", 512),
        ("1.5kB", 1500),
        ("1KiB", 1024),
        ("307.2MiB", 307.2 * 1024**2),
        ("1GiB", 1024**3),
        ("2.25MB", 2.25e6),
        ("--", 0),
        ("", 0),
    ],
)
def test_bytes_de(texto, esperado):
    assert recursos.bytes_de(texto) == pytest.approx(esperado)


def test_bytes_de_rechaza_unidades_raras():
    with pytest.raises(ValueError):
        recursos.bytes_de("3 lunas")


def test_porcentaje():
    assert recursos.porcentaje("85.50%") == 85.5
    assert recursos.porcentaje("--") == 0


def test_parsear_docker():
    registros = recursos.registros_de_salida(DOCKER)
    api = recursos.parsear_registro(registros[0], 10.0)
    assert api.contenedor == "src-api-1"
    assert api.cpu_pct == 85.5
    assert api.memoria_bytes == pytest.approx(307.2 * 1024**2)
    assert api.limite_memoria_bytes == 1024**3
    assert api.red_salida_bytes == pytest.approx(2.25e6)
    mlflow = recursos.parsear_registro(registros[1], 10.0)
    assert mlflow.red_entrada_bytes == 0


def test_parsear_podman_lista_json():
    texto = json.dumps(
        [
            {
                "name": "src_api_1",
                "cpu_percent": "12.5%",
                "mem_usage": "100MB / 1GB",
                "net_io": "1kB / 2kB",
            }
        ]
    )
    muestra = recursos.parsear_registro(recursos.registros_de_salida(texto)[0], 1.0)
    assert (muestra.contenedor, muestra.cpu_pct, muestra.memoria_bytes) == ("src_api_1", 12.5, 1e8)


def test_comando_stats():
    assert recursos.comando_stats("docker", ["a"]) == [
        "docker",
        "stats",
        "--no-stream",
        "--format",
        "{{json .}}",
        "a",
    ]
    assert recursos.comando_stats("podman", [])[-1] == "json"


def _ejecutar_falso(salida: str, codigo: int = 0):
    def ejecutar(comando, **_):
        return subprocess.CompletedProcess(comando, codigo, stdout=salida, stderr="falló")

    return ejecutar


def test_muestreador_toma_muestras_en_segundo_plano():
    with recursos.Muestreador(
        ["src-api-1"], intervalo=0.02, ejecutar=_ejecutar_falso(DOCKER)
    ) as muestreador:
        time.sleep(0.2)
    assert len(muestreador.muestras) >= 4
    assert {m.contenedor for m in muestreador.muestras} == {"src-api-1", "src-mlflow-1"}
    assert not muestreador._hilo.is_alive()


def test_muestreador_registra_errores_sin_detenerse():
    muestreador = recursos.Muestreador(["x"], ejecutar=_ejecutar_falso("", codigo=1))
    assert muestreador.tomar() == []
    assert muestreador.errores == ["falló"]


def test_csv_ida_y_vuelta(tmp_path):
    muestras = [recursos.parsear_registro(r, 5.0) for r in recursos.registros_de_salida(DOCKER)]
    ruta = tmp_path / "sub" / "recursos.csv"
    recursos.guardar_csv(muestras, ruta)
    assert recursos.leer_csv(ruta) == muestras


def test_percentil_como_numpy():
    assert recursos.percentil([1, 2, 3, 4], 50) == 2.5
    assert recursos.percentil([1, 2, 3, 4], 95) == pytest.approx(3.85)
    assert math.isnan(recursos.percentil([], 95))


def test_resumir_en_nucleos_y_mib():
    muestras = [
        recursos.Muestra(t, "api", cpu, mem * 1024**2, 1024**3, 0, 0)
        for t, cpu, mem in [(0, 50, 300), (1, 100, 310), (2, 150, 320)]
    ]
    resumen = recursos.resumir(muestras)["api"]
    assert resumen["muestras"] == 3
    assert resumen["cpu_nucleos_promedio"] == pytest.approx(1.0)
    assert resumen["cpu_nucleos_max"] == pytest.approx(1.5)
    assert resumen["memoria_mib_max"] == pytest.approx(320)
    assert resumen["limite_memoria_mib"] == pytest.approx(1024)


def _muestras_en(marcas):
    return [recursos.Muestra(t, "api", 100.0, 0, 0, 0, 0) for t in marcas]


def test_en_ventana_usa_las_marcas_del_historial():
    muestras = _muestras_en([998, 1002, 1012, 1020, 1030, 1035])
    # Historial de 1000 a 1030 y 10 s de rampa: quedan las muestras de 1010 a 1030.
    estables = recursos.en_ventana(muestras, [1000.0, 1005.0, 1030.0], 10, 0, 0)
    assert [m.marca for m in estables] == [1012, 1020, 1030]


def test_en_ventana_sin_marcas_usa_inicio_y_fin():
    muestras = _muestras_en([998, 1002, 1012, 1020, 1030, 1035])
    estables = recursos.en_ventana(muestras, [math.nan], 10, inicio=1000, fin=1025)
    assert [m.marca for m in estables] == [1012, 1020]
    assert recursos.en_ventana(muestras, [], 10, inicio=1000, fin=1025) == estables


def test_en_ventana_vacia_devuelve_todas():
    muestras = _muestras_en([1, 2, 3])
    assert recursos.en_ventana(muestras, [1000.0, 1030.0], 10, 0, 0) == muestras
    assert recursos.en_ventana([], [1000.0], 10, 0, 0) == []


def test_muestreador_decodifica_utf8():
    recibidos = {}

    def ejecutar(comando, **opciones):
        recibidos.update(opciones)
        return subprocess.CompletedProcess(comando, 0, stdout=DOCKER, stderr="")

    recursos.Muestreador(["src-api-1"], ejecutar=ejecutar).tomar()
    assert recibidos["encoding"] == "utf-8" and recibidos["errors"] == "replace"


def test_contenedor_de_servicio():
    resumen = {"src-api-1": {}, "src-mlflow-1": {}}
    assert recursos.contenedor_de_servicio(resumen, "api") == "src-api-1"
    assert recursos.contenedor_de_servicio(resumen, "redis") is None
