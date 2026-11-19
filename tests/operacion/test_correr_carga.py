"""Funciones del orquestador de la prueba de carga que no necesitan Docker ni Locust."""

import importlib.util
import json
from pathlib import Path
import urllib.error

import pytest

RUTA = Path(__file__).resolve().parents[2] / "scripts" / "carga" / "correr_carga.py"


@pytest.fixture(scope="module")
def orquestador():
    spec = importlib.util.spec_from_file_location("correr_carga", RUTA)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_preguntas_sin_modelo(orquestador):
    assert orquestador.preguntas_sin_modelo("Models not loaded: p2 (RestException)") == {"p2"}
    assert orquestador.preguntas_sin_modelo(
        "Models not loaded: p2 (RestException), p3 (OSError)"
    ) == {"p2", "p3"}
    assert orquestador.preguntas_sin_modelo("No model is loaded") == {"p1", "p2", "p3"}


def test_nombres_compose_un_objeto_por_renglon_y_lista(orquestador):
    renglones = (
        json.dumps({"Service": "mlflow", "Name": "src-mlflow-1"})
        + "\n"
        + json.dumps({"Service": "api", "Name": "src-api-1"})
    )
    esperado = {"mlflow": "src-mlflow-1", "api": "src-api-1"}
    assert orquestador.nombres_compose(renglones) == esperado
    lista = json.dumps([{"Service": "api", "Name": "src-api-1"}])
    assert orquestador.nombres_compose(lista) == {"api": "src-api-1"}
    assert orquestador.nombres_compose("") == {}


def test_reemplazar_bloque(orquestador):
    texto = (
        "antes\n"
        f"{orquestador.INICIO_BLOQUE}\nviejo\n{orquestador.FIN_BLOQUE}\n"
        "después\n"
    )
    nuevo = orquestador.reemplazar_bloque(
        texto, f"{orquestador.INICIO_BLOQUE}\nnuevo\n{orquestador.FIN_BLOQUE}"
    )
    assert "nuevo" in nuevo and "viejo" not in nuevo
    assert nuevo.startswith("antes\n") and nuevo.endswith("después\n")
    with pytest.raises(ValueError):
        orquestador.reemplazar_bloque("sin bloque", "x")


def test_el_readme_trae_el_bloque(orquestador):
    texto = orquestador.README.read_text(encoding="utf-8")
    assert orquestador.INICIO_BLOQUE in texto and orquestador.FIN_BLOQUE in texto


def test_tabla_de_etapas(orquestador):
    etapa = {
        "usuarios": 50,
        "rps_sostenido": 40.0,
        "p50_ms": 20.0,
        "p95_ms": 90.0,
        "p99_ms": 120.0,
        "tasa_error": 0.0,
        "recursos": {"api": {"cpu_nucleos_p95": 0.7, "memoria_mib_max": 320.0}},
    }
    tabla = orquestador.tabla_etapas([etapa])
    assert "| 50 | 40.0 | 20 | 90 | 120 | 0.00% | 0.70 | 320 |" in tabla


class _Respuesta:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return b"{}"


def test_sonda_de_lotes_sigue_si_un_endpoint_falla(orquestador, monkeypatch):
    llamadas = []

    def urlopen(peticion, timeout):
        llamadas.append(peticion.full_url)
        if "/p2/" in peticion.full_url:
            raise urllib.error.URLError("conexión rechazada")
        return _Respuesta()

    monkeypatch.setattr(orquestador.urllib.request, "urlopen", urlopen)
    lotes = orquestador.sonda_de_lotes("http://api", 5, 3, ["p1", "p2", "p3"])
    assert "conexión rechazada" in lotes["p2_error"]
    assert "p2_ms" not in lotes
    assert lotes["p3_ms"] >= 0
    assert sum("/p3/" in url for url in llamadas) == 3


def test_sonda_de_lotes_con_tiempo_agotado(orquestador, monkeypatch):
    def urlopen(peticion, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(orquestador.urllib.request, "urlopen", urlopen)
    lotes = orquestador.sonda_de_lotes("http://api", 5, 2, ["p2", "p3"])
    assert lotes["p2_error"].startswith("TimeoutError")
    assert lotes["p3_error"].startswith("TimeoutError")
    assert lotes["estaciones"] == 5


def test_opcion_vcpu_max(orquestador):
    assert orquestador.argumentos([]).vcpu_max == 1.0
    assert orquestador.argumentos(["--vcpu-max", "2"]).vcpu_max == 2.0


def test_sin_csv_de_locust_termina_con_codigo_2(orquestador, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(orquestador, "endpoints_disponibles", lambda host: ["p1"])
    monkeypatch.setattr(orquestador, "correr_etapa", lambda *a: (0.0, 1.0))
    codigo = orquestador.main(["--sin-recursos", "--etapas", "10,50", "--salida", str(tmp_path)])
    salida = capsys.readouterr().out
    assert codigo == 2
    assert salida.count("no dejó los CSV de Locust") == 2
    assert "Ninguna etapa dejó resultados" in salida
    assert not (tmp_path / "resumen.json").exists()
