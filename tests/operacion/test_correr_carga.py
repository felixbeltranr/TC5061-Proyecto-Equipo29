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


def test_primera_peticion_mide_cada_pregunta_con_modelo(orquestador, monkeypatch):
    llamadas = []

    def urlopen(peticion, timeout):
        llamadas.append(peticion.full_url)
        return _Respuesta()

    monkeypatch.setattr(orquestador.urllib.request, "urlopen", urlopen)
    primera = orquestador.primera_peticion("http://api", ["p1", "p3"])
    assert llamadas == ["http://api/api/v1/predict", "http://api/api/v1/p3/predict"]
    assert primera["p1_ms"] >= 0 and primera["p3_ms"] >= 0
    assert "p2_ms" not in primera
    texto = orquestador.texto_primera_peticion({"p1_ms": 120.4, "p2_ms": 2600.0})
    assert texto.endswith("P1 120 ms, P2 2,600 ms.")
    assert texto.startswith(
        "Primera inferencia de cada pregunta después de que /ready responde 200"
    )
    assert "no incluye el arranque del contenedor ni la carga de los modelos" in texto
    assert "en frío" not in texto


def test_primera_peticion_guarda_el_error_y_sigue(orquestador, monkeypatch):
    def urlopen(peticion, timeout):
        if peticion.full_url.endswith("/api/v1/predict"):
            raise urllib.error.URLError("conexión rechazada")
        return _Respuesta()

    monkeypatch.setattr(orquestador.urllib.request, "urlopen", urlopen)
    primera = orquestador.primera_peticion("http://api", ["p1", "p2"])
    assert "conexión rechazada" in primera["p1_error"]
    assert primera["p2_ms"] >= 0
    assert orquestador.texto_primera_peticion({"p1_error": "x"}) == ""


def test_bloque_del_readme_incluye_la_primera_peticion(orquestador):
    resumen = {
        "fecha": "2026-11-19_1800",
        "commit": "abc1234",
        "etapas": [],
        "duracion_s": 60,
        "endpoints": ["p1"],
        "p95_objetivo_ms": 500.0,
        "capacidad": None,
        "primera_peticion": {"p1_ms": 85.0},
    }
    bloque = orquestador.bloque_readme(resumen, "tabla", "reports/carga/x")
    assert "ni la carga de los modelos): P1 85 ms." in bloque
    sin = orquestador.bloque_readme({**resumen, "primera_peticion": {}}, "tabla", "reports/carga/x")
    assert "Primera inferencia" not in sin
    assert "ninguna etapa cumplió el objetivo." in sin
    assert "![Latencia por endpoint](reports/carga/x/endpoints.png)" in bloque


def test_bloque_del_readme_con_capacidad_y_saturacion(orquestador):
    resumen = {
        "fecha": "2026-11-19_1800",
        "commit": "abc1234",
        "etapas": [],
        "duracion_s": 90,
        "endpoints": ["p1", "p2", "p3"],
        "p95_objetivo_ms": 500.0,
        "capacidad": {
            "usuarios": 20,
            "rps_sostenido": 15.5,
            "rps_inferencias": 13.9,
            "p95_ms": 240.0,
        },
        "saturacion": {"usuarios": 60, "rps_sostenido": 22.6, "p95_ms": 2300.0},
        "primera_peticion": {},
    }
    bloque = orquestador.bloque_readme(resumen, "tabla", "reports/carga/x")
    assert (
        "Capacidad sostenida con p95 menor a 500 ms: 13.9 RPS de inferencias (15.5 en total "
        "con GET /live) con 20 usuarios y p95 de 240 ms. Saturación observada: 22.6 RPS con "
        "60 usuarios y p95 de 2,300 ms."
    ) in bloque


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


def test_opcion_vcpu_medidos(orquestador):
    assert orquestador.argumentos([]).vcpu_medidos == 1.0
    assert orquestador.argumentos(["--vcpu-medidos", "0.5"]).vcpu_medidos == 0.5


def test_etapas_y_duracion_por_omision(orquestador):
    args = orquestador.argumentos([])
    assert args.etapas == "10,20,30,40,60"
    assert args.duracion == 90


def test_main_guarda_la_saturacion_y_la_pasa_a_finops(orquestador, monkeypatch, tmp_path):
    def correr_etapa(args, usuarios, salida, endpoints):
        for sufijo in ("_stats.csv", "_stats_history.csv"):
            (salida / f"etapa_{usuarios:03d}{sufijo}").write_text("x\n", encoding="utf-8")
        return 0.0, 1.0

    def resumen_etapa(stats, historial, usuarios, duracion, descartar):
        return {
            "usuarios": usuarios,
            "rps_sostenido": {20: 15.5, 60: 22.6}[usuarios],
            "rps_inferencias": {20: 13.9, 60: 21.5}[usuarios],
            "p50_ms": 20.0,
            "p95_ms": {20: 240.0, 60: 2300.0}[usuarios],
            "p99_ms": 300.0,
            "tasa_error": 0.0,
            "promedio_ms": 60.0,
            "endpoints": [],
        }

    api = {"cpu_nucleos_promedio": 0.35, "cpu_nucleos_p95": 0.66, "memoria_mib_max": 245.0}
    llamadas = {}
    analizar = orquestador.finops.analizar

    def analizar_espia(*args, **kwargs):
        llamadas.update(kwargs)
        return analizar(*args, **kwargs)

    monkeypatch.setattr(orquestador, "endpoints_disponibles", lambda host: ["p1"])
    monkeypatch.setattr(orquestador, "primera_peticion", lambda host, endpoints: {})
    monkeypatch.setattr(orquestador, "correr_etapa", correr_etapa)
    monkeypatch.setattr(orquestador, "sonda_de_lotes", lambda *a: {})
    monkeypatch.setattr(orquestador, "_commit", lambda: "abc1234")
    monkeypatch.setattr(orquestador.carga, "leer_stats", lambda ruta: None)
    monkeypatch.setattr(orquestador.carga, "leer_historial", lambda ruta: {})
    monkeypatch.setattr(orquestador.carga, "resumen_etapa", resumen_etapa)
    class Muestreador:
        muestras, errores = [], []

        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(orquestador.recursos, "Muestreador", Muestreador)
    monkeypatch.setattr(orquestador.recursos, "en_ventana", lambda *a: [])
    monkeypatch.setattr(orquestador.recursos, "resumir", lambda muestras: {"api-1": api})
    monkeypatch.setattr(orquestador.recursos, "guardar_csv", lambda *a: None)
    for figura in ("grafica_desempeno", "grafica_capacidad", "grafica_endpoints"):
        monkeypatch.setattr(orquestador.graficas, figura, lambda *a: None)
    monkeypatch.setattr(orquestador.finops, "analizar", analizar_espia)

    codigo = orquestador.main(
        [
            "--contenedores",
            "api-1",
            "--etapas",
            "20,60",
            "--vcpu-medidos",
            "0.5",
            "--salida",
            str(tmp_path),
        ]
    )
    assert codigo == 0
    resumen = json.loads((tmp_path / "resumen.json").read_text(encoding="utf-8"))
    assert resumen["capacidad"]["usuarios"] == 20
    assert resumen["saturacion"]["usuarios"] == 60
    assert llamadas["saturacion"]["usuarios"] == 60
    assert llamadas["vcpu_medidos"] == 0.5


def test_sin_csv_de_locust_termina_con_codigo_2(orquestador, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(orquestador, "endpoints_disponibles", lambda host: ["p1"])
    monkeypatch.setattr(orquestador, "correr_etapa", lambda *a: (0.0, 1.0))
    codigo = orquestador.main(["--sin-recursos", "--etapas", "10,50", "--salida", str(tmp_path)])
    salida = capsys.readouterr().out
    assert codigo == 2
    assert salida.count("no dejó los CSV de Locust") == 2
    assert "Ninguna etapa dejó resultados" in salida
    assert not (tmp_path / "resumen.json").exists()
