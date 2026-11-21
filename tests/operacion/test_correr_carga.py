"""Funciones del orquestador de la prueba de carga que no necesitan Docker ni Locust."""

import importlib.util
import json
import math
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
        "Capacidad sostenida con p95 de 500 ms o menos: 13.9 RPS de inferencias (15.5 en total "
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
    # Sin la opción, el límite se lee del contenedor de la API.
    assert orquestador.argumentos([]).vcpu_medidos is None
    assert orquestador.argumentos(["--vcpu-medidos", "0.5"]).vcpu_medidos == 0.5


def test_etapas_y_duracion_por_omision(orquestador):
    args = orquestador.argumentos([])
    assert args.etapas == "10,15,20,30,50"
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
    limites = {"api": {"cpus": 0.5, "memoria_mib": 1024.0}}
    monkeypatch.setattr(orquestador, "limites_de", lambda contenedores, motor: limites)
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
            "--salida",
            str(tmp_path),
        ]
    )
    assert codigo == 0
    resumen = json.loads((tmp_path / "resumen.json").read_text(encoding="utf-8"))
    assert resumen["capacidad"]["usuarios"] == 20
    assert resumen["saturacion"]["usuarios"] == 60
    assert llamadas["saturacion"]["usuarios"] == 60
    # El límite de CPU de la API sale del contenedor, no de un valor supuesto.
    assert llamadas["vcpu_medidos"] == 0.5
    assert resumen["limites"] == limites
    assert resumen["vcpu_medidos"] == 0.5
    assert llamadas["recursos_mlflow_sin_ajuste"] is None


def test_sin_csv_de_locust_termina_con_codigo_2(orquestador, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(orquestador, "endpoints_disponibles", lambda host: ["p1"])
    monkeypatch.setattr(orquestador, "primera_peticion", lambda host, endpoints: {})
    monkeypatch.setattr(orquestador, "correr_etapa", lambda *a: (0.0, 1.0))
    codigo = orquestador.main(["--sin-recursos", "--etapas", "10,50", "--salida", str(tmp_path)])
    salida = capsys.readouterr().out
    assert codigo == 2
    assert salida.count("no dejó los CSV de Locust") == 2
    assert "Ninguna etapa dejó resultados" in salida
    assert not (tmp_path / "resumen.json").exists()


def test_leer_limites(orquestador):
    con_limites = orquestador.leer_limites("1000000000 1073741824")
    assert con_limites == {"cpus": 1.0, "memoria_mib": 1024.0}
    assert orquestador.leer_limites("500000000 0") == {"cpus": 0.5, "memoria_mib": None}
    # Sin límites, o si inspect no respondió, no se inventa un valor.
    assert orquestador.leer_limites("0 0") == {"cpus": None, "memoria_mib": None}
    assert orquestador.leer_limites("") == {"cpus": None, "memoria_mib": None}


def test_limites_de_sin_motor_no_falla(orquestador):
    limites = orquestador.limites_de({"api": "api-1"}, "motor-que-no-existe")
    assert limites == {"api": {"cpus": None, "memoria_mib": None}}


def test_texto_limites(orquestador):
    limites = {
        "api": {"cpus": 1.0, "memoria_mib": 1024.0},
        "mlflow": {"cpus": None, "memoria_mib": None},
    }
    texto = orquestador.texto_limites(limites)
    assert texto == (
        "Límites durante la corrida: api con 1 vCPU y 1,024 MiB; "
        "mlflow con CPU sin límite y memoria sin límite."
    )
    assert orquestador.texto_limites({}) == ""


def test_sin_limite_de_cpu_avisa_y_toma_un_nucleo(orquestador, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(orquestador, "endpoints_disponibles", lambda host: ["p1"])
    monkeypatch.setattr(orquestador, "primera_peticion", lambda host, endpoints: {})
    monkeypatch.setattr(orquestador, "correr_etapa", lambda *a: (0.0, 1.0))
    sin_limites = {"api": {"cpus": None, "memoria_mib": None}}
    monkeypatch.setattr(orquestador, "limites_de", lambda contenedores, motor: sin_limites)

    class Muestreador:
        muestras, errores = [], []

        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(orquestador.recursos, "Muestreador", Muestreador)
    codigo = orquestador.main(
        ["--contenedores", "api-1", "--etapas", "10", "--salida", str(tmp_path)]
    )
    salida = capsys.readouterr().out
    assert codigo == 2  # la etapa simulada no deja CSV; el aviso sale antes
    assert "api con CPU sin límite y memoria sin límite" in salida
    assert "La API no tiene límite de CPU" in salida


def test_tabla_preguntas(orquestador):
    etapa = {
        "endpoints": [
            {
                "endpoint": "P1 POST /api/v1/predict",
                "peticiones": 38,
                "promedio_ms": 91.4,
                "p95_ms": 850.0,
                "p99_ms": 1000.0,
            },
            {
                "endpoint": "P2 POST /api/v1/p2/predict",
                "peticiones": 1860,
                "promedio_ms": 40.2,
                "p95_ms": 190.0,
                "p99_ms": 800.0,
            },
        ]
    }
    tabla = orquestador.tabla_preguntas(etapa)
    assert "| P1 POST /api/v1/predict | 38 | 91 | 850 | 1000 |" in tabla
    assert "| P2 POST /api/v1/p2/predict | 1,860 | 40 | 190 | 800 |" in tabla
    assert orquestador.tabla_preguntas({"endpoints": []}) == ""


def test_bloque_del_readme_con_limites_y_endpoints(orquestador):
    capacidad = {
        "usuarios": 15,
        "rps_sostenido": 11.4,
        "rps_inferencias": 10.9,
        "p95_ms": 190.0,
        "endpoints": [
            {
                "endpoint": "P1 POST /api/v1/predict",
                "peticiones": 38,
                "promedio_ms": 91.4,
                "p95_ms": 850.0,
                "p99_ms": 1000.0,
            }
        ],
    }
    etapa = dict(capacidad, p50_ms=25.0, p99_ms=840.0, tasa_error=0.0)
    resumen = {
        "fecha": "2026-11-20_2100",
        "commit": "abc1234",
        "duracion_s": 90,
        "endpoints": ["p1"],
        "p95_objetivo_ms": 500.0,
        "limites": {"api": {"cpus": 1.0, "memoria_mib": 1024.0}},
        "etapas": [etapa],
        "capacidad": capacidad,
        "saturacion": {},
        "primera_peticion": {},
    }
    bloque = orquestador.bloque_readme(resumen, "FINOPS", "reports/carga/x")
    assert "Límites durante la corrida: api con 1 vCPU y 1,024 MiB." in bloque
    assert "Por endpoint en la etapa de capacidad (15 usuarios)" in bloque
    assert "| P1 POST /api/v1/predict | 38 | 91 | 850 | 1000 |" in bloque
    assert bloque.index("Límites durante") < bloque.index("| Usuarios |")
    assert bloque.index("| Usuarios |") < bloque.index("Por endpoint") < bloque.index("FINOPS")


def test_reposo_escribe_el_consumo_sin_carga(orquestador, monkeypatch, tmp_path):
    medido = {"cpu_nucleos_promedio": 0.02, "cpu_nucleos_p95": 0.3, "memoria_mib_max": 1884.0}

    class Muestreador:
        muestras, errores = [], []

        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    esperas = []
    monkeypatch.setattr(orquestador.recursos, "Muestreador", Muestreador)
    monkeypatch.setattr(orquestador.recursos, "resumir", lambda muestras: {"x-mlflow-1": medido})
    monkeypatch.setattr(orquestador.recursos, "guardar_csv", lambda *a: None)
    monkeypatch.setattr(orquestador, "limites_de", lambda contenedores, motor: {})
    monkeypatch.setattr(orquestador.time, "sleep", esperas.append)
    monkeypatch.setattr(
        orquestador, "endpoints_disponibles", lambda host: pytest.fail("no debe tocar la API")
    )

    codigo = orquestador.main(
        ["--reposo", "60", "--contenedores", "x-mlflow-1", "--salida", str(tmp_path)]
    )
    assert codigo == 0
    assert esperas == [60]
    reposo = json.loads((tmp_path / "reposo.json").read_text(encoding="utf-8"))
    assert reposo["segundos"] == 60
    assert reposo["recursos"]["mlflow"] == medido
    # El archivo es el que después se pasa con --mlflow-sin-ajuste.
    assert orquestador.mlflow_sin_ajuste(tmp_path / "reposo.json") == medido
    assert orquestador.mlflow_sin_ajuste(None) is None


def test_limites_de_lee_inspect(orquestador, monkeypatch):
    comandos = []

    def run(comando, **_):
        comandos.append(comando)
        return orquestador.subprocess.CompletedProcess(comando, 0, stdout="500000000 1073741824")

    monkeypatch.setattr(orquestador.subprocess, "run", run)
    limites = orquestador.limites_de({"api": "x-api-1"}, "docker")
    assert limites == {"api": {"cpus": 0.5, "memoria_mib": 1024.0}}
    assert comandos == [
        [
            "docker",
            "inspect",
            "--format",
            "{{.HostConfig.NanoCpus}} {{.HostConfig.Memory}}",
            "x-api-1",
        ]
    ]


def test_limites_de_con_inspect_fallido_no_inventa(orquestador, monkeypatch):
    def run(comando, **_):
        return orquestador.subprocess.CompletedProcess(comando, 1, stdout="1000000000 1")

    monkeypatch.setattr(orquestador.subprocess, "run", run)
    limites = orquestador.limites_de({"api": "x-api-1"}, "docker")
    assert limites == {"api": {"cpus": None, "memoria_mib": None}}


@pytest.mark.parametrize(
    "nombre,servicio",
    [
        ("tc5061-carga-r2-api-1", "api"),
        ("tc5061-carga-r2-mlflow-1", "mlflow"),
        ("src_api_1", "api"),
        ("api", "api"),
        ("api-1", "api"),
        # El proyecto se llama «mibici-api»: su MLflow no es la API.
        ("mibici-api-mlflow-1", "mlflow"),
        ("mibici-api-api-1", "api"),
        ("rapido-mlflow-2", "mlflow"),
        ("otro-servicio-1", "otro-servicio-1"),
    ],
)
def test_servicio_de_decide_por_el_final_del_nombre(orquestador, nombre, servicio):
    assert orquestador.servicio_de(nombre) == servicio


def test_contenedores_de_un_proyecto_que_se_llama_como_un_servicio(orquestador):
    args = orquestador.argumentos(["--contenedores", "mibici-api-api-1, mibici-api-mlflow-1"])
    assert orquestador.contenedores_del_sistema(args) == {
        "api": "mibici-api-api-1",
        "mlflow": "mibici-api-mlflow-1",
    }


def test_tabla_de_etapas_sin_recursos_dice_nd(orquestador):
    etapa = {
        "usuarios": 10,
        "rps_sostenido": 7.8,
        "p50_ms": 19.0,
        "p95_ms": 110.0,
        "p99_ms": 390.0,
        "tasa_error": 0.0,
        "recursos": {},
    }
    tabla = orquestador.tabla_etapas([etapa, dict(etapa, recursos={"api": None})])
    assert tabla.count("| 10 | 7.8 | 19 | 110 | 390 | 0.00% | n/d | n/d |") == 2
    assert "nan" not in tabla


def _etapa(usuarios, rps, p95, recursos):
    return {
        "usuarios": usuarios,
        "rps_sostenido": rps,
        "rps_inferencias": rps * 0.95,
        "p50_ms": 20.0,
        "p95_ms": p95,
        "p99_ms": p95 * 2,
        "tasa_error": 0.0,
        "promedio_ms": 60.0,
        "endpoints": [],
        "recursos": recursos,
    }


def _resumen_guardado(recursos):
    capacidad = _etapa(15, 11.2, 260.0, recursos)
    saturacion = _etapa(50, 19.2, 2200.0, recursos)
    return {
        "fecha": "2026-11-20_2224",
        "commit": "abc1234",
        "duracion_s": 90,
        "endpoints": ["p1", "p2", "p3"],
        "p95_objetivo_ms": 500.0,
        "limites": {"api": {"cpus": 1.0, "memoria_mib": 1024.0}},
        "vcpu_medidos": 1.0,
        "mlflow_sin_ajuste": None,
        "etapas": [capacidad, saturacion],
        "capacidad": capacidad,
        "saturacion": saturacion,
        "lotes": {"estaciones": 468, "p2_ms": 52.0, "p3_ms": None},
        "primera_peticion": {},
    }


API = {"cpu_nucleos_promedio": 0.37, "cpu_nucleos_p95": 0.75, "memoria_mib_max": 261.0}


def test_sin_muestras_de_la_api_no_dice_que_falto_capacidad(orquestador, tmp_path):
    args = orquestador.argumentos([])
    # Un costo calculado antes no debe quedarse junto al texto que dice que no se calculó.
    (tmp_path / "finops.json").write_text("{}", encoding="utf-8")
    texto = orquestador.analisis_de_costo(_resumen_guardado({}), tmp_path, args)
    assert texto == orquestador.SIN_RECURSOS
    assert "docker stats sin datos" in texto and texto != orquestador.SIN_CAPACIDAD
    assert not (tmp_path / "finops.json").exists()
    assert "| 15 | 11.2 | 20 | 260 | 520 | 0.00% | n/d | n/d |" in (
        tmp_path / "resumen.md"
    ).read_text(encoding="utf-8")
    sin_etapa = dict(_resumen_guardado({}), capacidad={})
    (tmp_path / "finops.json").write_text("{}", encoding="utf-8")
    assert orquestador.analisis_de_costo(sin_etapa, tmp_path, args) == orquestador.SIN_CAPACIDAD
    assert not (tmp_path / "finops.json").exists()


def test_sin_recursos_no_avisa_de_limites_que_no_leyo(orquestador, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(orquestador, "endpoints_disponibles", lambda host: ["p1"])
    monkeypatch.setattr(orquestador, "primera_peticion", lambda host, endpoints: {})
    monkeypatch.setattr(orquestador, "correr_etapa", lambda *a: (0.0, 1.0))
    orquestador.main(["--sin-recursos", "--etapas", "10", "--salida", str(tmp_path)])
    assert "La API no tiene límite de CPU" not in capsys.readouterr().out


def test_recalcular_regenera_los_costos_sin_correr_la_carga(orquestador, monkeypatch, tmp_path):
    mlflow = {"cpu_nucleos_promedio": 0.04, "cpu_nucleos_p95": 0.3, "memoria_mib_max": 415.0}
    reposo = {"cpu_nucleos_promedio": 0.03, "cpu_nucleos_p95": 0.2, "memoria_mib_max": 1975.0}
    resumen = _resumen_guardado({"api": API, "mlflow": mlflow})
    resumen["mlflow_sin_ajuste"] = reposo
    resumen["estaciones"] = 368
    # Una medida que faltó se guarda como null y debe volver como dato faltante.
    resumen["etapas"][1] = dict(resumen["etapas"][1], p99_ms=None)
    orquestador._escribir_json(tmp_path / "resumen.json", resumen)
    for prohibida in ("endpoints_disponibles", "correr_etapa", "primera_peticion"):
        monkeypatch.setattr(
            orquestador, prohibida, lambda *a, nombre=prohibida: pytest.fail(f"llamó a {nombre}")
        )

    assert orquestador.main(["--recalcular", str(tmp_path)]) == 0
    finops_json = json.loads((tmp_path / "finops.json").read_text(encoding="utf-8"))
    claves = [fila["clave"] for fila in finops_json["escenarios"]]
    assert claves[:4] == ["piso", "siempre_encendido", "horario", "arm"]
    # La medición en reposo guardada en la corrida vuelve a costearse.
    assert claves[-2:] == ["mlflow", "mlflow_sin_ajuste"]
    assert finops_json["medido"]["usuarios"] == 15
    # Los parámetros son los de la corrida, no los valores por omisión de la línea de comandos.
    assert finops_json["supuestos"]["demanda"]["estaciones"] == 368
    assert "| 50 | 19.2 | 20 | 2200 | nan |" in (tmp_path / "resumen.md").read_text(
        encoding="utf-8"
    )
    texto = (tmp_path / "finops.md").read_text(encoding="utf-8")
    assert texto == orquestador.finops.a_markdown(
        orquestador.finops.analizar(
            resumen["capacidad"],
            API,
            lotes={"estaciones": 468, "p2_ms": 52.0, "p3_ms": float("nan")},
            recursos_mlflow=mlflow,
            demanda=orquestador.finops.Demanda(estaciones=368),
            saturacion=resumen["saturacion"],
            recursos_mlflow_sin_ajuste=reposo,
        )
    )
    assert "Apagar el ejecutor de trabajos de MLflow" in texto
    assert texto in (tmp_path / "resumen.md").read_text(encoding="utf-8")


def test_recalcular_sin_resumen_termina_con_codigo_2(orquestador, tmp_path, capsys):
    assert orquestador.main(["--recalcular", str(tmp_path)]) == 2
    assert "necesita la carpeta de una corrida" in capsys.readouterr().out


def test_actualizar_readme_fuera_del_repositorio_se_rechaza_antes_de_correr(
    orquestador, monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(
        orquestador, "endpoints_disponibles", lambda host: pytest.fail("corrió la carga")
    )
    codigo = orquestador.main(["--actualizar-readme", "--salida", str(tmp_path / "fuera")])
    assert codigo == 2
    assert "--actualizar-readme necesita que la salida esté dentro de" in capsys.readouterr().out
    assert not (tmp_path / "fuera").exists()


def test_con_nan_respeta_los_contenedores_sin_muestras(orquestador):
    leido = {"p95_ms": None, "recursos": {"api": None, "mlflow": {"cpu_nucleos_p95": None}}}
    vuelto = orquestador._con_nan(leido)
    assert math.isnan(vuelto["p95_ms"])
    assert vuelto["recursos"]["api"] is None
    assert math.isnan(vuelto["recursos"]["mlflow"]["cpu_nucleos_p95"])


def test_recalcular_con_actualizar_readme_reescribe_solo_el_bloque(
    orquestador, monkeypatch, tmp_path
):
    raiz = tmp_path.resolve()
    carpeta = raiz / "reports" / "carga" / "2026-11-20_2224"
    carpeta.mkdir(parents=True)
    orquestador._escribir_json(carpeta / "resumen.json", _resumen_guardado({"api": API}))
    readme = raiz / "README.md"
    original = "\n".join(
        ["antes", orquestador.INICIO_BLOQUE, "viejo", orquestador.FIN_BLOQUE, "después", ""]
    )
    readme.write_text(original, encoding="utf-8")
    monkeypatch.setattr(orquestador, "RAIZ", raiz)
    monkeypatch.setattr(orquestador, "README", readme)

    # Sin la opción, el README no se toca.
    assert orquestador.main(["--recalcular", str(carpeta)]) == 0
    assert readme.read_text(encoding="utf-8") == original

    assert orquestador.main(["--recalcular", str(carpeta), "--actualizar-readme"]) == 0
    texto = readme.read_text(encoding="utf-8")
    assert texto.startswith("antes\n") and texto.endswith("después\n")
    assert "viejo" not in texto
    assert "Corrida del 2026-11-20_2224 (commit `abc1234`)" in texto
    assert "![Capacidad](reports/carga/2026-11-20_2224/capacidad.png)" in texto
    assert (carpeta / "finops.md").read_text(encoding="utf-8").strip() in texto
