"""Siembra del Registry y recorrido E2E de P1: verificación del almacén y comprobaciones.

No usa Docker: prueba las funciones puras de ``scripts/p1/sembrar_registry_p1.py`` y
``scripts/p1/e2e_api_p1.py`` con un almacén sintético que imita las tablas del Registry de MLflow.
"""

import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

import pytest

from mibici.api.schemas.p1 import EJEMPLO_P1
from mibici.api.services.p1_inference import DEFAULT_MODEL_URI

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/p1"))
import e2e_api_p1 as e2e
import sembrar_registry_p1 as siembra

NOMBRE = "mibici-equipo29-p1-demanda"
UBICACION = "mlflow-artifacts:/1/models/m-abc/artifacts"
SKOPS = b"modelo de prueba"


def crear_store(
    raiz: Path, *, version_alias=1, run_id="run-1", ubicacion=UBICACION, estado="READY"
):
    raiz.mkdir()
    con = sqlite3.connect(raiz / "mlflow.db")
    con.executescript(
        """
        CREATE TABLE registered_models (name TEXT);
        CREATE TABLE registered_model_aliases (alias TEXT, version INT, name TEXT, workspace TEXT);
        CREATE TABLE model_versions (
            name TEXT, version INT, run_id TEXT, storage_location TEXT, status TEXT
        );
        CREATE TABLE alembic_version (version_num TEXT);
        """
    )
    con.execute("INSERT INTO registered_models VALUES (?)", (NOMBRE,))
    con.execute(
        "INSERT INTO registered_model_aliases VALUES ('champion', ?, ?, 'default')",
        (version_alias, NOMBRE),
    )
    con.execute(
        "INSERT INTO model_versions VALUES (?, 1, ?, ?, ?)", (NOMBRE, run_id, ubicacion, estado)
    )
    con.execute("INSERT INTO alembic_version VALUES ('b7e2c1a4d9f3')")
    con.commit()
    con.close()
    artefactos = raiz / "mlartifacts/1/models/m-abc/artifacts"
    artefactos.mkdir(parents=True)
    (artefactos / "model.skops").write_bytes(SKOPS)
    (artefactos / "MLmodel").write_text("flavors: {}\n", encoding="utf-8")
    return raiz


@pytest.fixture
def modelo():
    return {
        "nombre": NOMBRE,
        "alias": "champion",
        "version": "1",
        "run_id": "run-1",
        "storage_location": UBICACION,
        "model_skops_sha256": hashlib.sha256(SKOPS).hexdigest(),
    }


def test_verificar_store_acepta_el_campeon_y_no_modifica_el_almacen(tmp_path, modelo):
    store = crear_store(tmp_path / "store")
    db = store / "mlflow.db"
    antes = (db.read_bytes(), db.stat().st_mtime_ns, sorted(p.name for p in store.iterdir()))

    resultado = siembra.verificar_store(store, modelo)

    assert resultado["version"] == "1"
    assert resultado["run_id"] == "run-1"
    assert resultado["archivos_artefactos"] == 2
    assert resultado["mlflow_db_sha256"] == hashlib.sha256(antes[0]).hexdigest()
    assert resultado["skops_en_contenedor"] == (
        "/mlflow/artifacts/1/models/m-abc/artifacts/model.skops"
    )
    assert (db.read_bytes(), db.stat().st_mtime_ns, sorted(p.name for p in store.iterdir())) == (
        antes
    )


@pytest.mark.parametrize(
    ("cambio", "mensaje"),
    [
        ({"version_alias": 2}, "versión 2"),
        ({"run_id": "otro-run"}, "run_id"),
        ({"ubicacion": "file:///C:/mlruns/1/models/m-abc/artifacts"}, "storage_location"),
        ({"estado": "FAILED_REGISTRATION"}, "status"),
    ],
)
def test_verificar_store_rechaza_otro_registro(tmp_path, modelo, cambio, mensaje):
    store = crear_store(tmp_path / "store", **cambio)
    with pytest.raises(siembra.ErrorSiembra, match=mensaje):
        siembra.verificar_store(store, modelo)


def test_verificar_store_rechaza_un_model_skops_distinto(tmp_path, modelo):
    store = crear_store(tmp_path / "store")
    (store / "mlartifacts/1/models/m-abc/artifacts/model.skops").write_bytes(b"otro")
    with pytest.raises(siembra.ErrorSiembra, match="SHA256"):
        siembra.verificar_store(store, modelo)


def test_verificar_store_rechaza_carpetas_incompletas(tmp_path, modelo):
    (tmp_path / "vacia").mkdir()
    with pytest.raises(siembra.ErrorSiembra, match="mlflow.db"):
        siembra.verificar_store(tmp_path / "vacia", modelo)
    (tmp_path / "vacia/mlflow.db").write_bytes(b"")
    with pytest.raises(siembra.ErrorSiembra, match="mlartifacts"):
        siembra.verificar_store(tmp_path / "vacia", modelo)


@pytest.mark.parametrize(
    "ubicacion", ["file:///C:/mlruns/1", "mlflow-artifacts:/", "mlflow-artifacts:/1/../../etc"]
)
def test_ruta_relativa_rechaza_ubicaciones_no_servidas(ubicacion):
    with pytest.raises(siembra.ErrorSiembra):
        siembra.ruta_relativa_artefactos(ubicacion)


def test_leer_huellas_del_contenedor():
    texto = "DB_SHA256 abc\nARCHIVOS 167\nruido\nSKOPS_SHA256 def\n"
    assert siembra.leer_huellas(texto) == {
        "DB_SHA256": "abc",
        "ARCHIVOS": "167",
        "SKOPS_SHA256": "def",
    }


def test_campeon_coincide_con_la_api():
    campeon = json.loads(
        (Path(siembra.__file__).with_name("campeon_api_p1.json")).read_text(encoding="utf-8")
    )
    assert campeon["peticion_real"]["cuerpo"]["features"] == EJEMPLO_P1
    nombre, _, alias = DEFAULT_MODEL_URI.removeprefix("models:/").partition("@")
    assert (campeon["modelo"]["nombre"], campeon["modelo"]["alias"]) == (nombre, alias)
    assert campeon["modelo"]["storage_location"].startswith("mlflow-artifacts:/")


@pytest.mark.parametrize(
    "texto",
    [
        '[{"Service": "api", "State": "exited", "ExitCode": 3}]',
        '{"Service": "api", "State": "exited", "ExitCode": 3}\n',
    ],
)
def test_parsear_ps_acepta_arreglo_y_lineas(texto):
    assert e2e.parsear_ps(texto) == [{"Service": "api", "State": "exited", "ExitCode": 3}]
    assert e2e.parsear_ps("") == []


@pytest.mark.parametrize(
    ("publicado", "url", "esperado"),
    [
        ("127.0.0.1:8000", "http://127.0.0.1:8000", True),
        ("127.0.0.1:8000", "http://localhost:8000", True),
        ("127.0.0.1:8001", "http://127.0.0.1:8000", False),
        ("", "http://127.0.0.1:8000", False),
    ],
)
def test_puerto_coincide(publicado, url, esperado):
    assert e2e.puerto_coincide(publicado, url) is esperado


def test_predicciones_iguales_con_tolerancia_relativa():
    assert e2e.predicciones_iguales(280.9855056086529, 280.9855056086529, 1e-9)
    assert not e2e.predicciones_iguales(280.99, 280.9855056086529, 1e-9)
    assert not e2e.predicciones_iguales(math.nan, 280.9855056086529, 1e-9)


def test_cuerpos_fuera_de_dominio_llevan_el_cambio_indicado():
    cuerpos = e2e.cuerpos_fuera_de_dominio(EJEMPLO_P1)
    assert '"lag_2": NaN' in cuerpos["lag_2 = NaN"]
    assert json.loads(cuerpos["festivo = 7"])["features"]["festivo"] == 7
    assert "lag_28" not in json.loads(cuerpos["falta lag_28"])["features"]
    assert json.loads(cuerpos["campo extra fuera de features"])["modelo"] == "p2"


def versiones(instaladas, python="3.14.10"):
    return {
        "python": python,
        "python_modelo": "3.14.8",
        "requisitos_modelo": ["mlflow==3.16.1", "scikit-learn==1.9.1"],
        "instaladas": instaladas,
    }


def test_comparar_versiones_pasa_si_coinciden():
    checks = e2e.comparar_versiones(versiones({"mlflow": "3.16.1", "scikit-learn": "1.9.1"}))
    assert [c.ok for c in checks] == [True, True, True]


def test_comparar_versiones_falla_con_otra_version_o_python():
    checks = e2e.comparar_versiones(
        versiones({"mlflow": "3.16.1", "scikit-learn": "1.8.0"}, python="3.13.5")
    )
    assert {c.nombre: c.ok for c in checks} == {
        "dependencia mlflow": True,
        "dependencia scikit-learn": False,
        "python mayor.menor": False,
    }


def test_comparar_versiones_falla_si_el_modelo_no_fija_version():
    datos = versiones({"mlflow": "3.16.1"})
    datos["requisitos_modelo"] = ["mlflow"]
    checks = e2e.comparar_versiones(datos)
    assert checks[0].ok is False


def test_argumentos_run_monta_el_almacen_solo_lectura_con_v(tmp_path):
    store = tmp_path / "almacén con espacios"
    store.mkdir()

    args = siembra.argumentos_run(store, "/mlflow/artifacts/1/models/m-abc/artifacts/model.skops")

    assert "--mount" not in args  # docker compose run no lo acepta
    montaje = args[args.index("-v") + 1]
    assert montaje == f"{store.resolve()}:/seed:ro"
    assert args[args.index("--user") + 1] == "root"
    assert args[-3:] == ["mlflow", "-c", siembra.COPIA_EN_CONTENEDOR]


def test_argumentos_run_rechaza_dos_puntos_fuera_de_la_unidad(tmp_path, monkeypatch):
    monkeypatch.setattr(siembra.Path, "resolve", lambda self: self)
    with pytest.raises(siembra.ErrorSiembra, match="':'"):
        siembra.argumentos_run(Path("/datos/a:b"), "/x")
