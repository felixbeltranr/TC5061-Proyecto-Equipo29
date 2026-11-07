"""API de P3: asignación de grupos, validación de perfiles y carga independiente por modelo."""

import json

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from mibici.api.schemas.p1 import EJEMPLO_P1
from mibici.api.schemas.p3 import EJEMPLO_P3
from mibici.api.services import p3_inference
from mibici.api.services.model_registry import RegisteredModelInfo
from mibici.config import CONFIG_P3

INFO_P1 = RegisteredModelInfo(
    name="mibici-equipo29-p1-demanda",
    version="1",
    alias="champion",
    run_id="run-p1",
    model_uri="models:/mibici-equipo29-p1-demanda/1",
)


INFO_P3 = RegisteredModelInfo(
    name="mibici-equipo29-p3-grupos",
    version="1",
    alias="champion",
    run_id="run-p3",
    model_uri="models:/mibici-equipo29-p3-grupos/1",
)


class FakeModelP3:
    def __init__(self, grupos):
        self.grupos = grupos
        self.entrada = None

    def predict(self, frame):
        self.entrada = frame
        return self.grupos


class FakeModelP1:
    def predict(self, frame):
        return [7.25] * len(frame)


def _falla():
    raise RuntimeError("Registry no disponible")


@pytest.fixture
def api_main(monkeypatch):
    import mibici.api.main as api_main

    monkeypatch.setattr(api_main, "load_p1_model_with_info", lambda: (FakeModelP1(), INFO_P1))
    return api_main


@pytest.fixture
def cliente(api_main):
    with TestClient(api_main.app) as client:
        yield client


def _peticion(*perfiles, ids=None):
    ids = ids or [None] * len(perfiles)
    return {"stations": [{"station_id": i, "features": p} for i, p in zip(ids, perfiles)]}


# --- asignación -----------------------------------------------------------------------------


def test_modelo_por_omision_es_el_campeon_registrado_de_p3():
    assert p3_inference.DEFAULT_MODEL_URI == "models:/mibici-equipo29-p3-grupos@champion"


def test_asigna_con_las_12_variables_en_el_orden_de_la_firma():
    modelo = FakeModelP3([2])

    grupos = p3_inference.assign_groups(modelo, [dict(reversed(EJEMPLO_P3.items()))])

    assert grupos == [2]
    assert modelo.entrada.columns.tolist() == list(CONFIG_P3.prioridad_variables)
    assert modelo.entrada.dtypes.tolist() == [np.dtype("float64")] * 12


@pytest.mark.parametrize(
    "salida", [[float("nan")], [-1], [1.5], [0, 1]], ids=["nan", "negativo", "fraccion", "largo"]
)
def test_rechaza_salidas_invalidas_del_modelo(salida):
    with pytest.raises(ValueError, match="grupos inválidos"):
        p3_inference.assign_groups(FakeModelP3(salida), [EJEMPLO_P3])


def test_nombres_salen_del_run_de_la_version_cargada(tmp_path, monkeypatch):
    tabla = tmp_path / "grupos_interpretabilidad.csv"
    pd.DataFrame({"grupo": [0, 1], "nombre": ["a", "b"], "n_estaciones": [3, 4]}).to_csv(
        tabla, index=False
    )
    pedidos = []

    def descargar(run_id, artifact_path):
        pedidos.append((run_id, artifact_path))
        return str(tabla)

    monkeypatch.setattr(p3_inference.mlflow.artifacts, "download_artifacts", descargar)

    assert p3_inference.load_p3_group_names(INFO_P3) == {0: "a", 1: "b"}
    assert pedidos == [("run-p3", "data/grupos_interpretabilidad.csv")]


def test_sin_tabla_de_nombres_la_asignacion_sigue(monkeypatch):
    def descargar(**_):
        raise OSError("sin artefacto")

    monkeypatch.setattr(p3_inference.mlflow.artifacts, "download_artifacts", descargar)
    sin_run = RegisteredModelInfo("m", "1", None, None, "models:/m/1")

    assert p3_inference.load_p3_group_names(INFO_P3) == {}
    assert p3_inference.load_p3_group_names(sin_run) == {}


# --- endpoint -------------------------------------------------------------------------------


def test_predict_devuelve_grupo_nombre_y_version(cliente, p3_falso):
    p3_falso.grupos = [1]

    respuesta = cliente.post("/api/v1/p3/predict", json=_peticion(EJEMPLO_P3, ids=["51"]))

    assert respuesta.status_code == 200
    assert respuesta.json() == {
        "assignments": [{"station_id": "51", "group": 1, "group_name": "se vacía en la mañana"}],
        "method": "mlflow_registry",
        "input_count": 1,
        "model_name": "mibici-equipo29-p3-grupos",
        "model_version": "1",
    }


def test_predict_por_lotes_conserva_el_orden(cliente, p3_falso):
    p3_falso.grupos = [2, 0, 1]

    respuesta = cliente.post(
        "/api/v1/p3/predict", json=_peticion(*[EJEMPLO_P3] * 3, ids=["c", "a", "b"])
    )

    asignaciones = respuesta.json()["assignments"]
    assert [(a["station_id"], a["group"]) for a in asignaciones] == [("c", 2), ("a", 0), ("b", 1)]
    assert respuesta.json()["input_count"] == 3


def test_grupo_sin_nombre_se_devuelve_nulo(cliente, p3_falso):
    p3_falso.grupos = [7]

    respuesta = cliente.post("/api/v1/p3/predict", json=_peticion(EJEMPLO_P3))

    assert respuesta.json()["assignments"][0]["group_name"] is None


def test_salida_invalida_del_modelo_responde_500_con_detalle(cliente, p3_falso):
    p3_falso.grupos = [-1]

    respuesta = cliente.post("/api/v1/p3/predict", json=_peticion(EJEMPLO_P3))

    assert respuesta.status_code == 500
    assert respuesta.json() == {"detail": "El modelo produjo grupos inválidos"}


def test_model_reporta_la_version_de_p3(cliente):
    respuesta = cliente.get("/api/v1/p3/model")

    assert respuesta.status_code == 200
    assert respuesta.json()["model_uri"] == "models:/mibici-equipo29-p3-grupos/1"


# --- validación de la entrada ---------------------------------------------------------------


def _con(**cambios):
    return {**EJEMPLO_P3, **cambios}


@pytest.mark.parametrize(
    "cuerpo",
    [
        _peticion(_con(estacion=51)),
        _peticion({k: v for k, v in EJEMPLO_P3.items() if k != "net_pm"}),
        _peticion(_con(weekend_share=1.2)),
        _peticion(_con(dep_am=-0.1)),
        _peticion(_con(net_am=-1.5)),
        _peticion(_con(log_trips_day=-0.5)),
        _peticion(_con(dep_am=0.5, dep_mid=0.4, dep_pm=0.3)),
        _peticion(_con(arr_night=0.9)),
        _peticion(_con(dep_pm="0.24")),
        {"stations": []},
        {"stations": [{"features": EJEMPLO_P3}] * 1001},
        _peticion(EJEMPLO_P3, ids=["51; DROP"]),
        {"stations": [{"features": EJEMPLO_P3}], "extra": 1},
    ],
    ids=[
        "campo-extra",
        "variable-faltante",
        "proporcion-mayor-a-1",
        "proporcion-negativa",
        "net-fuera-de-rango",
        "log-negativo",
        "salidas-suman-mas-de-1",
        "llegadas-suman-mas-de-1",
        "numero-como-texto",
        "lista-vacia",
        "mas-de-1000",
        "id-con-caracteres-raros",
        "campo-extra-afuera",
    ],
)
def test_entradas_invalidas_responden_422(cliente, p3_falso, cuerpo):
    respuesta = cliente.post("/api/v1/p3/predict", json=cuerpo)

    assert respuesta.status_code == 422
    assert p3_falso.entrada is None  # el modelo no se llamó


def test_nan_responde_422(cliente):
    cuerpo = json.dumps(_peticion(_con(net_am=float("nan"))))
    assert '"net_am": NaN' in cuerpo

    respuesta = cliente.post(
        "/api/v1/p3/predict", content=cuerpo, headers={"content-type": "application/json"}
    )

    assert respuesta.status_code == 422


# --- carga independiente por modelo ---------------------------------------------------------


def test_sin_p3_la_api_sirve_p1_y_ready_dice_que_falta(api_main, monkeypatch):
    monkeypatch.setattr(api_main, "load_p3_model_with_info", _falla)

    with TestClient(api_main.app) as client:
        ready = client.get("/ready")
        p3 = client.post("/api/v1/p3/predict", json=_peticion(EJEMPLO_P3))
        p1 = client.post("/api/v1/predict", json={"features": EJEMPLO_P1})

    assert ready.status_code == 503
    assert "p3 (RuntimeError)" in ready.json()["detail"]
    assert p3.status_code == 503
    assert p1.status_code == 200


def test_sin_p1_la_api_sirve_p3(api_main, monkeypatch):
    monkeypatch.setattr(api_main, "load_p1_model_with_info", _falla)

    with TestClient(api_main.app) as client:
        ready = client.get("/ready")
        p1 = client.post("/api/v1/predict", json={"features": EJEMPLO_P1})
        p3 = client.post("/api/v1/p3/predict", json=_peticion(EJEMPLO_P3))

    assert ready.status_code == 503
    assert p1.status_code == 503
    assert p3.status_code == 200


def test_con_ambos_modelos_ready_responde_igual_que_antes(cliente):
    assert cliente.get("/ready").json() == {"status": "ready"}
