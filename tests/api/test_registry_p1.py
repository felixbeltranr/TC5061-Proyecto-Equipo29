"""Resolución del alias a una versión fija, trazabilidad en la respuesta y validación de P1."""

from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from mibici.api.schemas.p1 import EJEMPLO_P1
from mibici.api.services.model_registry import (
    RegisteredModelInfo,
    load_registered_model_with_info,
    parse_model_uri,
)

INFO = RegisteredModelInfo(
    name="mibici-equipo29-p1-demanda",
    version="1",
    alias="champion",
    run_id="a8fea49c708d4e849bd92ec54536809f",
    model_uri="models:/mibici-equipo29-p1-demanda/1",
)


class FakeModel:
    def predict(self, frame):
        return [7.25] * len(frame)


@pytest.mark.parametrize(
    ("uri", "esperado"),
    [
        (
            "models:/mibici-equipo29-p1-demanda@champion",
            ("mibici-equipo29-p1-demanda", "champion", None),
        ),
        ("models:/mibici-equipo29-p1-demanda/3", ("mibici-equipo29-p1-demanda", None, "3")),
    ],
)
def test_parse_model_uri_separa_alias_y_version(uri, esperado):
    assert parse_model_uri(uri) == esperado


@pytest.mark.parametrize(
    "uri", ["runs:/abc/model", "models:/nombre", "models:/@champion", "models:/nombre/latest"]
)
def test_parse_model_uri_rechaza_uris_sin_alias_ni_version(uri):
    with pytest.raises(ValueError):
        parse_model_uri(uri)


class FakeClient:
    llamadas: list = []

    def __init__(self, tracking_uri, registry_uri):
        FakeClient.llamadas.append(("init", tracking_uri, registry_uri))

    def get_model_version_by_alias(self, name, alias):
        FakeClient.llamadas.append(("alias", name, alias))
        return SimpleNamespace(version="4", run_id="run-alias")

    def get_model_version(self, name, version):
        FakeClient.llamadas.append(("version", name, version))
        return SimpleNamespace(version=version, run_id="run-version")


@pytest.fixture
def registro_falso(monkeypatch):
    import mibici.api.services.model_registry as model_registry

    cargadas: list[str] = []
    FakeClient.llamadas = []
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setattr(model_registry, "MlflowClient", FakeClient)
    monkeypatch.setattr(model_registry.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(model_registry.mlflow, "set_registry_uri", lambda uri: None)
    monkeypatch.setattr(
        model_registry.mlflow.pyfunc, "load_model", lambda uri: cargadas.append(uri) or FakeModel()
    )
    return cargadas


def test_alias_se_resuelve_y_se_carga_la_version_inmutable(registro_falso, monkeypatch):
    monkeypatch.setenv("P1_MODEL_URI", "models:/mibici-equipo29-p1-demanda@champion")

    _, info = load_registered_model_with_info("P1_MODEL_URI", "models:/otro@champion")

    assert ("alias", "mibici-equipo29-p1-demanda", "champion") in FakeClient.llamadas
    assert registro_falso == ["models:/mibici-equipo29-p1-demanda/4"]
    assert info == RegisteredModelInfo(
        name="mibici-equipo29-p1-demanda",
        version="4",
        alias="champion",
        run_id="run-alias",
        model_uri="models:/mibici-equipo29-p1-demanda/4",
    )


def test_version_explicita_no_consulta_alias(registro_falso, monkeypatch):
    monkeypatch.setenv("P1_MODEL_URI", "models:/mibici-equipo29-p1-demanda/2")

    _, info = load_registered_model_with_info("P1_MODEL_URI", "models:/otro@champion")

    assert ("version", "mibici-equipo29-p1-demanda", "2") in FakeClient.llamadas
    assert not any(llamada[0] == "alias" for llamada in FakeClient.llamadas)
    assert registro_falso == ["models:/mibici-equipo29-p1-demanda/2"]
    assert (info.version, info.alias, info.run_id) == ("2", None, "run-version")


@pytest.fixture
def cliente(monkeypatch):
    import mibici.api.main as api_main

    monkeypatch.setattr(api_main, "load_p1_model_with_info", lambda: (FakeModel(), INFO))
    with TestClient(api_main.app) as client:
        yield client


def test_endpoint_model_reporta_la_version_cargada(cliente):
    respuesta = cliente.get("/api/v1/model")

    assert respuesta.status_code == 200
    assert respuesta.json() == {
        "name": "mibici-equipo29-p1-demanda",
        "version": "1",
        "alias": "champion",
        "run_id": "a8fea49c708d4e849bd92ec54536809f",
        "model_uri": "models:/mibici-equipo29-p1-demanda/1",
    }


def test_prediccion_reporta_nombre_y_version(cliente):
    respuesta = cliente.post("/api/v1/predict", json={"features": EJEMPLO_P1})

    assert respuesta.status_code == 200
    assert respuesta.json()["model_name"] == "mibici-equipo29-p1-demanda"
    assert respuesta.json()["model_version"] == "1"


@pytest.mark.parametrize(
    ("cambio", "campo"),
    [
        ({"lag_2": -1.0}, "lag_2"),
        ({"media_28d": -0.5}, "media_28d"),
        ({"festivo": 2}, "festivo"),
        ({"festivo_lag7": 0.5}, "festivo_lag7"),
        ({"dia_semana": 7}, "dia_semana"),
        ({"estacion": "51"}, "estacion"),
    ],
)
def test_prediccion_rechaza_entradas_fuera_de_dominio(cliente, cambio, campo):
    respuesta = cliente.post("/api/v1/predict", json={"features": {**EJEMPLO_P1, **cambio}})

    assert respuesta.status_code == 422
    assert any(campo in error["loc"] for error in respuesta.json()["detail"])


def test_prediccion_rechaza_valores_no_finitos(cliente):
    cuerpo = (
        '{"features": {"dia_semana": 3, "festivo": 0, "festivo_lag7": 0, "lag_2": NaN, '
        '"lag_7": 1, "lag_14": 1, "lag_21": 1, "lag_28": 1, "media_7d": 1, "media_28d": 1, '
        '"media_misma_dsem_4": 1}}'
    )

    respuesta = cliente.post(
        "/api/v1/predict", content=cuerpo, headers={"Content-Type": "application/json"}
    )

    assert respuesta.status_code == 422


def test_prediccion_rechaza_campos_extra_fuera_de_features(cliente):
    respuesta = cliente.post("/api/v1/predict", json={"features": EJEMPLO_P1, "modelo": "otro"})

    assert respuesta.status_code == 422


def test_model_sin_modelo_cargado_responde_503():
    from fastapi import FastAPI

    from mibici.api.routes import p1

    app = FastAPI()
    app.include_router(p1.router, prefix="/api/v1")

    assert TestClient(app).get("/api/v1/model").status_code == 503
