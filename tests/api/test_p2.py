"""API de P2: clasificación de riesgo, validación del panel y carga independiente por modelo."""

import json

from fastapi.testclient import TestClient
import numpy as np
import pytest

from mibici.api.schemas.p1 import EJEMPLO_P1
from mibici.api.schemas.p2 import EJEMPLO_P2, P2Features
from mibici.api.schemas.p3 import EJEMPLO_P3
from mibici.api.services import p2_inference
from mibici.api.services.model_registry import RegisteredModelInfo
from mibici.features_p2 import FEATURES

INFO_P1 = RegisteredModelInfo(
    name="mibici-equipo29-p1-demanda",
    version="1",
    alias="champion",
    run_id="run-p1",
    model_uri="models:/mibici-equipo29-p1-demanda/1",
)


class FakeModelP2:
    def __init__(self, clases):
        self.clases = clases
        self.entrada = None

    def predict(self, frame):
        self.entrada = frame
        return self.clases


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


def _peticion(*paneles, ids=None):
    ids = ids or [None] * len(paneles)
    return {"stations": [{"station_id": i, "features": p} for i, p in zip(ids, paneles)]}


# --- clasificación --------------------------------------------------------------------------


def test_esquema_sigue_el_orden_de_features():
    assert list(P2Features.model_fields) == FEATURES


def test_modelo_por_omision_es_el_campeon_registrado_de_p2():
    assert p2_inference.DEFAULT_MODEL_URI == "models:/mibici-equipo29-p2-riesgo@champion"


def test_clasifica_con_las_18_variables_en_el_orden_de_la_firma():
    modelo = FakeModelP2(["segura"])

    clases = p2_inference.classify_risk(modelo, [dict(reversed(list(EJEMPLO_P2.items())))])

    assert clases == ["segura"]
    assert modelo.entrada.columns.tolist() == FEATURES
    assert modelo.entrada.dtypes.tolist() == [np.dtype("float64")] * 18


def test_nulos_del_panel_llegan_como_nan():
    modelo = FakeModelP2(["segura"])
    fila = {**EJEMPLO_P2, "neto_lag24": None, "est_tasa_sin_bicis": None}

    p2_inference.classify_risk(modelo, [fila])

    assert modelo.entrada["neto_lag24"].isna().iloc[0]
    assert modelo.entrada["est_tasa_sin_bicis"].isna().iloc[0]


@pytest.mark.parametrize(
    "salida",
    [["desconocida"], [0], ["segura", "sin_bicicletas"]],
    ids=["etiqueta", "indice", "largo"],
)
def test_rechaza_salidas_invalidas_del_modelo(salida):
    with pytest.raises(ValueError, match="clases inválidas"):
        p2_inference.classify_risk(FakeModelP2(salida), [EJEMPLO_P2])


# --- endpoint -------------------------------------------------------------------------------


def test_predict_devuelve_clase_y_version(cliente, p2_falso):
    p2_falso.clases = ["sin_bicicletas"]

    respuesta = cliente.post("/api/v1/p2/predict", json=_peticion(EJEMPLO_P2, ids=["51"]))

    assert respuesta.status_code == 200
    assert respuesta.json() == {
        "predictions": [{"station_id": "51", "risk_class": "sin_bicicletas"}],
        "method": "mlflow_registry",
        "input_count": 1,
        "model_name": "mibici-equipo29-p2-riesgo",
        "model_version": "1",
    }


def test_predict_por_lotes_conserva_el_orden(cliente, p2_falso):
    p2_falso.clases = ["sin_anclajes", "segura", "sin_bicicletas"]

    respuesta = cliente.post(
        "/api/v1/p2/predict", json=_peticion(*[EJEMPLO_P2] * 3, ids=["c", "a", "b"])
    )

    predicciones = respuesta.json()["predictions"]
    assert [(p["station_id"], p["risk_class"]) for p in predicciones] == [
        ("c", "sin_anclajes"),
        ("a", "segura"),
        ("b", "sin_bicicletas"),
    ]
    assert respuesta.json()["input_count"] == 3


def test_salida_invalida_del_modelo_responde_500_con_detalle(cliente, p2_falso):
    p2_falso.clases = ["otra"]

    respuesta = cliente.post("/api/v1/p2/predict", json=_peticion(EJEMPLO_P2))

    assert respuesta.status_code == 500
    assert respuesta.json() == {"detail": "El modelo produjo clases inválidas"}


def test_model_reporta_la_version_de_p2(cliente):
    respuesta = cliente.get("/api/v1/p2/model")

    assert respuesta.status_code == 200
    assert respuesta.json()["model_uri"] == "models:/mibici-equipo29-p2-riesgo/1"


# --- validación de la entrada ---------------------------------------------------------------


def _con(**cambios):
    return {**EJEMPLO_P2, **cambios}


@pytest.mark.parametrize(
    "cuerpo",
    [
        _peticion(_con(estacion=51)),
        _peticion({k: v for k, v in EJEMPLO_P2.items() if k != "hora"}),
        _peticion(_con(hora=5)),
        _peticion(_con(hora=24)),
        _peticion(_con(dia_semana=7)),
        _peticion(_con(fin_semana=1)),
        _peticion(_con(dia_semana=6, fin_semana=0)),
        _peticion(_con(capacidad=0)),
        _peticion(_con(ocupacion_actual=1.2)),
        _peticion(_con(salidas_lag1=-1)),
        _peticion(_con(neto_lag1=0.0)),
        _peticion(_con(est_tasa_sin_bicis=0.7, est_tasa_sin_anclajes=0.4)),
        _peticion(_con(ocupacion_lag24=1.5)),
        _peticion(_con(capacidad="15")),
        {"stations": []},
        {"stations": [{"features": EJEMPLO_P2}] * 1001},
        _peticion(EJEMPLO_P2, ids=["51; DROP"]),
        {"stations": [{"features": EJEMPLO_P2}], "extra": 1},
        _peticion({k: v for k, v in EJEMPLO_P2.items() if k != "neto_lag24"}),
    ],
    ids=[
        "campo-extra",
        "variable-faltante",
        "hora-antes-de-6",
        "hora-despues-de-23",
        "dia-fuera-de-rango",
        "fin-semana-inconsistente-habil",
        "fin-semana-inconsistente-domingo",
        "capacidad-cero",
        "ocupacion-fuera-de-rango",
        "salidas-negativas",
        "neto-no-cuadra",
        "tasas-suman-mas-de-1",
        "ocupacion-lag-fuera-de-rango",
        "numero-como-texto",
        "lista-vacia",
        "mas-de-1000",
        "id-con-caracteres-raros",
        "campo-extra-afuera",
        "sin-neto-lag24",
    ],
)
def test_entradas_invalidas_responden_422(cliente, p2_falso, cuerpo):
    respuesta = cliente.post("/api/v1/p2/predict", json=cuerpo)

    assert respuesta.status_code == 422
    assert p2_falso.entrada is None


def test_nulos_permitidos_no_llaman_mal_al_validador(cliente, p2_falso):
    p2_falso.clases = ["segura"]
    fila = {
        **EJEMPLO_P2,
        "neto_lag24": None,
        "neto_lag168": None,
        "neto_media168_misma_hora": None,
        "ocupacion_lag24": None,
        "ocupacion_lag168": None,
        "est_tasa_sin_bicis": None,
        "est_tasa_sin_anclajes": None,
        "est_neto_medio_hora": None,
    }

    respuesta = cliente.post("/api/v1/p2/predict", json=_peticion(fila, ids=["51"]))

    assert respuesta.status_code == 200
    assert p2_falso.entrada["neto_lag24"].isna().iloc[0]
    assert p2_falso.entrada["est_tasa_sin_bicis"].isna().iloc[0]


def test_nan_responde_422(cliente):
    cuerpo = json.dumps(_peticion(_con(ocupacion_actual=float("nan"))))
    assert '"ocupacion_actual": NaN' in cuerpo

    respuesta = cliente.post(
        "/api/v1/p2/predict", content=cuerpo, headers={"content-type": "application/json"}
    )

    assert respuesta.status_code == 422


# --- carga independiente por modelo ---------------------------------------------------------


def test_sin_p2_la_api_sirve_p1_y_p3_y_ready_dice_que_falta(api_main, monkeypatch):
    monkeypatch.setattr(api_main, "load_p2_model_with_info", _falla)

    with TestClient(api_main.app) as client:
        ready = client.get("/ready")
        p2 = client.post("/api/v1/p2/predict", json=_peticion(EJEMPLO_P2))
        p1 = client.post("/api/v1/predict", json={"features": EJEMPLO_P1})
        p3 = client.post("/api/v1/p3/predict", json={"stations": [{"features": EJEMPLO_P3}]})

    assert ready.status_code == 503
    assert "p2 (RuntimeError)" in ready.json()["detail"]
    assert p2.status_code == 503
    assert p1.status_code == 200
    assert p3.status_code == 200


def test_sin_p1_la_api_sirve_p2(api_main, monkeypatch):
    monkeypatch.setattr(api_main, "load_p1_model_with_info", _falla)

    with TestClient(api_main.app) as client:
        ready = client.get("/ready")
        p1 = client.post("/api/v1/predict", json={"features": EJEMPLO_P1})
        p2 = client.post("/api/v1/p2/predict", json=_peticion(EJEMPLO_P2))

    assert ready.status_code == 503
    assert p1.status_code == 503
    assert p2.status_code == 200


def test_con_los_tres_modelos_ready_responde_ok(cliente):
    assert cliente.get("/ready").json() == {"status": "ready"}
