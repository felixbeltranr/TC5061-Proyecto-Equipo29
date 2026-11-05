from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from starlette.requests import Request

from mibici.api.routes.health import liveness, readiness
from mibici.api.routes.p1 import predict
from mibici.api.schemas.p1 import P1Features, PredictionRequest
from mibici.api.services.model_registry import load_registered_model
from mibici.api.services.p1_inference import predict_with_model
from mibici.features import FEATURES


def _features() -> dict[str, int | float]:
    return {name: 0 for name in FEATURES}


def _request(app: FastAPI) -> Request:
    return Request({"type": "http", "app": app, "headers": [], "method": "GET", "path": "/"})


class FakeModel:
    def __init__(self, prediction: float):
        self.prediction = prediction
        self.input: pd.DataFrame | None = None

    def predict(self, frame: pd.DataFrame) -> list[float]:
        self.input = frame
        return [self.prediction]


def test_p1_prediction_uses_expected_feature_order():
    model = FakeModel(12.5)

    prediction = predict_with_model(model, _features())

    assert prediction == 12.5
    assert model.input is not None
    assert model.input.columns.tolist() == FEATURES
    assert model.input.dtypes.tolist() == [np.dtype("float64")] * len(FEATURES)


def test_p1_route_returns_prediction_metadata():
    model = FakeModel(12.5)
    app = FastAPI()
    app.state.models = {"p1": model}

    response = predict(PredictionRequest(features=P1Features(**_features())), _request(app))

    assert response.model_dump() == {
        "prediction": 12.5,
        "method": "mlflow_registry",
        "input_count": 1,
    }


@pytest.mark.parametrize("prediction", [float("nan"), float("inf"), -1.0])
def test_p1_prediction_rejects_invalid_model_output(prediction: float):
    with pytest.raises(ValueError, match="predicción inválida"):
        predict_with_model(FakeModel(prediction), _features())


def test_prediction_request_requires_all_p1_features():
    with pytest.raises(ValidationError):
        PredictionRequest(features={})


def test_liveness_reports_alive():
    assert liveness().status == "alive"


def test_readiness_requires_p1_model():
    app = FastAPI()
    app.state.models = {"p1": FakeModel(0.0)}

    assert readiness(_request(app)).status == "ready"

    app.state.models = {}
    with pytest.raises(HTTPException) as exc_info:
        readiness(_request(app))
    assert exc_info.value.status_code == 503


def test_app_loads_p1_and_exposes_health_and_prediction_routes(monkeypatch):
    import mibici.api.main as api_main

    monkeypatch.setattr(api_main, "load_p1_model", lambda: FakeModel(8.5))

    with TestClient(api_main.app) as client:
        assert client.get("/live").json() == {"status": "alive"}
        assert client.get("/ready").json() == {"status": "ready"}
        response = client.post(
            "/api/v1/predict",
            json={"features": _features()},
        )

    assert response.status_code == 200
    assert response.json() == {
        "prediction": 8.5,
        "method": "mlflow_registry",
        "input_count": 1,
    }


def test_model_registry_loader_uses_configured_tracking_and_model_uris(monkeypatch):
    import mibici.api.services.model_registry as model_registry

    calls: list[str] = []
    model = object()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://host.containers.internal:5000")
    monkeypatch.setenv("P1_MODEL_URI", "models:/example/2")
    monkeypatch.setattr(model_registry.mlflow, "set_tracking_uri", calls.append)
    monkeypatch.setattr(model_registry.mlflow, "set_registry_uri", calls.append)
    monkeypatch.setattr(model_registry.mlflow.pyfunc, "load_model", lambda uri: model)

    result = load_registered_model("P1_MODEL_URI", "models:/example@champion")

    assert calls == [
        "http://host.containers.internal:5000",
        "http://host.containers.internal:5000",
    ]
    assert result is model
