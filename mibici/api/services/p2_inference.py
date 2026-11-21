"""Carga del campeón de P2 desde el Registry y clasificación de riesgo."""

from mlflow.pyfunc import PyFuncModel
import numpy as np
import pandas as pd

from mibici.api.schemas.p2 import P2Prediction, P2PredictionResponse
from mibici.api.services.model_registry import (
    RegisteredModelInfo,
    load_registered_model_with_info,
)
from mibici.config import CONFIG_P2
from mibici.features_p2 import CLASES, FEATURES

DEFAULT_MODEL_URI = f"models:/{CONFIG_P2.modelo_registrado}@{CONFIG_P2.alias_campeon}"
VARIABLES = list(FEATURES)


def load_p2_model_with_info() -> tuple[PyFuncModel, RegisteredModelInfo]:
    return load_registered_model_with_info("P2_MODEL_URI", DEFAULT_MODEL_URI)


def classify_risk(model: PyFuncModel, rows: list[dict]) -> list[str]:
    """Clase de cada estación, en el mismo orden; las columnas van en el orden de la firma."""
    frame = pd.DataFrame([{name: row[name] for name in VARIABLES} for row in rows])
    frame = frame[VARIABLES].astype("float64")

    clases = np.asarray(model.predict(frame)).reshape(-1)
    etiquetas = [str(clase) for clase in clases]
    if len(etiquetas) != len(rows) or any(clase not in CLASES for clase in etiquetas):
        raise ValueError("El modelo produjo clases inválidas")
    return etiquetas


def predict_response(
    model: PyFuncModel, info: RegisteredModelInfo | None, stations: list[dict]
) -> P2PredictionResponse:
    """Respuesta de P2 para estaciones con ``station_id`` y ``features``, en el mismo orden."""
    clases = classify_risk(model, [station["features"] for station in stations])
    return P2PredictionResponse(
        predictions=[
            P2Prediction(station_id=station["station_id"], risk_class=clase)
            for station, clase in zip(stations, clases, strict=True)
        ],
        method="mlflow_registry",
        input_count=len(clases),
        model_name=info.name if info else None,
        model_version=info.version if info else None,
    )
