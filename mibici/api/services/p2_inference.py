"""Carga del campeón de P2 desde el Registry y clasificación de riesgo."""

from mlflow.pyfunc import PyFuncModel
import numpy as np
import pandas as pd

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
