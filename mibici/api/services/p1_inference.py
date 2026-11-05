from mlflow.pyfunc import PyFuncModel
import numpy as np
import pandas as pd

from mibici.api.services.model_registry import load_registered_model
from mibici.features import FEATURES

DEFAULT_MODEL_URI = "models:/mibici-equipo29-p1-demanda@champion"


def load_p1_model() -> PyFuncModel:
    return load_registered_model("P1_MODEL_URI", DEFAULT_MODEL_URI)


def predict_with_model(model: PyFuncModel, features: dict[str, int | float]) -> float:
    row = {name: features[name] for name in FEATURES}
    frame = pd.DataFrame([row], columns=FEATURES).astype("float64")

    predictions = np.asarray(model.predict(frame), dtype=float).reshape(-1)
    if len(predictions) != 1 or not np.isfinite(predictions[0]) or predictions[0] < 0:
        raise ValueError("El modelo produjo una predicción inválida")

    return float(predictions[0])
