import os

import mlflow
from mlflow.pyfunc import PyFuncModel


def load_registered_model(uri_environment_variable: str, default_model_uri: str) -> PyFuncModel:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("Falta configurar MLFLOW_TRACKING_URI")

    model_uri = os.environ.get(uri_environment_variable, default_model_uri)
    if not model_uri:
        raise RuntimeError(f"Falta configurar {uri_environment_variable}")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    return mlflow.pyfunc.load_model(model_uri)