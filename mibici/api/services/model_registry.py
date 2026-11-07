from dataclasses import dataclass
import os

import mlflow
from mlflow import MlflowClient
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


@dataclass(frozen=True)
class RegisteredModelInfo:
    """Versión del Registry que quedó cargada; un alias se resuelve a su versión fija."""

    name: str
    version: str
    alias: str | None
    run_id: str | None
    model_uri: str


def parse_model_uri(model_uri: str) -> tuple[str, str | None, str | None]:
    """Separa ``models:/<nombre>@<alias>`` o ``models:/<nombre>/<versión>`` en sus partes."""
    if not model_uri.startswith("models:/"):
        raise ValueError(f"Se esperaba una URI del Registry (models:/...): {model_uri}")
    resto = model_uri.removeprefix("models:/")
    if "@" in resto:
        nombre, alias = resto.split("@", 1)
        if nombre and alias:
            return nombre, alias, None
    else:
        nombre, _, version = resto.rpartition("/")
        if nombre and version.isdigit():
            return nombre, None, version
    raise ValueError(f"URI del Registry sin nombre y alias o versión: {model_uri}")


def load_registered_model_with_info(
    uri_environment_variable: str, default_model_uri: str
) -> tuple[PyFuncModel, RegisteredModelInfo]:
    """Resuelve el alias a su versión y carga esa versión inmutable.

    El modelo servido y la versión que reporta la API coinciden aunque el alias se mueva
    después de arrancar el contenedor.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("Falta configurar MLFLOW_TRACKING_URI")

    model_uri = os.environ.get(uri_environment_variable, default_model_uri)
    if not model_uri:
        raise RuntimeError(f"Falta configurar {uri_environment_variable}")
    nombre, alias, version = parse_model_uri(model_uri)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    if alias is not None:
        model_version = client.get_model_version_by_alias(nombre, alias)
    else:
        model_version = client.get_model_version(nombre, version)

    immutable_uri = f"models:/{nombre}/{model_version.version}"
    model = mlflow.pyfunc.load_model(immutable_uri)
    info = RegisteredModelInfo(
        name=nombre,
        version=str(model_version.version),
        alias=alias,
        run_id=model_version.run_id,
        model_uri=immutable_uri,
    )
    return model, info
