"""Carga del campeón de P3 desde el Registry y asignación de grupos."""

import math

from loguru import logger
import mlflow
from mlflow.pyfunc import PyFuncModel
import numpy as np
import pandas as pd

from mibici.api.services.model_registry import (
    RegisteredModelInfo,
    load_registered_model_with_info,
)
from mibici.config import CONFIG_P3

DEFAULT_MODEL_URI = f"models:/{CONFIG_P3.modelo_registrado}@{CONFIG_P3.alias_campeon}"
VARIABLES = list(CONFIG_P3.prioridad_variables)
# Tabla que la selección de P3 deja en cada run candidato: un renglón por grupo con su nombre.
GRUPOS = "data/grupos_interpretabilidad.csv"


def load_p3_model_with_info() -> tuple[PyFuncModel, RegisteredModelInfo]:
    return load_registered_model_with_info("P3_MODEL_URI", DEFAULT_MODEL_URI)


def load_p3_group_names(info: RegisteredModelInfo) -> dict[int, str]:
    """Nombres de los grupos según el run que produjo la versión cargada.

    Los nombres salen del mismo run que el modelo, así que cambian junto con la versión. Si la
    tabla no se puede leer, la API sigue asignando grupos y devuelve el nombre como nulo.
    """
    if not info.run_id:
        return {}
    try:
        ruta = mlflow.artifacts.download_artifacts(run_id=info.run_id, artifact_path=GRUPOS)
        tabla = pd.read_csv(ruta)
        return {int(g): str(n) for g, n in zip(tabla["grupo"], tabla["nombre"], strict=True)}
    except Exception as exc:  # noqa: BLE001 - sin nombres la asignación sigue siendo válida
        logger.warning(f"No se leyeron los nombres de los grupos de P3 ({GRUPOS}): {exc}")
        return {}


def assign_groups(model: PyFuncModel, rows: list[dict[str, float]]) -> list[int]:
    """Grupo de cada perfil, en el mismo orden; las columnas van en el orden de la firma."""
    frame = pd.DataFrame([{name: row[name] for name in VARIABLES} for row in rows])
    frame = frame[VARIABLES].astype("float64")

    groups = np.asarray(model.predict(frame), dtype=float).reshape(-1)
    valido = len(groups) == len(rows) and all(
        math.isfinite(g) and g >= 0 and g == int(g) for g in groups
    )
    if not valido:
        raise ValueError("El modelo produjo grupos inválidos")
    return [int(g) for g in groups]
