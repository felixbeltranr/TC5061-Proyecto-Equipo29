"""Promocion segura del campeon de P1 en el Model Registry de MLflow.

La politica cientifica vive en :mod:`mibici.modeling.train_p1`: el modelo se elige
exclusivamente por ``val_mae``. Este modulo recibe esa decision ya tomada y se limita a
registrar el artefacto exacto de su run, validar una version inmutable y mover el alias
``champion`` solamente si la carga y la prediccion terminan correctamente.
"""

from __future__ import annotations

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd

from mibici.config import CONFIG_P1

CRITERIO_CAMPEON = (
    "menor MAE de validacion entre semana_previa y las candidatas con predicciones de "
    "validacion numericas, finitas y no negativas; "
    "persistencia se reporta pero no es elegible"
)


def registrar_campeon(
    run_id: str,
    model_uri: str,
    seleccionado: str,
    val_mae: float,
    linaje: dict[str, str],
    ejemplo: pd.DataFrame,
    nombre: str = CONFIG_P1.modelo_registrado,
    alias: str = CONFIG_P1.alias_campeon,
) -> tuple[mlflow.entities.model_registry.ModelVersion, list[float]]:
    """Registra, valida y solo entonces promueve el artefacto elegido en validacion.

    La version nueva queda en el Registry aunque falle la carga o la prediccion, para que el
    fallo sea auditable. En ese caso el alias anterior no se modifica.
    """
    cliente = MlflowClient()
    try:
        cliente.get_registered_model(nombre)
    except MlflowException:
        cliente.create_registered_model(
            nombre,
            tags={"question": "P1", "team": CONFIG_P1.equipo},
            description=(
                "Campeon de P1 (demanda diaria de salidas por estacion). El alias champion "
                "apunta a la version vigente validada. "
                f"Criterio: {CRITERIO_CAMPEON}."
            ),
        )

    run = cliente.get_run(run_id)
    etiquetas = {
        "run_id": run_id,
        "git_commit": run.data.tags.get("git_commit", ""),
        "git_dirty": run.data.tags.get("git_dirty", ""),
        "data_version": run.data.tags.get("data_version", ""),
        "criterio_seleccion": CRITERIO_CAMPEON,
        "candidata": seleccionado,
        **linaje,
    }
    version = mlflow.register_model(model_uri, nombre, tags=etiquetas)
    cliente.update_model_version(
        nombre,
        version.version,
        description=(
            f"Candidata {seleccionado}, elegida exclusivamente en validacion con MAE "
            f"{val_mae:.6f}. Criterio: {CRITERIO_CAMPEON}."
        ),
    )

    predicciones = verificar_version(nombre, version.version, ejemplo)
    cliente.set_registered_model_alias(nombre, alias, version.version)
    return cliente.get_model_version(nombre, version.version), predicciones


def verificar_version(nombre: str, version: str | int, ejemplo: pd.DataFrame) -> list[float]:
    """Carga una version inmutable y exige demanda numerica, finita, no negativa y fila a fila."""
    modelo = mlflow.pyfunc.load_model(f"models:/{nombre}/{version}")
    predicciones = np.asarray(modelo.predict(ejemplo), dtype=float).reshape(-1)
    if len(predicciones) != len(ejemplo):
        raise ValueError(
            "La prediccion de humo no conserva el numero de filas: "
            f"{len(predicciones)} != {len(ejemplo)}"
        )
    if not np.isfinite(predicciones).all():
        raise ValueError("La prediccion de humo contiene valores no finitos")
    if (predicciones < 0).any():
        raise ValueError("La prediccion de humo contiene demanda negativa")
    return predicciones.tolist()


def verificar_campeon(
    ejemplo: pd.DataFrame,
    nombre: str = CONFIG_P1.modelo_registrado,
    alias: str = CONFIG_P1.alias_campeon,
) -> list[float]:
    """Carga el alias vigente y valida una prediccion de demanda."""
    modelo = mlflow.pyfunc.load_model(f"models:/{nombre}@{alias}")
    predicciones = np.asarray(modelo.predict(ejemplo), dtype=float).reshape(-1)
    if (
        len(predicciones) != len(ejemplo)
        or not np.isfinite(predicciones).all()
        or (predicciones < 0).any()
    ):
        raise ValueError(
            "La prediccion del campeon no es numerica, finita, no negativa y fila a fila"
        )
    return predicciones.tolist()
