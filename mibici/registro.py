"""Seleccion del campeon de P2 y su registro en el Model Registry de MLflow.

Regla de seleccion (CRITERIO_CAMPEON): entre las candidatas balanceadas se elige la de mayor
recall medio de las clases de riesgo (sin_bicicletas y sin_anclajes) en validacion, en el nivel
inicial de referencia; si dos candidatas difieren menos de EMPATE_RECALL en ese recall se
consideran empatadas y decide el PR AUC medio de las mismas clases. No se elige con prueba y
tampoco con macro F1.

Supuesto operativo: para quien reabastece las estaciones, no avisar un riesgo cuesta mas que una
falsa alarma, por eso se prioriza el recall de riesgo. El macro F1 no respalda esta eleccion: la
baseline de persistencia lo supera (ver README). Si el supuesto cambia, cambia el campeon.

Se registra solo el campeon, en un modelo con un unico nombre (una pregunta, un modelo), y el
alias champion apunta a su version vigente. Cargarlo:
    mlflow.pyfunc.load_model("models:/mibici-equipo29-p2-riesgo@champion")
"""

from collections.abc import Sequence

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
import pandas as pd

from mibici.features_p2 import CLASES

RIESGO = ("sin_bicicletas", "sin_anclajes")
EMPATE_RECALL = 0.005  # diferencia de recall medio por debajo de la cual se considera empate
CRITERIO_CAMPEON = (
    "mayor recall medio de sin_bicicletas y sin_anclajes en validacion al nivel inicial de "
    f"referencia; diferencias menores a {EMPATE_RECALL} se deciden por PR AUC medio"
)


def seleccionar_campeon(
    resultados: dict[str, dict[str, float]],
    candidatas: Sequence[str],
    conjunto: str = "val",
    empate: float = EMPATE_RECALL,
) -> dict:
    """Elige la candidata de mayor recall medio de riesgo, con PR AUC medio como desempate.

    resultados mapea el nombre de cada candidata a sus metricas (como las de train_p2) y solo se
    consideran las claves de candidatas. conjunto es el prefijo de las metricas, val por omision.
    Regresa la clave elegida, sus promedios y el ranking completo, de mejor a peor.
    """
    filas = []
    for clave in candidatas:
        if clave not in resultados:
            continue
        m = resultados[clave]
        try:
            recall = sum(m[f"{conjunto}_recall_{c}"] for c in RIESGO) / len(RIESGO)
            pr_auc = sum(m[f"{conjunto}_pr_auc_{c}"] for c in RIESGO) / len(RIESGO)
        except KeyError as exc:
            raise ValueError(f"Falta la metrica {exc.args[0]} en la candidata {clave}") from exc
        filas.append({"clave": clave, "recall_medio": recall, "pr_auc_medio": pr_auc})
    if not filas:
        raise ValueError(f"Hay resultados sin candidatas entre {list(candidatas)}")
    mejor_recall = max(f["recall_medio"] for f in filas)
    # empatan con el mejor las que quedan a menos de `empate` (con 0 solo las de recall igual);
    # entre ellas gana el mayor PR AUC
    empatadas = [
        f
        for f in filas
        if f["recall_medio"] == mejor_recall or mejor_recall - f["recall_medio"] < empate
    ]
    elegida = max(empatadas, key=lambda f: f["pr_auc_medio"])
    ranking = [elegida] + sorted(
        (f for f in filas if f is not elegida),
        key=lambda f: (f["recall_medio"], f["pr_auc_medio"]),
        reverse=True,
    )
    return {**elegida, "conjunto": conjunto, "empate": empate, "ranking": ranking}


def registrar_campeon(
    nombre: str,
    alias: str,
    run_id: str,
    model_uri: str,
    campeon: dict,
    linaje: dict[str, str],
    ejemplo: pd.DataFrame,
) -> mlflow.entities.model_registry.ModelVersion:
    """Registra una version y comprueba carga/prediccion antes de mover el alias.

    El modelo registrado se crea una sola vez, con descripcion y etiquetas de la pregunta; cada
    version lleva su descripcion y el linaje: run de origen, commit y version de los datos (del
    run), el criterio de seleccion y lo que se pase en linaje (experimento, nivel inicial).
    """
    cliente = MlflowClient()
    try:
        cliente.get_registered_model(nombre)
    except MlflowException:
        cliente.create_registered_model(
            nombre,
            tags={"question": "P2", "team": "29"},
            description=(
                "Campeon de P2 (riesgo de quedarse sin bicicletas o sin anclajes por estacion y "
                "hora). Un solo modelo para la pregunta; el alias champion apunta a la version "
                f"vigente. Criterio: {CRITERIO_CAMPEON}."
            ),
        )
    run = cliente.get_run(run_id)
    etiquetas = {
        "run_id": run_id,
        "git_commit": run.data.tags.get("git_commit", ""),
        "data_version": run.data.tags.get("data_version", ""),
        "criterio_seleccion": CRITERIO_CAMPEON,
        "candidata": campeon["clave"],
        **linaje,
    }
    version = mlflow.register_model(model_uri, nombre, tags=etiquetas)
    cliente.update_model_version(
        nombre,
        version.version,
        description=(
            f"Candidata {campeon['clave']}. En {campeon['conjunto']}: recall medio de riesgo "
            f"{campeon['recall_medio']:.4f} y PR AUC medio de riesgo {campeon['pr_auc_medio']:.4f}. "
            f"Criterio: {CRITERIO_CAMPEON}."
        ),
    )
    try:
        comprobar_version(nombre, version.version, ejemplo)
    except Exception:
        cliente.set_model_version_tag(nombre, version.version, "comprobacion", "fallida")
        raise
    cliente.set_model_version_tag(nombre, version.version, "comprobacion", "ok")
    cliente.set_registered_model_alias(nombre, alias, version.version)
    return cliente.get_model_version(nombre, version.version)


def comprobar_version(nombre: str, version: str, ejemplo: pd.DataFrame) -> list[str]:
    """Carga una version inmutable y exige una clase valida por fila."""
    modelo = mlflow.pyfunc.load_model(f"models:/{nombre}/{version}")
    clases = list(modelo.predict(ejemplo))
    if len(clases) != len(ejemplo) or any(clase not in CLASES for clase in clases):
        raise ValueError(f"La version {version} produjo clases invalidas o un numero incorrecto")
    return clases


def verificar_campeon(nombre: str, alias: str, ejemplo: pd.DataFrame) -> list[str]:
    """Carga el modelo por alias desde el Registry y predice el ejemplo (regresa las clases)."""
    modelo = mlflow.pyfunc.load_model(f"models:/{nombre}@{alias}")
    return [str(c) for c in modelo.predict(ejemplo)]
