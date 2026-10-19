"""Convencion de MLflow del equipo 29: experimento, etiquetas y registro de runs.

Es la misma que usa mibici.modeling.train_p2 (experimento unico, etiquetas con pregunta,
equipo, autor, rango de datos, semilla, version de datos y commit) para que los runs de
P1, P2 y P3 se puedan comparar en la misma interfaz. Solo se registran metricas agregadas;
nunca columnas de usuario.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import mlflow

from mibici.config import CONFIG_P1, PROJ_ROOT

EXPERIMENTO = CONFIG_P1.experimento
EQUIPO = CONFIG_P1.equipo
TRACKING_URI = CONFIG_P1.tracking_uri
MANIFIESTO = CONFIG_P1.manifiesto_path


def git_commit() -> tuple[str, bool]:
    """Commit actual y si hay cambios sin guardar, para etiquetar el run."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, text=True
        ).strip()
        sucio = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJ_ROOT, text=True
            ).strip()
        )
        return sha, sucio
    except subprocess.CalledProcessError, FileNotFoundError:
        return "desconocido", True


def autor_git() -> str:
    """Nombre configurado en git, para la etiqueta author; 'desconocido' si no hay."""
    try:
        nombre = subprocess.check_output(["git", "config", "user.name"], text=True).strip()
    except subprocess.CalledProcessError, FileNotFoundError:
        return "desconocido"
    return nombre or "desconocido"


def version_datos(manifiesto: Path = MANIFIESTO) -> str:
    """Version de los datos: campo version del manifiesto (hash de los archivos de entrada)."""
    if not manifiesto.exists():
        raise FileNotFoundError(
            f"No existe el manifiesto de datos {manifiesto}. Se genera con python -m "
            "mibici.manifiesto_p2 (modulo de P2) o se puede indicar otro con --manifiesto"
        )
    return json.loads(manifiesto.read_text())["version"]


def configurar(
    tracking_uri: str = TRACKING_URI, carpeta_artefactos: Path = CONFIG_P1.artefactos_dir
) -> None:
    """Apunta MLflow al almacen indicado y deja activo el experimento del equipo.

    La carpeta de artefactos puede ser relativa; solo se usa si el experimento aun no existe.
    """
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(EXPERIMENTO) is None:
        mlflow.create_experiment(
            EXPERIMENTO, artifact_location=carpeta_artefactos.resolve().as_uri()
        )
    mlflow.set_experiment(EXPERIMENTO)


def etiquetas_base(
    pregunta: str, autor: str, semilla: int, rango_datos: str, data_version: str
) -> dict[str, str]:
    """Etiquetas comunes a todos los runs: pregunta, equipo, autor, datos y commit."""
    sha, sucio = git_commit()
    return {
        "question": pregunta,
        "team": EQUIPO,
        "author": autor,
        "data_range": rango_datos,
        "seed": str(semilla),
        "data_version": data_version,
        "mlflow.source.git.commit": sha,
        "git_commit": sha,
        "git_dirty": str(sucio),
    }


def registrar(
    nombre: str, tags: dict, params: dict, mets: dict[str, float], artefactos: list[Path]
) -> str:
    """Crea un run con nombre descriptivo, etiquetas, parametros, metricas y artefactos."""
    with mlflow.start_run(run_name=nombre) as run:
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        mlflow.log_metrics(mets)
        for a in artefactos:
            mlflow.log_artifact(str(a))
    return run.info.run_id
