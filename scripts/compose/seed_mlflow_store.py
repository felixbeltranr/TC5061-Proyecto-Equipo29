"""Initialize empty Compose MLflow volumes from a verified local model-store copy."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import sys

MODEL_STORE_DIR = Path(os.environ.get("MODEL_STORE_DIR", "/seed"))
BACKEND_DIR = Path("/mlflow/backend")
ARTIFACTS_DIR = Path("/mlflow/artifacts")
P1_MANIFEST = Path(__file__).resolve().parents[1] / "p1" / "campeon_api_p1.json"
ARTIFACT_URI_PREFIX = "mlflow-artifacts:/"
REQUIRED_MODELS = (
    ("mibici-equipo29-p1-demanda", "champion"),
    ("mibici-equipo29-p2-riesgo", "champion"),
    ("mibici-equipo29-p3-grupos", "champion"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_path(storage_location: str, artifact_root: Path) -> Path:
    if not storage_location.startswith(ARTIFACT_URI_PREFIX):
        raise ValueError(f"El artefacto no está servido por MLflow: {storage_location}")

    relative = PurePosixPath(storage_location[len(ARTIFACT_URI_PREFIX) :].lstrip("/"))
    if not relative.parts or ".." in relative.parts:
        raise ValueError(f"Ruta de artefacto inválida: {storage_location}")

    root = artifact_root.resolve(strict=True)
    resolved = root.joinpath(*relative.parts).resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"El artefacto sale del almacén: {storage_location}")
    return resolved


def validate_store(database: Path, artifact_root: Path, p1_manifest: Path) -> None:
    if not database.is_file():
        raise ValueError(f"Falta {database}")
    if not artifact_root.is_dir():
        raise ValueError(f"Falta {artifact_root}")

    expected_p1 = json.loads(p1_manifest.read_text(encoding="utf-8"))["modelo"]
    database_uri = f"{database.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True)
    try:
        for name, alias in REQUIRED_MODELS:
            row = connection.execute(
                "SELECT version FROM registered_model_aliases WHERE name = ? AND alias = ?",
                (name, alias),
            ).fetchone()
            if row is None:
                raise ValueError(f"Falta el modelo {name}@{alias}")

            version = str(row[0])
            model = connection.execute(
                "SELECT run_id, storage_location, status FROM model_versions "
                "WHERE name = ? AND version = ?",
                (name, int(version)),
            ).fetchone()
            if model is None or model[2] != "READY":
                raise ValueError(f"{name}@{alias} no apunta a una versión READY")

            run_id, storage_location, _ = model
            directory = artifact_path(storage_location, artifact_root)
            if not (directory / "MLmodel").is_file():
                raise ValueError(f"Falta MLmodel para {name}@{alias}: {directory}")

            if name == expected_p1["nombre"]:
                if (
                    alias != expected_p1["alias"]
                    or version != expected_p1["version"]
                    or run_id != expected_p1["run_id"]
                    or storage_location != expected_p1["storage_location"]
                ):
                    raise ValueError("El campeón P1 no coincide con campeon_api_p1.json")
                model_file = directory / "model.skops"
                if not model_file.is_file():
                    raise ValueError(f"Falta el modelo serializado de P1: {model_file}")
                if sha256(model_file) != expected_p1["model_skops_sha256"]:
                    raise ValueError("El SHA256 de model.skops no coincide con el manifiesto P1")
    finally:
        connection.close()


def _is_empty(backend: Path, artifacts: Path) -> bool:
    return not any(backend.iterdir()) and not any(artifacts.iterdir())


def seed() -> None:
    source_database = MODEL_STORE_DIR / "mlflow.db"
    source_artifacts = MODEL_STORE_DIR / "mlartifacts"
    BACKEND_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not _is_empty(BACKEND_DIR, ARTIFACTS_DIR):
        validate_store(BACKEND_DIR / "mlflow.db", ARTIFACTS_DIR, P1_MANIFEST)
        print("El Registry existente es válido; no se modificaron sus volúmenes.")
        return

    validate_store(source_database, source_artifacts, P1_MANIFEST)
    shutil.copytree(source_artifacts, ARTIFACTS_DIR, dirs_exist_ok=True)
    temporary_database = BACKEND_DIR / ".mlflow.db.seed"
    shutil.copy2(source_database, temporary_database)
    temporary_database.replace(BACKEND_DIR / "mlflow.db")

    validate_store(BACKEND_DIR / "mlflow.db", ARTIFACTS_DIR, P1_MANIFEST)
    print("Registry P1/P2/P3 inicializado y verificado.")


def main() -> int:
    try:
        seed()
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as error:
        print(f"No se pudo inicializar el Registry: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
