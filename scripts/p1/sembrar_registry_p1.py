"""Siembra el Registry del Compose con el campeón de P1, copiado de un almacén de MLflow.

Uso, desde cualquier carpeta (el script ubica la raíz del repositorio):

    python scripts/p1/sembrar_registry_p1.py --store <carpeta con mlflow.db y mlartifacts/>
    python scripts/p1/sembrar_registry_p1.py --store <carpeta> --solo-verificar

1. Verifica el almacén sin modificarlo (SQLite en sólo lectura): el alias, la versión, el run,
   la ubicación ``mlflow-artifacts:/`` y el SHA256 de ``model.skops`` deben coincidir con
   ``campeon_api_p1.json``.
2. Se detiene si los volúmenes del proyecto de Compose ya tienen base o artefactos: no
   sobrescribe un Registry.
3. Copia con el almacén montado sólo de lectura y comprueba en el volumen las mismas huellas.

No inicia MLflow ni la API y no borra contenedores ni volúmenes. Sólo usa la biblioteca
estándar; requiere Docker con Compose v2 para sembrar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sqlite3
import subprocess
import sys

RAIZ = Path(__file__).resolve().parents[2]
CAMPEON = Path(__file__).resolve().with_name("campeon_api_p1.json")
PREFIJO_ARTEFACTOS = "mlflow-artifacts:/"

# Código de salida cuando los volúmenes ya tienen datos.
VOLUMENES_OCUPADOS = 42

# Corre dentro del contenedor de mlflow como root, con el almacén en /seed (sólo lectura).
COPIA_EN_CONTENEDOR = r"""
set -eu
if [ -e /mlflow/backend/mlflow.db ] || [ -n "$(ls -A /mlflow/artifacts)" ]; then
  echo "VOLUMENES_OCUPADOS" >&2
  exit 42
fi
cp /seed/mlflow.db /mlflow/backend/mlflow.db
cp -R /seed/mlartifacts/. /mlflow/artifacts/
chown -R 10001:10001 /mlflow/backend /mlflow/artifacts
echo "DB_SHA256 $(sha256sum /mlflow/backend/mlflow.db | cut -d' ' -f1)"
echo "ARCHIVOS $(find /mlflow/artifacts -type f | wc -l)"
echo "SKOPS_SHA256 $(sha256sum "$SKOPS" | cut -d' ' -f1)"
"""


class ErrorSiembra(Exception):
    """El almacén o los volúmenes no cumplen lo esperado; no se copió nada."""


def cargar_campeon(ruta: Path = CAMPEON) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))["modelo"]


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def ruta_relativa_artefactos(storage_location: str) -> PurePosixPath:
    """``mlflow-artifacts:/1/models/m-x/artifacts`` -> ``1/models/m-x/artifacts``."""
    if not storage_location.startswith(PREFIJO_ARTEFACTOS):
        raise ErrorSiembra(
            f"Los artefactos no están en el servidor ({storage_location}); "
            "el contenedor no puede cargar rutas file:///"
        )
    relativa = PurePosixPath(storage_location.removeprefix(PREFIJO_ARTEFACTOS).lstrip("/"))
    if not relativa.parts or ".." in relativa.parts:
        raise ErrorSiembra(f"Ubicación de artefactos no válida: {storage_location}")
    return relativa


def leer_registry(db: Path, nombre: str, alias: str) -> dict:
    """Lee la versión a la que apunta el alias, sin escribir en la base ni crear archivos."""
    uri = f"{db.resolve().as_uri()}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    try:
        fila = con.execute(
            "SELECT version FROM registered_model_aliases WHERE name = ? AND alias = ?",
            (nombre, alias),
        ).fetchone()
        if fila is None:
            raise ErrorSiembra(f"El almacén no tiene {nombre}@{alias}")
        version = str(fila[0])
        fila = con.execute(
            "SELECT run_id, storage_location, status FROM model_versions "
            "WHERE name = ? AND version = ?",
            (nombre, int(version)),
        ).fetchone()
        if fila is None:
            raise ErrorSiembra(f"El alias apunta a la versión {version}, que no existe")
        esquema = con.execute("SELECT version_num FROM alembic_version").fetchone()
        modelos = con.execute("SELECT count(*) FROM registered_models").fetchone()[0]
    except sqlite3.DatabaseError as exc:
        raise ErrorSiembra(f"{db} no es una base de MLflow legible: {exc}") from exc
    finally:
        con.close()
    return {
        "version": version,
        "run_id": fila[0],
        "storage_location": fila[1],
        "status": fila[2],
        "esquema": esquema[0] if esquema else None,
        "modelos_registrados": modelos,
    }


def verificar_store(store: Path, modelo: dict) -> dict:
    """Comprueba que el almacén contiene el campeón esperado. No modifica nada."""
    db = store / "mlflow.db"
    artefactos = store / "mlartifacts"
    if not db.is_file():
        raise ErrorSiembra(f"Falta {db}")
    if not artefactos.is_dir():
        raise ErrorSiembra(f"Falta la carpeta {artefactos}")

    registro = leer_registry(db, modelo["nombre"], modelo["alias"])
    esperado = {
        "version": modelo["version"],
        "run_id": modelo["run_id"],
        "storage_location": modelo["storage_location"],
        "status": "READY",
    }
    for campo, valor in esperado.items():
        if registro[campo] != valor:
            raise ErrorSiembra(
                f"{modelo['nombre']}@{modelo['alias']}: {campo} es {registro[campo]!r}, "
                f"se esperaba {valor!r}"
            )

    relativa = ruta_relativa_artefactos(registro["storage_location"])
    skops = artefactos.joinpath(*relativa.parts, "model.skops")
    if not skops.is_file():
        raise ErrorSiembra(f"Falta el artefacto {skops}")
    huella = sha256(skops)
    if huella != modelo["model_skops_sha256"]:
        raise ErrorSiembra(
            f"model.skops tiene SHA256 {huella}, se esperaba {modelo['model_skops_sha256']}"
        )

    return {
        "store": str(store.resolve()),
        "mlflow_db_sha256": sha256(db),
        "archivos_artefactos": sum(1 for p in artefactos.rglob("*") if p.is_file()),
        "model_skops": str(skops.resolve()),
        "model_skops_sha256": huella,
        "skops_en_contenedor": str(PurePosixPath("/mlflow/artifacts", relativa, "model.skops")),
        **registro,
    }


def compose(*args: str, mostrar_stderr: bool = False) -> subprocess.CompletedProcess:
    """``docker compose`` desde la raíz del repositorio; stdout siempre se captura."""
    return subprocess.run(
        ["docker", "compose", *args],
        check=False,
        cwd=RAIZ,
        stdout=subprocess.PIPE,
        stderr=None if mostrar_stderr else subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def nombre_proyecto() -> str | None:
    salida = compose("config", "--format", "json")
    if salida.returncode != 0:
        return None
    try:
        return json.loads(salida.stdout).get("name")
    except json.JSONDecodeError:
        return None


def leer_huellas(texto: str) -> dict:
    huellas = {}
    for linea in texto.splitlines():
        clave, _, valor = linea.strip().partition(" ")
        if clave in {"DB_SHA256", "ARCHIVOS", "SKOPS_SHA256"}:
            huellas[clave] = valor.strip()
    return huellas


def argumentos_run(store: Path, skops_en_contenedor: str) -> list[str]:
    """``docker compose run`` sólo acepta ``-v``; el almacén se monta sólo de lectura."""
    origen = str(store.resolve())
    sin_unidad = origen[2:] if len(origen) > 1 and origen[1] == ":" else origen
    if ":" in sin_unidad:
        raise ErrorSiembra(f"La ruta del almacén no puede llevar ':' (docker -v): {origen}")
    return [
        "run",
        "--rm",
        "--no-deps",
        "--user",
        "root",
        "-v",
        f"{origen}:/seed:ro",
        "-e",
        f"SKOPS={skops_en_contenedor}",
        "--entrypoint",
        "sh",
        "mlflow",
        "-c",
        COPIA_EN_CONTENEDOR,
    ]


def sembrar(store: Path, verificado: dict) -> dict:
    salida = compose(
        *argumentos_run(store, verificado["skops_en_contenedor"]), mostrar_stderr=True
    )
    if salida.returncode == VOLUMENES_OCUPADOS:
        raise ErrorSiembra(
            "Los volúmenes de este proyecto de Compose ya tienen una base o artefactos; no se "
            "copió nada. Usa otro COMPOSE_PROJECT_NAME para sembrar volúmenes nuevos."
        )
    if salida.returncode != 0:
        raise ErrorSiembra(f"docker compose run terminó con código {salida.returncode}")

    huellas = leer_huellas(salida.stdout)
    esperadas = {
        "DB_SHA256": verificado["mlflow_db_sha256"],
        "ARCHIVOS": str(verificado["archivos_artefactos"]),
        "SKOPS_SHA256": verificado["model_skops_sha256"],
    }
    if huellas != esperadas:
        raise ErrorSiembra(f"La copia no coincide: volumen {huellas}, almacén {esperadas}")
    return huellas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", required=True, type=Path, help="mlflow.db y mlartifacts/")
    parser.add_argument(
        "--solo-verificar", action="store_true", help="verifica el almacén y no copia"
    )
    args = parser.parse_args(argv)

    modelo = cargar_campeon()
    try:
        verificado = verificar_store(args.store, modelo)
        print(f"Almacén verificado: {modelo['nombre']} v{verificado['version']}")
        print(f"  run {verificado['run_id']}, model.skops {verificado['model_skops_sha256']}")
        print(f"  mlflow.db {verificado['mlflow_db_sha256']}")
        print(f"  {verificado['archivos_artefactos']} archivos en mlartifacts/")
        if args.solo_verificar:
            return 0
        proyecto = nombre_proyecto()
        print(f"Sembrando los volúmenes del proyecto de Compose {proyecto!r}")
        print("  (si falta la imagen, docker compose la construye primero)")
        huellas = sembrar(args.store, verificado)
    except ErrorSiembra as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"ERROR: no se encontró {exc.filename} (¿Docker instalado?)", file=sys.stderr)
        return 1
    print("Copia comprobada en el volumen:")
    for clave, valor in huellas.items():
        print(f"  {clave} {valor}")
    print(json.dumps({"proyecto": proyecto, **verificado, "volumen": huellas}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
