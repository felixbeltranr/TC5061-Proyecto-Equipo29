"""Escribe el manifiesto (registros y sha256) de los archivos usados en P2.

La version resultante se registra como data_version en los runs de MLflow.
"""

import json
from pathlib import Path

from loguru import logger
import typer

from mibici.config import CONFIG, CONFIG_P2
from mibici.features_p2 import manifiesto_datos

app = typer.Typer()


@app.command()
def main(
    salida: Path = CONFIG_P2.manifiesto_path,
    carpeta: Path = CONFIG.raw_data_dir,
) -> None:
    """Calcula el manifiesto de los CSV mensuales y la nomenclatura en carpeta."""
    archivos = sorted(carpeta.glob("datos_abiertos_*.csv")) + [
        carpeta / CONFIG_P2.nomenclatura_path.name
    ]
    manifiesto = manifiesto_datos(archivos)
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(manifiesto, indent=2) + "\n")
    logger.info("Manifiesto {} con version {}", salida, manifiesto["version"])


if __name__ == "__main__":
    app()
