"""Exporta train y test de P2 con features y objetivo para validarlos con DeepChecks.

Las estadisticas por estacion salen solo de train, como en el entrenamiento. El resultado
queda en data/interim y lo lee tests/datos/test_deepchecks_p2.py en su entorno aparte.

    python -m mibici.modeling.exportar_p2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger
import pandas as pd

from mibici import features_p2 as fp
from mibici.config import CONFIG, CONFIG_P2
from mibici.modeling import train_p2 as t2


def exportar_particiones(panel: pd.DataFrame, nivel: float, destino: Path) -> dict[str, Path]:
    """Escribe p2_train.parquet y p2_test.parquet con FEATURES y objetivo; regresa las rutas."""
    partes = t2.preparar(nivel, panel)
    por_est, por_hora = fp.stats_estacion(partes["train"])
    destino.mkdir(parents=True, exist_ok=True)
    rutas = {}
    for nombre in ("train", "test"):
        df = fp.unir_stats(partes[nombre], por_est, por_hora)[[*fp.FEATURES, "objetivo"]]
        rutas[nombre] = destino / f"p2_{nombre}.parquet"
        df.to_parquet(rutas[nombre], index=False)
        logger.info("{}: {} filas en {}", nombre, len(df), rutas[nombre].name)
    return rutas


def main(argv: list[str] | None = None) -> None:
    """Punto de entrada: python -m mibici.modeling.exportar_p2."""
    ap = argparse.ArgumentParser(description="Exporta train y test de P2 para DeepChecks")
    ap.add_argument("--panel", default=str(CONFIG_P2.panel_path))
    ap.add_argument("--nivel", type=float, default=CONFIG_P2.nivel_inicial)
    ap.add_argument("--destino", default=str(CONFIG.interim_data_dir))
    args = ap.parse_args(argv)
    exportar_particiones(pd.read_parquet(args.panel), args.nivel, Path(args.destino))


if __name__ == "__main__":
    main()
