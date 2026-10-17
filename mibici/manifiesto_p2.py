"""Escribe el manifiesto (registros y sha256) de los archivos usados en P2."""

import json
from pathlib import Path
import sys

from mibici.config import CONFIG, PROJ_ROOT
from mibici.features_p2 import manifiesto_datos

if __name__ == "__main__":
    archivos = sorted(CONFIG.raw_data_dir.glob("datos_abiertos_*.csv")) + [
        CONFIG.raw_data_dir / "nomenclatura_2026_09.csv"
    ]
    salida = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else PROJ_ROOT / "references" / "manifiesto_datos_p2.json"
    )
    man = manifiesto_datos(archivos)
    salida.write_text(json.dumps(man, indent=2) + "\n")
    print(man["version"])
