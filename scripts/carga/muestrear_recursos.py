"""Muestrea el consumo de CPU y memoria de los contenedores durante la prueba de carga.

Corre ``docker stats --no-stream`` cada cierto intervalo y acumula las lecturas en un CSV, para
despues cruzarlas con los tiempos de Locust y estimar el costo por inferencia (Entregable 7).

Uso tipico, en otra terminal mientras corre Locust:

    python scripts/carga/muestrear_recursos.py --duracion 330 --salida reports/carga/recursos.csv

Con podman:

    python scripts/carga/muestrear_recursos.py --comando podman

Solo usa la biblioteca estandar. Las columnas de memoria estan en MiB (como las reporta el
motor de contenedores) y las de red y disco en MB decimales.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

COLUMNAS = [
    "ts",
    "segundo",
    "contenedor",
    "cpu_pct",
    "mem_mib",
    "mem_limite_mib",
    "mem_pct",
    "net_rx_mb",
    "net_tx_mb",
    "disco_lectura_mb",
    "disco_escritura_mb",
    "pids",
]
FACTORES = {
    "B": 1.0,
    "KB": 1000.0,
    "MB": 1000.0**2,
    "GB": 1000.0**3,
    "TB": 1000.0**4,
    "KIB": 1024.0,
    "MIB": 1024.0**2,
    "GIB": 1024.0**3,
    "TIB": 1024.0**4,
}
MIB = 1024.0**2
MB = 1000.0**2


def a_bytes(texto: str) -> float | None:
    """``"123.4MiB"`` -> bytes. Devuelve None cuando el motor reporta ``--``."""
    texto = (texto or "").strip()
    numero = ""
    indice = 0
    while indice < len(texto) and (texto[indice].isdigit() or texto[indice] in ".-+e"):
        numero += texto[indice]
        indice += 1
    sufijo = texto[indice:].strip().upper() or "B"
    if not numero or sufijo not in FACTORES:
        return None
    try:
        return float(numero) * FACTORES[sufijo]
    except ValueError:
        return None


def a_porcentaje(texto: str) -> float | None:
    texto = (texto or "").strip().rstrip("%")
    try:
        return float(texto)
    except ValueError:
        return None


def par(texto: str) -> tuple[float | None, float | None]:
    """``"123.4MiB / 7.6GiB"`` -> (bytes usados, bytes limite)."""
    izquierda, _, derecha = (texto or "").partition("/")
    return a_bytes(izquierda), a_bytes(derecha)


def redondear(valor: float | None, unidad: float) -> float | None:
    return None if valor is None else round(valor / unidad, 4)


def fila(lectura: dict, ts: datetime, segundo: float) -> dict:
    usado, limite = par(lectura.get("MemUsage", ""))
    recibido, enviado = par(lectura.get("NetIO", ""))
    leido, escrito = par(lectura.get("BlockIO", ""))
    return {
        "ts": ts.isoformat(timespec="seconds"),
        "segundo": round(segundo, 2),
        "contenedor": lectura.get("Name") or lectura.get("Container", ""),
        "cpu_pct": a_porcentaje(lectura.get("CPUPerc", "")),
        "mem_mib": redondear(usado, MIB),
        "mem_limite_mib": redondear(limite, MIB),
        "mem_pct": a_porcentaje(lectura.get("MemPerc", "")),
        "net_rx_mb": redondear(recibido, MB),
        "net_tx_mb": redondear(enviado, MB),
        "disco_lectura_mb": redondear(leido, MB),
        "disco_escritura_mb": redondear(escrito, MB),
        "pids": lectura.get("PIDs", ""),
    }


def leer_stats(comando: str) -> list[dict]:
    salida = subprocess.run(
        [comando, "stats", "--no-stream", "--format", "{{json .}}"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if salida.returncode != 0:
        raise RuntimeError(
            f"{comando} stats termino con codigo {salida.returncode}: {salida.stderr.strip()}"
        )
    lecturas = []
    for linea in salida.stdout.splitlines():
        linea = linea.strip()
        if linea:
            lecturas.append(json.loads(linea))
    return lecturas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--intervalo", type=float, default=2.0, help="segundos entre lecturas")
    parser.add_argument(
        "--duracion", type=float, default=0.0, help="segundos totales; 0 corre hasta Ctrl-C"
    )
    parser.add_argument("--salida", type=Path, default=Path("reports/carga/recursos.csv"))
    parser.add_argument(
        "--filtro", action="append", default=[], help="subcadena del nombre; repetible"
    )
    parser.add_argument("--comando", default="docker", help="docker o podman")
    args = parser.parse_args(argv)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    inicio = time.monotonic()
    lecturas_escritas = 0
    try:
        with args.salida.open("w", encoding="utf-8", newline="") as destino:
            escritor = csv.DictWriter(destino, fieldnames=COLUMNAS)
            escritor.writeheader()
            while True:
                momento = time.monotonic()
                segundo = momento - inicio
                if args.duracion and segundo > args.duracion:
                    break
                try:
                    lecturas = leer_stats(args.comando)
                except (RuntimeError, OSError, json.JSONDecodeError) as error:
                    print(f"ERROR: {error}", file=sys.stderr)
                    return 1
                ahora = datetime.now(timezone.utc).astimezone()
                for lectura in lecturas:
                    nombre = lectura.get("Name") or lectura.get("Container", "")
                    if args.filtro and not any(texto in nombre for texto in args.filtro):
                        continue
                    escritor.writerow(fila(lectura, ahora, segundo))
                    lecturas_escritas += 1
                destino.flush()
                descanso = args.intervalo - (time.monotonic() - momento)
                if descanso > 0:
                    time.sleep(descanso)
    except KeyboardInterrupt:
        print()
    print(f"{lecturas_escritas} lecturas en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
