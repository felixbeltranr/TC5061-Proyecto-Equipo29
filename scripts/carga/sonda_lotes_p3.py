"""Barrido de tamaños de lote de P3: cuánto baja el costo por inferencia al agrupar.

El escenario de carga manda una estación por petición, para que una petición sea una
inferencia y el costo se lea directo. En operación, P3 no se consulta así: el grupo de uso se
recalcula por lote cuando se actualizan los perfiles. Este barrido mide ese otro patrón.

La medición no usa ``sonda_de_lotes`` del orquestador, que abre una conexión HTTP por petición.
Esa diferencia no importa con un lote de todas las estaciones, que tarda decenas de
milisegundos, pero sí con uno de una sola: en Windows, abrir una conexión a través del proxy de
puertos de Docker Desktop cuesta entre 5 y 25 ms, de modo que el tiempo de conexión tapa por
completo el de la inferencia y el barrido sale sin sentido (lotes de 10 más rápidos que los de
1). Aquí se reutiliza una conexión para todas las repeticiones de cada tamaño.

Se reportan la mediana y el mínimo. El costo se calcula con el **mínimo**, que es la mejor
estimación del tiempo de servicio: el ruido del sistema sólo puede sumar tiempo, nunca restarlo.
Las tarifas son las de ``mibici.operacion.finops``, para que el costo sea comparable con el del
análisis de la prueba de carga.

Con el sistema levantado con el tamaño de la medición:

    docker compose -f compose.yaml -f carga/compose.medicion.yaml up -d --wait
    python scripts/carga/sonda_lotes_p3.py

Escribe ``reports/carga/<fecha>_lotes_p3/`` con la tabla en CSV, Markdown y JSON.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from http.client import HTTPConnection
import json
from pathlib import Path
import random
import statistics
import sys
import time
from urllib.parse import urlsplit

RAIZ = Path(__file__).resolve().parents[2]

from mibici.operacion import finops  # noqa: E402

RUTA = "/api/v1/p3/predict"
COLUMNAS = [
    "estaciones",
    "ms_peticion",
    "ms_mediana",
    "ms_por_inferencia",
    "factor_vs_una",
    "usd_1000_inferencias",
    "peticiones_mes",
    "usd_mes",
]


def medir(host: str, estaciones: int, repeticiones: int, calentamiento: int,
          rng: random.Random) -> dict[str, float]:
    """Tiempos de ``repeticiones`` peticiones de ``estaciones`` filas, en una sola conexión."""
    # Aqui y no arriba: generar los cuerpos arrastra scikit-learn, y las funciones de calculo
    # de este modulo deben poder probarse sin eso.
    from mibici.operacion.payloads import cuerpo_p3

    partes = urlsplit(host)
    conexion = HTTPConnection(partes.hostname, partes.port or 80, timeout=60)
    tiempos = []
    try:
        for indice in range(calentamiento + repeticiones):
            cuerpo = json.dumps(cuerpo_p3(rng, estaciones)).encode("utf-8")
            inicio = time.perf_counter()
            conexion.request("POST", RUTA, body=cuerpo,
                             headers={"Content-Type": "application/json"})
            respuesta = conexion.getresponse()
            datos = respuesta.read()
            transcurrido = (time.perf_counter() - inicio) * 1000
            if respuesta.status != 200:
                raise RuntimeError(f"HTTP {respuesta.status}: {datos[:120]!r}")
            if indice >= calentamiento:
                tiempos.append(transcurrido)
    finally:
        conexion.close()
    return {"mediana": statistics.median(tiempos), "minimo": min(tiempos),
            "maximo": max(tiempos), "n": len(tiempos)}


def tabla(mediciones: dict[int, float], tarifa: finops.Tarifa, vcpu: float, gb: float,
          demanda: finops.Demanda, medianas: dict[int, float] | None = None) -> list[dict]:
    """Arma la tabla del barrido. ``mediciones`` es {estaciones: ms de la peticion}."""
    medianas = medianas or {}
    base = None
    filas = []
    for estaciones in sorted(mediciones):
        ms = mediciones[estaciones]
        ms_por_inferencia = ms / estaciones
        if base is None:
            base = ms_por_inferencia
        usd_peticion = finops.usd_por_peticion(tarifa, vcpu, gb, ms / 1000)
        peticiones_mes = demanda.peticiones(estaciones)["p3"]
        filas.append({
            "estaciones": estaciones,
            "ms_peticion": round(ms, 2),
            "ms_mediana": round(medianas.get(estaciones, ms), 2),
            "ms_por_inferencia": round(ms_por_inferencia, 4),
            "factor_vs_una": round(base / ms_por_inferencia, 1) if ms_por_inferencia else None,
            "usd_1000_inferencias": round(usd_peticion / estaciones * 1000, 6),
            "peticiones_mes": round(peticiones_mes),
            "usd_mes": round(usd_peticion * peticiones_mes, 4),
        })
    return filas


def a_markdown(filas: list[dict], tarifa: finops.Tarifa, vcpu: float, gb: float,
               repeticiones: int, inferencias_mes: float,
               inicializacion: float | None = None) -> str:
    lineas = [
        f"Mínimo y mediana de {repeticiones} peticiones por tamaño, sobre una conexión "
        f"reutilizada y con el sistema en reposo salvo la propia sonda. El costo se calcula con "
        f"el mínimo, la mejor estimación del tiempo de servicio. Tarifa: {tarifa.nombre}, "
        f"{vcpu:g} vCPU y {gb:g} GB. La demanda de P3 es de {inferencias_mes:,.0f} inferencias "
        f"al mes.",
        "",
        "| Estaciones por petición | Mínimo (ms) | Mediana (ms) | ms por inferencia "
        "| Veces más barato | USD por 1,000 inferencias | Peticiones al mes | USD al mes |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for f in filas:
        lineas.append(
            f"| {f['estaciones']} | {f['ms_peticion']:.2f} | {f['ms_mediana']:.2f} | "
            f"{f['ms_por_inferencia']:.4f} | {f['factor_vs_una']:.1f} | "
            f"{f['usd_1000_inferencias']:.6f} | {f['peticiones_mes']:,} | {f['usd_mes']:.4f} |"
        )
    if inicializacion is not None:
        lineas += [
            "",
            f"El primer tamaño se midió dos veces, al principio y al final del barrido: la "
            f"diferencia de {inicializacion:.2f} ms entre ambas es el costo de inicialización "
            f"del camino de inferencia, que así queda separado del tiempo de servicio que se "
            f"costea.",
        ]
    return "\n".join(lineas) + "\n"


def main(argv: list[str] | None = None) -> int:
    demanda = finops.Demanda()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="http://127.0.0.1:8000")
    p.add_argument("--tamanos", default=f"1,10,100,{demanda.estaciones}",
                   help="estaciones por petición, separadas por coma")
    p.add_argument("--repeticiones", type=int, default=50)
    p.add_argument("--calentamiento", type=int, default=10,
                   help="peticiones que se descartan antes de medir cada tamaño")
    p.add_argument("--tarifa", default="cloud_run_peticion", choices=sorted(finops.TARIFAS))
    p.add_argument("--vcpu", type=float, default=1.0)
    p.add_argument("--gb", type=float, default=2.0)
    p.add_argument("--salida", type=Path, default=None)
    p.add_argument(
        "--sin-repetir-primero",
        dest="repetir_primero",
        action="store_false",
        help="no vuelve a medir el primer tamaño al final del barrido",
    )
    args = p.parse_args(argv)

    rng = random.Random(29)
    minimos, medianas, crudo = {}, {}, {}
    tamanos = [int(x) for x in args.tamanos.split(",") if x.strip()]
    for estaciones in tamanos:
        try:
            medida = medir(args.host, estaciones, args.repeticiones, args.calentamiento, rng)
        except (OSError, RuntimeError) as error:
            print(f"ERROR con {estaciones} estaciones: {error}", file=sys.stderr)
            return 1
        minimos[estaciones] = medida["minimo"]
        medianas[estaciones] = medida["mediana"]
        crudo[estaciones] = medida
        print(f"  {estaciones:>4} estaciones: mínimo {medida['minimo']:6.2f} ms | "
              f"mediana {medida['mediana']:6.2f} ms | máximo {medida['maximo']:6.2f} ms")

    # El primer tamaño otra vez, ya con todo caliente: la diferencia contra su primera
    # medición es el costo de inicialización, y así no se le carga al lote más chico.
    inicializacion = None
    if args.repetir_primero and tamanos:
        primero = tamanos[0]
        try:
            final = medir(args.host, primero, args.repeticiones, args.calentamiento, rng)
        except (OSError, RuntimeError) as error:
            print(f"ERROR al repetir {primero}: {error}", file=sys.stderr)
            return 1
        inicializacion = minimos[primero] - final["minimo"]
        crudo[f"{primero}_repetida"] = final
        print(f"  {primero:>4} estaciones (repetida): mínimo {final['minimo']:6.2f} ms | "
              f"mediana {final['mediana']:6.2f} ms  ->  inicialización {inicializacion:+.2f} ms")
        minimos[primero] = min(minimos[primero], final["minimo"])
        medianas[primero] = min(medianas[primero], final["mediana"])

    tarifa = finops.TARIFAS[args.tarifa]
    filas = tabla(minimos, tarifa, args.vcpu, args.gb, demanda, medianas)

    salida = args.salida or (
        RAIZ / "reports" / "carga" / f"{datetime.now():%Y-%m-%d_%H%M}_lotes_p3"
    )
    salida.mkdir(parents=True, exist_ok=True)
    with (salida / "lotes_p3.csv").open("w", encoding="utf-8", newline="") as destino:
        escritor = csv.DictWriter(destino, fieldnames=COLUMNAS)
        escritor.writeheader()
        escritor.writerows(filas)
    markdown = a_markdown(filas, tarifa, args.vcpu, args.gb, args.repeticiones,
                          demanda.inferencias()["p3"], inicializacion)
    (salida / "lotes_p3.md").write_text(markdown, encoding="utf-8")
    (salida / "lotes_p3.json").write_text(
        json.dumps({"host": args.host, "tarifa": tarifa.clave, "vcpu": args.vcpu, "gb": args.gb,
                    "repeticiones": args.repeticiones, "calentamiento": args.calentamiento,
                    "inicializacion_ms": inicializacion,
                    "medidas": {str(k): v for k, v in crudo.items()}, "filas": filas},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print()
    print(markdown)
    print(f"Escrito en {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
