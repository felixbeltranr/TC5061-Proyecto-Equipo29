"""Graficas y resumen de la prueba de carga (Entregable 7).

Lee los CSV que deja Locust (``--csv PREFIJO --csv-full-history``) y el CSV de
``muestrear_recursos.py``, y escribe las figuras del reporte mas un resumen por consola con las
cifras que necesita el analisis FinOps.

    python scripts/carga/graficas_carga.py --locust reports/carga/locust \
        --recursos reports/carga/recursos.csv --salida reports/carga/figuras --filtro tc5061

Usa pandas y matplotlib, que ya estan en requirements.txt.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

AGREGADO = "Aggregated"


def columna(datos: pd.DataFrame, *nombres: str) -> pd.Series | None:
    """Primera columna que exista, como numero; las versiones de Locust cambian los nombres."""
    for nombre in nombres:
        if nombre in datos.columns:
            return pd.to_numeric(datos[nombre], errors="coerce")
    return None


def historia(prefijo: Path) -> pd.DataFrame | None:
    ruta = prefijo.with_name(prefijo.name + "_stats_history.csv")
    if not ruta.is_file():
        print(f"AVISO: no existe {ruta}; se omiten las graficas de la corrida.")
        return None
    datos = pd.read_csv(ruta)
    marca = columna(datos, "Timestamp")
    if marca is not None:
        datos["segundo"] = marca - marca.min()
    return datos


def resumen_por_endpoint(prefijo: Path) -> pd.DataFrame | None:
    ruta = prefijo.with_name(prefijo.name + "_stats.csv")
    if not ruta.is_file():
        print(f"AVISO: no existe {ruta}; se omite el resumen por endpoint.")
        return None
    return pd.read_csv(ruta)


def figura_rps(datos: pd.DataFrame, salida: Path) -> None:
    agregado = datos[datos["Name"] == AGREGADO]
    if agregado.empty:
        return
    ejes = plt.subplots(figsize=(9, 4.5))[1]
    ejes.plot(agregado["segundo"], columna(agregado, "Requests/s"), color="tab:blue")
    ejes.set_xlabel("Segundos desde el inicio")
    ejes.set_ylabel("Peticiones por segundo", color="tab:blue")
    usuarios = columna(agregado, "User Count")
    if usuarios is not None:
        derecha = ejes.twinx()
        derecha.plot(agregado["segundo"], usuarios, color="tab:orange", linestyle="--")
        derecha.set_ylabel("Usuarios simultaneos", color="tab:orange")
    ejes.set_title("Peticiones por segundo y usuarios simultaneos")
    ejes.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(salida / "carga_rps_usuarios.png", dpi=150)
    plt.close("all")


def figura_latencia(datos: pd.DataFrame, salida: Path) -> None:
    agregado = datos[datos["Name"] == AGREGADO]
    if agregado.empty:
        return
    ejes = plt.subplots(figsize=(9, 4.5))[1]
    for etiqueta, nombres in (
        ("mediana", ("50%", "Total Median Response Time")),
        ("p95", ("95%",)),
        ("p99", ("99%",)),
    ):
        serie = columna(agregado, *nombres)
        if serie is not None:
            ejes.plot(agregado["segundo"], serie, label=etiqueta)
    ejes.set_xlabel("Segundos desde el inicio")
    ejes.set_ylabel("Latencia (ms)")
    ejes.set_title("Latencia de respuesta durante la corrida")
    ejes.legend()
    ejes.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(salida / "carga_latencia.png", dpi=150)
    plt.close("all")


def figura_errores(datos: pd.DataFrame, salida: Path) -> None:
    agregado = datos[datos["Name"] == AGREGADO]
    if agregado.empty:
        return
    peticiones = columna(agregado, "Requests/s")
    fallas = columna(agregado, "Failures/s")
    if peticiones is None or fallas is None:
        return
    ejes = plt.subplots(figsize=(9, 4.5))[1]
    ejes.plot(agregado["segundo"], fallas, color="tab:red", label="Fallas por segundo")
    # where en vez de replace: replace(0, pd.NA) vuelve la serie de tipo objeto y pandas
    # avisa del cambio de tipo al rellenar.
    porcentaje = (fallas / peticiones.where(peticiones > 0) * 100).fillna(0)
    derecha = ejes.twinx()
    derecha.plot(agregado["segundo"], porcentaje, color="tab:gray", linestyle=":")
    derecha.set_ylabel("Tasa de error (%)", color="tab:gray")
    ejes.set_xlabel("Segundos desde el inicio")
    ejes.set_ylabel("Fallas por segundo", color="tab:red")
    ejes.set_title("Errores durante la corrida")
    ejes.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(salida / "carga_errores.png", dpi=150)
    plt.close("all")


def figura_por_endpoint(estadisticas: pd.DataFrame, salida: Path) -> None:
    datos = estadisticas[estadisticas["Name"] != AGREGADO].copy()
    if datos.empty:
        return
    p95 = columna(datos, "95%")
    media = columna(datos, "Average Response Time")
    if p95 is None and media is None:
        return
    ejes = plt.subplots(figsize=(9, 0.6 * len(datos) + 2))[1]
    posiciones = range(len(datos))
    if media is not None:
        ejes.barh([p - 0.2 for p in posiciones], media, height=0.4, label="media")
    if p95 is not None:
        ejes.barh([p + 0.2 for p in posiciones], p95, height=0.4, label="p95")
    ejes.set_yticks(list(posiciones))
    ejes.set_yticklabels(datos["Name"])
    ejes.set_xlabel("Latencia (ms)")
    ejes.set_title("Latencia por endpoint")
    ejes.legend()
    ejes.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(salida / "carga_latencia_por_endpoint.png", dpi=150)
    plt.close("all")


def figuras_recursos(ruta: Path, salida: Path, filtro: list[str]) -> pd.DataFrame | None:
    if not ruta.is_file():
        print(f"AVISO: no existe {ruta}; se omiten las graficas de recursos.")
        return None
    datos = pd.read_csv(ruta)
    if filtro:
        fuera = sorted(
            nombre
            for nombre in datos["contenedor"].unique()
            if not any(texto in nombre for texto in filtro)
        )
        datos = datos[datos["contenedor"].apply(lambda n: any(t in n for t in filtro))]
        if fuera:
            print(f"Contenedores excluidos por --filtro: {', '.join(fuera)}")
    for campo, etiqueta, archivo in (
        ("cpu_pct", "CPU (%)", "recursos_cpu.png"),
        ("mem_mib", "Memoria (MiB)", "recursos_memoria.png"),
    ):
        if campo not in datos.columns:
            continue
        ejes = plt.subplots(figsize=(9, 4.5))[1]
        for contenedor, grupo in datos.groupby("contenedor"):
            ejes.plot(grupo["segundo"], pd.to_numeric(grupo[campo], errors="coerce"),
                      label=contenedor)
        ejes.set_xlabel("Segundos desde el inicio del muestreo")
        ejes.set_ylabel(etiqueta)
        ejes.set_title(f"{etiqueta} por contenedor")
        ejes.legend()
        ejes.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(salida / archivo, dpi=150)
        plt.close("all")
    return datos


def imprimir_resumen(estadisticas: pd.DataFrame | None, recursos: pd.DataFrame | None) -> None:
    if estadisticas is not None:
        print("\nPor endpoint (de Locust):")
        columnas = [c for c in ("Name", "Request Count", "Failure Count", "Requests/s",
                                "Average Response Time", "95%") if c in estadisticas.columns]
        print(estadisticas[columnas].to_string(index=False))
    if recursos is not None and not recursos.empty:
        print("\nConsumo por contenedor (del muestreo):")
        agrupado = recursos.groupby("contenedor").agg(
            cpu_media=("cpu_pct", "mean"),
            cpu_max=("cpu_pct", "max"),
            mem_media_mib=("mem_mib", "mean"),
            mem_max_mib=("mem_mib", "max"),
            lecturas=("cpu_pct", "size"),
        )
        print(agrupado.round(2).to_string())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--locust", type=Path, default=Path("reports/carga/locust"),
                        help="prefijo de los CSV de Locust, sin _stats.csv")
    parser.add_argument("--recursos", type=Path, default=Path("reports/carga/recursos.csv"))
    parser.add_argument("--salida", type=Path, default=Path("reports/carga/figuras"))
    parser.add_argument(
        "--filtro",
        action="append",
        default=[],
        help="subcadena del nombre del contenedor; repetible. Sin esto se grafican todos, "
        "incluidos los ajenos al proyecto que esten corriendo en la misma maquina",
    )
    args = parser.parse_args(argv)

    args.salida.mkdir(parents=True, exist_ok=True)
    datos = historia(args.locust)
    if datos is not None and "Name" in datos.columns:
        figura_rps(datos, args.salida)
        figura_latencia(datos, args.salida)
        figura_errores(datos, args.salida)
    estadisticas = resumen_por_endpoint(args.locust)
    if estadisticas is not None and "Name" in estadisticas.columns:
        figura_por_endpoint(estadisticas, args.salida)
    recursos = figuras_recursos(args.recursos, args.salida, args.filtro)
    imprimir_resumen(estadisticas, recursos)

    figuras = sorted(p.name for p in args.salida.glob("*.png"))
    if not figuras:
        print("No se genero ninguna figura: revisa las rutas.", file=sys.stderr)
        return 1
    print(f"\n{len(figuras)} figuras en {args.salida}: {', '.join(figuras)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
