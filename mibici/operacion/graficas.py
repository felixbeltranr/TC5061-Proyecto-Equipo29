"""Figuras de la prueba de carga para el reporte: desempeño, capacidad, endpoints y recursos.

Convenciones: un solo eje Y por panel (dos medidas de distinta escala van en paneles
separados), percentiles en una rampa de un solo tono (p50 claro, p99 oscuro), contenedores con
colores categóricos fijos (API azul, MLflow naranja), líneas delgadas, rejilla tenue y
etiquetas directas en texto gris. Las paletas se validaron con el validador de daltonismo y
contraste (azul y naranja: ΔE CVD 9.2; rampa ordinal monótona con el extremo claro a 2.06:1).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from mibici.operacion.carga import AGREGADO
from mibici.operacion.recursos import MIB, Muestra

SUPERFICIE = "#fcfcfb"
TEXTO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
REJILLA = "#e4e3de"
REFERENCIA = "#8a8984"
AZUL, NARANJA = "#2a78d6", "#eb6834"
PERCENTILES = {"50%": ("p50", "#86b6ef"), "95%": ("p95", "#2a78d6"), "99%": ("p99", "#104281")}
ANCHO_LINEA = 1.6
DPI = 200


def _estilo() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SUPERFICIE,
            "axes.facecolor": SUPERFICIE,
            "savefig.facecolor": SUPERFICIE,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.edgecolor": REJILLA,
            "axes.labelcolor": TEXTO_SECUNDARIO,
            "axes.titlecolor": TEXTO,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": REJILLA,
            "grid.linewidth": 0.6,
            "xtick.color": TEXTO_SECUNDARIO,
            "ytick.color": TEXTO_SECUNDARIO,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )


def _guardar(figura, salida: Path) -> Path:
    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(salida, dpi=DPI, bbox_inches="tight")
    plt.close(figura)
    return salida


def _etiqueta_final(eje, x, y, texto: str, dy: float = 0) -> None:
    """Etiqueta directa al final de una serie, en texto gris (el color lo lleva la línea)."""
    eje.annotate(
        texto,
        xy=(x, y),
        xytext=(4, dy),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color=TEXTO_SECUNDARIO,
    )


def _etiquetas_sin_choque(eje, finales: list[tuple[float, float, str]]) -> None:
    """Etiquetas al final de cada serie; si dos quedan muy juntas, sube la de arriba."""
    finales = sorted(finales, key=lambda f: f[1])
    alto = (eje.get_ylim()[1] - eje.get_ylim()[0]) or 1
    previa, desfase = None, 0.0
    for x, y, texto in finales:
        desfase = desfase + 9 if previa is not None and abs(y - previa) < 0.08 * alto else 0.0
        _etiqueta_final(eje, x, y, texto, desfase)
        previa = y


def serie_continua(historiales: list[tuple[int, pd.DataFrame]]) -> tuple[pd.DataFrame, list]:
    """Une las etapas en un solo eje de tiempo; devuelve también dónde empieza cada etapa.

    Cada fila conserva su ``etapa`` para dibujar un trazo por etapa (sin unir el final de una
    con el inicio de la siguiente). Los percentiles sin peticiones todavía quedan vacíos en
    lugar de cero, para no dibujar latencias que no existieron.
    """
    partes, cortes, desfase = [], [], 0.0
    for usuarios, historial in historiales:
        agregado = historial[historial["Name"] == AGREGADO].copy()
        if agregado.empty:
            continue
        agregado["t"] = agregado["segundo"] + desfase
        agregado["etapa"] = usuarios
        sin_peticiones = (
            agregado.get("Total Request Count", pd.Series(1, index=agregado.index)) <= 0
        )
        for columna in PERCENTILES:
            if columna in agregado:
                agregado.loc[sin_peticiones | (agregado[columna] <= 0), columna] = float("nan")
        cortes.append((desfase, usuarios))
        partes.append(agregado)
        desfase = float(agregado["t"].max()) + 1
    if not partes:
        return pd.DataFrame(columns=["t", "etapa"]), cortes
    return pd.concat(partes, ignore_index=True), cortes


def _por_etapa(eje, datos: pd.DataFrame, columna: str, color: str, etiqueta: str | None = None):
    """Un trazo por etapa con el mismo color; sólo el primero lleva la etiqueta de la leyenda."""
    for indice, (_, parte) in enumerate(datos.groupby("etapa", sort=False)):
        eje.plot(
            parte["t"],
            parte[columna],
            color=color,
            linewidth=ANCHO_LINEA,
            label=etiqueta if indice == 0 else None,
        )


def _marcar_etapas(eje, cortes: list, alto: bool = True) -> None:
    for inicio, usuarios in cortes:
        eje.axvline(inicio, color=REJILLA, linewidth=1.0, zorder=0)
        if alto:
            eje.annotate(
                f"{usuarios} usuarios",
                xy=(inicio, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(3, -10),
                textcoords="offset points",
                fontsize=7,
                color=TEXTO_SECUNDARIO,
            )


def tasa_error(datos: pd.DataFrame) -> pd.Series:
    """Fallas por segundo entre peticiones por segundo, en porcentaje (vacío si no hubo RPS)."""
    if datos.empty or "Failures/s" not in datos or "Requests/s" not in datos:
        return pd.Series(dtype=float, index=datos.index)
    rps = datos["Requests/s"].where(datos["Requests/s"] > 0)
    return datos["Failures/s"].fillna(0) / rps * 100


def grafica_desempeno(historiales: list[tuple[int, pd.DataFrame]], salida: Path) -> Path:
    """RPS, latencia (p50, p95, p99) y tasa de error a lo largo de las etapas, un panel cada uno."""
    _estilo()
    datos, cortes = serie_continua(historiales)
    figura, ejes = plt.subplots(3, 1, figsize=(7.2, 6.6), sharex=True)
    eje_rps, eje_lat, eje_err = ejes

    _por_etapa(eje_rps, datos, "Requests/s", AZUL)
    eje_rps.set_title("Peticiones por segundo")
    eje_rps.set_ylabel("RPS")
    eje_rps.set_ylim(bottom=0)
    _marcar_etapas(eje_rps, cortes)

    finales = []
    for columna, (nombre, color) in PERCENTILES.items():
        if columna in datos:
            _por_etapa(eje_lat, datos, columna, color, nombre)
            ultimo = datos[["t", columna]].dropna().tail(1)
            if not ultimo.empty:
                finales.append((ultimo["t"].iloc[0], ultimo[columna].iloc[0], nombre))
    eje_lat.set_title("Latencia por percentil")
    eje_lat.set_ylabel("milisegundos")
    eje_lat.set_ylim(bottom=0)
    eje_lat.legend(loc="upper left", ncols=3)
    _etiquetas_sin_choque(eje_lat, finales)
    _marcar_etapas(eje_lat, cortes, alto=False)

    errores = datos.assign(tasa_error=tasa_error(datos))
    sin_fallas = "Failures/s" not in datos or datos["Failures/s"].fillna(0).sum() == 0
    if not errores.empty:
        _por_etapa(eje_err, errores, "tasa_error", NARANJA)
    eje_err.set_title("Tasa de error")
    eje_err.set_ylabel("% de peticiones")
    if sin_fallas:
        # Sin fallas la línea queda pegada al cero: se fija una escala y se anota el resultado.
        eje_err.set_ylim(0, 1)
        eje_err.annotate(
            "0.00 % de errores en todas las etapas",
            xy=(0.5, 0.5),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=9,
            color=TEXTO_SECUNDARIO,
        )
    else:
        eje_err.set_ylim(bottom=0)
    _marcar_etapas(eje_err, cortes, alto=False)
    eje_err.set_xlabel("segundos desde el inicio de la prueba")
    figura.tight_layout()
    return _guardar(figura, salida)


def grafica_capacidad(resumenes: list[dict], salida: Path, p95_objetivo_ms: float = 500.0) -> Path:
    """Usuarios contra RPS sostenido y contra p95, en dos paneles (sin doble eje)."""
    _estilo()
    usuarios = [r["usuarios"] for r in resumenes]
    figura, (eje_rps, eje_p95) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    eje_rps.plot(
        usuarios,
        [r["rps_sostenido"] for r in resumenes],
        color=AZUL,
        linewidth=ANCHO_LINEA,
        marker="o",
        markersize=6,
    )
    eje_rps.set_title("RPS sostenido")
    eje_rps.set_xlabel("usuarios simultáneos")
    eje_rps.set_ylabel("RPS")
    eje_rps.set_ylim(bottom=0)

    eje_p95.plot(
        usuarios,
        [r["p95_ms"] for r in resumenes],
        color=AZUL,
        linewidth=ANCHO_LINEA,
        marker="o",
        markersize=6,
    )
    eje_p95.axhline(p95_objetivo_ms, color=REFERENCIA, linewidth=1.0, linestyle=(0, (4, 3)))
    eje_p95.annotate(
        f"objetivo {p95_objetivo_ms:.0f} ms",
        xy=(0, p95_objetivo_ms),
        xycoords=("axes fraction", "data"),
        xytext=(2, 3),
        textcoords="offset points",
        fontsize=7,
        color=TEXTO_SECUNDARIO,
    )
    eje_p95.set_title("Latencia p95")
    eje_p95.set_xlabel("usuarios simultáneos")
    eje_p95.set_ylabel("milisegundos")
    eje_p95.set_ylim(bottom=0)
    figura.tight_layout()
    return _guardar(figura, salida)


def grafica_endpoints(resumen: dict, salida: Path) -> Path:
    """p50 y p95 por endpoint en una etapa, en barras horizontales."""
    _estilo()
    filas = [e for e in resumen["endpoints"] if e["peticiones"] > 0]
    filas.sort(key=lambda e: e["p95_ms"])
    nombres = [e["endpoint"] for e in filas]
    posiciones = range(len(filas))
    figura, eje = plt.subplots(figsize=(7.2, 0.55 * max(len(filas), 2) + 0.9))
    alto = 0.36
    for desplazamiento, columna, nombre, color in (
        (-alto / 2 - 0.02, "p50_ms", "p50", PERCENTILES["50%"][1]),
        (alto / 2 + 0.02, "p95_ms", "p95", PERCENTILES["95%"][1]),
    ):
        valores = [e[columna] for e in filas]
        barras = eje.barh(
            [p + desplazamiento for p in posiciones],
            valores,
            height=alto,
            color=color,
            label=nombre,
        )
        eje.bar_label(barras, fmt="%.0f", padding=3, fontsize=7, color=TEXTO_SECUNDARIO)
    eje.set_yticks(list(posiciones), nombres)
    eje.set_xlabel("milisegundos")
    eje.set_title(f"Latencia por endpoint con {resumen['usuarios']} usuarios")
    eje.grid(axis="y", visible=False)
    eje.legend(loc="lower right", ncols=2)
    figura.tight_layout()
    return _guardar(figura, salida)


def grafica_recursos(
    muestras: list[Muestra],
    salida: Path,
    cortes: list[tuple[float, int]] | None = None,
    nombres: dict[str, str] | None = None,
) -> Path:
    """CPU (núcleos) y memoria (MiB) de cada contenedor; los cortes marcan cada etapa."""
    _estilo()
    if not muestras:
        raise ValueError("No hay muestras de recursos para graficar")
    inicio = min(m.marca for m in muestras)
    nombres = nombres or {}
    contenedores = sorted({m.contenedor for m in muestras}, key=lambda c: (c not in nombres, c))
    colores = [AZUL, NARANJA]
    figura, (eje_cpu, eje_mem) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    finales_cpu, finales_mem = [], []
    for indice, contenedor in enumerate(contenedores[:2]):
        propias = sorted(
            (m for m in muestras if m.contenedor == contenedor), key=lambda m: m.marca
        )
        t = [m.marca - inicio for m in propias]
        etiqueta = nombres.get(contenedor, contenedor)
        cpu = [m.cpu_pct / 100 for m in propias]
        mem = [m.memoria_bytes / MIB for m in propias]
        color = colores[indice]
        eje_cpu.plot(t, cpu, color=color, linewidth=ANCHO_LINEA, label=etiqueta)
        eje_mem.plot(t, mem, color=color, linewidth=ANCHO_LINEA)
        finales_cpu.append((t[-1], cpu[-1], etiqueta))
        finales_mem.append((t[-1], mem[-1], etiqueta))
    for eje, finales in ((eje_cpu, finales_cpu), (eje_mem, finales_mem)):
        eje.set_ylim(bottom=0)
        _etiquetas_sin_choque(eje, finales)
        for corte, usuarios in cortes or []:
            eje.axvline(corte - inicio, color=REJILLA, linewidth=1.0, zorder=0)
    for corte, usuarios in cortes or []:
        eje_cpu.annotate(
            f"{usuarios} usuarios",
            xy=(corte - inicio, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -10),
            textcoords="offset points",
            fontsize=7,
            color=TEXTO_SECUNDARIO,
        )
    eje_cpu.set_title("CPU por contenedor")
    eje_cpu.set_ylabel("núcleos")
    eje_mem.set_title("Memoria por contenedor")
    eje_mem.set_ylabel("MiB")
    eje_mem.set_xlabel("segundos desde la primera muestra")
    figura.tight_layout(rect=(0, 0.06, 1, 1))
    figura.legend(loc="lower center", ncols=2, bbox_to_anchor=(0.5, 0.0))
    return _guardar(figura, salida)
