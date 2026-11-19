"""Lectura de los CSV de Locust y resumen de cada etapa de la prueba de carga.

Locust escribe, con ``--csv <prefijo> --csv-full-history``, ``<prefijo>_stats.csv`` (totales por
endpoint al final) y ``<prefijo>_stats_history.csv`` (una fila por segundo y endpoint). El RPS
sostenido de una etapa se calcula con el historial, descontando la rampa de arranque, para que
no lo diluyan los segundos en los que todavía se estaban creando usuarios.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

AGREGADO = "Aggregated"
PERCENTILES = ("50%", "95%", "99%")
SONDA = "/live"  # endpoints de salud, que no cuentan como inferencias


def _numerico(frame: pd.DataFrame, excluir: tuple[str, ...]) -> pd.DataFrame:
    frame = frame.copy()
    for columna in frame.columns:
        if columna not in excluir:
            frame[columna] = pd.to_numeric(frame[columna], errors="coerce")
    return frame


def leer_stats(ruta: Path) -> pd.DataFrame:
    """Totales por endpoint; la fila ``Aggregated`` es el total de la etapa."""
    frame = pd.read_csv(ruta, dtype=str, keep_default_na=False)
    return _numerico(frame, ("Type", "Name"))


def leer_historial(ruta: Path) -> pd.DataFrame:
    """Historial por segundo, con ``segundo`` = segundos desde el inicio de la etapa."""
    frame = pd.read_csv(ruta, dtype=str, keep_default_na=False)
    frame = _numerico(frame, ("Type", "Name"))
    if not frame.empty:
        frame["segundo"] = frame["Timestamp"] - frame["Timestamp"].min()
    else:
        frame["segundo"] = pd.Series(dtype=float)
    return frame


def fila_agregada(stats: pd.DataFrame) -> pd.Series:
    filas = stats[stats["Name"] == AGREGADO]
    if filas.empty:
        raise ValueError("El CSV de Locust no trae la fila Aggregated")
    return filas.iloc[-1]


def _valor(fila: pd.Series, columna: str) -> float:
    valor = fila.get(columna, math.nan)
    return float(valor) if pd.notna(valor) else math.nan


def rps_sostenido(historial: pd.DataFrame, usuarios: int, descartar_s: float = 10.0) -> float:
    """RPS medio una vez creados todos los usuarios, sin los primeros ``descartar_s``."""
    agregado = historial[historial["Name"] == AGREGADO]
    estable = agregado[(agregado["User Count"] >= usuarios) & (agregado["segundo"] >= descartar_s)]
    if estable.empty:
        estable = agregado[agregado["segundo"] >= descartar_s]
    if estable.empty:
        return math.nan
    return float(estable["Requests/s"].mean())


def resumen_etapa(
    stats: pd.DataFrame,
    historial: pd.DataFrame,
    usuarios: int,
    duracion_s: float,
    descartar_s: float = 10.0,
) -> dict:
    """Totales de la etapa, RPS sostenido y latencias por endpoint (en milisegundos)."""
    total = fila_agregada(stats)
    peticiones = int(_valor(total, "Request Count") or 0)
    fallas = int(_valor(total, "Failure Count") or 0)
    endpoints = []
    for _, fila in stats[stats["Name"] != AGREGADO].iterrows():
        endpoints.append(
            {
                # Los nombres del escenario ya traen el método ("P1 POST /api/v1/predict").
                "endpoint": str(fila["Name"]),
                "peticiones": int(_valor(fila, "Request Count") or 0),
                "fallas": int(_valor(fila, "Failure Count") or 0),
                "promedio_ms": _valor(fila, "Average Response Time"),
                "p50_ms": _valor(fila, "50%"),
                "p95_ms": _valor(fila, "95%"),
                "p99_ms": _valor(fila, "99%"),
                "rps": _valor(fila, "Requests/s"),
            }
        )
    rps = rps_sostenido(historial, usuarios, descartar_s)
    # Las sondas GET /live no son inferencias: se separan para el análisis de costo.
    inferencias = [e for e in endpoints if SONDA not in e["endpoint"]]
    total_endpoints = sum(e["peticiones"] for e in endpoints)
    total_inferencias = sum(e["peticiones"] for e in inferencias)
    ponderadas = [e for e in inferencias if e["peticiones"] and not math.isnan(e["promedio_ms"])]
    peso = sum(e["peticiones"] for e in ponderadas)
    suma_ms = sum(e["promedio_ms"] * e["peticiones"] for e in ponderadas)
    return {
        "usuarios": usuarios,
        "duracion_s": duracion_s,
        "peticiones": peticiones,
        "fallas": fallas,
        "tasa_error": fallas / peticiones if peticiones else math.nan,
        "rps_promedio": _valor(total, "Requests/s"),
        "rps_sostenido": rps,
        "rps_inferencias": rps * total_inferencias / total_endpoints
        if total_endpoints
        else math.nan,
        "promedio_ms": _valor(total, "Average Response Time"),
        "promedio_inferencia_ms": suma_ms / peso if peso else math.nan,
        "p50_ms": _valor(total, "50%"),
        "p95_ms": _valor(total, "95%"),
        "p99_ms": _valor(total, "99%"),
        "max_ms": _valor(total, "Max Response Time"),
        "endpoints": endpoints,
    }


def capacidad_sostenida(
    resumenes: list[dict], p95_max_ms: float = 500.0, error_max: float = 0.01
) -> dict | None:
    """La etapa con más RPS sostenido que cumple el objetivo de latencia y de errores."""
    validas = [
        r
        for r in resumenes
        if r["p95_ms"] <= p95_max_ms
        and (r["tasa_error"] <= error_max or math.isnan(r["tasa_error"]))
        and not math.isnan(r["rps_sostenido"])
    ]
    return max(validas, key=lambda r: r["rps_sostenido"]) if validas else None
