"""Peticiones válidas y variadas para la prueba de carga de la API.

Cada generador parte del ejemplo real de su esquema (estación 51) y lo varía dentro de las
reglas que valida la API, para que la carga no repita siempre el mismo cuerpo. Las pruebas de
``tests/operacion/test_payloads.py`` validan los cuerpos contra los esquemas de Pydantic con
``model_validate_json``, igual que lo hace FastAPI al recibir la petición.
"""

from __future__ import annotations

import math
import random

from mibici.api.schemas.p1 import EJEMPLO_P1
from mibici.api.schemas.p2 import HORA_MAX, HORA_MIN
from mibici.api.schemas.p3 import EJEMPLO_P3

LAGS_P1 = ("lag_2", "lag_7", "lag_14", "lag_21", "lag_28")
CAPACIDADES_P2 = (11, 15, 19, 23, 27)
FRANJAS_P3 = ("am", "mid", "pm", "night")


def _redondeo_hacia_abajo(valor: float, decimales: int = 4) -> float:
    """Redondea hacia abajo para que una suma de proporciones nunca pase de 1."""
    factor = 10**decimales
    return math.floor(valor * factor) / factor


def _estacion(rng: random.Random) -> str:
    return str(rng.randint(1, 400))


def features_p1(rng: random.Random) -> dict:
    """Una estación con más o menos demanda que la del ejemplo (factor de 0.2 a 1.8)."""
    escala = rng.uniform(0.2, 1.8)
    lags = {
        clave: round(EJEMPLO_P1[clave] * escala * rng.uniform(0.85, 1.15), 1) for clave in LAGS_P1
    }
    misma_dsem = (lags["lag_7"] + lags["lag_14"] + lags["lag_21"] + lags["lag_28"]) / 4
    return {
        "dia_semana": rng.randrange(7),
        "festivo": int(rng.random() < 0.03),
        "festivo_lag7": int(rng.random() < 0.03),
        **lags,
        "media_7d": round(EJEMPLO_P1["media_7d"] * escala * rng.uniform(0.9, 1.1), 4),
        "media_28d": round(EJEMPLO_P1["media_28d"] * escala * rng.uniform(0.9, 1.1), 4),
        "media_misma_dsem_4": round(misma_dsem, 4),
    }


def cuerpo_p1(rng: random.Random) -> dict:
    """Cuerpo de ``POST /api/v1/predict``."""
    return {"features": features_p1(rng)}


def _opcional(rng: random.Random, valor: float, probabilidad_nulo: float = 0.05):
    """Las estadísticas largas pueden venir nulas en estaciones sin historial."""
    return None if rng.random() < probabilidad_nulo else valor


def features_p2(rng: random.Random) -> dict:
    """Una estación y una hora objetivo; respeta las consistencias de ``P2Features``."""
    dia = rng.randrange(7)
    salidas = rng.randint(0, 12)
    llegadas = rng.randint(0, 12)
    tasa_sin_bicis = round(rng.uniform(0, 0.3), 3)
    tasa_sin_anclajes = round(rng.uniform(0, 0.3), 3)
    return {
        "hora": rng.randint(HORA_MIN, HORA_MAX),
        "dia_semana": dia,
        "fin_semana": int(dia >= 5),
        "capacidad": rng.choice(CAPACIDADES_P2),
        "ocupacion_actual": round(rng.uniform(0, 1), 3),
        "neto_lag1": float(llegadas - salidas),
        "neto_lag24": _opcional(rng, round(rng.uniform(-6, 6), 1)),
        "neto_lag168": _opcional(rng, round(rng.uniform(-6, 6), 1)),
        "salidas_lag1": salidas,
        "llegadas_lag1": llegadas,
        "neto_media3": round(rng.uniform(-4, 4), 3),
        "neto_media24": round(rng.uniform(-2, 2), 3),
        "neto_media168_misma_hora": _opcional(rng, round(rng.uniform(-4, 4), 3)),
        "ocupacion_lag24": _opcional(rng, round(rng.uniform(0, 1), 3)),
        "ocupacion_lag168": _opcional(rng, round(rng.uniform(0, 1), 3)),
        "est_tasa_sin_bicis": tasa_sin_bicis,
        "est_tasa_sin_anclajes": tasa_sin_anclajes,
        "est_neto_medio_hora": round(rng.uniform(-2, 2), 3),
    }


def cuerpo_p2(rng: random.Random, estaciones: int = 1) -> dict:
    """Cuerpo de ``POST /api/v1/p2/predict`` con una o varias estaciones."""
    return {
        "stations": [
            {"station_id": _estacion(rng), "features": features_p2(rng)} for _ in range(estaciones)
        ]
    }


def _proporciones(rng: random.Random) -> list[float]:
    """Cuatro proporciones que suman menos de 1: la quinta parte queda para las 10 y 16 h."""
    pesos = [rng.uniform(0.05, 1.0) for _ in range(5)]
    total = sum(pesos)
    return [_redondeo_hacia_abajo(peso / total) for peso in pesos[:4]]


def features_p3(rng: random.Random) -> dict:
    """Un perfil de uso con proporciones por franja que suman a lo más 1."""
    salidas = _proporciones(rng)
    llegadas = _proporciones(rng)
    return {
        "log_trips_day": round(EJEMPLO_P3["log_trips_day"] * rng.uniform(0.4, 1.2), 4),
        "weekend_share": round(rng.uniform(0.1, 0.4), 4),
        "net_am": round(rng.uniform(-0.8, 0.8), 4),
        "net_pm": round(rng.uniform(-0.8, 0.8), 4),
        **{f"dep_{franja}": valor for franja, valor in zip(FRANJAS_P3, salidas)},
        **{f"arr_{franja}": valor for franja, valor in zip(FRANJAS_P3, llegadas)},
    }


def cuerpo_p3(rng: random.Random, estaciones: int = 1) -> dict:
    """Cuerpo de ``POST /api/v1/p3/predict`` con una o varias estaciones."""
    return {
        "stations": [
            {"station_id": _estacion(rng), "features": features_p3(rng)} for _ in range(estaciones)
        ]
    }
