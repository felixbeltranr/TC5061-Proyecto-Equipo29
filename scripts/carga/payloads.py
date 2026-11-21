"""Cuerpos de peticion para la prueba de carga de la API de MiBici (Entregable 7).

Las peticiones salen de los ejemplos que ya estan versionados, para no duplicar el contrato de
cada pregunta ni inventar valores:

- P1: ``scripts/p1/campeon_api_p1.json`` -> ``peticion_real.cuerpo``.
- P2: ``scripts/p2/ejemplo_p2.json``.
- P3: ``scripts/p3/ejemplo_p3.json``.

Las variaciones respetan los validadores de ``mibici/api/schemas``, para que la carga mida el
servicio y no el manejador de errores 422:

- P1 escala los rezagos y las medias con un mismo factor (todos ``ge=0``) y recalcula
  ``media_misma_dsem_4`` como la media de los cuatro rezagos semanales.
- P2 mantiene ``neto_lag1 == llegadas_lag1 - salidas_lag1`` y ``fin_semana == (dia_semana >= 5)``,
  y usa horas de 6 a 23, que son las de ``CONFIG_P2.horas_etiqueta``.
- P3 solo mueve ``log_trips_day`` y ``weekend_share``: las proporciones ``dep_*`` y ``arr_*``
  tienen que sumar a lo mas 1 y moverlas por separado rompe esa validacion.

Este modulo no importa locust ni el paquete ``mibici``, para que las pruebas lo puedan usar sin
levantar nada.
"""

from __future__ import annotations

import json
from pathlib import Path
import random

RAIZ = Path(__file__).resolve().parents[2]
EJEMPLOS = {
    "p1": RAIZ / "scripts" / "p1" / "campeon_api_p1.json",
    "p2": RAIZ / "scripts" / "p2" / "ejemplo_p2.json",
    "p3": RAIZ / "scripts" / "p3" / "ejemplo_p3.json",
}

# Campos de P1 que escalan juntos; media_misma_dsem_4 se recalcula a partir de los semanales.
ESCALABLES_P1 = ("lag_2", "lag_7", "lag_14", "lag_21", "lag_28", "media_7d", "media_28d")
SEMANALES_P1 = ("lag_7", "lag_14", "lag_21", "lag_28")
# CONFIG_P2.horas_etiqueta = tuple(range(6, 24)).
HORA_MIN_P2, HORA_MAX_P2 = 6, 23


def _cuerpo_ejemplo(pregunta: str) -> dict:
    datos = json.loads(EJEMPLOS[pregunta].read_text(encoding="utf-8"))
    return datos["peticion_real"]["cuerpo"] if pregunta == "p1" else datos


BASE_P1 = _cuerpo_ejemplo("p1")
BASE_P2 = _cuerpo_ejemplo("p2")
BASE_P3 = _cuerpo_ejemplo("p3")


def estacion_id(indice: int) -> str:
    """Identificador que cumple el patron de los esquemas (alfanumerico, hasta 20 caracteres)."""
    return f"c{indice:04d}"


def cuerpo_p1(rng: random.Random) -> dict:
    """Una peticion de P1: un dia objetivo para una estacion."""
    features = dict(BASE_P1["features"])
    factor = rng.uniform(0.8, 1.2)
    for campo in ESCALABLES_P1:
        features[campo] = round(float(features[campo]) * factor, 4)
    features["media_misma_dsem_4"] = round(
        sum(features[campo] for campo in SEMANALES_P1) / len(SEMANALES_P1), 4
    )
    features["dia_semana"] = rng.randrange(7)
    return {"features": features}


def cuerpo_p2(rng: random.Random, estaciones: int = 1) -> dict:
    """Una peticion de P2 con ``estaciones`` filas: el riesgo de la siguiente hora."""
    base = BASE_P2["stations"][0]["features"]
    filas = []
    for indice in range(estaciones):
        features = dict(base)
        dia = rng.randrange(7)
        salidas = rng.randint(0, 12)
        llegadas = rng.randint(0, 12)
        features["dia_semana"] = dia
        features["fin_semana"] = 1 if dia >= 5 else 0
        features["hora"] = rng.randint(HORA_MIN_P2, HORA_MAX_P2)
        features["ocupacion_actual"] = round(rng.uniform(0.05, 0.95), 4)
        features["salidas_lag1"] = salidas
        features["llegadas_lag1"] = llegadas
        features["neto_lag1"] = float(llegadas - salidas)
        filas.append({"station_id": estacion_id(indice), "features": features})
    return {"stations": filas}


def cuerpo_p3(rng: random.Random, estaciones: int = 1) -> dict:
    """Una peticion de P3 con ``estaciones`` filas: el grupo de uso de cada estacion."""
    base = BASE_P3["stations"][0]["features"]
    filas = []
    for indice in range(estaciones):
        features = dict(base)
        features["log_trips_day"] = round(
            max(0.0, float(base["log_trips_day"]) * rng.uniform(0.7, 1.3)), 4
        )
        features["weekend_share"] = round(
            min(1.0, max(0.0, float(base["weekend_share"]) + rng.uniform(-0.1, 0.1))), 4
        )
        filas.append({"station_id": estacion_id(indice), "features": features})
    return {"stations": filas}
