"""Escenario de carga de la API de MiBici para Locust.

Simula a los sistemas que consumirían la API en operación:

- P2 (peso 18): el tablero de rebalanceo pide, cada hora de 6 a 23 h, el riesgo de cada
  estación. Es la pregunta con más tráfico, porque se consulta por estación y por hora.
- P1 (peso 1): la planeación nocturna pide, una vez al día, la demanda del día siguiente de
  cada estación.
- P3 (peso 1): planeación urbana consulta el grupo de uso de alguna estación.
- ``GET /live`` (peso 1): la sonda de salud del orquestador.

P2 y P1 siguen la proporción de la operación que usa el análisis de costo: 18 consultas de P2
por cada una de P1, por estación y día. P3, que en operación se recalcula una vez al mes, va
sobrerrepresentada a propósito para ejercitar su endpoint. Es una prueba de estrés por petición
individual (una estación por petición); el envío en lote se mide aparte, con la sonda de 468
estaciones del orquestador.

Cada usuario espera de 0.5 a 2 segundos entre peticiones, y cada cuerpo es distinto
(``mibici.operacion.payloads``), para que no se mida una respuesta repetida. Variables de
entorno:

- ``MIBICI_CARGA_ENDPOINTS``: qué preguntas se prueban (``p1,p2,p3`` por omisión). El
  orquestador ``scripts/carga/correr_carga.py`` deja fuera las que ``/ready`` reporta sin modelo.
- ``MIBICI_CARGA_LOTE_MAX``: estaciones por petición de P2 y P3 (1 por omisión, para que una
  petición sea una inferencia y el costo por inferencia se lea directo).

Uso directo: ``locust -f carga/locustfile.py --host http://127.0.0.1:8000`` (con ``localhost``,
en Windows cada conexión nueva tarda unos 2 segundos mientras se intenta IPv6).
"""

import os
import random

from locust import HttpUser, between

from mibici.operacion.payloads import cuerpo_p1, cuerpo_p2, cuerpo_p3

ENDPOINTS = {
    clave.strip()
    for clave in os.environ.get("MIBICI_CARGA_ENDPOINTS", "p1,p2,p3").split(",")
    if clave.strip()
}
LOTE_MAX = max(1, int(os.environ.get("MIBICI_CARGA_LOTE_MAX", "1")))


def _registrar(respuesta, clave: str) -> None:
    """Una respuesta cuenta como éxito sólo si es 200 y trae la clave esperada."""
    if respuesta.status_code != 200:
        respuesta.failure(f"HTTP {respuesta.status_code}: {respuesta.text[:120]}")
        return
    try:
        cuerpo = respuesta.json()
    except ValueError:
        respuesta.failure("La respuesta no es JSON")
        return
    if clave not in cuerpo:
        respuesta.failure(f"La respuesta no trae '{clave}'")
    else:
        respuesta.success()


def p1_demanda(usuario) -> None:
    with usuario.client.post(
        "/api/v1/predict",
        json=cuerpo_p1(usuario.rng),
        name="P1 POST /api/v1/predict",
        catch_response=True,
    ) as respuesta:
        _registrar(respuesta, "prediction")


def p2_riesgo(usuario) -> None:
    estaciones = usuario.rng.randint(1, LOTE_MAX)
    with usuario.client.post(
        "/api/v1/p2/predict",
        json=cuerpo_p2(usuario.rng, estaciones),
        name="P2 POST /api/v1/p2/predict",
        catch_response=True,
    ) as respuesta:
        _registrar(respuesta, "predictions")


def p3_grupos(usuario) -> None:
    estaciones = usuario.rng.randint(1, LOTE_MAX)
    with usuario.client.post(
        "/api/v1/p3/predict",
        json=cuerpo_p3(usuario.rng, estaciones),
        name="P3 POST /api/v1/p3/predict",
        catch_response=True,
    ) as respuesta:
        _registrar(respuesta, "assignments")


def salud(usuario) -> None:
    usuario.client.get("/live", name="GET /live")


PESOS = {p2_riesgo: 18, p1_demanda: 1, p3_grupos: 1}
TAREAS = {tarea: peso for tarea, peso in PESOS.items() if tarea.__name__[:2] in ENDPOINTS}
TAREAS[salud] = 1


class UsuarioMiBici(HttpUser):
    wait_time = between(0.5, 2.0)
    tasks = TAREAS

    def on_start(self) -> None:
        self.rng = random.Random()
