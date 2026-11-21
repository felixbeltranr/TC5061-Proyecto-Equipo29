"""Prueba de carga de la API de MiBici (Entregable 7).

Escenario ponderado sobre las tres preguntas, con P3 en lotes de 1, 10 y 100 estaciones para
poder comparar el costo por inferencia segun el tamano del lote en el analisis FinOps.

Requisitos: el sistema levantado (``docker compose up --build -d``) y ``/ready`` respondiendo
200. Instalar con ``pip install -r requirements-carga.txt``.

Corrida con interfaz web:

    locust -f scripts/carga/locustfile.py --host http://127.0.0.1:8000

Corrida sin interfaz, escalonada, guardando CSV y reporte HTML (lo que se usa en el reporte):

    locust -f scripts/carga/locustfile.py --host http://127.0.0.1:8000 --headless ^
        --csv reports/carga/locust --csv-full-history --html reports/carga/locust.html

En PowerShell, antes de la corrida escalonada:

    $env:MIBICI_CARGA_FORMA = "escalonada"

En bash:

    export MIBICI_CARGA_FORMA=escalonada

Sin esa variable no se define la forma de carga y mandan ``-u`` y ``-r`` de la linea de
comandos. Los escalones se cambian con ``MIBICI_CARGA_ESCALONES`` (pares ``usuarios:segundos``
separados por comas) y la velocidad de arranque con ``MIBICI_CARGA_SPAWN``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import random
import sys

from locust import HttpUser, LoadTestShape, between, events, task
import requests

# locust ejecuta este archivo por ruta: el modulo de cuerpos vive junto a el, no en el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import payloads  # noqa: E402

# 127.0.0.1 y no localhost: el Compose publica los puertos solo en 127.0.0.1, y en Windows
# localhost resuelve primero a ::1, donde no hay nada escuchando; cada intento se cuelga
# unos 4 segundos antes de fallar y la medicion sale inservible.
HOST_POR_OMISION = "http://127.0.0.1:8000"
RUTAS_MODELO = {
    "P1": "/api/v1/model",
    "P2": "/api/v1/p2/model",
    "P3": "/api/v1/p3/model",
}
ESCALONES_POR_OMISION = "10:60,25:60,50:60,100:60,150:60"
SPAWN_POR_OMISION = 10.0
# Un lote de P2 representa el barrido horario de un grupo de estaciones.
ESTACIONES_P2 = 5


@events.test_start.add_listener
def registrar_versiones(environment, **_) -> None:
    """Deja en el log que modelo sirvio cada pregunta: el reporte tiene que poder citarlo."""
    host = (environment.host or HOST_POR_OMISION).rstrip("/")
    for pregunta, ruta in RUTAS_MODELO.items():
        try:
            respuesta = requests.get(f"{host}{ruta}", timeout=10)
            respuesta.raise_for_status()
            datos = respuesta.json()
        except requests.RequestException as error:
            logging.warning("No se pudo leer el modelo de %s: %s", pregunta, error)
            continue
        logging.info(
            "%s servido por %s version %s",
            pregunta,
            datos.get("name", json.dumps(datos)),
            datos.get("version", "?"),
        )


class UsuarioMiBici(HttpUser):
    """Cliente que mezcla las tres preguntas contra la API."""

    host = HOST_POR_OMISION
    wait_time = between(0.5, 2.0)

    def on_start(self) -> None:
        # Semilla por usuario: las peticiones varian entre usuarios y la corrida se repite.
        self.rng = random.Random(hash((id(self), os.getpid())) & 0xFFFFFFFF)

    def _revisar(self, respuesta) -> None:
        """Marca la falla con el detalle de la API.

        Con ``status_code`` 0 no hubo respuesta (la conexion fallo o expiro): ahi no se llama a
        ``failure``, para que Locust reporte la excepcion real en vez de un "HTTP 0" sin
        informacion.
        """
        if respuesta.status_code == 0:
            return
        if respuesta.status_code != 200:
            respuesta.failure(f"HTTP {respuesta.status_code}: {respuesta.text[:200]}")

    def _predecir(self, ruta: str, cuerpo: dict, nombre: str) -> None:
        with self.client.post(ruta, json=cuerpo, name=nombre, catch_response=True) as respuesta:
            self._revisar(respuesta)

    @task(10)
    def p1_demanda(self) -> None:
        self._predecir("/api/v1/predict", payloads.cuerpo_p1(self.rng), "P1 predict")

    @task(6)
    def p2_riesgo(self) -> None:
        self._predecir(
            "/api/v1/p2/predict",
            payloads.cuerpo_p2(self.rng, ESTACIONES_P2),
            f"P2 predict x{ESTACIONES_P2}",
        )

    @task(6)
    def p3_una_estacion(self) -> None:
        self._predecir("/api/v1/p3/predict", payloads.cuerpo_p3(self.rng, 1), "P3 predict x1")

    @task(3)
    def p3_diez_estaciones(self) -> None:
        self._predecir("/api/v1/p3/predict", payloads.cuerpo_p3(self.rng, 10), "P3 predict x10")

    @task(1)
    def p3_cien_estaciones(self) -> None:
        self._predecir("/api/v1/p3/predict", payloads.cuerpo_p3(self.rng, 100), "P3 predict x100")

    @task(1)
    def disponibilidad(self) -> None:
        with self.client.get("/ready", name="GET /ready", catch_response=True) as respuesta:
            self._revisar(respuesta)


def _escalones() -> list[tuple[int, int]]:
    """``"10:60,25:60"`` -> ``[(10, 60), (25, 60)]``; usuarios y segundos de cada escalon."""
    crudo = os.environ.get("MIBICI_CARGA_ESCALONES", ESCALONES_POR_OMISION)
    escalones = []
    for par in crudo.split(","):
        usuarios, _, segundos = par.strip().partition(":")
        escalones.append((int(usuarios), int(segundos)))
    return escalones


if os.environ.get("MIBICI_CARGA_FORMA", "").strip().lower() == "escalonada":

    class CargaEscalonada(LoadTestShape):
        """Sube la carga por escalones para que la grafica muestre donde satura el servicio.

        Los escalones son atributos de clase y no se sobrescribe ``__init__``: la firma del
        constructor de ``LoadTestShape`` ha cambiado entre versiones de Locust.
        """

        escalones = _escalones()
        spawn = float(os.environ.get("MIBICI_CARGA_SPAWN", SPAWN_POR_OMISION))

        def tick(self):
            transcurrido = self.get_run_time()
            acumulado = 0
            for usuarios, segundos in self.escalones:
                acumulado += segundos
                if transcurrido < acumulado:
                    return usuarios, self.spawn
            return None
