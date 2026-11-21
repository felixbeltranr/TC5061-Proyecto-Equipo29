"""Predicción de P2 en lote por una cola de Redis atendida por un worker de RQ.

El worker corre con ``rq.worker.SimpleWorker``, que ejecuta los trabajos en su propio proceso:
el campeón de P2 se carga en el primer trabajo y se reusa en los siguientes.
"""

from functools import cache

from mibici.api.services.p2_inference import load_p2_model_with_info, predict_response

COLA = "p2"


@cache
def _modelo():
    return load_p2_model_with_info()


def predecir_lote(stations: list[dict]) -> dict:
    model, info = _modelo()
    return predict_response(model, info, stations).model_dump()
