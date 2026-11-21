"""Los cuerpos de la prueba de carga deben pasar la misma validación que hace la API."""

import json
import random

import pytest

from mibici.api.schemas.p1 import PredictionRequest
from mibici.api.schemas.p2 import P2PredictionRequest
from mibici.api.schemas.p3 import P3PredictionRequest
from mibici.operacion.payloads import cuerpo_p1, cuerpo_p2, cuerpo_p3, features_p3


@pytest.mark.parametrize("semilla", range(300))
def test_cuerpos_validos_como_json(semilla):
    rng = random.Random(semilla)
    # model_validate_json valida igual que FastAPI al recibir el cuerpo (tipos estrictos de P2 y P3).
    PredictionRequest.model_validate_json(json.dumps(cuerpo_p1(rng)))
    P2PredictionRequest.model_validate_json(json.dumps(cuerpo_p2(rng, estaciones=3)))
    P3PredictionRequest.model_validate_json(json.dumps(cuerpo_p3(rng, estaciones=3)))


def test_lote_con_todas_las_estaciones():
    rng = random.Random(29)
    cuerpo = cuerpo_p2(rng, estaciones=468)
    assert len(P2PredictionRequest.model_validate_json(json.dumps(cuerpo)).stations) == 468
    cuerpo = cuerpo_p3(rng, estaciones=468)
    assert len(P3PredictionRequest.model_validate_json(json.dumps(cuerpo)).stations) == 468


def test_misma_semilla_mismo_cuerpo():
    assert cuerpo_p1(random.Random(7)) == cuerpo_p1(random.Random(7))
    assert cuerpo_p2(random.Random(7), 2) == cuerpo_p2(random.Random(7), 2)


def test_cuerpos_distintos_entre_peticiones():
    rng = random.Random(1)
    cuerpos = {json.dumps(cuerpo_p1(rng), sort_keys=True) for _ in range(50)}
    assert len(cuerpos) == 50


def test_proporciones_p3_suman_a_lo_mas_uno():
    rng = random.Random(3)
    for _ in range(500):
        perfil = features_p3(rng)
        for sentido in ("dep", "arr"):
            suma = sum(perfil[f"{sentido}_{franja}"] for franja in ("am", "mid", "pm", "night"))
            assert suma <= 1
