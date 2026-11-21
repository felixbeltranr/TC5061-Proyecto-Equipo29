"""Los cuerpos que genera la prueba de carga tienen que pasar los esquemas de la API.

Si no, la corrida mediria respuestas 422 en vez del servicio. Las validaciones cruzadas de P2
(neto_lag1, fin_semana) y la suma de proporciones de P3 son las que mas facil se rompen al
variar los valores, asi que se comprueban contra los esquemas reales cuando el paquete esta
instalado, y a mano en cualquier caso.
"""

from pathlib import Path
import random
import sys

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "scripts" / "carga"))
import payloads  # noqa: E402

FRANJAS = ("am", "mid", "pm", "night")


@pytest.fixture
def rng():
    return random.Random(29)


def test_p1_respeta_los_rangos_y_recalcula_la_media_semanal(rng):
    for _ in range(50):
        cuerpo = payloads.cuerpo_p1(rng)
        features = cuerpo["features"]
        assert set(features) == set(payloads.BASE_P1["features"])
        assert 0 <= features["dia_semana"] <= 6
        assert all(features[campo] >= 0 for campo in payloads.ESCALABLES_P1)
        esperada = sum(features[campo] for campo in payloads.SEMANALES_P1) / 4
        assert features["media_misma_dsem_4"] == pytest.approx(esperada, abs=1e-4)


def test_p2_cumple_sus_validaciones_cruzadas(rng):
    cuerpo = payloads.cuerpo_p2(rng, estaciones=25)
    assert len(cuerpo["stations"]) == 25
    assert len({fila["station_id"] for fila in cuerpo["stations"]}) == 25
    for fila in cuerpo["stations"]:
        features = fila["features"]
        assert payloads.HORA_MIN_P2 <= features["hora"] <= payloads.HORA_MAX_P2
        assert features["fin_semana"] == (1 if features["dia_semana"] >= 5 else 0)
        assert features["neto_lag1"] == features["llegadas_lag1"] - features["salidas_lag1"]
        # El esquema de P2 es estricto: neto_lag1 es float y no acepta un entero.
        assert isinstance(features["neto_lag1"], float)
        assert 0 <= features["ocupacion_actual"] <= 1


def test_p3_no_rompe_la_suma_de_proporciones(rng):
    cuerpo = payloads.cuerpo_p3(rng, estaciones=25)
    assert len(cuerpo["stations"]) == 25
    for fila in cuerpo["stations"]:
        features = fila["features"]
        assert set(features) == set(payloads.BASE_P3["stations"][0]["features"])
        for sentido in ("dep", "arr"):
            assert sum(features[f"{sentido}_{franja}"] for franja in FRANJAS) <= 1 + 1e-6
        assert features["log_trips_day"] >= 0
        assert 0 <= features["weekend_share"] <= 1


@pytest.mark.parametrize(
    ("modulo", "clase", "generador"),
    [
        ("mibici.api.schemas.p1", "PredictionRequest", lambda r: payloads.cuerpo_p1(r)),
        ("mibici.api.schemas.p2", "P2PredictionRequest", lambda r: payloads.cuerpo_p2(r, 10)),
        ("mibici.api.schemas.p3", "P3PredictionRequest", lambda r: payloads.cuerpo_p3(r, 10)),
    ],
)
def test_los_esquemas_de_la_api_aceptan_los_cuerpos(modulo, clase, generador, rng):
    esquemas = pytest.importorskip(modulo)
    peticion = getattr(esquemas, clase)
    for _ in range(20):
        peticion.model_validate(generador(rng))
