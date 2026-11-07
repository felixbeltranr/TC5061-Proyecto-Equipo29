"""Modelo falso de P3 por omisión para las pruebas de la app.

La API carga todos los modelos al arrancar. Las pruebas de P1 sustituyen sólo el cargador de
P1; este fixture sustituye el de P3 para que no busque un servidor de MLflow. Las pruebas de
P3 lo reemplazan cuando necesitan otro comportamiento.
"""

import pytest

from mibici.api.services.model_registry import RegisteredModelInfo

INFO_P3 = RegisteredModelInfo(
    name="mibici-equipo29-p3-grupos",
    version="1",
    alias="champion",
    run_id="run-p3",
    model_uri="models:/mibici-equipo29-p3-grupos/1",
)
NOMBRES_P3 = {0: "baja demanda", 1: "se vacía en la mañana", 2: "alta demanda"}


class FakeModelP3:
    """Asigna grupos fijos, uno por renglón, y guarda la entrada recibida."""

    def __init__(self, grupos=None):
        self.grupos = grupos
        self.entrada = None

    def predict(self, frame):
        self.entrada = frame
        return self.grupos if self.grupos is not None else [1] * len(frame)


@pytest.fixture(autouse=True)
def p3_falso(monkeypatch):
    import mibici.api.main as api_main

    modelo = FakeModelP3()
    monkeypatch.setattr(api_main, "load_p3_model_with_info", lambda: (modelo, INFO_P3))
    monkeypatch.setattr(api_main, "load_p3_group_names", lambda info: dict(NOMBRES_P3))
    return modelo
