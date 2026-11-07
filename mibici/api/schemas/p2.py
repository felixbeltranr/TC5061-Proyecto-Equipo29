"""Esquemas de entrada y salida de P2: clase de riesgo horario de una o varias estaciones.

Las 18 variables son las de ``features_p2.FEATURES``, en ese orden, y coinciden con la firma
del modelo registrado. La API no las calcula: recibe el panel ya armado (estadísticas de
entrenamiento incluidas). ``hora``, ``dia_semana`` y ``fin_semana`` son del instante objetivo
(t+1). El modelo publicado cubre el horizonte de 1 hora.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mibici.config import CONFIG_P2
from mibici.features_p2 import CLASES

TOLERANCIA_SUMA = 1e-6
HORA_MIN = min(CONFIG_P2.horas_etiqueta)
HORA_MAX = max(CONFIG_P2.horas_etiqueta)
# Ejemplo que cumple el contrato: estación 51, hora 8 de un miércoles. Los valores ilustran el
# formato; para una predicción significativa hay que calcular las características con
# mibici.features_p2 (el modelo acepta nulos en las estadísticas de estaciones nuevas y en
# rezagos largos sin historial).
EJEMPLO_P2 = {
    "hora": 8,
    "dia_semana": 2,
    "fin_semana": 0,
    "capacidad": 15,
    "ocupacion_actual": 0.47,
    "neto_lag1": 2.0,
    "neto_lag24": -1.0,
    "neto_lag168": 3.0,
    "salidas_lag1": 4,
    "llegadas_lag1": 6,
    "neto_media3": 1.0,
    "neto_media24": 0.2,
    "neto_media168_misma_hora": 1.5,
    "ocupacion_lag24": 0.40,
    "ocupacion_lag168": 0.53,
    "est_tasa_sin_bicis": 0.08,
    "est_tasa_sin_anclajes": 0.05,
    "est_neto_medio_hora": 0.4,
}


class P2Features(BaseModel):
    """Características del panel de P2 para una estación y una hora objetivo (t+1)."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)

    hora: int = Field(ge=HORA_MIN, le=HORA_MAX, description="Hora objetivo (t+1), de 6 a 23")
    dia_semana: int = Field(ge=0, le=6, description="Día de la semana de t+1; 0 es lunes")
    fin_semana: int = Field(ge=0, le=1, description="1 si t+1 es sábado o domingo")
    capacidad: int = Field(gt=0, description="Anclajes de la estación")
    ocupacion_actual: float = Field(ge=0, le=1, description="Ocupación al inicio de t+1")
    neto_lag1: float = Field(description="Llegadas menos salidas de la hora previa")
    neto_lag24: float | None = Field(description="Neto de la misma hora del día anterior")
    neto_lag168: float | None = Field(description="Neto de la misma hora de la semana anterior")
    salidas_lag1: int = Field(ge=0, description="Salidas de la hora previa")
    llegadas_lag1: int = Field(ge=0, description="Llegadas de la hora previa")
    neto_media3: float = Field(description="Media del neto de las 3 horas previas")
    neto_media24: float = Field(description="Media del neto de las 24 horas previas")
    neto_media168_misma_hora: float | None = Field(
        description="Media del neto a la misma hora en las 7 semanas previas"
    )
    ocupacion_lag24: float | None = Field(ge=0, le=1, description="Ocupación de hace 24 horas")
    ocupacion_lag168: float | None = Field(ge=0, le=1, description="Ocupación de hace 168 horas")
    est_tasa_sin_bicis: float | None = Field(
        ge=0, le=1, description="Tasa histórica de sin_bicicletas de la estación en train"
    )
    est_tasa_sin_anclajes: float | None = Field(
        ge=0, le=1, description="Tasa histórica de sin_anclajes de la estación en train"
    )
    est_neto_medio_hora: float | None = Field(
        description="Neto medio de la estación a esa hora en train"
    )

    @model_validator(mode="after")
    def consistencias(self):
        if (self.fin_semana == 1) != (self.dia_semana >= 5):
            raise ValueError(
                "fin_semana debe ser 1 si dia_semana es 5 o 6 (sábado o domingo), y 0 en otro caso"
            )
        neto_esperado = self.llegadas_lag1 - self.salidas_lag1
        if self.neto_lag1 != neto_esperado:
            raise ValueError(
                f"neto_lag1 es {self.neto_lag1} y no coincide con "
                f"llegadas_lag1 - salidas_lag1 ({neto_esperado})"
            )
        if (
            self.est_tasa_sin_bicis is not None
            and self.est_tasa_sin_anclajes is not None
            and self.est_tasa_sin_bicis + self.est_tasa_sin_anclajes > 1 + TOLERANCIA_SUMA
        ):
            suma = self.est_tasa_sin_bicis + self.est_tasa_sin_anclajes
            raise ValueError(
                f"est_tasa_sin_bicis y est_tasa_sin_anclajes suman {suma:.4f}; "
                "deben sumar a lo más 1"
            )
        return self


class P2Station(BaseModel):
    model_config = ConfigDict(extra="forbid")

    station_id: str | None = Field(
        default=None,
        max_length=20,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Identificador opcional; se devuelve igual para ubicar cada predicción",
    )
    features: P2Features


class P2PredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"stations": [{"station_id": "51", "features": EJEMPLO_P2}]}]
        },
    )

    stations: list[P2Station] = Field(
        min_length=1, max_length=1000, description="Una o varias estaciones (hasta 1000)"
    )


class P2Prediction(BaseModel):
    station_id: str | None
    risk_class: str = Field(description=f"Una de {', '.join(CLASES)}")


class P2PredictionResponse(BaseModel):
    predictions: list[P2Prediction]
    method: str
    input_count: int
    model_name: str | None = None
    model_version: str | None = None

