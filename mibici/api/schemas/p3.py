"""Esquemas de entrada y salida de P3: grupo de uso de una o varias estaciones.

Las 12 variables son las de ``CONFIG_P3.prioridad_variables`` y se calculan con
``mibici.features_p3.perfil_estaciones`` sobre los viajes de cada estación. El modelo elige
internamente (por VIF) cuáles usa, así que la API recibe siempre las 12.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Las franjas am, mid, pm y night no cubren las 10 y 16 h, así que sus proporciones suman a lo
# más 1. Margen para el redondeo de quien calcula el perfil.
TOLERANCIA_SUMA = 1e-6
FRANJAS = ("am", "mid", "pm", "night")

# Petición real: estación 51 con su perfil del periodo de evaluación (mayo a septiembre de
# 2026, data/interim/p3_perfiles_evaluacion.parquet), redondeado a 4 decimales. El campeón la
# asigna al grupo 2 con el perfil exacto y con el redondeado.
EJEMPLO_P3 = {
    "log_trips_day": 5.7487,
    "weekend_share": 0.2622,
    "net_am": -0.4885,
    "net_pm": 0.3124,
    "dep_am": 0.3048,
    "dep_pm": 0.281,
    "dep_mid": 0.2113,
    "dep_night": 0.1137,
    "arr_am": 0.0904,
    "arr_pm": 0.4629,
    "arr_mid": 0.2306,
    "arr_night": 0.1018,
}


def _proporcion(descripcion: str):
    return Field(ge=0, le=1, description=descripcion)


class P3Features(BaseModel):
    """Perfil de uso de una estación (12 variables candidatas de P3)."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)

    log_trips_day: float = Field(ge=0, description="log(1 + viajes por día activo)")
    weekend_share: float = _proporcion("Proporción de salidas en fin de semana")
    net_am: float = Field(
        ge=-1, le=1, description="(llegadas - salidas) / total de 6 a 9 h; < 0 se vacía"
    )
    net_pm: float = Field(
        ge=-1, le=1, description="(llegadas - salidas) / total de 17 a 20 h; < 0 se vacía"
    )
    dep_am: float = _proporcion("Proporción de salidas en día hábil de 6 a 9 h")
    dep_pm: float = _proporcion("Proporción de salidas en día hábil de 17 a 20 h")
    dep_mid: float = _proporcion("Proporción de salidas en día hábil de 11 a 15 h")
    dep_night: float = _proporcion("Proporción de salidas en día hábil de 21 a 5 h")
    arr_am: float = _proporcion("Proporción de llegadas en día hábil de 6 a 9 h")
    arr_pm: float = _proporcion("Proporción de llegadas en día hábil de 17 a 20 h")
    arr_mid: float = _proporcion("Proporción de llegadas en día hábil de 11 a 15 h")
    arr_night: float = _proporcion("Proporción de llegadas en día hábil de 21 a 5 h")

    @model_validator(mode="after")
    def franjas_suman_a_lo_mas_uno(self):
        for sentido in ("dep", "arr"):
            suma = sum(getattr(self, f"{sentido}_{franja}") for franja in FRANJAS)
            if suma > 1 + TOLERANCIA_SUMA:
                raise ValueError(
                    f"Las proporciones {sentido}_* suman {suma:.4f}; deben sumar a lo más 1"
                )
        return self


class P3Station(BaseModel):
    model_config = ConfigDict(extra="forbid")

    station_id: str | None = Field(
        default=None,
        max_length=20,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Identificador opcional; se devuelve igual para ubicar cada asignación",
    )
    features: P3Features


class P3PredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"stations": [{"station_id": "51", "features": EJEMPLO_P3}]}]
        },
    )

    stations: list[P3Station] = Field(
        min_length=1, max_length=1000, description="Una o varias estaciones (hasta 1000)"
    )


class P3Assignment(BaseModel):
    station_id: str | None
    group: int = Field(ge=0)
    group_name: str | None = Field(description="Nombre del grupo; nulo si no se pudo leer")


class P3PredictionResponse(BaseModel):
    assignments: list[P3Assignment]
    method: str
    input_count: int
    model_name: str | None = None
    model_version: str | None = None
