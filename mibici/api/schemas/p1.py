from pydantic import BaseModel, ConfigDict, Field

# Petición real: estación 51, pronóstico del 1 de octubre de 2026 con el panel hasta el 29 de
# septiembre (reglas de mibici.features).
EJEMPLO_P1 = {
    "dia_semana": 3,
    "festivo": 0,
    "festivo_lag7": 0,
    "lag_2": 404.0,
    "lag_7": 369.0,
    "lag_14": 413.0,
    "lag_21": 332.0,
    "lag_28": 273.0,
    "media_7d": 398.7142857142857,
    "media_28d": 364.10714285714283,
    "media_misma_dsem_4": 346.75,
}


class P1Features(BaseModel):
    """Variables de un día objetivo D, con salidas observadas hasta D-2."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    dia_semana: int = Field(ge=0, le=6, description="Día de la semana de D: 0=lunes ... 6=domingo")
    festivo: int = Field(ge=0, le=1, description="1 si D es festivo")
    festivo_lag7: int = Field(ge=0, le=1, description="1 si D-7 fue festivo")
    lag_2: float = Field(ge=0, description="Salidas en D-2")
    lag_7: float = Field(ge=0, description="Salidas en D-7")
    lag_14: float = Field(ge=0, description="Salidas en D-14")
    lag_21: float = Field(ge=0, description="Salidas en D-21")
    lag_28: float = Field(ge=0, description="Salidas en D-28")
    media_7d: float = Field(ge=0, description="Media diaria de D-8 a D-2")
    media_28d: float = Field(ge=0, description="Media diaria de D-29 a D-2")
    media_misma_dsem_4: float = Field(ge=0, description="Media de lag_7, lag_14, lag_21 y lag_28")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"examples": [{"features": EJEMPLO_P1}]}
    )

    features: P1Features
