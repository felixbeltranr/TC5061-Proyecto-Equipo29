from pydantic import BaseModel, Field


class P1Features(BaseModel):
    dia_semana: int = Field(ge=0, le=6)
    festivo: float
    festivo_lag7: float
    lag_2: float
    lag_7: float
    lag_14: float
    lag_21: float
    lag_28: float
    media_7d: float
    media_28d: float
    media_misma_dsem_4: float


class PredictionRequest(BaseModel):
    features: P1Features