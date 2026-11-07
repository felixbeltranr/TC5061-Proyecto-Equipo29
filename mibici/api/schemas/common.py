from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    alias: str | None
    run_id: str | None
    model_uri: str


class PredictionResponse(BaseModel):
    prediction: float
    method: str
    input_count: int
    model_name: str | None = None
    model_version: str | None = None
