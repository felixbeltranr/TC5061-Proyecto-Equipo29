from fastapi import APIRouter, Request

from mibici.api.schemas.common import PredictionResponse
from mibici.api.schemas.p1 import PredictionRequest
from mibici.api.services.p1_inference import predict_with_model

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, http_request: Request) -> PredictionResponse:
    prediction = predict_with_model(
        http_request.app.state.models["p1"],
        payload.features.model_dump(),
    )
    return PredictionResponse(
        prediction=prediction,
        method="mlflow_registry",
        input_count=1,
    )