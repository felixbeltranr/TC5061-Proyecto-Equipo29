from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request

from mibici.api.schemas.common import ModelInfoResponse, PredictionResponse
from mibici.api.schemas.p1 import PredictionRequest
from mibici.api.services.p1_inference import predict_with_model

router = APIRouter()


def _p1_info(http_request: Request):
    return getattr(http_request.app.state, "model_info", {}).get("p1")


@router.get("/model", response_model=ModelInfoResponse)
def model_info(http_request: Request) -> ModelInfoResponse:
    info = _p1_info(http_request)
    if info is None:
        raise HTTPException(status_code=503, detail="P1 model is not loaded")
    return ModelInfoResponse(**asdict(info))


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, http_request: Request) -> PredictionResponse:
    prediction = predict_with_model(
        http_request.app.state.models["p1"],
        payload.features.model_dump(),
    )
    info = _p1_info(http_request)
    return PredictionResponse(
        prediction=prediction,
        method="mlflow_registry",
        input_count=1,
        model_name=info.name if info else None,
        model_version=info.version if info else None,
    )
