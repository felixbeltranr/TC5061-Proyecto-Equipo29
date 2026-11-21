from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request

from mibici.api.schemas.common import ModelInfoResponse
from mibici.api.schemas.p2 import P2PredictionRequest, P2PredictionResponse
from mibici.api.services.p2_inference import predict_response

router = APIRouter()


def _estado(http_request: Request):
    state = http_request.app.state
    model = getattr(state, "models", {}).get("p2")
    if model is None:
        raise HTTPException(status_code=503, detail="P2 model is not loaded")
    info = getattr(state, "model_info", {}).get("p2")
    return model, info


@router.get("/model", response_model=ModelInfoResponse)
def model_info(http_request: Request) -> ModelInfoResponse:
    _, info = _estado(http_request)
    if info is None:
        raise HTTPException(status_code=503, detail="P2 model is not loaded")
    return ModelInfoResponse(**asdict(info))


@router.post("/predict", response_model=P2PredictionResponse)
def predict(payload: P2PredictionRequest, http_request: Request) -> P2PredictionResponse:
    model, info = _estado(http_request)
    try:
        return predict_response(model, info, payload.model_dump()["stations"])
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
