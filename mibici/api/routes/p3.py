from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request

from mibici.api.schemas.common import ModelInfoResponse
from mibici.api.schemas.p3 import P3Assignment, P3PredictionRequest, P3PredictionResponse
from mibici.api.services.p3_inference import assign_groups

router = APIRouter()


def _estado(http_request: Request):
    state = http_request.app.state
    model = getattr(state, "models", {}).get("p3")
    if model is None:
        raise HTTPException(status_code=503, detail="P3 model is not loaded")
    info = getattr(state, "model_info", {}).get("p3")
    names = getattr(state, "group_names", {}).get("p3", {})
    return model, info, names


@router.get("/model", response_model=ModelInfoResponse)
def model_info(http_request: Request) -> ModelInfoResponse:
    _, info, _ = _estado(http_request)
    if info is None:
        raise HTTPException(status_code=503, detail="P3 model is not loaded")
    return ModelInfoResponse(**asdict(info))


@router.post("/predict", response_model=P3PredictionResponse)
def predict(payload: P3PredictionRequest, http_request: Request) -> P3PredictionResponse:
    model, info, names = _estado(http_request)
    try:
        groups = assign_groups(model, [s.features.model_dump() for s in payload.stations])
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return P3PredictionResponse(
        assignments=[
            P3Assignment(station_id=s.station_id, group=g, group_name=names.get(g))
            for s, g in zip(payload.stations, groups, strict=True)
        ],
        method="mlflow_registry",
        input_count=len(groups),
        model_name=info.name if info else None,
        model_version=info.version if info else None,
    )
