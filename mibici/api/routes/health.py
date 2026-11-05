from fastapi import APIRouter, HTTPException, Request

from mibici.api.schemas.common import HealthResponse

router = APIRouter()


@router.get("/live", response_model=HealthResponse)
def liveness() -> HealthResponse:
    return HealthResponse(status="alive")


@router.get("/ready", response_model=HealthResponse)
def readiness(request: Request) -> HealthResponse:
    models = getattr(request.app.state, "models", {})
    if not isinstance(models, dict) or models.get("p1") is None:
        raise HTTPException(status_code=503, detail="P1 model is not loaded")
    return HealthResponse(status="ready")