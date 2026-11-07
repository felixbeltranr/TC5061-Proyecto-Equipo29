from fastapi import APIRouter, HTTPException, Request

from mibici.api.schemas.common import HealthResponse

router = APIRouter()


@router.get("/live", response_model=HealthResponse)
def liveness() -> HealthResponse:
    return HealthResponse(status="alive")


@router.get("/ready", response_model=HealthResponse)
def readiness(request: Request) -> HealthResponse:
    """Lista sólo si cargaron todos los modelos; si falta alguno, 503 con cuál falta."""
    models = getattr(request.app.state, "models", {})
    errors = getattr(request.app.state, "model_errors", {})
    if not isinstance(models, dict) or not models:
        raise HTTPException(status_code=503, detail="No model is loaded")
    if errors:
        faltantes = ", ".join(f"{clave} ({error})" for clave, error in sorted(errors.items()))
        raise HTTPException(status_code=503, detail=f"Models not loaded: {faltantes}")
    return HealthResponse(status="ready")
