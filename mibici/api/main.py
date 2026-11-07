from contextlib import asynccontextmanager
import math

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from mibici.api.routes import health, p1
from mibici.api.services.p1_inference import load_p1_model_with_info


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, info = load_p1_model_with_info()
    app.state.models = {"p1": model}
    app.state.model_info = {"p1": info}
    yield


app = FastAPI(
    title="MiBici P1 API",
    description="Predicción de demanda P1 usando el campeón de MLflow.",
    version="0.2.0",
    lifespan=lifespan,
)


def _json_seguro(valor):
    """NaN e infinito no caben en JSON: se devuelven como texto en el detalle del error."""
    if isinstance(valor, float) and not math.isfinite(valor):
        return str(valor)
    if isinstance(valor, dict):
        return {clave: _json_seguro(v) for clave, v in valor.items()}
    if isinstance(valor, list):
        return [_json_seguro(v) for v in valor]
    return valor


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # El manejador por omisión repite la entrada; con NaN o infinito falla al serializar y
    # respondería 500 en lugar de 422.
    return JSONResponse(
        status_code=422, content={"detail": _json_seguro(jsonable_encoder(exc.errors()))}
    )


app.include_router(health.router, tags=["health"])
app.include_router(p1.router, prefix="/api/v1", tags=["predictions"])
