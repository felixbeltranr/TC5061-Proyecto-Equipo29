from contextlib import asynccontextmanager
import math

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger

from mibici.api.routes import health, p1, p3
from mibici.api.services.p1_inference import load_p1_model_with_info
from mibici.api.services.p3_inference import load_p3_group_names, load_p3_model_with_info


def _cargadores() -> dict:
    """Un cargador por pregunta; se leen al arrancar para que las pruebas puedan sustituirlos."""
    return {"p1": load_p1_model_with_info, "p3": load_p3_model_with_info}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cada modelo carga por separado: si uno falla, los demás se sirven y /ready reporta 503.
    app.state.models, app.state.model_info, app.state.model_errors = {}, {}, {}
    for clave, cargar in _cargadores().items():
        try:
            model, info = cargar()
        except Exception as exc:  # noqa: BLE001 - se registra y la API sigue con los demás
            logger.error(f"No se cargó el modelo de {clave}: {exc}")
            app.state.model_errors[clave] = type(exc).__name__
            continue
        app.state.models[clave] = model
        app.state.model_info[clave] = info
    p3_info = app.state.model_info.get("p3")
    app.state.group_names = {"p3": load_p3_group_names(p3_info) if p3_info else {}}
    yield


app = FastAPI(
    title="MiBici API",
    description="P1: demanda diaria por estación. P3: grupo de uso de cada estación. "
    "Ambos modelos se cargan desde el Model Registry de MLflow.",
    version="0.3.0",
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
app.include_router(p3.router, prefix="/api/v1/p3", tags=["p3"])
