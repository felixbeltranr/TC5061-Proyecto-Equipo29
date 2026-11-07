from contextlib import asynccontextmanager

from fastapi import FastAPI

from mibici.api.routes import health, p1
from mibici.api.services.p1_inference import load_p1_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = {"p1": load_p1_model()}
    yield


app = FastAPI(
    title="MiBici P1 API",
    description="Predicción de demanda P1 usando el campeón de MLflow.",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(health.router, tags=["health"])
app.include_router(p1.router, prefix="/api/v1", tags=["predictions"])