from dataclasses import asdict
from functools import cache
import os
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from redis import Redis
from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job, JobStatus

from mibici.api.schemas.common import ModelInfoResponse
from mibici.api.schemas.p2 import P2JobResponse, P2PredictionRequest, P2PredictionResponse
from mibici.api.services.p2_cola import COLA, predecir_lote
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


@cache
def _redis(url: str) -> Redis:
    return Redis.from_url(url)


def cola_p2() -> Queue:
    url = os.environ.get("REDIS_URL")
    if not url:
        raise HTTPException(status_code=503, detail="P2 queue is not configured")
    return Queue(COLA, connection=_redis(url))


ColaP2 = Annotated[Queue, Depends(cola_p2)]


@router.post("/jobs", status_code=202, response_model=P2JobResponse)
def enqueue(payload: P2PredictionRequest, cola: ColaP2) -> P2JobResponse:
    job = cola.enqueue(predecir_lote, payload.model_dump()["stations"])
    return P2JobResponse(job_id=job.id, status=job.get_status())


@router.get("/jobs/{job_id}", response_model=P2JobResponse)
def job_result(job_id: str, cola: ColaP2) -> P2JobResponse:
    try:
        job = Job.fetch(job_id, connection=cola.connection)
    except NoSuchJobError as exc:
        raise HTTPException(status_code=404, detail="P2 job not found") from exc
    status = job.get_status()
    result = job.return_value() if status == JobStatus.FINISHED else None
    error = None
    if status == JobStatus.FAILED:
        error = job.latest_result().exc_string.strip().splitlines()[-1]
    return P2JobResponse(job_id=job.id, status=status, result=result, error=error)
