"""Cola de P2: el trabajo en lote da la misma respuesta que /predict y se consulta por su id."""

from fakeredis import FakeRedis
from fastapi.testclient import TestClient
import pytest
from rq import Queue, SimpleWorker
from tests.api.conftest import INFO_P2, FakeModelP2

from mibici.api.routes.p2 import cola_p2
from mibici.api.schemas.p2 import EJEMPLO_P2
from mibici.api.services import p2_cola


def _peticion(n):
    return {"stations": [{"station_id": str(i), "features": EJEMPLO_P2} for i in range(n)]}


@pytest.fixture
def modelo_worker(monkeypatch):
    modelo = FakeModelP2()
    cargas = []

    def cargar():
        cargas.append(1)
        return modelo, INFO_P2

    monkeypatch.setattr(p2_cola, "load_p2_model_with_info", cargar)
    p2_cola._modelo.cache_clear()
    yield modelo, cargas
    p2_cola._modelo.cache_clear()


@pytest.fixture
def cola():
    return Queue(p2_cola.COLA, connection=FakeRedis())


@pytest.fixture
def cliente(cola):
    import mibici.api.main as api_main

    api_main.app.dependency_overrides[cola_p2] = lambda: cola
    with TestClient(api_main.app) as client:
        yield client
    api_main.app.dependency_overrides.clear()


def _atender(cola):
    SimpleWorker([cola], connection=cola.connection).work(burst=True)


def test_el_trabajo_da_la_misma_respuesta_que_predict(cliente, cola, modelo_worker):
    encolado = cliente.post("/api/v1/p2/jobs", json=_peticion(3))
    assert encolado.status_code == 202
    assert encolado.json()["status"] == "queued"

    _atender(cola)
    trabajo = cliente.get(f"/api/v1/p2/jobs/{encolado.json()['job_id']}")

    assert trabajo.json()["status"] == "finished"
    assert trabajo.json()["result"] == cliente.post("/api/v1/p2/predict", json=_peticion(3)).json()


def test_el_worker_carga_el_modelo_una_sola_vez(cliente, cola, modelo_worker):
    _, cargas = modelo_worker
    for _ in range(2):
        cliente.post("/api/v1/p2/jobs", json=_peticion(1))

    _atender(cola)

    assert len(cargas) == 1


def test_un_trabajo_fallido_reporta_el_error(cliente, cola, modelo_worker):
    modelo, _ = modelo_worker
    modelo.clases = ["otra"]
    job_id = cliente.post("/api/v1/p2/jobs", json=_peticion(1)).json()["job_id"]

    _atender(cola)
    trabajo = cliente.get(f"/api/v1/p2/jobs/{job_id}").json()

    assert trabajo["status"] == "failed"
    assert "clases inválidas" in trabajo["error"]
    assert trabajo["result"] is None


def test_un_id_desconocido_responde_404(cliente):
    assert cliente.get("/api/v1/p2/jobs/no-existe").status_code == 404


def test_una_peticion_invalida_no_se_encola(cliente, cola):
    respuesta = cliente.post("/api/v1/p2/jobs", json={"stations": []})

    assert respuesta.status_code == 422
    assert cola.count == 0


def test_sin_redis_configurado_responde_503(monkeypatch):
    import mibici.api.main as api_main

    monkeypatch.delenv("REDIS_URL", raising=False)
    with TestClient(api_main.app) as client:
        respuesta = client.post("/api/v1/p2/jobs", json=_peticion(1))

    assert respuesta.status_code == 503
