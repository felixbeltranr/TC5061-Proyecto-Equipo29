import os

import pytest

from mibici import features
from mibici import mlflow_project as mp
from mibici.modeling import train_p1


def test_run_p1_propaga_el_tracking_uri_del_proyecto(monkeypatch):
    llamadas = []
    monkeypatch.setattr(features, "main", lambda: llamadas.append(("features", None)))
    monkeypatch.setattr(train_p1, "main", lambda argv=None: llamadas.append(("train_p1", argv)))
    uri = "sqlite:///evidence/p1-unificado.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    mp._run_p1("hgb_poisson", {"max_iter": 30})

    assert [nombre for nombre, _ in llamadas] == ["features", "train_p1"]
    assert llamadas[1][1] == [
        "--model",
        "hgb_poisson",
        "--hyperparameters",
        '{"max_iter":30}',
        "--tracking-uri",
        uri,
    ]


def test_run_p1_directo_conserva_el_tracking_uri_por_defecto(monkeypatch):
    visto = []
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(features, "main", lambda: None)
    monkeypatch.setattr(train_p1, "main", lambda argv=None: visto.append(argv))

    mp._run_p1("auto", {})

    assert visto == [["--model", "auto", "--hyperparameters", "{}"]]


def test_run_p1_acepta_los_candidatos_de_train_p1_y_rechaza_otros(monkeypatch):
    vistos = []
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(features, "main", lambda: None)
    monkeypatch.setattr(train_p1, "main", lambda argv=None: vistos.append(argv[1]))

    for modelo in ("auto", *train_p1.CANDIDATOS):
        mp._run_p1(modelo, {})

    assert vistos == ["auto", "poisson", "hgb_poisson", "hgb_absoluto"]
    with pytest.raises(ValueError, match="P1 model"):
        mp._run_p1("random_forest", {})


def test_run_train_p1_quita_run_exterior_y_lo_restaura_aun_si_falla(monkeypatch):
    observado = []
    monkeypatch.setenv("MLFLOW_RUN_ID", "run_exterior")
    uri = "http://localhost:5000"
    monkeypatch.setenv("MIBICI_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    def falla(model, hyperparameters):
        observado.append((os.environ.get("MLFLOW_RUN_ID"), os.environ.get("MLFLOW_TRACKING_URI")))
        raise RuntimeError("fallo simulado")

    monkeypatch.setattr(mp, "_run_p1", falla)
    with pytest.raises(RuntimeError, match="fallo simulado"):
        mp.run_train("p1")

    assert observado == [(None, uri)]
    assert os.environ["MLFLOW_RUN_ID"] == "run_exterior"
