import json
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
import pytest

from mibici import tracking


@pytest.fixture(autouse=True)
def restaurar_estado_mlflow():
    """Cada prueba deja tracking y Registry como estaban al comenzar."""
    tracking_previo = mlflow.get_tracking_uri()
    registry_previo = mlflow.get_registry_uri()
    registry_env_previo = os.environ.get("MLFLOW_REGISTRY_URI")
    yield
    mlflow.set_tracking_uri(tracking_previo)
    if registry_env_previo is None:
        os.environ.pop("MLFLOW_REGISTRY_URI", None)
        mlflow.set_registry_uri(None)
    else:
        mlflow.set_registry_uri(registry_previo)


def test_version_datos_lee_el_manifiesto(tmp_path):
    ruta = tmp_path / "manifiesto.json"
    ruta.write_text(json.dumps({"version": "66dd3d190bba", "archivos": []}))
    assert tracking.version_datos(ruta) == "66dd3d190bba"


def test_version_datos_falla_si_no_hay_manifiesto(tmp_path):
    with pytest.raises(FileNotFoundError, match="--manifiesto"):
        tracking.version_datos(tmp_path / "no_existe.json")


def test_configurar_acepta_carpeta_de_artefactos_relativa(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    tracking_previo = mlflow.get_tracking_uri()
    registry_previo = mlflow.get_registry_uri()
    registry_env_previo = os.environ.get("MLFLOW_REGISTRY_URI")
    try:
        tracking.configurar(uri, Path("mlartifacts"))
        experimento = MlflowClient(tracking_uri=uri).get_experiment_by_name(tracking.EXPERIMENTO)
        assert experimento.artifact_location == (tmp_path / "mlartifacts").resolve().as_uri()
        assert mlflow.get_tracking_uri() == uri
        assert mlflow.get_registry_uri() == uri
    finally:
        mlflow.set_tracking_uri(tracking_previo)
        if registry_env_previo is None:
            os.environ.pop("MLFLOW_REGISTRY_URI", None)
            mlflow.set_registry_uri(None)
        else:
            mlflow.set_registry_uri(registry_previo)


def test_etiquetas_base_siguen_la_convencion_del_equipo():
    tags = tracking.etiquetas_base("P1", "Alguien", 29, "2025-01..2026-09", "abc")
    assert set(tags) == {
        "question", "team", "author", "data_range", "seed", "data_version",
        "mlflow.source.git.commit", "git_commit", "git_dirty",
    }  # fmt: skip
    assert tags["question"] == "P1" and tags["team"] == "29" and tags["seed"] == "29"
    assert tags["git_dirty"] in {"True", "False"}


def test_git_commit_es_un_sha_o_desconocido():
    sha, sucio = tracking.git_commit()
    assert sha == "desconocido" or len(sha) == 40
    assert isinstance(sucio, bool)


def test_experimento_y_etiquetas_iguales_a_los_de_p2():
    p2 = pytest.importorskip("mibici.modeling.train_p2")
    assert tracking.EXPERIMENTO == p2.EXPERIMENTO


def test_registrar_crea_el_run_con_todo(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    tracking.configurar(uri, tmp_path / "mlartifacts")
    artefacto = tmp_path / "salida.txt"
    artefacto.write_text("hola")
    run_id = tracking.registrar(
        "p1_prueba", {"question": "P1"}, {"semilla": 29}, {"test_mae": 1.5}, [artefacto]
    )
    cliente = MlflowClient(tracking_uri=uri)
    run = cliente.get_run(run_id)
    assert run.info.run_name == "p1_prueba"
    assert run.data.tags["question"] == "P1" and run.data.params["semilla"] == "29"
    assert run.data.metrics["test_mae"] == 1.5
    assert [a.path for a in cliente.list_artifacts(run_id)] == ["salida.txt"]
    assert cliente.get_experiment(run.info.experiment_id).name == tracking.EXPERIMENTO
