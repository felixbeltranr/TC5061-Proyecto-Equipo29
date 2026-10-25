"""Dispatch contract for the integrated MLflow Project, without model training."""
import os

import pytest

from mibici import mlflow_project as mp


@pytest.mark.parametrize("uri", [None, "sqlite:///evidence/common.db"])
def test_all_runs_selection_and_uses_one_tracking_uri(monkeypatch, uri):
    calls = []
    monkeypatch.delenv("MIBICI_TRACKING_URI", raising=False)
    if uri is not None:
        monkeypatch.setenv("MIBICI_TRACKING_URI", uri)
    expected_uri = mp._project_tracking_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", expected_uri)
    monkeypatch.setenv("MLFLOW_RUN_ID", "outer-run")
    for number in (1, 2, 3):
        def capture(model, params, number=number):
            calls.append((number, model, params, os.environ.get("MLFLOW_TRACKING_URI")))
            assert "MLFLOW_RUN_ID" not in os.environ
        monkeypatch.setattr(mp, f"_run_p{number}", capture)

    mp.run_train("all")

    assert calls == [
        (1, "auto", {}, expected_uri),
        (2, mp.CONFIG_P2.model, {}, expected_uri),
        (3, "seleccion", {}, expected_uri),
    ]
    assert os.environ.get("MLFLOW_TRACKING_URI") == expected_uri
    assert os.environ["MLFLOW_RUN_ID"] == "outer-run"


@pytest.mark.parametrize("kwargs", [{"model": "auto"}, {"hyperparameters": {"k": 3}}])
def test_all_rejects_shared_options_before_training(monkeypatch, kwargs):
    def unexpected(*args):
        raise AssertionError("Training must not start for ambiguous all options")
    for number in (1, 2, 3):
        monkeypatch.setattr(mp, f"_run_p{number}", unexpected)
    with pytest.raises(ValueError, match="question=all"):
        mp.run_train("all", **kwargs)


def test_dispatch_restores_environment_after_failure(monkeypatch):
    monkeypatch.delenv("MIBICI_TRACKING_URI", raising=False)
    uri = mp._project_tracking_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_RUN_ID", "outer-run")
    def fail(*args):
        raise RuntimeError("training failed")
    monkeypatch.setattr(mp, "_run_p1", fail)
    with pytest.raises(RuntimeError, match="training failed"):
        mp.run_train("all")
    assert os.environ["MLFLOW_TRACKING_URI"] == uri
    assert os.environ["MLFLOW_RUN_ID"] == "outer-run"


def test_default_uri_is_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("MIBICI_TRACKING_URI", raising=False)
    monkeypatch.chdir(tmp_path)
    assert mp._project_tracking_uri() == f"sqlite:///{(mp.PROJ_ROOT / 'mlflow.db').resolve().as_posix()}"


def test_project_override_resolves_relative_sqlite_against_project(tmp_path, monkeypatch):
    monkeypatch.setenv("MIBICI_TRACKING_URI", "sqlite:///evidence/custom.db")
    monkeypatch.chdir(tmp_path)
    assert mp._project_tracking_uri() == f"sqlite:///{(mp.PROJ_ROOT / 'evidence/custom.db').resolve().as_posix()}"


def test_direct_run_ignores_incidental_mlflow_uri(monkeypatch):
    monkeypatch.setenv("MIBICI_TRACKING_URI", "http://localhost:6000")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    seen = []
    monkeypatch.setattr(mp, "_run_p1", lambda *args: seen.append(os.environ["MLFLOW_TRACKING_URI"]))
    mp.run_train("p1")
    assert seen == ["http://localhost:6000"]
    assert os.environ["MLFLOW_TRACKING_URI"] == "http://localhost:5000"


@pytest.mark.parametrize("injected_uri", ["http://localhost:5001", "sqlite:///mlflow.db"])
def test_outer_mismatch_fails_before_training_or_environment_changes(monkeypatch, injected_uri):
    monkeypatch.delenv("MIBICI_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", injected_uri)
    monkeypatch.setenv("MLFLOW_RUN_ID", "outer-run")
    before = dict(os.environ)
    def unexpected(*args):
        raise AssertionError("Training must not start on another outer backend")
    monkeypatch.setattr(mp, "_run_p1", unexpected)
    with pytest.raises(ValueError, match="Antes de mlflow run"):
        mp.run_train("p1")
    assert dict(os.environ) == before


def test_ingest_creates_manifest_after_data_download(monkeypatch):
    from mibici import dataset, manifiesto_p2
    calls = []
    monkeypatch.setattr(dataset, "main", lambda: calls.append("data"))
    monkeypatch.setattr(manifiesto_p2, "main", lambda: calls.append("manifest"))
    mp.main(["ingest"])
    assert calls == ["data", "manifest"]
