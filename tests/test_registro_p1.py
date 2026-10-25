"""Pruebas del Model Registry local de P1."""

import os

import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from mibici import features as fe
from mibici import registro_p1
from mibici.config import CONFIG_P1
from mibici.modeling import train_p1 as t1


@pytest.fixture
def tracking(tmp_path):
    """Tracking y Registry en el mismo SQLite temporal; restaura ambas URI al terminar."""
    tracking_previo = mlflow.get_tracking_uri()
    registry_previo = mlflow.get_registry_uri()
    registry_env_previo = os.environ.get("MLFLOW_REGISTRY_URI")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    mlflow.create_experiment("registro-p1", artifact_location=(tmp_path / "arts").as_uri())
    mlflow.set_experiment("registro-p1")
    yield MlflowClient(tracking_uri=uri, registry_uri=uri)
    mlflow.set_tracking_uri(tracking_previo)
    if registry_env_previo is None:
        os.environ.pop("MLFLOW_REGISTRY_URI", None)
        mlflow.set_registry_uri(None)
    else:
        mlflow.set_registry_uri(registry_previo)


def entrenar_y_loguear(semilla: int = 0) -> tuple[str, str, pd.DataFrame]:
    """Crea un run pequeno de regresion y devuelve run, URI y ejemplo compatible."""
    rng = np.random.default_rng(semilla)
    x = pd.DataFrame(rng.uniform(0, 10, size=(100, len(fe.FEATURES))), columns=fe.FEATURES)
    y = rng.poisson(6, size=len(x)).astype(float)
    modelo = HistGradientBoostingRegressor(loss="poisson", max_iter=5, random_state=semilla).fit(
        x, y
    )
    ejemplo = x.head(4)
    with mlflow.start_run(run_name=f"p1-candidata-{semilla}") as run:
        mlflow.set_tags(
            {
                "git_commit": "abc1234",
                "git_dirty": "True",
                "data_version": "v-prueba",
            }
        )
        info = mlflow.sklearn.log_model(
            modelo,
            name="model",
            signature=infer_signature(ejemplo, modelo.predict(ejemplo)),
            input_example=ejemplo,
            serialization_format="skops",
            skops_trusted_types=[
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
            ],
            pip_requirements=[],
        )
    return run.info.run_id, info.model_uri, ejemplo


def promover(run_id: str, uri: str, ejemplo: pd.DataFrame):
    return registro_p1.registrar_campeon(
        run_id,
        uri,
        "hgb_poisson",
        5.61,
        {"experimento": "mibici-equipo29"},
        ejemplo,
    )


def test_registra_campeon_con_alias_descripcion_y_linaje(tracking):
    run_id, uri, ejemplo = entrenar_y_loguear()
    version, predicciones = promover(run_id, uri, ejemplo)

    assert version.name == CONFIG_P1.modelo_registrado and version.version == 1
    assert len(predicciones) == len(ejemplo) and np.isfinite(predicciones).all()
    registrado = tracking.get_registered_model(CONFIG_P1.modelo_registrado)
    assert registrado.aliases == {CONFIG_P1.alias_campeon: 1}
    assert "p1" in registrado.description.lower() and "campeon" in registrado.description.lower()
    assert {"question": "P1", "team": "29"}.items() <= registrado.tags.items()
    detalle = tracking.get_model_version(CONFIG_P1.modelo_registrado, "1")
    assert detalle.run_id == run_id and "validacion" in detalle.description
    assert {
        "run_id": run_id,
        "git_commit": "abc1234",
        "git_dirty": "True",
        "data_version": "v-prueba",
        "experimento": "mibici-equipo29",
        "criterio_seleccion": registro_p1.CRITERIO_CAMPEON,
        "candidata": "hgb_poisson",
    }.items() <= detalle.tags.items()


def test_carga_version_inmutable_y_alias_predice_demanda(tracking):
    run_id, uri, ejemplo = entrenar_y_loguear()
    version, esperada = promover(run_id, uri, ejemplo)

    por_version = registro_p1.verificar_version(
        CONFIG_P1.modelo_registrado, version.version, ejemplo
    )
    por_alias = registro_p1.verificar_campeon(ejemplo)
    assert por_version == pytest.approx(esperada)
    assert por_alias == pytest.approx(esperada)


def test_si_falla_version_nueva_conserva_champion_anterior(tracking, monkeypatch):
    run_1, uri_1, ejemplo_1 = entrenar_y_loguear(1)
    promover(run_1, uri_1, ejemplo_1)
    run_2, uri_2, ejemplo_2 = entrenar_y_loguear(2)

    original = mlflow.pyfunc.load_model

    def falla_segunda(uri):
        if uri.endswith("/2"):
            raise RuntimeError("fallo de carga simulado")
        return original(uri)

    monkeypatch.setattr(mlflow.pyfunc, "load_model", falla_segunda)
    with pytest.raises(RuntimeError, match="fallo de carga simulado"):
        promover(run_2, uri_2, ejemplo_2)

    registrado = tracking.get_registered_model(CONFIG_P1.modelo_registrado)
    assert registrado.aliases == {CONFIG_P1.alias_campeon: 1}
    assert (
        tracking.get_model_version_by_alias(
            CONFIG_P1.modelo_registrado, CONFIG_P1.alias_campeon
        ).run_id
        == run_1
    )
    versiones = tracking.search_model_versions(f"name='{CONFIG_P1.modelo_registrado}'")
    assert [int(v.version) for v in versiones] == [2, 1]
    assert tracking.get_model_version(CONFIG_P1.modelo_registrado, "2").run_id == run_2


def test_prediccion_negativa_de_version_nueva_conserva_champion_anterior(
    tracking, monkeypatch
):
    run_1, uri_1, ejemplo_1 = entrenar_y_loguear(3)
    promover(run_1, uri_1, ejemplo_1)
    run_2, uri_2, ejemplo_2 = entrenar_y_loguear(4)
    original = mlflow.pyfunc.load_model

    class ModeloNegativo:
        def predict(self, ejemplo):
            return np.full(len(ejemplo), -1.0)

    def cargar(uri):
        return ModeloNegativo() if uri.endswith("/2") else original(uri)

    monkeypatch.setattr(mlflow.pyfunc, "load_model", cargar)
    with pytest.raises(ValueError, match="demanda negativa"):
        promover(run_2, uri_2, ejemplo_2)

    registrado = tracking.get_registered_model(CONFIG_P1.modelo_registrado)
    assert registrado.aliases == {CONFIG_P1.alias_campeon: 1}
    assert tracking.get_model_version_by_alias(
        CONFIG_P1.modelo_registrado, CONFIG_P1.alias_campeon
    ).run_id == run_1
    versiones = tracking.search_model_versions(f"name='{CONFIG_P1.modelo_registrado}'")
    assert [int(v.version) for v in versiones] == [2, 1]


def diagnosticos_elegibles(valor: bool) -> dict:
    return {
        candidato: {
            "eligible": valor,
            "reason": "eligible" if valor else "predicciones_negativas_validacion",
            "prediction_min": 0.0 if valor else -1.0,
            "negative_n": 0 if valor else 1,
            "nonfinite_n": 0,
            "n": 4,
        }
        for candidato in t1.CANDIDATOS
    }


def test_ejecucion_individual_no_mueve_champion_existente(tracking):
    run_1, uri_1, ejemplo_1 = entrenar_y_loguear(5)
    promover(run_1, uri_1, ejemplo_1)
    run_2, uri_2, ejemplo_2 = entrenar_y_loguear(6)

    resultado = t1.resolver_registro(
        "hgb_poisson",
        False,
        "hgb_poisson",
        {"hgb_poisson": {"val_mae": 1.0}},
        {"hgb_poisson": {"run_id": run_2, "model_uri": uri_2}},
        diagnosticos_elegibles(True),
        {"train": ejemplo_2},
        29,
    )

    assert resultado["reason"] == "promotion_requires_model_auto"
    assert tracking.get_model_version_by_alias(
        CONFIG_P1.modelo_registrado, CONFIG_P1.alias_campeon
    ).run_id == run_1
    assert len(tracking.search_model_versions(f"name='{CONFIG_P1.modelo_registrado}'")) == 1


def test_candidata_invalida_no_se_registra_y_conserva_champion(tracking):
    run_1, uri_1, ejemplo_1 = entrenar_y_loguear(7)
    promover(run_1, uri_1, ejemplo_1)
    run_2, uri_2, ejemplo_2 = entrenar_y_loguear(8)
    elegibilidad = diagnosticos_elegibles(True)
    elegibilidad["hgb_absoluto"] = {
        "eligible": False,
        "reason": "predicciones_negativas_validacion",
        "prediction_min": -2.0,
        "negative_n": 2,
        "nonfinite_n": 0,
        "n": 4,
    }

    resultado = t1.resolver_registro(
        "auto",
        False,
        "hgb_absoluto",
        {"hgb_absoluto": {"val_mae": 0.5}},
        {"hgb_absoluto": {"run_id": run_2, "model_uri": uri_2}},
        elegibilidad,
        {"train": ejemplo_2},
        29,
    )

    assert resultado["reason"] == "selected_candidate_ineligible_nonnegative_domain"
    assert tracking.get_model_version_by_alias(
        CONFIG_P1.modelo_registrado, CONFIG_P1.alias_campeon
    ).run_id == run_1
    assert len(tracking.search_model_versions(f"name='{CONFIG_P1.modelo_registrado}'")) == 1


def test_sin_candidatas_elegibles_conserva_champion(tracking):
    run_1, uri_1, ejemplo_1 = entrenar_y_loguear(9)
    promover(run_1, uri_1, ejemplo_1)

    resultado = t1.resolver_registro(
        "auto",
        False,
        "semana_previa",
        {},
        {},
        diagnosticos_elegibles(False),
        {},
        29,
    )

    assert resultado["reason"] == "no_eligible_candidate_nonnegative_domain"
    assert tracking.get_model_version_by_alias(
        CONFIG_P1.modelo_registrado, CONFIG_P1.alias_campeon
    ).run_id == run_1
    assert len(tracking.search_model_versions(f"name='{CONFIG_P1.modelo_registrado}'")) == 1
