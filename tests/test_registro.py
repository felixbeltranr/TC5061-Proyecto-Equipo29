"""Pruebas de la seleccion y el registro del campeon de P2 en el Model Registry de MLflow."""

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from mibici import features_p2 as fp
from mibici import registro
from mibici.modeling import train_p2 as t2

NOMBRE = "mibici-equipo29-p2-riesgo"


def val(macro_f1, rec_sb, rec_sa, pr_sb, pr_sa, conjunto="val"):
    """Metricas de una candidata con los nombres que registra train_p2."""
    return {
        f"{conjunto}_macro_f1": macro_f1,
        f"{conjunto}_recall_sin_bicicletas": rec_sb,
        f"{conjunto}_recall_sin_anclajes": rec_sa,
        f"{conjunto}_pr_auc_sin_bicicletas": pr_sb,
        f"{conjunto}_pr_auc_sin_anclajes": pr_sa,
    }


# Metricas de validacion de la corrida real con nivel inicial 0.5 y 0.3 (macro F1, recall y PR AUC
# de sin_bicicletas y de sin_anclajes). hgb_sin_pesos tiene el macro F1 mas alto pero un recall de
# riesgo mucho menor.
NIVEL_050 = {
    "logistica": val(0.740091, 0.944160, 0.956184, 0.767750, 0.805865),
    "hgb": val(0.756155, 0.944261, 0.959248, 0.836131, 0.858906),
    "random_forest": val(0.755679, 0.932611, 0.951128, 0.810983, 0.836062),
    "hgb_sin_pesos": val(0.831868, 0.714106, 0.764066, 0.836896, 0.860166),
}
NIVEL_030 = {
    "logistica": val(0.749842, 0.906430, 0.969674, 0.804374, 0.798287),
    "hgb": val(0.777795, 0.908237, 0.964694, 0.866637, 0.857993),
    "random_forest": val(0.771312, 0.890119, 0.961949, 0.852995, 0.827609),
}
BALANCEADAS = ("logistica", "hgb", "random_forest")


@pytest.fixture
def tracking(tmp_path):
    """Tracking en SQLite dentro de tmp_path; restaura la direccion global de MLflow al terminar."""
    previa = mlflow.get_tracking_uri()
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("registro", artifact_location=(tmp_path / "arts").as_uri())
    mlflow.set_experiment("registro")
    yield MlflowClient(tracking_uri=uri)
    mlflow.set_tracking_uri(previa)


def entrenar_y_loguear(semilla: int = 0) -> tuple[str, str]:
    """Run con un HGB chico sobre las features reales; regresa (run_id, model_uri)."""
    rng = np.random.default_rng(semilla)
    x = pd.DataFrame(rng.normal(size=(120, len(fp.FEATURES))), columns=fp.FEATURES)
    y = np.array(fp.CLASES)[rng.integers(0, 3, 120)]
    modelo = HistGradientBoostingClassifier(max_iter=5, random_state=semilla).fit(x, y)
    with mlflow.start_run(run_name="candidata") as run:
        mlflow.set_tags({"git_commit": "abc1234", "data_version": "v-prueba"})
        info = t2.registrar_modelo(modelo, x.head(5))
    return run.info.run_id, info.model_uri


# seleccionar_campeon


def test_elige_hgb_por_recall_de_riesgo_en_el_nivel_de_referencia():
    campeon = registro.seleccionar_campeon(NIVEL_050, BALANCEADAS)
    assert campeon["clave"] == "hgb"
    assert campeon["recall_medio"] == pytest.approx((0.944261 + 0.959248) / 2)
    assert campeon["pr_auc_medio"] == pytest.approx((0.836131 + 0.858906) / 2)
    assert [r["clave"] for r in campeon["ranking"]] == ["hgb", "logistica", "random_forest"]
    assert campeon["conjunto"] == "val"


def test_ignora_las_candidatas_fuera_de_la_lista_aunque_ganen_en_macro_f1():
    # hgb_sin_pesos tiene el mayor macro F1 de validacion pero no es una candidata balanceada
    assert max(NIVEL_050, key=lambda k: NIVEL_050[k]["val_macro_f1"]) == "hgb_sin_pesos"
    assert registro.seleccionar_campeon(NIVEL_050, BALANCEADAS)["clave"] == "hgb"
    claves = [r["clave"] for r in registro.seleccionar_campeon(NIVEL_050, BALANCEADAS)["ranking"]]
    assert "hgb_sin_pesos" not in claves


def test_el_macro_f1_no_decide_la_eleccion():
    resultados = {k: dict(v) for k, v in NIVEL_050.items()}
    resultados["random_forest"]["val_macro_f1"] = 0.99  # mucho mejor macro F1, menor recall
    assert registro.seleccionar_campeon(resultados, BALANCEADAS)["clave"] == "hgb"


def test_con_recall_casi_igual_decide_el_pr_auc():
    # en el nivel 0.3 la logistica supera al HGB por 0.0016 de recall medio: sin margen de empate
    # gana la logistica; con el margen por omision es un empate y gana el HGB por PR AUC
    exacto = registro.seleccionar_campeon(NIVEL_030, BALANCEADAS, empate=0.0)
    assert exacto["clave"] == "logistica"
    con_margen = registro.seleccionar_campeon(NIVEL_030, BALANCEADAS)
    assert con_margen["clave"] == "hgb"
    assert registro.EMPATE_RECALL == 0.005


def test_un_recall_claramente_mayor_gana_aunque_el_pr_auc_sea_peor():
    resultados = {k: dict(v) for k, v in NIVEL_050.items()}
    resultados["logistica"]["val_recall_sin_bicicletas"] = 0.99  # +0.02 de recall medio
    resultados["logistica"]["val_recall_sin_anclajes"] = 0.99
    assert registro.seleccionar_campeon(resultados, BALANCEADAS)["clave"] == "logistica"


def test_usa_el_conjunto_pedido_y_valida_las_entradas():
    resultados = {
        "a": val(0.7, 0.9, 0.9, 0.5, 0.5, conjunto="test"),
        "b": val(0.6, 0.95, 0.95, 0.4, 0.4, conjunto="test"),
    }
    assert registro.seleccionar_campeon(resultados, ("a", "b"), conjunto="test")["clave"] == "b"
    with pytest.raises(ValueError, match="sin candidatas"):
        registro.seleccionar_campeon(resultados, ("c",), conjunto="test")
    with pytest.raises(ValueError, match="val_recall_sin_bicicletas"):
        registro.seleccionar_campeon(resultados, ("a",))


# registrar_campeon y carga por alias


def test_registra_el_campeon_con_alias_descripciones_y_linaje(tracking):
    run_id, uri = entrenar_y_loguear()
    campeon = registro.seleccionar_campeon(NIVEL_050, BALANCEADAS)
    version = registro.registrar_campeon(
        NOMBRE, "champion", run_id, uri, campeon, {"nivel_inicial": "0.5", "experimento": "x"}
    )
    assert version.version == 1 and version.name == NOMBRE
    modelo = tracking.get_registered_model(NOMBRE)
    assert modelo.aliases == {"champion": 1}
    assert "campeon" in modelo.description.lower() and "p2" in modelo.description.lower()
    assert {"question": "P2", "team": "29"}.items() <= modelo.tags.items()
    v = tracking.get_model_version(NOMBRE, "1")
    assert v.run_id == run_id and "recall" in v.description
    assert {
        "run_id": run_id,
        "git_commit": "abc1234",
        "data_version": "v-prueba",
        "nivel_inicial": "0.5",
        "experimento": "x",
        "criterio_seleccion": registro.CRITERIO_CAMPEON,
        "candidata": "hgb",
    }.items() <= v.tags.items()
    assert len(tracking.search_registered_models()) == 1  # una sola pregunta, un solo modelo


def test_carga_el_campeon_por_alias_y_predice_un_ejemplo(tracking):
    run_id, uri = entrenar_y_loguear()
    registro.registrar_campeon(
        NOMBRE, "champion", run_id, uri, registro.seleccionar_campeon(NIVEL_050, BALANCEADAS), {}
    )
    ejemplo = pd.DataFrame(np.zeros((3, len(fp.FEATURES))), columns=fp.FEATURES)
    clases = registro.verificar_campeon(NOMBRE, "champion", ejemplo)
    assert len(clases) == 3 and set(clases) <= set(fp.CLASES)
    modelo = mlflow.pyfunc.load_model(f"models:/{NOMBRE}@champion")
    assert list(modelo.predict(ejemplo)) == clases


def test_una_nueva_version_mueve_el_alias_sin_crear_otro_modelo(tracking):
    campeon = registro.seleccionar_campeon(NIVEL_050, BALANCEADAS)
    primero, uri1 = entrenar_y_loguear(1)
    registro.registrar_campeon(NOMBRE, "champion", primero, uri1, campeon, {})
    segundo, uri2 = entrenar_y_loguear(2)
    version = registro.registrar_campeon(NOMBRE, "champion", segundo, uri2, campeon, {})
    assert version.version == 2
    assert tracking.get_registered_model(NOMBRE).aliases == {"champion": 2}
    assert tracking.get_model_version_by_alias(NOMBRE, "champion").run_id == segundo
    assert len(tracking.search_registered_models()) == 1
    assert len(tracking.search_model_versions(f"name='{NOMBRE}'")) == 2
