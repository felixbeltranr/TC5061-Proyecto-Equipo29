"""Pruebas de la selección del campeón de P3 y su registro, con datos sintéticos."""

from dataclasses import replace

import mlflow
import numpy as np
import pandas as pd
import pytest

from mibici.config import CONFIG_P3
from mibici.features_p3 import DatasetP3
from mibici.modeling import seleccion_p3 as sel
from mibici.modeling.train_p3 import entrenar_candidata, registrar_candidata

VARIABLES = list(CONFIG_P3.prioridad_variables)
CFG = replace(CONFIG_P3, n_init=2)
CRITERIOS_OK = {"I1_tamano": True, "I2_distinguibles": True, "I3_nombrables": True}


def _perfiles(n: int = 60) -> tuple[pd.DataFrame, np.ndarray]:
    """Tres grupos: se vacía en la mañana, se llena en la mañana y mucho uso."""
    rng = np.random.default_rng(0)
    grupo = np.repeat([0, 1, 2], n // 3)
    datos = {c: rng.normal(0, 0.05, len(grupo)) for c in VARIABLES}
    datos["net_am"] = np.array([-0.5, 0.5, 0.0])[grupo] + rng.normal(0, 0.05, len(grupo))
    datos["log_trips_day"] = np.array([2.0, 2.0, 4.0])[grupo] + rng.normal(0, 0.1, len(grupo))
    return pd.DataFrame(datos), grupo


def test_nombre_grupo_usa_las_dos_variables_dominantes():
    z = pd.Series({"log_trips_day": 0.2, "net_am": -1.4, "arr_night": 0.9, "weekend_share": 0.6})
    assert sel.nombre_grupo(z, 0.5) == "se vacía en la mañana · llegadas de noche"
    assert sel.nombre_grupo(z * 0.1, 0.5) == ""  # ninguna llega al umbral


def test_criterios_cumplen_con_grupos_claros():
    perfiles, grupo = _perfiles()
    resumen, tabla = sel.criterios(perfiles, grupo, ["log_trips_day", "net_am"], CFG)
    assert resumen["cumple"]
    assert tabla["n_estaciones"].tolist() == [20, 20, 20]
    assert tabla["nombre"].is_unique


def test_criterios_detectan_grupo_pequeno_y_falta_de_rebalanceo():
    perfiles, grupo = _perfiles()
    grupo = grupo.copy()
    grupo[:2] = 3  # un grupo con 2 de 60 estaciones (3 %)
    resumen, _ = sel.criterios(perfiles, grupo, ["log_trips_day", "net_am"], CFG)
    assert not resumen["I1_tamano"] and not resumen["cumple"]

    perfiles, grupo = _perfiles()
    perfiles["net_am"] = perfiles["net_am"].abs() + 0.1  # todas se llenan en la mañana
    resumen, _ = sel.criterios(perfiles, grupo, ["log_trips_day", "net_am"], CFG)
    assert not resumen["U1_rebalanceo"]


def test_criterios_detectan_grupos_con_el_mismo_nombre():
    perfiles, _ = _perfiles()
    grupo = np.repeat([0, 1, 2], 20)
    grupo[grupo == 2] = np.array([2, 3] * 10)  # parte "mucho uso" en dos grupos casi iguales
    resumen, tabla = sel.criterios(perfiles, grupo, ["log_trips_day", "net_am"], CFG)
    assert not resumen["I3_nombrables"] and not tabla["nombre"].is_unique
    assert not resumen["I2_distinguibles"]


def _r(k, sil, cumple, model="kmeans_uso"):
    params = {"k": k} if model == "kmeans_uso" else {"n_components": k, "covariance_type": "full"}
    return {
        "model": model,
        "params": params,
        "val_silhouette_uso": sil,
        "val_ari_estabilidad": 0.5,
        "val_davies_bouldin_uso": 1.5,
        "criterios": {**CRITERIOS_OK, "U1_rebalanceo": cumple, "cumple": cumple},
    }


def test_elegir_candidatas_salta_k_base_en_c2():
    ranking_km = [_r(5, 0.2, True), _r(3, 0.19, True), _r(4, 0.18, True)]
    ranking_gmm = [_r(4, 0.15, True, "gmm_uso")]
    c = sel.elegir_candidatas(ranking_km, ranking_gmm, CFG)
    assert c["C1"]["params"]["k"] == 5 and c["C2"]["params"]["k"] == 3
    assert c["C3"]["model"] == "gmm_uso"


def test_campeon_es_la_primera_que_cumple_y_se_documenta_el_motivo():
    ranking = sel.elegir_campeon(
        {"C1": _r(5, 0.17, True), "C2": _r(3, 0.20, False), "C3": _r(4, 0.15, True, "gmm_uso")},
        CFG,
    )
    papeles = {r["candidata"]: (r["posicion"], r["papel"]) for r in ranking}
    assert papeles == {"C2": (1, ""), "C1": (2, "champion"), "C3": (3, "challenger")}
    motivos = {r["candidata"]: r["motivo"] for r in ranking}
    assert motivos["C2"] == "no cumple U1_rebalanceo"
    assert motivos["C1"] == "las anteriores del ranking no cumplen los criterios"
    tabla = sel.tabla_seleccion(ranking)
    assert list(tabla["candidata"]) == ["C2", "C1", "C3"]


def test_sin_candidatas_aptas_falla():
    with pytest.raises(ValueError, match="Ninguna candidata"):
        sel.elegir_campeon({"C1": _r(5, 0.17, False)}, CFG)


def _dataset() -> DatasetP3:
    perfiles, _ = _perfiles()
    perfiles["trips_day"] = np.exp(perfiles["log_trips_day"])
    idx = pd.Index(range(1, len(perfiles) + 1), name="estacion")
    perfiles.index = idx
    rng = np.random.default_rng(1)
    nom = pd.DataFrame(
        {
            "name": [f"est {i}" for i in idx],
            "municipio": "GDL",
            "latitude": 20.6 + rng.random(len(idx)) / 10,
            "longitude": -103.4 + rng.random(len(idx)) / 10,
        },
        index=idx,
    )
    return DatasetP3(
        bitacora=pd.DataFrame({"mes": ["2025_01"], "viajes_validos": [10]}),
        nom=nom,
        entrenamiento=perfiles,
        evaluacion=perfiles,
        variables=["log_trips_day", "net_am"],
        nuevas=pd.Index([], name="estacion"),
        data_version="prueba",
        nomenclatura="nomenclatura_prueba.csv",
    )


def _registrar_dos_candidatas(tmp_path):
    """Dos runs candidatos en un MLflow temporal; regresa (ds, run_ids, ranking)."""
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("p3-prueba")
    ds = _dataset()
    run_ids, filas = {}, []
    for clave, k in (("C1", 3), ("C2", 4)):
        modelo, m = entrenar_candidata(ds, "kmeans_uso", {"k": k}, CFG)
        run_ids[clave] = registrar_candidata(ds, "kmeans_uso", {"k": k}, modelo, m, CFG)
        filas.append({**_r(k, 0.3 - k / 100, True), "candidata": clave})
    ranking = sel.elegir_campeon({f["candidata"]: f for f in filas}, CFG)
    return ds, run_ids, ranking


def test_registro_con_aliases_y_carga_por_nombre(tmp_path):
    ds, run_ids, ranking = _registrar_dos_candidatas(tmp_path)
    ejemplo = ds.entrenamiento.head(5)
    versiones = sel.registrar_en_registry(ranking, run_ids, "p3-prueba", ejemplo, CFG)
    assert set(versiones) == {"champion", "challenger"}

    cliente = mlflow.MlflowClient()
    campeon = cliente.get_model_version_by_alias(CFG.modelo_registrado, "champion")
    assert campeon.run_id == run_ids["C1"]
    assert {"run_id", "git_commit", "data_version", "experimento", "candidata"} <= set(
        campeon.tags
    )
    assert campeon.tags["data_version"] == "prueba" and campeon.description
    assert campeon.tags["comprobacion"] == "ok"
    assert cliente.get_registered_model(CFG.modelo_registrado).description
    grupos = sel.verificar_alias(ejemplo, "champion", CFG)
    assert len(grupos) == 5


class _ModeloRoto:
    """Modelo que carga pero predice mal (un faltante y menos filas de las pedidas)."""

    def predict(self, X):
        return np.array([np.nan] * (len(X) - 1))


@pytest.mark.parametrize("falla", ["carga", "prediccion"])
def test_si_la_comprobacion_falla_se_conservan_los_aliases_anteriores(
    tmp_path, monkeypatch, falla
):
    ds, run_ids, ranking = _registrar_dos_candidatas(tmp_path)
    ejemplo = ds.entrenamiento.head(5)
    anteriores = sel.registrar_en_registry(ranking, run_ids, "p3-prueba", ejemplo, CFG)

    cargar_original = mlflow.pyfunc.load_model

    def cargar_con_falla(uri, *args, **kwargs):
        es_version_nueva = uri.startswith(f"models:/{CFG.modelo_registrado}/") and int(
            uri.rsplit("/", 1)[1]
        ) > max(int(v) for v in anteriores.values())
        if not es_version_nueva:
            return cargar_original(uri, *args, **kwargs)
        if falla == "carga":
            raise OSError("no se pudo cargar el modelo")
        return _ModeloRoto()

    monkeypatch.setattr(mlflow.pyfunc, "load_model", cargar_con_falla)
    with pytest.raises((OSError, ValueError)):
        sel.registrar_en_registry(ranking, run_ids, "p3-prueba", ejemplo, CFG)

    cliente = mlflow.MlflowClient()
    for alias, version in anteriores.items():  # los aliases siguen en las versiones anteriores
        assert cliente.get_model_version_by_alias(CFG.modelo_registrado, alias).version == version
    nuevas = [
        v
        for v in cliente.search_model_versions(f"name='{CFG.modelo_registrado}'")
        if v.version not in anteriores.values()
    ]
    assert len(nuevas) == 2 and all(v.tags.get("comprobacion") == "fallida" for v in nuevas)
    monkeypatch.setattr(mlflow.pyfunc, "load_model", cargar_original)
    assert len(sel.verificar_alias(ejemplo, "champion", CFG)) == 5  # producción sigue sirviendo


def test_comprobar_version_rechaza_grupos_invalidos(monkeypatch):
    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda uri: _ModeloRoto())
    perfiles, _ = _perfiles()
    with pytest.raises(ValueError, match="predijo"):
        sel.comprobar_version("m", "1", perfiles.head(5), CFG)


@pytest.mark.parametrize("valor", [np.inf, -np.inf, np.nan, -1, 0.5])
def test_comprobar_version_rechaza_no_finitos_negativos_y_fraccionarios(monkeypatch, valor):
    class ModeloInvalido:
        def predict(self, entrada):
            return np.full(len(entrada), valor)
    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda uri: ModeloInvalido())
    perfiles, _ = _perfiles()
    with pytest.raises(ValueError, match="inválidos"):
        sel.comprobar_version("m", "1", perfiles.head(5), CFG)


@pytest.mark.parametrize("uri", [None, "sqlite:///evidence/tracking-comun.db"])
def test_mlflow_project_seleccion_no_recibe_hiperparametros(monkeypatch, uri):
    from mibici import features_p3, mlflow_project
    from mibici.modeling import train_p3

    llamadas = {}
    monkeypatch.delenv("MIBICI_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    if uri is not None:
        monkeypatch.setenv("MIBICI_TRACKING_URI", uri)
    monkeypatch.setattr(features_p3, "main", lambda: None)
    monkeypatch.setattr(train_p3, "main", lambda **kw: llamadas.update(kw))
    mlflow_project.run_train("p3", "seleccion", {})
    assert llamadas == {
        "model": "seleccion",
        "tracking_uri": mlflow_project._project_tracking_uri(),
    }
    with pytest.raises(ValueError, match="seleccion"):
        mlflow_project.run_train("p3", "seleccion", {"k": 3})


def test_el_sentido_del_flujo_usa_el_signo_original():
    # el grupo se llena poco en la mañana (+0.05): muy debajo del promedio, pero se llena
    z = pd.Series({"net_am": -1.5, "log_trips_day": 0.1})
    original = pd.Series({"net_am": 0.05, "log_trips_day": 3.0})
    assert sel.nombre_grupo(z, 0.5, original) == "se llena en la mañana"
    assert sel.nombre_grupo(z, 0.5) == "se vacía en la mañana"  # sin original, signo de z


def test_criterios_nombran_con_el_signo_original():
    perfiles, grupo = _perfiles()
    perfiles["net_am"] = np.array([0.02, 0.6, 0.3])[grupo]  # todas se llenan en la mañana
    _, tabla = sel.criterios(perfiles, grupo, ["log_trips_day", "net_am"], CFG)
    assert not tabla["nombre"].str.contains("se vacía en la mañana").any()


def test_validacion_incompleta_no_puede_ser_champion_ni_challenger():
    c1 = {**_r(5, np.nan, True), "val_davies_bouldin_uso": np.nan}  # una sola partición válida
    with pytest.raises(ValueError, match="validación completa"):
        sel.elegir_campeon({"C1": c1, "C2": _r(3, 0.20, False)}, CFG)
    ranking = sel.elegir_campeon({"C1": c1, "C2": _r(3, 0.20, True), "C3": _r(4, 0.1, False)}, CFG)
    papeles = {r["candidata"]: r["papel"] for r in ranking}
    assert papeles == {"C2": "champion", "C1": "", "C3": ""}
    motivo_c1 = next(r["motivo"] for r in ranking if r["candidata"] == "C1")
    assert motivo_c1.startswith("validación incompleta")


def test_flujo_cero_se_nombra_equilibrado():
    z = pd.Series({"net_pm": -1.2, "log_trips_day": 0.1})
    original = pd.Series({"net_pm": 0.0, "log_trips_day": 3.0})
    assert sel.nombre_grupo(z, 0.5, original) == "equilibrada en la tarde"
