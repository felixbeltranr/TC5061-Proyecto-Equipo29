"""Pruebas del A/B de P1 con viajes sintéticos pequeños (sin datos reales ni red)."""

from dataclasses import replace
import json
import os

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from mibici import features as fe
from mibici import tracking
from mibici.config import CONFIG_P1
from mibici.modeling import train_p1
from mibici.monitoring import p1_ab
from mibici.monitoring.p1_drift import file_sha256

INICIO, FIN = pd.Timestamp("2026-10-01"), pd.Timestamp("2026-11-01")
# estación -> (primer día con viajes, salidas medias por día); la 4 abre después de abril
ESTACIONES = {
    1: ("2025-01-01", 6.0),
    2: ("2025-01-01", 12.0),
    3: ("2025-01-01", 3.0),
    4: ("2026-06-01", 5.0),
}


def _viajes(desde: str, hasta: str, semilla: int) -> pd.DataFrame:
    """Viajes con al menos una salida diaria por estación y patrón semanal."""
    rng = np.random.default_rng(semilla)
    partes = []
    for estacion, (primer_dia, media) in ESTACIONES.items():
        dias = pd.date_range(max(pd.Timestamp(desde), pd.Timestamp(primer_dia)), hasta, freq="D")
        if dias.empty:
            continue
        conteo = rng.poisson(media * np.where(dias.dayofweek < 5, 1.3, 0.6)) + 1
        minutos = pd.to_timedelta(rng.integers(6 * 60, 22 * 60, conteo.sum()), unit="min")
        partes.append(
            pd.DataFrame(
                {"Inicio_del_viaje": dias.repeat(conteo) + minutos, "Origen_Id": estacion}
            )
        )
    return pd.concat(partes, ignore_index=True)


@pytest.fixture(scope="module")
def archivos(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("viajes")
    historico, octubre = carpeta / "historico.parquet", carpeta / "octubre.csv"
    _viajes("2025-01-01", "2026-09-30", 1).to_parquet(historico, index=False)
    _viajes("2026-10-01", "2026-10-31", 2).to_csv(octubre, index=False)
    return historico, octubre


@pytest.fixture(scope="module")
def particion(archivos):
    panel, _ = p1_ab.construir_panel(*archivos, INICIO, FIN)
    return p1_ab.particionar(fe.construir_features(panel), INICIO, FIN)


def _guardar(modelo, carpeta):
    mlflow.sklearn.save_model(
        modelo,
        str(carpeta),
        serialization_format="skops",
        skops_trusted_types=train_p1.tipos_confiables(modelo),
    )
    return carpeta


@pytest.fixture(scope="module")
def campeon_local(particion, tmp_path_factory):
    """Carpeta de MLflow con un A entrenado como el campeón (ventana y parámetros de A)."""
    train_a = particion[0]
    modelo = train_p1.ajustar_candidato(p1_ab.MODELO, train_a, p1_ab.PARAMS_CAMPEON, 29)
    return _guardar(modelo, tmp_path_factory.mktemp("campeon") / "modelo")


# --- Métricas, bootstrap y regla -------------------------------------------------------------


def test_metricas_con_valores_conocidos():
    m = p1_ab.metricas(np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 2.0, 2.0, 6.0]))

    assert m == pytest.approx(
        {"mae": 1.0, "rmse": np.sqrt(1.5), "wape": 0.4, "sesgo": 0.5, "n": 4}
    )


def _errores(n_estaciones: int = 6, filas_por_estacion: int = 5, semilla: int = 0):
    rng = np.random.default_rng(semilla)
    n = n_estaciones * filas_por_estacion
    return pd.DataFrame(
        {
            "estacion": np.repeat(np.arange(n_estaciones), filas_por_estacion),
            "fecha": np.tile(
                pd.date_range("2026-10-01", periods=filas_por_estacion), n_estaciones
            ),
            "abs_a": rng.uniform(0, 10, n),
        }
    )


def test_bootstrap_con_a_igual_a_b_da_cero_exacto():
    filas = _errores().assign(abs_b=lambda d: d["abs_a"])

    assert p1_ab.bootstrap_pareado(filas, np.random.default_rng(29), 200) == (0.0, 0.0, 0.0)


def test_bootstrap_es_determinista_y_pareado():
    filas = _errores().assign(abs_b=lambda d: d["abs_a"] - 1.0)

    primera = p1_ab.bootstrap_pareado(filas, np.random.default_rng(29), 300)
    segunda = p1_ab.bootstrap_pareado(filas, np.random.default_rng(29), 300)
    por_dia = p1_ab.bootstrap_pareado(filas, np.random.default_rng(29), 300, "fecha")

    assert primera == segunda
    # pareado: cada réplica toma las mismas estaciones para A y B, así que la mejora
    # constante de 1 por fila no se diluye con el ruido entre estaciones
    assert primera == pytest.approx((-1.0, -1.0, -1.0), abs=1e-12)
    assert por_dia == pytest.approx((-1.0, -1.0, -1.0), abs=1e-12)


def test_comparar_consume_la_semilla_en_orden_todas_existentes_nuevas():
    filas = _errores(n_estaciones=8)
    y = np.full(len(filas), 5.0)
    evalua = filas[["estacion", "fecha"]].assign(objetivo=y)
    pred_a = y + filas["abs_a"].to_numpy()
    pred_b = y + filas["abs_a"].to_numpy() * 0.9
    existentes = {0, 1, 2, 3, 4}

    tabla, con_error = p1_ab.comparar(evalua, pred_a, pred_b, existentes, n_bootstrap=100)

    rng = np.random.default_rng(29)
    esperado = {
        "todas": p1_ab.bootstrap_pareado(con_error, rng, 100),
        "existentes": p1_ab.bootstrap_pareado(
            con_error[con_error["estacion"].isin(existentes)], rng, 100
        ),
        "nuevas": p1_ab.bootstrap_pareado(
            con_error[~con_error["estacion"].isin(existentes)], rng, 100
        ),
    }
    for grupo, (dif, bajo, alto) in esperado.items():
        assert tabla[grupo]["dif_mae_B_menos_A"] == dif and tabla[grupo]["ic95"] == [bajo, alto]
    assert (tabla["existentes"]["estaciones"], tabla["nuevas"]["estaciones"]) == (5, 3)
    assert tabla["todas"]["A"]["n"] == len(filas)
    assert set(con_error["grupo"]) == {"existentes", "nuevas"}


def _tabla(todas, existentes, nuevas=None):
    tabla = {"todas": {"ic95": list(todas)}, "existentes": {"ic95": list(existentes)}}
    if nuevas is not None:
        tabla["nuevas"] = {"ic95": list(nuevas)}
    return tabla


BIEN = np.array([0.0, 1.5, 7.0])


@pytest.mark.parametrize(
    ("todas", "existentes", "nuevas", "pred_b", "fallan"),
    [
        ((-0.3, -0.02), (-0.4, -0.03), (-0.02, 0.04), BIEN, []),
        ((-0.3, -0.02), (-0.4, -0.03), None, BIEN, []),  # sin estaciones nuevas
        ((-0.3, 0.01), (-0.4, -0.03), (-0.02, 0.04), BIEN, ["c1_mejora_mae_ic_bajo_cero"]),
        ((-0.3, -0.02), (-0.4, -0.03), (0.01, 0.05), BIEN, ["c2_no_peor_en_grupos"]),
        ((-0.3, -0.02), (0.001, 0.2), (-0.02, 0.04), BIEN, ["c2_no_peor_en_grupos"]),
        ((-0.3, -0.02), (-0.4, -0.03), (-0.02, 0.04), np.array([1.0, -0.1]), ["c3_dominio_b"]),
        ((-0.3, -0.02), (-0.4, -0.03), (-0.02, 0.04), np.array([1.0, np.nan]), ["c3_dominio_b"]),
        ((-0.3, -0.02), (-0.4, -0.03), (-0.02, 0.04), np.array([np.inf]), ["c3_dominio_b"]),
    ],
)
def test_regla_de_decision(todas, existentes, nuevas, pred_b, fallan):
    resultado = p1_ab.decidir(_tabla(todas, existentes, nuevas), pred_b)

    assert [c for c, cumple in resultado["regla"].items() if not cumple] == fallan
    esperado = p1_ab.DECISION_MANTENER if fallan else p1_ab.DECISION_CUMPLE
    assert resultado["decision"] == esperado
    assert "no lo promueve" in resultado["lectura_decision"]


# --- Ventanas, embargo y validación de entradas ----------------------------------------------


def test_embargo_b_termina_dos_dias_antes_del_mes_y_a_en_la_ventana_del_campeon(particion):
    train_a, train_b, evalua = particion
    v = p1_ab.ventanas("2026-10-01")

    assert v["B"] == (pd.Timestamp("2025-01-01"), INICIO - pd.Timedelta(days=fe.LAG_MINIMO))
    assert v["B"][1] == pd.Timestamp("2026-09-29")
    assert v["A"][1] == pd.Timestamp("2026-04-29")
    assert p1_ab.ventanas("2026-12-01")["B"][1] == pd.Timestamp("2026-11-29")
    assert train_b["fecha"].max() == pd.Timestamp("2026-09-29")
    assert not (train_b["fecha"] >= INICIO).any()
    assert train_a["fecha"].max() == pd.Timestamp("2026-04-29")
    assert set(evalua["fecha"]) == set(pd.date_range("2026-10-01", "2026-10-31"))
    assert 4 in set(train_b["estacion"]) and 4 not in set(train_a["estacion"])


def test_mes_incompleto_es_error(archivos, tmp_path):
    historico, _ = archivos
    parcial = tmp_path / "octubre_parcial.csv"
    _viajes("2026-10-01", "2026-10-20", 3).to_csv(parcial, index=False)

    with pytest.raises(ValueError, match="no está completo"):
        p1_ab.ejecutar(historico, parcial, "2026-10-01", "2026-11-01", tmp_path / "salida")
    assert not (tmp_path / "salida").exists()


@pytest.mark.parametrize(
    ("desde", "hasta", "mensaje"),
    [
        ("2026-09-01", "2026-09-30", "dentro del mes"),
        ("2026-10-01", "2026-11-03", "dentro del mes"),
    ],
)
def test_viajes_actuales_fuera_del_mes_son_error(archivos, tmp_path, desde, hasta, mensaje):
    historico, _ = archivos
    actual = tmp_path / "actual.csv"
    _viajes(desde, hasta, 4).to_csv(actual, index=False)

    with pytest.raises(ValueError, match=mensaje):
        p1_ab.construir_panel(historico, actual, INICIO, FIN)


def test_historico_que_llega_al_mes_evaluado_es_error(archivos, tmp_path):
    _, octubre = archivos
    historico = tmp_path / "historico_largo.parquet"
    _viajes("2025-01-01", "2026-10-05", 5).to_parquet(historico, index=False)

    with pytest.raises(ValueError, match="terminar antes"):
        p1_ab.construir_panel(historico, octubre, INICIO, FIN)


def test_periodo_que_no_es_un_mes_calendario_es_error(archivos, tmp_path):
    with pytest.raises(ValueError, match="one calendar month"):
        p1_ab.ejecutar(*archivos, "2026-10-01", "2026-10-15", tmp_path / "salida")


def test_registrar_con_modelo_b_es_error(archivos, tmp_path):
    resultado = CliRunner().invoke(
        p1_ab.app, ["--registrar", "--modelo-b", "models:/x/2", "--output", str(tmp_path)]
    )

    assert resultado.exit_code == 2
    with pytest.raises(ValueError, match="no se combina"):
        p1_ab.ejecutar(
            *archivos, "2026-10-01", "2026-11-01", tmp_path, modelo_b="models:/x/2", registrar=True
        )
    assert list(tmp_path.iterdir()) == []


# --- Corrida completa sin registro -----------------------------------------------------------


def test_cli_completo_escribe_salidas_sin_tocar_mlflow(
    archivos, particion, campeon_local, tmp_path
):
    historico, octubre = archivos
    salida = tmp_path / "salida"
    almacen = tmp_path / "no_se_usa.db"

    resultado = CliRunner().invoke(
        p1_ab.app,
        [
            "--historical-trips",
            str(historico),
            "--current-trips",
            str(octubre),
            "--modelo-a",
            str(campeon_local),
            "--tracking-uri",
            f"sqlite:///{almacen.as_posix()}",
            "--output",
            str(salida),
            "--n-bootstrap",
            "50",
        ],
    )

    assert resultado.exit_code == 0, resultado.output
    assert not almacen.exists()  # sin --registrar no se abre ningún almacén de MLflow
    assert {p.name for p in salida.iterdir()} == {
        "p1_ab_resumen.json",
        "p1_ab_resultados.csv",
        "p1_ab_por_estacion.csv",
        "p1_ab_por_dia.csv",
        "p1_ab_predicciones.parquet",
    }
    resumen = json.loads((salida / "p1_ab_resumen.json").read_text(encoding="utf-8"))
    train_a, train_b, evalua = particion
    assert resumen["reproduccion_A"]["max_abs_dif_pred"] <= p1_ab.TOLERANCIA
    assert resumen["A"]["entrenamiento"] == ["2025-01-01", "2026-04-29"]
    assert resumen["B"]["entrenamiento"] == ["2025-01-01", "2026-09-29"]
    assert (resumen["A"]["filas"], resumen["B"]["filas"]) == (len(train_a), len(train_b))
    assert resumen["entradas"]["viajes_historicos"]["sha256"] == file_sha256(historico)
    assert resumen["entradas"]["viajes_actuales"]["sha256"] == file_sha256(octubre)
    assert resumen["entradas"]["modelo_a"]["model_skops_sha256"] == file_sha256(
        campeon_local / "model.skops"
    )
    assert list(resumen["tabla"]) == ["todas", "existentes", "nuevas"]
    assert resumen["tabla"]["nuevas"]["estaciones"] == 1
    assert resumen["decision"] == (
        p1_ab.DECISION_CUMPLE if all(resumen["regla"].values()) else p1_ab.DECISION_MANTENER
    )
    assert resumen["bootstrap"] == {"replicas": 50, "semilla": 29, "unidad": "estacion"}
    assert "registro" not in resumen

    predicciones = pd.read_parquet(salida / "p1_ab_predicciones.parquet")
    assert list(predicciones.columns) == [
        "estacion",
        "fecha",
        "objetivo",
        "pred_a",
        "pred_b",
        "grupo",
    ]
    assert len(predicciones) == len(evalua)
    assert set(predicciones.loc[predicciones["estacion"] == 4, "grupo"]) == {"nuevas"}
    resultados = pd.read_csv(salida / "p1_ab_resultados.csv")
    assert resultados["grupo"].tolist() == ["todas", "existentes", "nuevas"]
    assert resultados.loc[0, "mae_a"] == pytest.approx(resumen["tabla"]["todas"]["A"]["mae"])
    por_estacion = pd.read_csv(salida / "p1_ab_por_estacion.csv")
    assert list(por_estacion.columns) == ["estacion", "abs_a", "abs_b", "mejora"]
    assert len(pd.read_csv(salida / "p1_ab_por_dia.csv")) == 31


def test_a_que_no_se_reproduce_detiene_el_ab(archivos, particion, tmp_path):
    otro = train_p1.ajustar_candidato(p1_ab.MODELO, particion[0], {"max_iter": 20}, 29)

    with pytest.raises(ValueError, match="no reproduce a A"):
        p1_ab.ejecutar(
            *archivos,
            "2026-10-01",
            "2026-11-01",
            tmp_path / "salida",
            modelo_a=str(_guardar(otro, tmp_path / "otro")),
        )


# --- Registro de B en un almacén temporal ----------------------------------------------------


@pytest.fixture
def almacen(tmp_path):
    """Tracking y Registry en un SQLite temporal; restaura ambas URI al terminar."""
    tracking_previo = mlflow.get_tracking_uri()
    registry_previo = mlflow.get_registry_uri()
    registry_env_previo = os.environ.get("MLFLOW_REGISTRY_URI")
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    manifiesto = tmp_path / "manifiesto.json"
    manifiesto.write_text(json.dumps({"version": "v-prueba"}))
    cfg = replace(CONFIG_P1, artefactos_dir=tmp_path / "arts", manifiesto_path=manifiesto)
    yield uri, cfg, MlflowClient(tracking_uri=uri, registry_uri=uri)
    if mlflow.active_run():
        mlflow.end_run()
    mlflow.set_tracking_uri(tracking_previo)
    if registry_env_previo is None:
        os.environ.pop("MLFLOW_REGISTRY_URI", None)
        mlflow.set_registry_uri(None)
    else:
        mlflow.set_registry_uri(registry_previo)


def _campeon_v1(uri, cfg, train_a):
    """Campeón sintético: versión 1 del modelo de P1 con el alias champion."""
    tracking.configurar_experimento(uri, cfg.experimento, cfg.artefactos_dir.resolve().as_uri())
    modelo = train_p1.ajustar_hgb(train_a, {"max_iter": 5}, 0)
    with mlflow.start_run(run_name="campeon-sintetico"):
        info = train_p1.registrar_modelo(modelo, train_a.head(5))
    version = mlflow.register_model(info.model_uri, cfg.modelo_registrado)
    MlflowClient(uri, uri).set_registered_model_alias(
        cfg.modelo_registrado, cfg.alias_campeon, version.version
    )
    return modelo


@pytest.fixture
def ab_registrable(almacen, particion, tmp_path):
    """Campeón v1 en el almacén, B entrenado y el resumen del A/B listo para registrar."""
    uri, cfg, cliente = almacen
    train_a, train_b, evalua = particion
    campeon = _campeon_v1(uri, cfg, train_a)
    modelo_b = train_p1.ajustar_hgb(train_b, {"max_iter": 8}, 1)
    pred_a = np.asarray(campeon.predict(evalua[fe.FEATURES].astype("float64")), dtype=float)
    pred_b = np.asarray(modelo_b.predict(evalua[fe.FEATURES]), dtype=float)
    tabla, filas = p1_ab.comparar(evalua, pred_a, pred_b, set(train_a["estacion"]), 20)
    sens, _, _ = p1_ab.sensibilidad(filas, 20)
    resumen = {
        "ab_periodo": "2026-10-01..2026-10-31",
        "entradas": {
            "viajes_historicos": {"sha256": "a" * 64},
            "viajes_actuales": {"sha256": "b" * 64},
            "modelo_a": {"referencia": p1_ab.MODELO_A_POR_OMISION},
        },
        "B": {"params": {"max_iter": 8}, "entrenamiento": ["2025-01-01", "2026-09-29"]},
        "tabla": tabla,
        "sensibilidad": sens,
        "reproduccion_A": {"max_abs_dif_pred": 0.0},
        "bootstrap": {"replicas": 20, "semilla": 29, "unidad": "estacion"},
        **p1_ab.decidir(tabla, pred_b),
    }
    archivo = tmp_path / "p1_ab_resumen.json"
    archivo.write_text(json.dumps(resumen), encoding="utf-8")
    return {
        "uri": uri,
        "cfg": cfg,
        "cliente": cliente,
        "args": (modelo_b, train_b, filas, pred_a, pred_b, resumen, [archivo], uri, cfg),
    }


def _aliases(cliente, cfg):
    aliases = cliente.get_registered_model(cfg.modelo_registrado).aliases
    return {alias: int(version) for alias, version in aliases.items()}


def test_registrar_b_crea_v2_challenger_y_champion_sigue_en_v1(ab_registrable):
    cliente, cfg = ab_registrable["cliente"], ab_registrable["cfg"]
    resumen = ab_registrable["args"][5]

    registro = p1_ab.registrar_version_b(*ab_registrable["args"], autor="prueba")

    assert registro["version"] == 2 and registro["alias"] == "challenger"
    assert registro["retador_version_anterior"] is None
    assert registro["campeon_version_antes"] == registro["campeon_version_despues"] == 1
    assert registro["max_abs_dif_pred"] <= p1_ab.TOLERANCIA
    assert registro["campeon_max_abs_dif_pred_a"] <= p1_ab.TOLERANCIA
    assert registro["filas_comprobadas"] == len(ab_registrable["args"][2])
    assert len(registro["model_skops_sha256"]) == 64
    assert _aliases(cliente, cfg) == {"champion": 1, "challenger": 2}

    v2 = cliente.get_model_version(cfg.modelo_registrado, "2")
    assert v2.run_id == registro["run_id"]
    assert {
        "run_id": registro["run_id"],
        "origen": "mibici/monitoring/p1_ab.py",
        "ab_decision": resumen["decision"],
        "ab_periodo": "2026-10-01..2026-10-31",
        "entrenamiento_fin": "2026-09-29",
        "comprobacion": "ok",
    }.items() <= v2.tags.items()
    assert {"git_commit", "git_dirty"} <= set(v2.tags)
    assert "no lo promueve" in v2.description

    run = cliente.get_run(registro["run_id"])
    assert {
        "question": "P1",
        "author": "prueba",
        "data_version": "v-prueba",
        "stage_of_work": "ab_test",
        "origen": "mibici/monitoring/p1_ab.py",
        "ab_periodo": "2026-10-01..2026-10-31",
        "entrenamiento_inicio": "2025-01-01",
        "entrenamiento_fin": "2026-09-29",
        "sha256_viajes_historicos": "a" * 64,
        "sha256_viajes_actuales": "b" * 64,
        "campeon_version": "1",
    }.items() <= run.data.tags.items()
    assert run.data.params["entrenamiento_fin"] == "2026-09-29"
    assert run.data.metrics["ab_todas_dif_mae_b_menos_a"] == pytest.approx(
        resumen["tabla"]["todas"]["dif_mae_B_menos_A"]
    )
    artefactos = [a.path for a in cliente.list_artifacts(registro["run_id"], "ab_test")]
    assert artefactos == ["ab_test/p1_ab_resumen.json"]


def test_comprobacion_fallida_no_asigna_alias(ab_registrable):
    cliente, cfg = ab_registrable["cliente"], ab_registrable["cfg"]
    args = list(ab_registrable["args"])
    args[4] = args[4] + 0.5  # B en memoria distinto del registrado

    with pytest.raises(ValueError, match="queda sin alias"):
        p1_ab.registrar_version_b(*args, autor="prueba")

    assert cliente.get_model_version(cfg.modelo_registrado, "2").tags["comprobacion"] == "fallida"
    assert _aliases(cliente, cfg) == {"champion": 1}


def test_a_distinto_del_champion_no_registra_nada(ab_registrable):
    cliente, cfg = ab_registrable["cliente"], ab_registrable["cfg"]
    args = list(ab_registrable["args"])
    args[3] = args[3] + 1.0  # el A/B no se hizo contra el campeón del Registry

    with pytest.raises(ValueError, match="no es"):
        p1_ab.registrar_version_b(*args, autor="prueba")

    versiones = cliente.search_model_versions(f"name='{cfg.modelo_registrado}'")
    assert [int(v.version) for v in versiones] == [1]
    assert _aliases(cliente, cfg) == {"champion": 1}


def test_sin_champion_no_registra(almacen, particion):
    uri, cfg, cliente = almacen
    _, train_b, evalua = particion
    modelo_b = train_p1.ajustar_hgb(train_b, {"max_iter": 3}, 1)
    pred = np.asarray(modelo_b.predict(evalua[fe.FEATURES]), dtype=float)

    with pytest.raises(ValueError, match="No existe"):
        p1_ab.registrar_version_b(modelo_b, train_b, evalua, pred, pred, {}, [], uri, cfg)

    assert cliente.search_registered_models() == []


def test_alias_del_retador_igual_al_del_campeon_es_error():
    cfg = replace(CONFIG_P1, alias_retador=CONFIG_P1.alias_campeon)

    with pytest.raises(ValueError, match="no puede ser el del campeón"):
        p1_ab.registrar_version_b(None, None, None, None, None, {}, [], "sqlite://", cfg)
