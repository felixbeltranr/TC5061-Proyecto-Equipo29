"""Pruebas del A/B de P1 con viajes sintéticos pequeños (sin datos reales ni red)."""

from dataclasses import replace
import io
import json
import os

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.tracking._tracking_service.client import TrackingServiceClient
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


def _ajuste_independiente(train: pd.DataFrame):
    """HGB de Poisson con los parámetros del campeón, sin pasar por ``ajustar_candidato``."""
    perdida = train_p1.PERDIDAS_HGB[p1_ab.MODELO]
    return train_p1.ajustar_hgb(train, p1_ab.PARAMS_CAMPEON, 29, perdida)


@pytest.fixture(scope="module")
def corrida_cli(archivos, campeon_local, tmp_path_factory):
    """Una corrida completa por el CLI, sin --registrar, que comparten varias pruebas."""
    historico, octubre = archivos
    base = tmp_path_factory.mktemp("cli")
    salida, almacen = base / "salida", base / "no_se_usa.db"
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
    return resultado, salida, almacen


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


@pytest.mark.parametrize(
    ("a", "b", "esperado"),
    [
        ([1.0, 2.0], [1.0, 2.5], 0.5),
        ([1.0, 2.0], [1.0, 2.0], 0.0),
        ([1.0, np.nan], [1.0, 2.0], np.inf),  # con max a secas daba NaN y NaN > tol es falso
        ([np.nan, np.nan], [np.nan, np.nan], np.inf),
        ([1.0, 2.0], [1.0, np.inf], np.inf),
        ([1.0, 2.0], [1.0, 2.0, 3.0], np.inf),  # longitudes distintas
        ([], [], np.inf),  # sin filas no hay comprobación
    ],
)
def test_diferencia_pred_falla_cerrada(a, b, esperado):
    dif = p1_ab.diferencia_pred(np.array(a), np.array(b))

    assert dif == esperado
    assert (dif <= p1_ab.TOLERANCIA) == (esperado == 0.0)


def test_reproduccion_con_nan_en_a_es_error(particion):
    train_a, _, evalua = particion

    with pytest.raises(ValueError, match="no reproduce a A"):
        p1_ab.comprobar_reproduccion(
            train_a, evalua, np.full(len(evalua), np.nan), params={"max_iter": 3}
        )


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


def test_historico_con_hueco_antes_del_mes_es_error(archivos, tmp_path):
    _, octubre = archivos
    historico = tmp_path / "historico_corto.parquet"
    _viajes("2025-01-01", "2026-09-20", 1).to_parquet(historico, index=False)

    # sin la validación, del 21 al 29 de septiembre entraban ceros falsos a la ventana de B
    with pytest.raises(ValueError, match="termina el 2026-09-20 y debe llegar al 2026-09-30"):
        p1_ab.construir_panel(historico, octubre, INICIO, FIN)


def test_dia_sin_viajes_dentro_del_mes_es_error(archivos, tmp_path):
    historico, _ = archivos
    viajes = _viajes("2026-10-01", "2026-10-31", 2)
    sin_15 = tmp_path / "octubre_sin_15.csv"
    viajes[viajes["Inicio_del_viaje"].dt.normalize() != "2026-10-15"].to_csv(sin_15, index=False)

    with pytest.raises(ValueError, match=r"no está completo.*1 días sin viajes.*2026-10-15"):
        p1_ab.construir_panel(historico, sin_15, INICIO, FIN)


def test_ventana_de_b_no_depende_del_csv_del_mes(archivos, particion):
    historico, _ = archivos
    panel_hist, _ = fe.construir_panel_diario(
        fe.cargar_salidas(historico), inicio=p1_ab.INICIO_PANEL, fin=INICIO - pd.Timedelta(days=1)
    )
    feats = fe.construir_features(panel_hist)
    solo_hist = feats[feats["elegible"] & feats["fecha"].between(*p1_ab.ventanas(INICIO)["B"])]
    columnas = ["estacion", "fecha", "objetivo", *fe.FEATURES]

    pd.testing.assert_frame_equal(
        particion[1][columnas].reset_index(drop=True), solo_hist[columnas].reset_index(drop=True)
    )


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


def test_forzar_sin_registrar_es_error(archivos, tmp_path):
    resultado = CliRunner().invoke(p1_ab.app, ["--forzar", "--output", str(tmp_path)])

    assert resultado.exit_code == 2
    with pytest.raises(ValueError, match="solo aplica con --registrar"):
        p1_ab.ejecutar(*archivos, "2026-10-01", "2026-11-01", tmp_path, forzar=True)
    assert list(tmp_path.iterdir()) == []


def test_modelo_b_en_carpeta_local_es_error(archivos, campeon_local, tmp_path):
    # una carpeta no declara su ventana: no se puede descartar que B viera el mes evaluado
    with pytest.raises(ValueError, match="debe ser una versión del Registry"):
        p1_ab.ejecutar(
            *archivos,
            "2026-10-01",
            "2026-11-01",
            tmp_path / "salida",
            modelo_a=str(campeon_local),
            modelo_b=str(campeon_local),
        )
    assert not (tmp_path / "salida").exists()


# --- Corrida completa sin registro -----------------------------------------------------------


def test_cli_completo_escribe_salidas_sin_tocar_mlflow(
    archivos, particion, campeon_local, corrida_cli
):
    historico, octubre = archivos
    resultado, salida, almacen = corrida_cli

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

    # lo que se ajustó, no solo el conteo: B es el HGB del campeón con train_b y nada más, y A
    # es el campeón cargado
    pd.testing.assert_frame_equal(
        predicciones[["estacion", "fecha", "objetivo"]],
        evalua[["estacion", "fecha", "objetivo"]].reset_index(drop=True),
        check_dtype=False,
    )
    X = evalua[fe.FEATURES]
    esperado_b = _ajuste_independiente(train_b).predict(X)
    assert p1_ab.diferencia_pred(predicciones["pred_b"], esperado_b) <= p1_ab.TOLERANCIA
    esperado_a = mlflow.sklearn.load_model(str(campeon_local)).predict(X.astype("float64"))
    assert p1_ab.diferencia_pred(predicciones["pred_a"], esperado_a) <= p1_ab.TOLERANCIA


def test_b_no_depende_de_las_etiquetas_del_mes_evaluado(
    archivos, campeon_local, corrida_cli, tmp_path
):
    """Triplicar las salidas del 31 de octubre cambia ese objetivo, no las features ni a B.

    Las features del 31 llegan hasta el 29 (embargo de 2 días), así que A y B deben dar
    exactamente las mismas predicciones; si B se ajustara con filas del mes evaluado, cambiaría.
    """
    historico, _ = archivos
    viajes = _viajes("2026-10-01", "2026-10-31", 2)  # misma semilla que el octubre de base
    dia_31 = viajes[viajes["Inicio_del_viaje"].dt.normalize() == "2026-10-31"]
    otro = tmp_path / "octubre_otro_31.csv"
    pd.concat([viajes, dia_31, dia_31], ignore_index=True).to_csv(otro, index=False)
    almacen = tmp_path / "no_se_usa.db"

    p1_ab.ejecutar(
        historico,
        otro,
        "2026-10-01",
        "2026-11-01",
        tmp_path / "salida",
        modelo_a=str(campeon_local),
        tracking_uri=f"sqlite:///{almacen.as_posix()}",
        n_bootstrap=20,
    )

    base = pd.read_parquet(corrida_cli[1] / "p1_ab_predicciones.parquet")
    nueva = pd.read_parquet(tmp_path / "salida" / "p1_ab_predicciones.parquet")
    es_31 = (base["fecha"] == "2026-10-31").to_numpy()
    assert es_31.sum() == 4
    pd.testing.assert_frame_equal(nueva[["estacion", "fecha"]], base[["estacion", "fecha"]])
    assert (nueva.loc[es_31, "objetivo"] == 3 * base.loc[es_31, "objetivo"]).all()
    assert nueva.loc[~es_31, "objetivo"].equals(base.loc[~es_31, "objetivo"])
    np.testing.assert_array_equal(nueva["pred_a"], base["pred_a"])
    np.testing.assert_array_equal(nueva["pred_b"], base["pred_b"])
    assert not almacen.exists()


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
    assert registro["version_equivalente_previa"] is None
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
    assert not {"retador_anterior", "reemplazada_por"} & set(v2.tags)
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
        "registro": "ok",
        "registro_version": "2",
    }.items() <= run.data.tags.items()
    assert run.info.status == "FINISHED"
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

    v2 = cliente.get_model_version(cfg.modelo_registrado, "2")
    assert v2.tags["comprobacion"] == "fallida"
    assert cliente.get_run(v2.run_id).data.tags["registro"] == "fallido"
    assert _aliases(cliente, cfg) == {"champion": 1}


def _a_con_nan(pred_a):
    """A muy distinto del campeón, con un NaN que antes hacía pasar la comprobación."""
    otro = pred_a + 100.0
    otro[0] = np.nan
    return otro


@pytest.mark.parametrize("alterar", [lambda p: p + 1.0, _a_con_nan], ids=["corrido", "con_nan"])
def test_a_distinto_del_champion_no_registra_nada(ab_registrable, alterar):
    cliente, cfg = ab_registrable["cliente"], ab_registrable["cfg"]
    args = list(ab_registrable["args"])
    args[3] = alterar(args[3])  # el A/B no se hizo contra el campeón del Registry

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


# --- Registro repetido, fallas a medio registro y consola -----------------------------------


def _runs_ab(cliente, cfg):
    experimento = cliente.get_experiment_by_name(cfg.experimento).experiment_id
    return cliente.search_runs([experimento], "tags.stage_of_work = 'ab_test'")


def _versiones(cliente, cfg):
    versiones = cliente.search_model_versions(f"name='{cfg.modelo_registrado}'")
    return sorted(int(v.version) for v in versiones)


def test_registro_repetido_se_detiene_y_forzar_deja_constancia_del_reemplazo(ab_registrable):
    cliente, cfg = ab_registrable["cliente"], ab_registrable["cfg"]
    nombre = cfg.modelo_registrado
    p1_ab.registrar_version_b(*ab_registrable["args"], autor="prueba")

    with pytest.raises(ValueError, match=r"v2 ya es un B .*--forzar"):
        p1_ab.registrar_version_b(*ab_registrable["args"], autor="prueba")

    # la segunda corrida no escribió nada: ni run, ni versión, ni alias movido
    assert _versiones(cliente, cfg) == [1, 2]
    assert len(_runs_ab(cliente, cfg)) == 1
    assert _aliases(cliente, cfg) == {"champion": 1, "challenger": 2}

    tercera = p1_ab.registrar_version_b(*ab_registrable["args"], autor="prueba", forzar=True)

    assert (tercera["version"], tercera["version_equivalente_previa"]) == (3, 2)
    assert tercera["retador_version_anterior"] == 2
    assert _aliases(cliente, cfg) == {"champion": 1, "challenger": 3}
    assert cliente.get_model_version(nombre, "2").tags["reemplazada_por"] == "3"
    assert cliente.get_model_version(nombre, "3").tags["retador_anterior"] == "2"


def test_ejecutar_con_registro_repetido_se_detiene_antes_de_entrenar(
    ab_registrable, archivos, tmp_path, monkeypatch
):
    p1_ab.registrar_version_b(*ab_registrable["args"], autor="prueba")

    def no_entrenar(*args, **kwargs):
        raise AssertionError("no debe entrenar si el registro se va a rechazar")

    monkeypatch.setattr(train_p1, "ajustar_candidato", no_entrenar)
    with pytest.raises(ValueError, match="--forzar"):
        p1_ab.ejecutar(
            *archivos,
            "2026-10-01",
            "2026-11-01",
            tmp_path / "salida",
            registrar=True,
            tracking_uri=ab_registrable["uri"],
        )
    assert not (tmp_path / "salida").exists()


def test_falla_al_cerrar_el_run_lo_deja_fallido_y_sin_version(ab_registrable, monkeypatch):
    cliente, cfg = ab_registrable["cliente"], ab_registrable["cfg"]
    llamadas = []

    def enlace_en_cp1252(self, run_id):
        # como MLflow con tracking HTTP en una consola cp1252: el emoji del enlace del run
        llamadas.append(run_id)
        if len(llamadas) == 1:
            raise UnicodeEncodeError(
                "charmap", "\U0001f3c3", 0, 1, "character maps to <undefined>"
            )

    monkeypatch.setattr(TrackingServiceClient, "_log_url", enlace_en_cp1252)

    with pytest.raises(UnicodeEncodeError):
        p1_ab.registrar_version_b(*ab_registrable["args"], autor="prueba")

    (run,) = _runs_ab(cliente, cfg)
    assert run.info.status == "FAILED"
    assert run.data.tags["registro"] == "fallido"
    assert run.data.tags["registro_detalle"].startswith("UnicodeEncodeError")
    assert _versiones(cliente, cfg) == [1]
    assert _aliases(cliente, cfg) == {"champion": 1}


@pytest.mark.parametrize(
    ("tambien_etiquetas", "comprobacion"), [(False, "fallida"), (True, "pendiente")]
)
def test_falla_tras_register_model_deja_la_version_marcada(
    ab_registrable, monkeypatch, tambien_etiquetas, comprobacion
):
    cliente, cfg = ab_registrable["cliente"], ab_registrable["cfg"]

    def caido(*args, **kwargs):
        raise RuntimeError("simulado: el servidor cae tras register_model")

    monkeypatch.setattr(MlflowClient, "update_model_version", caido)
    if tambien_etiquetas:  # ni siquiera se puede escribir fallida: queda la de nacimiento
        monkeypatch.setattr(MlflowClient, "set_model_version_tag", caido)

    with pytest.raises(ValueError, match="queda sin alias: simulado"):
        p1_ab.registrar_version_b(*ab_registrable["args"], autor="prueba")

    v2 = cliente.get_model_version(cfg.modelo_registrado, "2")
    assert v2.tags["comprobacion"] == comprobacion
    assert cliente.get_run(v2.run_id).data.tags["registro"] == "fallido"
    assert _aliases(cliente, cfg) == {"champion": 1}


def _falla_al_cerrar(original):
    def falla(self, run_id, key, value, *args, **kwargs):
        if (key, value) == ("registro", "ok"):
            raise RuntimeError("simulado: falla al cerrar el registro")
        return original(self, run_id, key, value, *args, **kwargs)

    return falla


@pytest.mark.parametrize("donde", ["alias", "cierre"])
def test_falla_al_terminar_el_registro_no_deja_ok_ni_bloquea_el_reintento(
    ab_registrable, monkeypatch, donde
):
    cliente, cfg = ab_registrable["cliente"], ab_registrable["cfg"]
    nombre = cfg.modelo_registrado
    with monkeypatch.context() as parche:
        if donde == "alias":
            original = MlflowClient.set_registered_model_alias

            def falla(self, name, alias, version):
                if alias == cfg.alias_retador:
                    raise RuntimeError("simulado: falla al asignar el alias")
                return original(self, name, alias, version)

            parche.setattr(MlflowClient, "set_registered_model_alias", falla)
        else:
            parche.setattr(MlflowClient, "set_tag", _falla_al_cerrar(MlflowClient.set_tag))
        with pytest.raises(ValueError, match="queda sin alias: simulado"):
            p1_ab.registrar_version_b(*ab_registrable["args"], autor="prueba")

    v2 = cliente.get_model_version(nombre, "2")
    assert v2.tags["comprobacion"] == "fallida"
    assert cliente.get_run(v2.run_id).data.tags["registro"] == "fallido"
    assert _aliases(cliente, cfg) == {"champion": 1}

    # el reintento normal, sin --forzar, registra otra versión y sí termina
    tercera = p1_ab.registrar_version_b(*ab_registrable["args"], autor="prueba")
    assert tercera["version"] == 3
    assert cliente.get_model_version(nombre, "3").tags["comprobacion"] == "ok"
    assert _aliases(cliente, cfg) == {"champion": 1, "challenger": 3}


def test_falla_al_terminar_un_registro_forzado_devuelve_el_alias_al_retador_anterior(
    ab_registrable, monkeypatch
):
    cliente, cfg = ab_registrable["cliente"], ab_registrable["cfg"]
    nombre = cfg.modelo_registrado
    p1_ab.registrar_version_b(*ab_registrable["args"], autor="prueba")

    monkeypatch.setattr(MlflowClient, "set_tag", _falla_al_cerrar(MlflowClient.set_tag))
    with pytest.raises(ValueError, match="queda sin alias: simulado"):
        p1_ab.registrar_version_b(*ab_registrable["args"], autor="prueba", forzar=True)

    assert _aliases(cliente, cfg) == {"champion": 1, "challenger": 2}
    assert "reemplazada_por" not in cliente.get_model_version(nombre, "2").tags
    assert cliente.get_model_version(nombre, "3").tags["comprobacion"] == "fallida"


def test_error_al_leer_el_retador_anterior_no_mueve_el_alias(ab_registrable, monkeypatch):
    from mlflow.exceptions import MlflowException
    from mlflow.protos.databricks_pb2 import TEMPORARILY_UNAVAILABLE

    cliente, cfg = ab_registrable["cliente"], ab_registrable["cfg"]
    nombre = cfg.modelo_registrado
    p1_ab.registrar_version_b(*ab_registrable["args"], autor="prueba")
    original = MlflowClient.get_model_version_by_alias

    def caido(self, name, alias):
        if alias == cfg.alias_retador:
            raise MlflowException("simulado: servidor no disponible", TEMPORARILY_UNAVAILABLE)
        return original(self, name, alias)

    with monkeypatch.context() as parche:
        parche.setattr(MlflowClient, "get_model_version_by_alias", caido)
        with pytest.raises(ValueError, match="alias sin verificar: simulado"):
            p1_ab.registrar_version_b(*ab_registrable["args"], autor="prueba", forzar=True)

    # no se sabía si había retador: el alias no se movió y v2 no quedó como reemplazada
    assert _aliases(cliente, cfg) == {"champion": 1, "challenger": 2}
    assert "reemplazada_por" not in cliente.get_model_version(nombre, "2").tags
    v3 = cliente.get_model_version(nombre, "3")
    assert v3.tags["comprobacion"] == "fallida"
    assert cliente.get_run(v3.run_id).data.tags["registro"] == "fallido"


@pytest.mark.parametrize("con_retador_previo", [False, True])
def test_alias_aplicado_con_respuesta_perdida_se_revierte(
    ab_registrable, monkeypatch, con_retador_previo
):
    cliente, cfg = ab_registrable["cliente"], ab_registrable["cfg"]
    nombre = cfg.modelo_registrado
    if con_retador_previo:
        p1_ab.registrar_version_b(*ab_registrable["args"], autor="prueba")
    nueva = "3" if con_retador_previo else "2"
    original = MlflowClient.set_registered_model_alias
    llamadas = []

    def aplica_y_falla(self, name, alias, version):
        resultado = original(self, name, alias, version)
        llamadas.append((alias, str(version)))
        if llamadas == [(cfg.alias_retador, nueva)]:  # solo la primera: el servidor sí lo aplicó
            raise RuntimeError("simulado: se perdió la respuesta del servidor")
        return resultado

    monkeypatch.setattr(MlflowClient, "set_registered_model_alias", aplica_y_falla)
    with pytest.raises(ValueError, match="queda sin alias: simulado"):
        p1_ab.registrar_version_b(
            *ab_registrable["args"], autor="prueba", forzar=con_retador_previo
        )

    esperado = {"champion": 1, "challenger": 2} if con_retador_previo else {"champion": 1}
    assert _aliases(cliente, cfg) == esperado
    assert cliente.get_model_version(nombre, nueva).tags["comprobacion"] == "fallida"
    if con_retador_previo:
        assert "reemplazada_por" not in cliente.get_model_version(nombre, "2").tags


def test_mes_dentro_de_la_ventana_de_a_es_error(archivos, tmp_path):
    with pytest.raises(ValueError, match="ventana de entrenamiento de A"):
        p1_ab.ejecutar(*archivos, "2026-04-01", "2026-05-01", tmp_path / "salida")
    assert not (tmp_path / "salida").exists()


def test_proteger_consola_evita_el_error_de_cp1252():
    consola = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="\n")
    with pytest.raises(UnicodeEncodeError):
        consola.write("\U0001f3c3 View run")
        consola.flush()

    consola = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="\n")
    p1_ab.proteger_consola(consola)
    consola.write("\U0001f3c3 View run Decisión\n")
    consola.flush()

    assert consola.buffer.getvalue() == b"\\U0001f3c3 View run Decisi\xf3n\n"


# --- B cargado del Registry: ventana y embargo ----------------------------------------------


def _version(uri, cfg, train, tags_version=None, tags_run=None):
    """Versión sintética del modelo de P1 con las etiquetas indicadas en la versión o en el run."""
    modelo = train_p1.ajustar_hgb(train, {"max_iter": 3}, 0)
    with mlflow.start_run(run_name="b-sintetico"):
        mlflow.set_tags(tags_run or {})
        info = train_p1.registrar_modelo(modelo, train.head(5))
    return mlflow.register_model(info.model_uri, cfg.modelo_registrado, tags=tags_version)


def test_ventana_de_b_registrado_exige_entrenamiento_fin_con_embargo(
    almacen, particion, archivos, tmp_path
):
    uri, cfg, _ = almacen
    train_a, train_b, _ = particion
    nombre, limite = cfg.modelo_registrado, pd.Timestamp("2026-09-29")
    _campeon_v1(uri, cfg, train_a)  # v1: sin etiquetas de ventana
    _version(uri, cfg, train_b, tags_version={"entrenamiento_fin": "2026-10-31"})  # v2: fuga
    _version(  # v3: la ventana solo está en el run
        uri,
        cfg,
        train_b,
        tags_run={"entrenamiento_inicio": "2025-01-01", "entrenamiento_fin": "2026-09-29"},
    )
    _version(uri, cfg, train_b, tags_version={"entrenamiento_fin": ""})  # v4: fecha vacía

    casos = {
        f"models:/{nombre}@champion": "no tiene la etiqueta entrenamiento_fin",
        f"models:/{nombre}/2": "pudo ver etiquetas del mes evaluado",
        f"models:/{nombre}/4": "no es una fecha",
        f"models:/{nombre}": "debe ser una versión del Registry",
    }
    for referencia, mensaje in casos.items():
        with pytest.raises(ValueError, match=mensaje):
            p1_ab.ventana_registrada(referencia, uri, limite)

    info = p1_ab.ventana_registrada(f"models:/{nombre}/3", uri, limite)
    assert (info["version"], info["entrenamiento_inicio"], info["entrenamiento_fin"]) == (
        3,
        "2025-01-01",
        "2026-09-29",
    )
    assert info["fuente_entrenamiento_fin"] == "run"

    with pytest.raises(ValueError, match="pudo ver etiquetas del mes evaluado"):
        p1_ab.ejecutar(
            *archivos,
            "2026-10-01",
            "2026-11-01",
            tmp_path / "salida",
            modelo_a=f"models:/{nombre}@champion",
            modelo_b=f"models:/{nombre}/2",
            tracking_uri=uri,
        )
    assert not (tmp_path / "salida").exists()


def test_ab_con_b_registrado_por_el_modulo_lee_su_ventana(ab_registrable, archivos, tmp_path):
    uri, cfg = ab_registrable["uri"], ab_registrable["cfg"]
    nombre = cfg.modelo_registrado
    pred_a, pred_b = ab_registrable["args"][3], ab_registrable["args"][4]
    p1_ab.registrar_version_b(*ab_registrable["args"], autor="prueba")

    resumen = p1_ab.ejecutar(
        *archivos,
        "2026-10-01",
        "2026-11-01",
        tmp_path / "salida",
        modelo_a=f"models:/{nombre}@champion",
        modelo_b=f"models:/{nombre}@challenger",
        tracking_uri=uri,
        n_bootstrap=20,
    )

    assert resumen["B"]["entrenamiento"] == ["2025-01-01", "2026-09-29"]
    assert resumen["B"]["filas"] is None and resumen["B"]["params"] is None
    b = resumen["entradas"]["modelo_b"]
    assert (b["version"], b["fuente_entrenamiento_fin"], b["comprobacion"]) == (2, "versión", "ok")
    assert "v2 del Registry" in resumen["reproduccion_A"]["nota"]
    assert "cumple el embargo (fin <= 2026-09-29)" in resumen["reproduccion_A"]["nota"]
    predicciones = pd.read_parquet(tmp_path / "salida" / "p1_ab_predicciones.parquet")
    assert p1_ab.diferencia_pred(predicciones["pred_a"], pred_a) <= p1_ab.TOLERANCIA
    assert p1_ab.diferencia_pred(predicciones["pred_b"], pred_b) <= p1_ab.TOLERANCIA
