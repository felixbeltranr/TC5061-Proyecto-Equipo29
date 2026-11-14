import json
import math

from loguru import logger
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from mibici import features
from mibici.features import FEATURES
from mibici.monitoring import p1_drift_calibrado as dc

NUMERICAS = [col for col in FEATURES if col not in ("dia_semana", "festivo", "festivo_lag7")]


def _filas(estaciones, fechas, rng, nivel=10.0) -> pd.DataFrame:
    """Filas estación-día con las columnas que usan las tablas de drift (sin panel)."""
    indice = pd.MultiIndex.from_product([estaciones, pd.DatetimeIndex(fechas)])
    filas = pd.DataFrame(
        {"estacion": indice.get_level_values(0), "fecha": indice.get_level_values(1)}
    )
    filas["mes"] = filas["fecha"].dt.to_period("M").astype(str)
    for col in NUMERICAS:
        filas[col] = rng.normal(nivel, 2.0, len(filas))
    filas["dia_semana"] = filas["fecha"].dt.dayofweek.astype("int8")
    filas["festivo"] = (filas["fecha"].dt.day == 1).astype("int8")
    filas["festivo_lag7"] = (filas["fecha"].dt.day == 8).astype("int8")
    filas["objetivo"] = rng.poisson(nivel, len(filas)).astype("float64")
    return filas


def _poblaciones(referencia, actual, control=None) -> dc.Poblaciones:
    control = referencia.iloc[:0] if control is None else control
    return dc.Poblaciones(
        referencia=referencia,
        actual_todas=actual.assign(sin_historia=False, inactiva=False),
        actual=actual,
        control=control,
        intermedias=actual.iloc[:0],
        referencia_inicio=referencia["fecha"].min(),
        referencia_fin=referencia["fecha"].max(),
        control_inicio=pd.Timestamp("2025-10-01"),
        control_fin_exclusivo=pd.Timestamp("2025-11-01"),
    )


def _umbrales(valor: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"umbral_calibrado": valor, "mes_del_maximo": "2025-02"}, index=pd.Index(FEATURES)
    )


def test_wasserstein_normada_contra_valores_conocidos():
    referencia = pd.Series([0.0, 1.0, 2.0, 3.0])  # desviación poblacional sqrt(1.25)

    assert dc.wasserstein_normada(referencia, referencia + 2) == pytest.approx(
        2 / math.sqrt(1.25), abs=1e-12
    )
    assert dc.wasserstein_normada(referencia, pd.Series([3.0, 2.0, 1.0, 0.0])) == 0.0
    # como Evidently, divide entre max(desviación, 0.001): una referencia constante no anula la
    # distancia (Wasserstein 3 entre 0.001) y una desviación de 1e-4 también usa el piso
    constante = pd.Series([5.0, 5.0])
    assert dc.wasserstein_normada(constante, pd.Series([7.0, 9.0])) == pytest.approx(3000.0)
    assert dc.wasserstein_normada(constante, constante) == 0.0
    chica = pd.Series([0.0, 0.0002] * 10)  # desviación poblacional 1e-4
    assert dc.wasserstein_normada(chica, chica + 0.0001) == pytest.approx(0.1, abs=1e-12)


@pytest.mark.parametrize(
    ("referencia", "actual"),
    [
        ([0.0, 1.0, 2.0, 3.0], [2.0, 3.5, 4.0, 9.0]),
        ([5.0, 5.0], [7.0, 9.0]),
        ([0.0, 0.0002] * 10, [0.0001, 0.0003] * 10),
    ],
    ids=["desviacion_normal", "referencia_constante", "desviacion_menor_al_piso"],
)
def test_wasserstein_normada_igual_a_evidently(referencia, actual):
    prueba = pytest.importorskip(
        "evidently.legacy.calculations.stattests.wasserstein_distance_norm"
    )
    from evidently.legacy.core import ColumnType

    referencia, actual = pd.Series(referencia), pd.Series(actual)
    esperado, _ = prueba._wasserstein_distance_norm(
        referencia, actual, ColumnType.Numerical, dc.UMBRAL_EVIDENTLY
    )

    assert dc.wasserstein_normada(referencia, actual) == pytest.approx(float(esperado), rel=1e-12)


def test_referencia_constante_marca_drift_si_el_mes_cambia():
    constante = pd.Series([5.0] * 30)
    tabla = pd.DataFrame(
        {
            "distancia": [dc.wasserstein_normada(constante, pd.Series([7.0, 9.0] * 15))],
            "distancia_existentes": [np.nan],
            "distancia_nuevas": [np.nan],
            "umbral_calibrado": [0.5],
        }
    )

    marcada = dc.marcar_drift(tabla)

    assert marcada.loc[0, "drift_evidently"] and marcada.loc[0, "drift_calibrado"]


def test_jensen_shannon_contra_valores_conocidos():
    mitad = pd.Series([0, 0, 1, 1])
    ceros = pd.Series([0, 0, 0, 0])
    m = (0.75, 0.25)
    divergencia = 0.5 * (0.5 * math.log(0.5 / m[0]) + 0.5 * math.log(0.5 / m[1])) + 0.5 * (
        math.log(1 / m[0])
    )

    assert dc.jensen_shannon(mitad, ceros) == pytest.approx(math.sqrt(divergencia), abs=1e-12)
    # sin categorías en común la divergencia es ln 2
    assert dc.jensen_shannon(ceros, pd.Series([1, 1])) == pytest.approx(
        math.sqrt(math.log(2)), abs=1e-12
    )
    assert dc.jensen_shannon(mitad, pd.Series([1, 0, 1, 0])) == 0.0


def test_distancia_usa_el_metodo_de_cada_variable_y_nan_si_no_hay_filas():
    referencia = pd.Series([0, 0, 1, 1])
    actual = pd.Series([0, 0, 0, 0])

    assert dc.metodo("festivo") == "jensen_shannon"
    assert dc.metodo("lag_2") == "wasserstein_normada"
    assert dc.distancia("festivo", referencia, actual) == dc.jensen_shannon(referencia, actual)
    assert dc.distancia("lag_2", referencia, actual) == dc.wasserstein_normada(referencia, actual)
    assert math.isnan(dc.distancia("lag_2", referencia, actual.iloc[:0]))


def test_calibracion_excluye_meses_con_menos_de_20_dias():
    rng = np.random.default_rng(29)
    referencia = _filas([1, 2], pd.date_range("2025-01-30", "2025-04-30"), rng)

    distancias, umbrales, dias = dc.calibrar(referencia)

    assert dias.to_dict() == {"2025-01": 2, "2025-02": 28, "2025-03": 31, "2025-04": 30}
    assert distancias.index.tolist() == ["2025-02", "2025-03", "2025-04"]
    assert distancias.columns.tolist() == FEATURES
    pd.testing.assert_series_equal(
        umbrales["umbral_calibrado"], distancias.max(), check_names=False
    )
    assert umbrales["mes_del_maximo"].tolist() == distancias.idxmax().tolist()
    # el resto contra el que se compara cada mes incluye los días del mes excluido
    febrero = referencia["mes"] == "2025-02"
    assert distancias.loc["2025-02", "lag_2"] == dc.wasserstein_normada(
        referencia.loc[~febrero, "lag_2"], referencia.loc[febrero, "lag_2"]
    )
    with pytest.raises(ValueError, match="al menos dos meses"):
        dc.calibrar(referencia, min_dias=31)


def test_separa_estaciones_existentes_y_nuevas():
    rng = np.random.default_rng(1)
    referencia = _filas([1, 2], pd.date_range("2025-02-01", "2025-03-31"), rng)
    actual = pd.concat(
        [
            _filas([1, 2], pd.date_range("2026-10-01", "2026-10-31"), rng, nivel=12.0),
            _filas([3], pd.date_range("2026-10-01", "2026-10-31"), rng, nivel=30.0),
        ],
        ignore_index=True,
    )
    pob = _poblaciones(referencia, actual)

    tabla = dc.tabla_mes_actual(pob, _umbrales(0.5))
    resumen = dc.resumir(
        pob, tabla, dc.drift_mensual(pob, _umbrales(0.5)), _umbrales(0.5), pd.Series(dtype="int64")
    )

    existentes = actual[actual["estacion"].isin([1, 2])]
    nuevas = actual[actual["estacion"] == 3]
    assert tabla.loc["lag_7", "distancia_existentes"] == dc.wasserstein_normada(
        referencia["lag_7"], existentes["lag_7"]
    )
    assert tabla.loc["lag_7", "distancia_nuevas"] == dc.wasserstein_normada(
        referencia["lag_7"], nuevas["lag_7"]
    )
    assert tabla.loc["lag_7", "distancia"] == dc.wasserstein_normada(
        referencia["lag_7"], actual["lag_7"]
    )
    assert tabla.loc["lag_7", "media_nuevas"] == pytest.approx(nuevas["lag_7"].mean())
    assert resumen["mes_actual"]["estaciones_existentes"] == 2
    assert resumen["mes_actual"]["estaciones_nuevas"] == 1
    assert resumen["mes_actual"]["filas_nuevas"] == 31
    assert resumen["objetivo_medio"]["control"] is None  # sin filas de control

    solo_existentes = dc.tabla_mes_actual(_poblaciones(referencia, existentes), _umbrales(0.5))
    assert solo_existentes["distancia_nuevas"].isna().all()
    assert not solo_existentes["drift_calibrado_nuevas"].any()
    assert not solo_existentes["drift_evidently_nuevas"].any()


def test_columnas_de_drift_coherentes_con_los_umbrales():
    rng = np.random.default_rng(7)
    referencia = _filas([1, 2], pd.date_range("2025-02-01", "2025-03-31"), rng)
    actual = pd.concat(
        [
            _filas([1, 2], pd.date_range("2026-10-01", "2026-10-31"), rng, nivel=11.0),
            _filas([3], pd.date_range("2026-10-01", "2026-10-31"), rng, nivel=14.0),
        ],
        ignore_index=True,
    )
    umbrales = _umbrales(0.4)
    umbrales.loc[["lag_2", "media_7d"], "umbral_calibrado"] = 1e6  # nunca se superan

    tabla = dc.tabla_mes_actual(_poblaciones(referencia, actual), umbrales)

    for sufijo in ("", "_existentes", "_nuevas"):
        valor = tabla[f"distancia{sufijo}"]
        assert (tabla[f"drift_evidently{sufijo}"] == (valor > 0.1)).all()
        assert (tabla[f"drift_calibrado{sufijo}"] == (valor > tabla["umbral_calibrado"])).all()
    assert not tabla.loc[["lag_2", "media_7d"], "drift_calibrado"].any()
    assert tabla.loc["lag_7", "drift_calibrado_nuevas"]  # nivel 14 contra 10 con sd 2
    listas = dc._variables_con_drift(tabla, "calibrado")
    assert listas["todas"] == tabla.index[tabla["drift_calibrado"]].tolist()


def _viajes(estacion: int, inicio: str, fin: str, lam: float, seed: int) -> pd.DataFrame:
    """Viajes sintéticos: un número Poisson de salidas por día a las 8:00."""
    dias = pd.date_range(inicio, fin, freq="D")
    rng = np.random.default_rng(seed)
    conteos = rng.poisson(lam + 2 * np.sin(dias.month.to_numpy()), len(dias))
    conteos[0] = max(conteos[0], 1)  # la estación entra al panel con su primera salida
    fechas = np.repeat(dias.to_numpy(), conteos)
    return pd.DataFrame(
        {
            "Inicio_del_viaje": pd.DatetimeIndex(fechas) + pd.Timedelta(hours=8),
            "Origen_Id": estacion,
        }
    )


@pytest.fixture(scope="module")
def viajes_sinteticos(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("viajes_p1")
    historico = pd.concat(
        [
            _viajes(42, "2025-01-01", "2026-09-30", 8, 1),
            _viajes(7, "2025-01-01", "2026-08-31", 5, 2),  # sin viajes en septiembre
            _viajes(99, "2026-06-01", "2026-09-30", 4, 3),  # nueva para la referencia
        ],
        ignore_index=True,
    )
    actual = pd.concat(
        [
            _viajes(42, "2026-10-01", "2026-10-31", 12, 4),
            _viajes(7, "2026-10-15", "2026-10-31", 5, 5),
            _viajes(99, "2026-10-01", "2026-10-31", 4, 6),
            _viajes(555, "2026-10-01", "2026-10-31", 3, 7),  # sin historia casi todo el mes
        ],
        ignore_index=True,
    )
    rutas = {"historico": carpeta / "historico.parquet", "actual": carpeta / "actual.csv"}
    historico.to_parquet(rutas["historico"], index=False)
    actual.to_csv(rutas["actual"], index=False)
    panel, _ = features.construir_panel_diario(
        historico, pd.Timestamp("2025-01-01"), pd.Timestamp("2026-09-30")
    )
    rutas["panel"] = carpeta / "panel_campeon.parquet"
    panel.to_parquet(rutas["panel"], index=False)
    return rutas


def test_error_si_el_historico_invade_el_mes_actual(tmp_path, viajes_sinteticos):
    historico = pd.concat(
        [
            pd.read_parquet(viajes_sinteticos["historico"]),
            _viajes(42, "2026-10-01", "2026-10-03", 5, 8),
        ]
    )
    invadido = tmp_path / "historico_con_octubre.parquet"
    historico.to_parquet(invadido, index=False)

    with pytest.raises(ValueError, match="debe terminar antes del mes actual"):
        dc.construir_panel(invadido, viajes_sinteticos["actual"], "2026-10-01", "2026-11-01")


def test_error_si_el_mes_actual_trae_otras_fechas(tmp_path, viajes_sinteticos):
    septiembre = tmp_path / "septiembre.csv"
    _viajes(42, "2026-09-25", "2026-10-05", 5, 9).to_csv(septiembre, index=False)

    with pytest.raises(ValueError, match="deben caer en"):
        dc.construir_panel(viajes_sinteticos["historico"], septiembre, "2026-10-01", "2026-11-01")
    with pytest.raises(ValueError, match="debe empezar después de la referencia"):
        dc.separar_poblaciones(pd.DataFrame(), "2026-04-01", "2026-05-01")


def test_panel_hasta_el_mes_anterior_no_cambia(viajes_sinteticos):
    panel, _, info = dc.construir_panel(
        viajes_sinteticos["historico"], viajes_sinteticos["actual"], "2026-10-01", "2026-11-01"
    )
    original = pd.read_parquet(viajes_sinteticos["panel"])

    assert dc.comparar_con_panel_campeon(panel, original)["igual"]
    assert info["historico_fin"] == "2026-09-30" and info["mes_actual_fin"] == "2026-10-31"
    alterado = original.copy()
    alterado.loc[0, "salidas"] += 1
    assert not dc.comparar_con_panel_campeon(panel, alterado)["igual"]


def test_comparacion_con_scores_de_evidently(tmp_path):
    reporte = {
        "metrics": [
            {"config": {"type": "evidently:metric_v2:DriftedColumnsCount"}, "value": {}},
            {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "lag_2",
                    "method": "Wasserstein distance (normed)",
                    "threshold": 0.1,
                },
                "value": 0.1,
            },
            {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "festivo",
                    "method": "Jensen-Shannon distance",
                    "threshold": 0.1,
                },
                "value": 0.05,
            },
        ]
    }
    ruta = tmp_path / "p1_data_drift.json"
    ruta.write_text(json.dumps(reporte), encoding="utf-8")
    tabla = pd.DataFrame(
        {
            "metodo": ["wasserstein_normada", "jensen_shannon"],
            "distancia": [0.1, 0.05 + 1e-10],
            "drift_evidently": [False, False],
        },
        index=pd.Index(["lag_2", "festivo"], name="variable"),
    )

    comparacion = dc.comparar_con_evidently(tabla, dc.leer_scores_evidently(ruta))

    assert comparacion["metodo_coincide"].all()
    assert comparacion["dif_abs"].max() == pytest.approx(1e-10, abs=1e-15)
    # Evidently marca drift con >= 0.1; la columna drift_evidently de este módulo usa > 0.1
    assert comparacion.loc["lag_2", "drift_segun_evidently"]
    assert not comparacion.loc["festivo", "drift_segun_evidently"]


def _resumen_verificado() -> dict:
    """Secciones de verificación de un resumen en que el panel y Evidently coinciden."""
    return {
        "panel_campeon": {
            "igual": True,
            "filas_campeon": 10,
            "filas_panel_hasta_esa_fecha": 10,
            "fecha_min": "2025-01-01",
            "fecha_max": "2026-09-30",
        },
        "equivalencia_evidently": {
            "variables": len(FEATURES),
            "variables_sin_score": [],
            "variables_solo_en_evidently": [],
            "metodos_coinciden": True,
            "max_dif_abs": 1e-17,
            "drift_evidently_coincide": True,
        },
    }


@pytest.mark.parametrize(
    ("seccion", "campo", "valor", "texto"),
    [
        ("panel_campeon", "igual", False, "panel"),
        ("equivalencia_evidently", "variables_sin_score", ["festivo"], "festivo"),
        ("equivalencia_evidently", "variables_solo_en_evidently", ["otra"], "otra"),
        ("equivalencia_evidently", "metodos_coinciden", False, "método"),
        ("equivalencia_evidently", "max_dif_abs", None, "Diferencia máxima"),
        ("equivalencia_evidently", "max_dif_abs", 1e-6, "Diferencia máxima"),
        ("equivalencia_evidently", "drift_evidently_coincide", False, "drift"),
    ],
)
def test_fallas_de_verificacion(seccion, campo, valor, texto):
    assert dc.fallas_de_verificacion({}) == []
    assert dc.fallas_de_verificacion(_resumen_verificado()) == []
    resumen = _resumen_verificado()
    resumen["equivalencia_evidently"]["max_dif_abs"] = dc.TOLERANCIA_EVIDENTLY
    assert dc.fallas_de_verificacion(resumen) == []  # la tolerancia es inclusiva

    resumen = _resumen_verificado()
    resumen[seccion][campo] = valor
    fallas = dc.fallas_de_verificacion(resumen)

    assert len(fallas) == 1 and texto in fallas[0]


def test_cli_escribe_las_cinco_salidas(tmp_path, viajes_sinteticos):
    salida = tmp_path / "p1"
    argumentos = [
        "--historical-trips",
        str(viajes_sinteticos["historico"]),
        "--current-trips",
        str(viajes_sinteticos["actual"]),
        "--current-start",
        "2026-10-01",
        "--current-end-exclusive",
        "2026-11-01",
        "--output",
        str(salida),
        "--panel-campeon",
        str(viajes_sinteticos["panel"]),
    ]

    resultado = CliRunner().invoke(dc.app, argumentos)

    assert resultado.exit_code == 0, resultado.output
    assert {ruta.name for ruta in salida.iterdir()} == set(dc.ARCHIVOS.values())
    resumen = json.loads((salida / dc.ARCHIVOS["resumen"]).read_text(encoding="utf-8"))
    assert resumen["referencia"]["inicio"] == "2025-01-01"
    assert resumen["referencia"]["fin"] == "2026-04-29"
    assert resumen["referencia"]["meses_excluidos_de_calibracion"] == {"2025-01": 2}
    assert resumen["panel_campeon"]["igual"]
    actual = resumen["mes_actual"]
    assert actual["estaciones_sin_historia"] == ["555"]
    assert actual["filas_inactivas"] == 16  # la estación 7 sin viajes del 1 de sep al 14 de oct
    assert actual["estaciones_existentes"] == 2 and actual["estaciones_nuevas"] == 2
    assert [m["mes"] for m in resumen["meses_intermedios"]] == [
        "2026-05",
        "2026-06",
        "2026-07",
        "2026-08",
        "2026-09",
    ]
    assert set(resumen["entradas"]) == {"historical_trips", "current_trips", "panel_campeon"}

    tabla = pd.read_csv(salida / dc.ARCHIVOS["tabla_mes_actual"], index_col="variable")
    mensual = pd.read_csv(salida / dc.ARCHIVOS["drift_mensual"])
    umbrales = pd.read_csv(salida / dc.ARCHIVOS["umbrales"], index_col="variable")
    assert tabla.index.tolist() == FEATURES and umbrales.index.tolist() == FEATURES
    assert len(mensual) == 5 * len(FEATURES)
    assert (tabla["drift_calibrado"] == (tabla["distancia"] > umbrales["umbral_calibrado"])).all()
    assert resumen["drift_mes_actual"]["calibrado"]["todas"] == (
        tabla.index[tabla["drift_calibrado"]].tolist()
    )
    assert resumen["verificaciones"]["realizadas"] == ["panel_campeon"]
    assert resumen["verificaciones"]["fallas"] == []


@pytest.fixture(scope="module")
def tabla_sintetica(tmp_path_factory, viajes_sinteticos) -> pd.DataFrame:
    """Tabla del mes actual de los viajes sintéticos, sin verificaciones."""
    salida = tmp_path_factory.mktemp("tabla_p1")
    dc.run(viajes_sinteticos["historico"], viajes_sinteticos["actual"], salida)
    return pd.read_csv(salida / dc.ARCHIVOS["tabla_mes_actual"], index_col="variable")


def _reporte_evidently(tabla: pd.DataFrame) -> dict:
    """Reporte con la forma del JSON de Evidently cuyos scores son las distancias del módulo."""
    return {
        "metrics": [
            {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": variable,
                    "method": dc.METODOS_EVIDENTLY[fila.metodo],
                    "threshold": dc.UMBRAL_EVIDENTLY,
                },
                "value": float(fila.distancia),
            }
            for variable, fila in tabla.iterrows()
        ]
    }


def _invocar_con_verificaciones(carpeta, viajes, panel: pd.DataFrame, reporte: dict):
    """Corre el CLI con un panel del campeón y un reporte de Evidently escritos en ``carpeta``.

    Regresa el resultado del CLI, el resumen escrito y los mensajes del log por nivel.
    """
    panel.to_parquet(carpeta / "panel.parquet", index=False)
    (carpeta / "p1_data_drift.json").write_text(json.dumps(reporte), encoding="utf-8")
    salida = carpeta / "p1"
    mensajes = []
    sumidero = logger.add(lambda m: mensajes.append((m.record["level"].name, m.record["message"])))
    try:
        resultado = CliRunner().invoke(
            dc.app,
            [
                "--historical-trips",
                str(viajes["historico"]),
                "--current-trips",
                str(viajes["actual"]),
                "--output",
                str(salida),
                "--panel-campeon",
                str(carpeta / "panel.parquet"),
                "--evidently-json",
                str(carpeta / "p1_data_drift.json"),
            ],
        )
    finally:
        logger.remove(sumidero)
    resumen = json.loads((salida / dc.ARCHIVOS["resumen"]).read_text(encoding="utf-8"))
    return resultado, resumen, mensajes


def test_cli_termina_en_0_si_el_panel_y_evidently_coinciden(
    tmp_path, viajes_sinteticos, tabla_sintetica
):
    resultado, resumen, mensajes = _invocar_con_verificaciones(
        tmp_path,
        viajes_sinteticos,
        pd.read_parquet(viajes_sinteticos["panel"]),
        _reporte_evidently(tabla_sintetica),
    )

    assert resultado.exit_code == 0, resultado.output
    assert not [texto for nivel, texto in mensajes if nivel == "ERROR"]
    assert any("idéntico al del campeón: True" in texto for _, texto in mensajes)
    assert any(nivel == "SUCCESS" for nivel, _ in mensajes)
    assert resumen["verificaciones"]["realizadas"] == ["panel_campeon", "evidently_json"]
    assert resumen["verificaciones"]["fallas"] == []
    equivalencia = resumen["equivalencia_evidently"]
    assert equivalencia["max_dif_abs"] <= dc.TOLERANCIA_EVIDENTLY
    assert equivalencia["variables_sin_score"] == []
    assert set(resumen["entradas"]) == {
        "historical_trips",
        "current_trips",
        "panel_campeon",
        "evidently_json",
    }


@pytest.mark.parametrize("alteracion", ["panel", "score", "variable_ausente"])
def test_cli_falla_si_el_panel_o_evidently_no_coinciden(
    tmp_path, viajes_sinteticos, tabla_sintetica, alteracion
):
    panel = pd.read_parquet(viajes_sinteticos["panel"])
    reporte = _reporte_evidently(tabla_sintetica)
    if alteracion == "panel":
        panel.loc[0, "salidas"] += 5  # una celda del panel del campeón distinta
    elif alteracion == "score":
        reporte["metrics"][3]["value"] += 1e-6  # lag_2 fuera de la tolerancia
    else:
        del reporte["metrics"][1]  # el reporte no trae festivo

    resultado, resumen, mensajes = _invocar_con_verificaciones(
        tmp_path, viajes_sinteticos, panel, reporte
    )

    # las salidas se escriben para revisar la falla, pero el código de salida la señala
    assert resultado.exit_code == 1
    fallas = resumen["verificaciones"]["fallas"]
    errores = [texto for nivel, texto in mensajes if nivel == "ERROR"]
    assert len(fallas) >= 1 and set(fallas) <= set(errores)
    assert any("no pasan las verificaciones" in texto for texto in errores)
    assert not any(nivel == "SUCCESS" for nivel, _ in mensajes)
    equivalencia = resumen["equivalencia_evidently"]
    if alteracion == "panel":
        assert not resumen["panel_campeon"]["igual"]
        assert equivalencia["max_dif_abs"] <= dc.TOLERANCIA_EVIDENTLY
    elif alteracion == "score":
        assert resumen["panel_campeon"]["igual"]
        assert equivalencia["max_dif_abs"] == pytest.approx(1e-6, rel=1e-6)
    else:
        assert equivalencia["variables_sin_score"] == ["festivo"]
        assert equivalencia["max_dif_abs"] is None
