import json
from pathlib import Path

from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest

from mibici import features as fe
from mibici.modeling import train_p1 as t1

ESTACIONES = (1, 2, 3, 4)
SEMANA = np.array([1.2, 1.2, 1.2, 1.2, 1.1, 0.6, 0.4])  # lunes ... domingo
INICIO_PRUEBA = t1.CORTES["test_inicio"]
# Desde la prueba cambia todo: patron semanal al reves, niveles de las estaciones invertidos y
# el triple de viajes. Sirve para comprobar que ninguna decision depende de la prueba.
OTRO_MUNDO = {
    "desde": INICIO_PRUEBA,
    "invertir_semana": True,
    "invertir_niveles": True,
    "factor": 3.0,
}
# Dos configuraciones por candidato: basta para que afinar tenga que elegir y es mas rapido
REJILLA_CORTA = {
    "ALPHAS_POISSON": [1e-3, 1.0],
    "REJILLA_HGB": [
        {"learning_rate": 0.1, "max_leaf_nodes": 7, "max_iter": 50, "l2_regularization": 0.0},
        {"learning_rate": 0.05, "max_leaf_nodes": 15, "max_iter": 100, "l2_regularization": 1.0},
    ],
}


def hacer_viajes(
    estaciones=ESTACIONES,
    *,
    desde: str | None = None,
    invertir_semana: bool = False,
    invertir_niveles: bool = False,
    factor: float = 1.0,
) -> pd.DataFrame:
    """Viajes sinteticos 2025-01 a 2026-09 con patron semanal y columnas de usuario ficticias.

    La estacion e tiene en promedio 10 * e salidas diarias moduladas por SEMANA. Desde la fecha
    desde, los conteos salen de otra secuencia aleatoria con el patron y los niveles indicados.
    Cada estacion usa sus propias secuencias, asi que los viajes anteriores a desde son los
    mismos haya o no cambio.
    """
    dias = pd.date_range("2025-01-01", "2026-09-30", freq="D")
    orden = tuple(reversed(estaciones)) if invertir_niveles else tuple(estaciones)
    niveles = dict(zip(estaciones, orden, strict=True))
    semana_nueva = SEMANA[::-1] if invertir_semana else SEMANA
    piezas = []
    for estacion in estaciones:
        n = np.random.default_rng([estacion, 0]).poisson(10 * estacion * SEMANA[dias.dayofweek])
        if desde is not None:
            media = factor * 10 * niveles[estacion] * semana_nueva[dias.dayofweek]
            nuevos = np.random.default_rng([estacion, 1]).poisson(media)
            n = np.where(dias >= pd.Timestamp(desde), nuevos, n)
        segundos = np.random.default_rng([estacion, 2]).integers(0, 86400, n.sum())
        fechas = dias.repeat(n) + pd.to_timedelta(segundos, unit="s")
        piezas.append(pd.DataFrame({"Inicio_del_viaje": fechas, "Origen_Id": estacion}))
    viajes = pd.concat(piezas, ignore_index=True)
    rng = np.random.default_rng(29)  # columnas de usuario: no deben llegar a ningun resultado
    viajes["Usuario_Id"] = rng.integers(1, 10**6, len(viajes))
    viajes["Genero"] = rng.choice(["F", "M"], len(viajes))
    viajes["Año_de_nacimiento"] = rng.integers(1950, 2008, len(viajes))
    return viajes


def particionar_panel(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Features y particiones de las filas elegibles, como las arma train_p1.main."""
    df = fe.construir_features(panel)
    return t1.particionar(df[df["elegible"]])


def salidas_de(panel: pd.DataFrame, filas: pd.DataFrame, dias_atras: int) -> np.ndarray:
    """Salidas de la estacion de cada fila, tomadas del panel dias_atras antes de su fecha."""
    serie = panel.set_index(["estacion", "fecha"])["salidas"]
    indice = pd.MultiIndex.from_arrays(
        [filas["estacion"], filas["fecha"] - pd.Timedelta(days=dias_atras)]
    )
    return serie.reindex(indice).to_numpy(dtype=float)


def correr_main(carpeta: Path, viajes: pd.DataFrame) -> dict:
    """Corre mibici.features y mibici.modeling.train_p1 sobre viajes, todo dentro de carpeta."""
    entrada = carpeta / "combined_dataset.parquet"
    viajes.to_parquet(entrada, index=False)
    panel = carpeta / "p1_panel_diario.parquet"
    fe.main(entrada=entrada, salida=panel, inicio=None, fin=None)
    manifiesto = carpeta / "manifiesto.json"
    manifiesto.write_text(json.dumps({"version": "abc123def456"}))
    figuras = carpeta / "figuras"
    uri = f"sqlite:///{carpeta / 'mlflow.db'}"
    resultados = t1.main(
        [
            "--panel", str(panel), "--tracking-uri", uri, "--figuras", str(figuras),
            "--artefactos", str(carpeta / "mlartifacts"), "--manifiesto", str(manifiesto),
            "--autor", "Tester",
        ]
    )  # fmt: skip
    return {"resultados": resultados, "panel": panel, "figuras": figuras, "uri": uri}


@pytest.fixture(scope="module")
def mundo_base() -> tuple[pd.DataFrame, dict]:
    """Panel y particiones de los viajes sinteticos sin cambios."""
    panel, _ = fe.construir_panel_diario(hacer_viajes())
    return panel, particionar_panel(panel)


@pytest.fixture(scope="module")
def corridas(tmp_path_factory) -> dict:
    """train_p1.main en dos mundos identicos hasta la prueba (con la rejilla corta)."""
    base, otro = hacer_viajes(), hacer_viajes(**OTRO_MUNDO)
    antes = [
        v.loc[v["Inicio_del_viaje"] < INICIO_PRUEBA, fe.COLUMNAS_VIAJES].reset_index(drop=True)
        for v in (base, otro)
    ]
    pd.testing.assert_frame_equal(*antes)  # el cambio empieza justo en la prueba
    with pytest.MonkeyPatch.context() as mp:
        for nombre, valor in REJILLA_CORTA.items():
            mp.setattr(t1, nombre, valor)
        return {
            "base": correr_main(tmp_path_factory.mktemp("base"), base),
            "otro": correr_main(tmp_path_factory.mktemp("otro"), otro),
        }


def test_cortes_coinciden_con_p2():
    p2 = pytest.importorskip("mibici.modeling.train_p2")
    assert t1.CORTES == p2.CORTES


def test_particiones_sin_solapes_y_con_etiquetas_disponibles():
    fechas = pd.date_range("2025-01-01", "2026-10-05", freq="D")
    df = pd.DataFrame({"fecha": fechas.repeat(2), "estacion": np.tile([1, 2], len(fechas))})
    partes = t1.particionar(df)
    fit, val, train, test = (partes[k]["fecha"] for k in ("fit", "val", "train", "test"))
    assert (fit.min(), fit.max()) == (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-02-27"))
    assert (val.min(), val.max()) == (pd.Timestamp("2026-03-01"), pd.Timestamp("2026-04-29"))
    assert (train.min(), train.max()) == (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-04-29"))
    assert (test.min(), test.max()) == (pd.Timestamp("2026-05-01"), pd.Timestamp("2026-09-30"))
    assert fit.max() < val.min() and train.max() < test.min()
    un_dia = pd.Timedelta(days=1)
    for ultimo_entrenamiento, primer_pronostico in (
        (fit.max(), val.min()),
        (train.max(), test.min()),
    ):
        # la etiqueta del ultimo dia de entrenamiento ya existe al emitir el primer pronostico...
        assert fe.disponible_desde(ultimo_entrenamiento) <= fe.emision(primer_pronostico)
        # ...y la del dia siguiente todavia no: el embargo no es mas largo de lo necesario
        assert fe.disponible_desde(ultimo_entrenamiento + un_dia) > fe.emision(primer_pronostico)


def test_validacion_no_usa_etiquetas_aun_no_disponibles_al_iniciar_test(mundo_base):
    panel, antes = mundo_base
    ultimo = fe.ultimo_dia_conocido(t1.CORTES["test_inicio"])
    cambiado = panel.copy()
    cambiado.loc[cambiado["fecha"] > ultimo, "salidas"] += 100000
    despues = particionar_panel(cambiado)
    for conjunto in ("fit", "val", "train"):
        pd.testing.assert_frame_equal(antes[conjunto], despues[conjunto])
    assert antes["val"]["fecha"].max() == ultimo
    assert fe.disponible_desde(ultimo) <= fe.emision(t1.CORTES["test_inicio"])


@pytest.mark.parametrize(
    ("conjunto", "primer_pronostico"), [("fit", "val_inicio"), ("train", "test_inicio")]
)
def test_entrenamiento_no_depende_de_etiquetas_que_no_existen_al_emitir(
    mundo_base, conjunto, primer_pronostico
):
    panel, partes = mundo_base
    ultimo = fe.ultimo_dia_conocido(t1.CORTES[primer_pronostico])

    def cambiar_desde(dia: pd.Timestamp) -> pd.DataFrame:
        alterado = panel.copy()
        desde = alterado["fecha"] >= dia
        alterado.loc[desde, "salidas"] = (alterado.loc[desde, "salidas"] * 3 + 7).astype("int32")
        return particionar_panel(alterado)[conjunto]

    # lo posterior al ultimo dia conocido no mueve ni features ni etiquetas de entrenamiento
    pd.testing.assert_frame_equal(cambiar_desde(ultimo + pd.Timedelta(days=1)), partes[conjunto])
    # y la prueba si detecta cambios: la etiqueta del ultimo dia conocido si entra
    assert not cambiar_desde(ultimo).equals(partes[conjunto])


def test_particionar_falla_si_el_panel_no_llega_a_la_prueba():
    fechas = pd.date_range("2025-01-01", "2026-04-30", freq="D")
    with pytest.raises(ValueError, match="test"):
        t1.particionar(pd.DataFrame({"fecha": fechas}))


def test_metricas_con_valores_conocidos():
    m = t1.metricas(np.array([10, 20, 30]), np.array([12, 18, 30]), "test")
    assert m["test_mae"] == pytest.approx(4 / 3)
    assert m["test_rmse"] == pytest.approx(np.sqrt(8 / 3))
    assert m["test_bias"] == pytest.approx(0.0)
    assert m["test_wape"] == pytest.approx(4 / 60)
    assert m["test_n"] == 3
    assert "test_wape" not in t1.metricas(np.zeros(3), np.ones(3), "test")
    # el sesgo es pronostico menos real: positivo si se sobreestima
    assert t1.metricas(np.array([10, 10]), np.array([14, 12]), "t")["t_bias"] == pytest.approx(3)
    assert t1.metricas(np.array([10, 10]), np.array([4, 8]), "t")["t_bias"] == pytest.approx(-4)


def test_segmentos_se_calculan_solo_con_train():
    train = pd.DataFrame({"estacion": [1, 1, 2, 2, 3, 3], "objetivo": [1, 1, 10, 10, 100, 100]})
    test = pd.DataFrame({"estacion": [1, 2, 3, 4], "objetivo": [9999.0] * 4})
    segmento = t1.segmentos(train, test)
    assert segmento.tolist() == ["baja", "media", "alta", "nuevas"]
    # cambiar los valores de prueba no mueve los cortes
    test["objetivo"] = 0.0
    assert t1.segmentos(train, test).tolist() == segmento.tolist()


def test_seleccion_prefiere_el_baseline_salvo_mejora_estricta():
    base = {"semana_previa": 5.0, "persistencia": 1.0, "poisson": 5.0, "hgb_poisson": 6.0}
    assert t1.seleccionar(base) == "semana_previa"  # empate: se queda el baseline
    assert t1.seleccionar(base | {"poisson": 4.9}) == "poisson"
    assert t1.seleccionar(base | {"poisson": 4.9, "hgb_poisson": 4.0}) == "hgb_poisson"


def test_rango_meses():
    assert t1.rango_meses("2025-01-01", "2026-05-01") == "2025-01..2026-04"
    assert t1.rango_meses("2026-05-01", "2026-10-01") == "2026-05..2026-09"


def test_afinar_entrena_con_fit_y_los_modelos_finales_con_train(mundo_base, monkeypatch):
    _, partes = mundo_base
    vistos = []
    originales = {"ajustar_poisson": t1.ajustar_poisson, "ajustar_hgb": t1.ajustar_hgb}

    def espiar(nombre):
        def ajustar(entrena, *args):
            vistos.append((entrena["fecha"].max(), len(entrena)))
            return originales[nombre](entrena, *args)

        return ajustar

    for nombre in originales:
        monkeypatch.setattr(t1, nombre, espiar(nombre))
    t1.predecir(partes, 29, completo=False)
    fit = (fe.ultimo_dia_conocido(t1.CORTES["val_inicio"]), len(partes["fit"]))
    train = (fe.ultimo_dia_conocido(t1.CORTES["test_inicio"]), len(partes["train"]))
    # afinar: un Poisson y un HGB solo con fit; finales: los dos con train (hasta el fin de val)
    assert sorted(vistos) == [fit, fit, train, train]


def test_pronostico_de_prueba_no_usa_etiquetas_que_ningun_feature_ve(mundo_base):
    panel, partes = mundo_base
    # las salidas de los dos ultimos dias no son lag de ninguna fila: solo son etiquetas
    alterado = panel.copy()
    ultimos = alterado["fecha"] >= alterado["fecha"].max() - pd.Timedelta(days=1)
    alterado.loc[ultimos, "salidas"] = (alterado.loc[ultimos, "salidas"] // 2).astype("int32")
    partes_alteradas = particionar_panel(alterado)
    assert not np.array_equal(partes["test"]["objetivo"], partes_alteradas["test"]["objetivo"])
    original = t1.predecir(partes, 29, completo=False)["pred_test"]
    alterada = t1.predecir(partes_alteradas, 29, completo=False)["pred_test"]
    for modelo, pred in original.items():
        np.testing.assert_allclose(alterada[modelo], pred, err_msg=modelo)


def test_decisiones_de_validacion_no_cambian_si_cambia_la_prueba(corridas):
    a, b = corridas["base"]["resultados"], corridas["otro"]["resultados"]
    # el escenario si separa validacion y prueba: en el otro mundo gana en prueba otro modelo
    mejor_en_prueba = min(b["resultados"], key=lambda m: b["resultados"][m]["test_mae"])
    assert mejor_en_prueba != b["seleccionado"]
    assert a["seleccionado"] == b["seleccionado"]
    assert a["elegidos"] == b["elegidos"]
    for candidato in t1.CANDIDATOS:
        for config_a, config_b in zip(
            a["rejilla"][candidato], b["rejilla"][candidato], strict=True
        ):
            assert config_a == pytest.approx(config_b)
    for modelo, metricas in a["resultados"].items():
        val = {k: v for k, v in metricas.items() if k.startswith("val_")}
        assert {k: b["resultados"][modelo][k] for k in val} == pytest.approx(val)
    test_a = a["resultados"]["semana_previa"]["test_mae"]
    assert b["resultados"]["semana_previa"]["test_mae"] != pytest.approx(test_a)


def test_cada_candidato_reporta_su_mejor_configuracion_y_se_elige_por_validacion(corridas):
    for corrida in corridas.values():
        r = corrida["resultados"]
        for candidato in t1.CANDIDATOS:
            mejor = min(r["rejilla"][candidato], key=lambda x: x["val_mae"])
            assert r["elegidos"][candidato] == {k: v for k, v in mejor.items() if k != "val_mae"}
            # el MAE de validacion del run es el del modelo entrenado solo con fit
            assert r["resultados"][candidato]["val_mae"] == pytest.approx(mejor["val_mae"])
        val_mae = {m: v["val_mae"] for m, v in r["resultados"].items()}
        esperado = min(t1.CANDIDATOS, key=val_mae.get)
        if val_mae[esperado] >= val_mae["semana_previa"]:
            esperado = "semana_previa"
        assert r["seleccionado"] == esperado


def test_metricas_de_los_baselines_coinciden_con_un_calculo_independiente(corridas):
    for corrida in corridas.values():
        resultados = corrida["resultados"]["resultados"]
        panel = pd.read_parquet(corrida["panel"])
        partes = particionar_panel(panel)
        for conjunto in ("val", "test"):
            filas = partes[conjunto]
            real = salidas_de(panel, filas, 0)
            for modelo, dias_atras in (("semana_previa", 7), ("persistencia", 2)):
                error = salidas_de(panel, filas, dias_atras) - real
                m = resultados[modelo]
                assert m[f"{conjunto}_mae"] == pytest.approx(np.abs(error).mean())
                assert m[f"{conjunto}_bias"] == pytest.approx(error.mean())
                assert m[f"{conjunto}_n"] == len(filas)


def test_segmentos_de_prueba_usan_los_terciles_de_train(corridas):
    # en el otro mundo los niveles se invierten en la prueba: sus terciles darian otros grupos
    corrida = corridas["otro"]
    m = corrida["resultados"]["resultados"]["semana_previa"]
    panel = pd.read_parquet(corrida["panel"])
    partes = particionar_panel(panel)
    train, test = partes["train"], partes["test"]
    nivel = train.groupby("estacion")["objetivo"].mean()
    corte_bajo, corte_alto = np.quantile(nivel.to_numpy(), [1 / 3, 2 / 3])
    grupo = {
        e: "baja" if v <= corte_bajo else "media" if v <= corte_alto else "alta"
        for e, v in nivel.items()
    }
    segmento = test["estacion"].map(grupo).fillna("nuevas").to_numpy()
    error = np.abs(salidas_de(panel, test, 7) - salidas_de(panel, test, 0))
    for nombre in t1.SEGMENTOS:
        mascara = segmento == nombre
        if mascara.any():
            assert m[f"test_n_{nombre}"] == mascara.sum()
            assert m[f"test_mae_{nombre}"] == pytest.approx(error[mascara].mean())
        else:
            assert f"test_n_{nombre}" not in m


def test_flujo_completo_registra_runs_en_mlflow_sin_datos_de_usuario(tmp_path):
    corrida = correr_main(tmp_path, hacer_viajes())
    resultados, figuras, panel = corrida["resultados"], corrida["figuras"], corrida["panel"]

    cliente = MlflowClient(tracking_uri=corrida["uri"])
    experimento = cliente.get_experiment_by_name("mibici-equipo29")
    runs = {r.info.run_name: r for r in cliente.search_runs([experimento.experiment_id])}
    assert set(runs) == {
        "p1_baseline_semana_previa",
        "p1_baseline_persistencia",
        "p1_poisson_regularizado",
        "p1_hgb_poisson",
    }
    assert resultados["seleccionado"] in {"semana_previa", "poisson", "hgb_poisson"}

    textos = []
    for run in runs.values():
        tags, params, metricas = run.data.tags, run.data.params, run.data.metrics
        assert tags["question"] == "P1" and tags["team"] == "29" and tags["author"] == "Tester"
        assert tags["data_version"] == "abc123def456" and tags["seed"] == "29"
        assert tags["data_range"] == "2025-01..2026-09"
        assert tags["git_commit"] and tags["mlflow.source.git.commit"] == tags["git_commit"]
        assert params["train_rango"] == "2025-01..2026-04"
        assert params["val_rango"] == "2026-03..2026-04"
        assert params["test_rango"] == "2026-05..2026-09"
        assert params["data_version"] == "abc123def456"
        assert {"val_mae", "test_mae", "test_rmse", "test_bias", "test_wape"} <= set(metricas)
        assert cliente.list_artifacts(run.info.run_id), "cada run debe tener al menos un artefacto"
        textos += [*tags, *tags.values(), *params, *params.values(), *metricas]
    assert sum(r.data.tags["seleccionado_validacion"] == "si" for r in runs.values()) == 1

    # los cuatro modelos se comparan sobre las mismas filas de prueba
    assert len({r.data.metrics["test_n"] for r in runs.values()}) == 1
    # con un patron semanal claro, los candidatos le ganan a la persistencia
    persistencia = runs["p1_baseline_persistencia"].data.metrics["test_mae"]
    assert runs["p1_poisson_regularizado"].data.metrics["test_mae"] < persistencia
    assert runs["p1_hgb_poisson"].data.metrics["test_mae"] < persistencia

    for archivo in [*figuras.glob("*.json"), *figuras.glob("*.csv"), panel.with_suffix(".json")]:
        textos.append(archivo.read_text())
    prohibidas = ("usuario", "genero", "nacimiento")
    assert not [t for t in textos if any(p in str(t).lower() for p in prohibidas)]
