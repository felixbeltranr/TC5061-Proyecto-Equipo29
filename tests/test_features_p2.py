import numpy as np
import pandas as pd
import pytest

from mibici.features_p2 import (
    FEATURES,
    acumular_ocupacion,
    agregar_ocupacion,
    cargar_viajes,
    construir_features,
    etiquetar_ocupacion,
)


def test_etiquetas_por_umbral():
    cap = np.array([20, 20, 20, 20, 20])
    occ = np.array([0, 1.9, 2, 18, 18.5])
    assert list(etiquetar_ocupacion(occ, cap)) == [
        "sin_bicicletas",
        "sin_bicicletas",
        "segura",
        "segura",
        "sin_anclajes",
    ]


def test_estacion_llena_es_sin_anclajes():
    assert etiquetar_ocupacion([15], [15])[0] == "sin_anclajes"


def test_rechaza_ocupacion_fuera_de_rango():
    with pytest.raises(ValueError):
        etiquetar_ocupacion([16], [15])
    with pytest.raises(ValueError):
        etiquetar_ocupacion([-1], [15])


def test_rechaza_capacidad_cero():
    with pytest.raises(ValueError):
        etiquetar_ocupacion([0], [0])


def test_acumulado_respeta_capacidad():
    neto = np.array([[5, 5, -20, 3]])
    occ = acumular_ocupacion(neto, np.array([10]), 0.5)
    assert occ.tolist() == [[10, 10, 0, 3]]
    assert (occ <= 10).all() and (occ >= 0).all()


def test_features_no_usan_la_hora_objetivo():
    horas = pd.date_range("2026-01-01 05:00", periods=24 * 10, freq="h")
    rng = np.random.default_rng(0)
    base = pd.DataFrame(
        {
            "ts": horas,
            "estacion": 1,
            "capacidad": np.float32(20),
            "salidas": rng.integers(0, 4, len(horas)).astype("float32"),
            "llegadas": rng.integers(0, 4, len(horas)).astype("float32"),
        }
    )
    base["neto"] = base["llegadas"] - base["salidas"]
    base["primer_dia"] = horas[0]
    t = 24 * 9 + 10
    alterado = base.copy()
    # cambiar solo lo que pasa en t+1 y despues
    alterado.loc[t + 1 :, "llegadas"] += 15
    alterado["neto"] = alterado["llegadas"] - alterado["salidas"]

    cols = [c for c in FEATURES if not c.startswith("est_")]
    a = construir_features(agregar_ocupacion(base, 0.5)).set_index("ts")
    b = construir_features(agregar_ocupacion(alterado, 0.5)).set_index("ts")
    ts_t = horas[t]
    pd.testing.assert_series_equal(a.loc[ts_t, cols], b.loc[ts_t, cols])


def test_media_misma_hora_usa_cuatro_semanas():
    # 36 dias de servicio por estacion; el neto de cada hora es distinto y conocido
    horas = pd.date_range("2026-01-01 05:00", periods=24 * 36, freq="h")
    piezas = []
    for est, desfase in [(1, 0.0), (2, 10_000.0)]:
        llegadas = np.arange(len(horas), dtype="float32") + desfase
        piezas.append(
            pd.DataFrame(
                {
                    "ts": horas,
                    "estacion": est,
                    "capacidad": np.float32(20),
                    "salidas": np.float32(0),
                    "llegadas": llegadas,
                }
            )
        )
    panel = pd.concat(piezas, ignore_index=True)
    panel["neto"] = panel["llegadas"] - panel["salidas"]
    panel["primer_dia"] = horas[0]
    neto = panel.set_index(["estacion", "ts"])["neto"]

    df = construir_features(agregar_ocupacion(panel, 0.5))
    completas = df[df["ts_objetivo"] - pd.Timedelta(weeks=4) >= horas[0]]
    assert len(completas) > 0
    for est, ts_obj, valor, lag168 in completas[
        ["estacion", "ts_objetivo", "neto_media168_misma_hora", "neto_lag168"]
    ].itertuples(index=False):
        # mismo instante del objetivo hace 1, 2, 3 y 4 semanas, dentro de la misma estacion
        esperado = np.mean([neto[(est, ts_obj - pd.Timedelta(weeks=k))] for k in range(1, 5)])
        assert valor == esperado
        assert lag168 == neto[(est, ts_obj - pd.Timedelta(weeks=1))]


def test_cargar_viajes_descarta_duplicados_y_menos_de_un_minuto(tmp_path):
    # misma limpieza que P3: el duplicado y el viaje de 59 s salen; el de 60 s se queda
    viajes = pd.DataFrame(
        {
            "Viaje_Id": [1, 1, 2, 3],
            "Usuario_Id": [9, 9, 9, 9],
            "Inicio_del_viaje": ["2025-03-03 08:00:00"] * 4,
            "Fin_del_viaje": [
                "2025-03-03 08:10:00",
                "2025-03-03 08:10:00",
                "2025-03-03 08:00:59",
                "2025-03-03 08:01:00",
            ],
            "Origen_Id": [1, 1, 1, 2],
            "Destino_Id": [2, 2, 1, 1],
        }
    )
    ruta = tmp_path / "combined_dataset.csv"
    viajes.to_csv(ruta, index=False)
    validos, bitacora = cargar_viajes(ruta)
    assert sorted(validos["Viaje_Id"]) == [1, 3]
    assert "Usuario_Id" not in validos
    tot = bitacora.drop(columns="mes").sum()
    assert tot.duplicados_exactos == 1
    assert tot.duracion_menor_1min == 1
    assert tot.viajes_validos == 2


def test_manifiesto_se_escribe_con_fin_de_linea_lf(tmp_path):
    from mibici import manifiesto_p2
    from mibici.config import CONFIG_P2

    (tmp_path / "datos_abiertos_2025_01.csv").write_bytes(b"a,b\n1,2\n")
    (tmp_path / CONFIG_P2.nomenclatura_path.name).write_bytes(b"id,name\n1,x\n")
    salida = tmp_path / "ref" / "manifiesto.json"
    manifiesto_p2.main(salida=salida, carpeta=tmp_path)
    assert b"\r\n" not in salida.read_bytes()


def test_umbrales_exactos_cuentan_como_segura():
    # justo en 10 % y 90 % la estacion es segura; una bicicleta menos o mas ya es riesgo
    cap = 30
    bajo, alto = 3, 27
    occ = [bajo - 1, bajo, alto, alto + 1]
    assert list(etiquetar_ocupacion(occ, [cap] * 4)) == [
        "sin_bicicletas",
        "segura",
        "segura",
        "sin_anclajes",
    ]


def test_features_con_menos_de_una_semana_de_historia():
    # tres dias de servicio: alcanza para los rezagos de 24 h pero no para los de 168 h
    horas = pd.date_range("2026-01-01 05:00", periods=24 * 3, freq="h")
    panel = pd.DataFrame(
        {
            "ts": horas,
            "estacion": 1,
            "capacidad": np.float32(20),
            "salidas": np.float32(1),
            "llegadas": np.arange(len(horas), dtype="float32") % 3,
        }
    )
    panel["neto"] = panel["llegadas"] - panel["salidas"]
    panel["primer_dia"] = horas[0]
    df = construir_features(agregar_ocupacion(panel, 0.5))
    # las filas no se descartan por falta de historia: los modelos manejan los faltantes
    assert len(df) > 0 and df["objetivo"].notna().all()
    for col in ("neto_lag168", "ocupacion_lag168", "neto_media168_misma_hora", "baseline_semana"):
        assert df[col].isna().all(), col
    # el rezago de 24 h existe desde que hay 23 horas previas en la estacion
    neto = panel.set_index("ts")["neto"]
    con_dia = df[df["ts_objetivo"] - pd.Timedelta(hours=24) >= horas[0]]
    sin_dia = df[df["ts_objetivo"] - pd.Timedelta(hours=24) < horas[0]]
    assert len(con_dia) > 0 and len(sin_dia) > 0
    assert sin_dia["neto_lag24"].isna().all()
    esperado = [neto[t - pd.Timedelta(hours=24)] for t in con_dia["ts_objetivo"]]
    assert con_dia["neto_lag24"].tolist() == esperado
    # las medias moviles usan lo que haya desde la primera hora
    assert df["neto_media24"].notna().all()


def panel_dos_estaciones(dias: int = 17) -> pd.DataFrame:
    horas = pd.date_range("2026-01-01 05:00", periods=24 * dias, freq="h")
    rng = np.random.default_rng(29)
    piezas = [
        pd.DataFrame(
            {
                "ts": horas,
                "estacion": est,
                "capacidad": np.float32(12),
                "salidas": rng.integers(0, 6, len(horas)).astype("float32"),
                "llegadas": rng.integers(0, 6, len(horas)).astype("float32"),
            }
        )
        for est in (1, 2)
    ]
    panel = pd.concat(piezas, ignore_index=True)
    panel["neto"] = panel["llegadas"] - panel["salidas"]
    panel["primer_dia"] = horas[0]
    return agregar_ocupacion(panel, 0.5)


def features_una_hora_congeladas(panel: pd.DataFrame) -> pd.DataFrame:
    """Calculo de construir_features antes de parametrizar el horizonte."""
    df = panel.sort_values(["estacion", "ts"]).reset_index(drop=True)
    g = df.groupby("estacion", sort=False)
    df["objetivo"] = g["clase_actual"].shift(-1)
    df["ts_objetivo"] = df["ts"] + pd.Timedelta(hours=1)
    df["hora"] = df["ts_objetivo"].dt.hour.astype("int8")
    df["dia_semana"] = df["ts_objetivo"].dt.dayofweek.astype("int8")
    df["fin_semana"] = (df["dia_semana"] >= 5).astype("int8")
    df["ocupacion_actual"] = (df["ocupacion"] / df["capacidad"]).astype("float32")
    df["neto_lag1"] = df["neto"]
    df["neto_lag24"] = g["neto"].shift(23)
    df["neto_lag168"] = g["neto"].shift(167)
    df["salidas_lag1"] = df["salidas"]
    df["llegadas_lag1"] = df["llegadas"]
    df["neto_media3"] = g["neto"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["neto_media24"] = g["neto"].transform(lambda s: s.rolling(24, min_periods=1).mean())
    sem = [g["neto"].shift(168 * k - 1) for k in range(1, 5)]
    df["neto_media168_misma_hora"] = pd.concat(sem, axis=1).mean(axis=1)
    razon = df["ocupacion"] / df["capacidad"]
    df["ocupacion_lag24"] = razon.groupby(df["estacion"]).shift(23)
    df["ocupacion_lag168"] = razon.groupby(df["estacion"]).shift(167)
    df["baseline_semana"] = g["clase_actual"].shift(167)
    df = df[df["objetivo"].notna() & df["hora"].isin(list(range(6, 24)))]
    return df.reset_index(drop=True)


def test_horizonte_uno_es_identico_al_calculo_entregado():
    panel = panel_dos_estaciones()
    pd.testing.assert_frame_equal(construir_features(panel), features_una_hora_congeladas(panel))


@pytest.mark.parametrize("h", [1, 2, 3, 4])
def test_horizonte_desplaza_objetivo_y_reancla_los_rezagos(h):
    panel = panel_dos_estaciones()
    df = construir_features(panel, horizonte=h)
    serie = panel.set_index(["estacion", "ts"])
    obj = list(zip(df["estacion"], df["ts_objetivo"]))
    assert (df["ts_objetivo"] - df["ts"] == pd.Timedelta(hours=h)).all()
    assert df["objetivo"].tolist() == serie.loc[obj, "clase_actual"].tolist()
    con_dia = df[df["neto_lag24"].notna()]
    hace_un_dia = list(zip(con_dia["estacion"], con_dia["ts_objetivo"] - pd.Timedelta(days=1)))
    assert con_dia["neto_lag24"].tolist() == serie.loc[hace_un_dia, "neto"].tolist()
    assert (df["ts"].dt.hour >= 5).all()
    assert set(df["hora"]) == set(range(5 + h, 24))


def test_horizonte_con_horas_objetivo_fijas_y_horizonte_invalido():
    panel = panel_dos_estaciones(3)
    df = construir_features(panel, horizonte=4, horas_objetivo=range(9, 24))
    assert set(df["hora"]) == set(range(9, 24))
    with pytest.raises(ValueError):
        construir_features(panel, horizonte=0)
