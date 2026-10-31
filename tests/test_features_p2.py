import numpy as np
import pandas as pd
import pytest

from mibici.features_p2 import acumular_ocupacion, cargar_viajes, etiquetar_ocupacion


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
    import pandas as pd

    from mibici.features_p2 import FEATURES, agregar_ocupacion, construir_features

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
    import pandas as pd

    from mibici.features_p2 import agregar_ocupacion, construir_features

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


@pytest.mark.parametrize("cap", [10, 20, 30, 40, 50])
def test_umbrales_exactos_cuentan_como_segura(cap):
    # justo en 10 % y 90 % la estacion es segura; una bicicleta menos o mas ya es riesgo
    bajo, alto = cap // 10, cap - cap // 10
    occ = [bajo - 1, bajo, alto, alto + 1]
    assert list(etiquetar_ocupacion(occ, [cap] * 4)) == [
        "sin_bicicletas",
        "segura",
        "segura",
        "sin_anclajes",
    ]


def test_features_con_menos_de_una_semana_de_historia():
    from mibici.features_p2 import agregar_ocupacion, construir_features

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
