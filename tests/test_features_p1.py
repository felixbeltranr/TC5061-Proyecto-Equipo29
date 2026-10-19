import numpy as np
import pandas as pd
import pytest

from mibici import features as fe


def hacer_panel(conteos: dict[int, list[int]], inicio: str = "2026-01-01") -> pd.DataFrame:
    """Panel estacion-dia con dias consecutivos a partir de inicio."""
    piezas = []
    for estacion, valores in conteos.items():
        fechas = pd.date_range(inicio, periods=len(valores), freq="D")
        piezas.append(
            pd.DataFrame(
                {
                    "estacion": np.int32(estacion),
                    "fecha": fechas,
                    "salidas": np.array(valores, dtype="int32"),
                    "primer_dia": fechas[0],
                }
            )
        )
    return pd.concat(piezas, ignore_index=True)


def test_lag_minimo_sale_de_la_hora_de_emision():
    objetivo = pd.Timestamp("2026-05-10")
    un_dia, dos_dias = pd.Timedelta(days=1), pd.Timedelta(days=2)
    assert fe.LAG_MINIMO == 2
    # a las 23:00 de D-1 el dia D-1 aun no cierra; D-2 ya esta disponible
    assert fe.emision(objetivo) == pd.Timestamp("2026-05-09 23:00")
    assert fe.disponible_desde(objetivo - un_dia) > fe.emision(objetivo)
    assert fe.disponible_desde(objetivo - dos_dias) <= fe.emision(objetivo)
    assert fe.ultimo_dia_conocido(objetivo) == objetivo - dos_dias


def test_pascua_de_fechas_conocidas():
    assert fe.pascua(2024).isoformat() == "2024-03-31"
    assert fe.pascua(2025).isoformat() == "2025-04-20"
    assert fe.pascua(2026).isoformat() == "2026-04-05"


def test_festivos_2025_y_2026():
    dias = set(fe.festivos([2025, 2026]).strftime("%Y-%m-%d"))
    esperados = {
        "2025-01-01", "2025-02-03", "2025-03-17", "2025-04-17", "2025-04-18", "2025-05-01",
        "2025-09-16", "2025-11-17", "2025-12-25",
        "2026-01-01", "2026-02-02", "2026-03-16", "2026-04-02", "2026-04-03", "2026-05-01",
        "2026-09-16", "2026-11-16", "2026-12-25",
    }  # fmt: skip
    assert dias == esperados


def test_panel_cuenta_por_fecha_local_y_rellena_ceros():
    viajes = pd.DataFrame(
        {
            "Inicio_del_viaje": pd.to_datetime(
                [
                    "2026-03-01 23:59:59",  # estacion 1, dia 1
                    "2026-03-02 00:00:00",  # estacion 1, dia 2 (la medianoche abre el dia nuevo)
                    "2026-03-04 10:00:00",  # estacion 1, dia 4 (el dia 3 queda en cero)
                    "2026-03-04 11:00:00",  # estacion 1, dia 4
                    "2026-03-03 08:00:00",  # estacion 2 entra al panel el dia 3
                ]
            ),
            "Origen_Id": pd.array([1, 1, 1, 1, 2], dtype="Int64"),
        }
    )
    panel, resumen = fe.construir_panel_diario(viajes, "2026-03-01", "2026-03-10")
    est1 = panel[panel["estacion"] == 1].set_index("fecha")["salidas"]
    est2 = panel[panel["estacion"] == 2].set_index("fecha")["salidas"]
    assert est1.tolist() == [1, 1, 0, 2]
    # no se inventan filas antes de la primera salida; el panel termina en el ultimo dia con viajes
    assert est2.index.min() == pd.Timestamp("2026-03-03") and est2.tolist() == [1, 0]
    assert panel["fecha"].max() == pd.Timestamp("2026-03-04")
    assert (panel.loc[panel["estacion"] == 2, "primer_dia"] == pd.Timestamp("2026-03-03")).all()
    assert resumen["salidas_contadas"] == 5 and resumen["dias_con_cero_rellenados"] == 2


def test_panel_descarta_filas_invalidas_y_las_cuenta():
    viajes = pd.DataFrame(
        {
            "Inicio_del_viaje": pd.to_datetime(
                [
                    "2026-03-01 10:00",
                    None,
                    "2026-03-01 11:00",
                    "2030-01-05 10:00",
                    "2026-03-02 09:00",
                ]
            ),
            "Origen_Id": pd.array([1, 1, None, 1, 1], dtype="Int64"),
        }
    )
    panel, resumen = fe.construir_panel_diario(viajes, "2026-03-01", "2026-03-31")
    assert resumen["viajes_totales"] == 5
    assert resumen["viajes_sin_fecha_o_estacion"] == 2
    assert resumen["viajes_fuera_de_rango"] == 1
    assert panel["salidas"].tolist() == [1, 1]


def test_cargar_salidas_no_lee_columnas_de_usuario(tmp_path, monkeypatch):
    viajes = pd.DataFrame(
        {
            "Viaje_Id": [1, 2],
            "Usuario_Id": [10, 11],
            "Genero": ["F", "M"],
            "Año_de_nacimiento": [1990, 1985],
            "Inicio_del_viaje": pd.to_datetime(["2026-03-01 10:00", "2026-03-01 11:00"]),
            "Fin_del_viaje": pd.to_datetime(["2026-03-01 10:20", "2026-03-01 11:20"]),
            "Origen_Id": [1, 2],
            "Destino_Id": [2, 1],
        }
    )
    ruta = tmp_path / "combined.parquet"
    viajes.to_parquet(ruta, index=False)
    ruta_csv = tmp_path / "combined.csv"
    viajes.to_csv(ruta_csv, index=False)
    # no basta con devolver dos columnas: al disco solo se le piden esas dos
    pedidas = []
    leer_parquet, leer_csv = pd.read_parquet, pd.read_csv

    def espiar_parquet(*args, **kwargs):
        pedidas.append(kwargs.get("columns"))
        return leer_parquet(*args, **kwargs)

    def espiar_csv(*args, **kwargs):
        pedidas.append(kwargs.get("usecols"))
        return leer_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", espiar_parquet)
    monkeypatch.setattr(pd, "read_csv", espiar_csv)
    assert list(fe.cargar_salidas(ruta).columns) == ["Inicio_del_viaje", "Origen_Id"]
    assert list(fe.cargar_salidas(ruta_csv).columns) == ["Inicio_del_viaje", "Origen_Id"]
    assert pedidas == [["Inicio_del_viaje", "Origen_Id"]] * 2


def test_cargar_salidas_lee_fechas_iso_con_formatos_mezclados(tmp_path):
    ruta = tmp_path / "combined.csv"
    ruta.write_text(
        "Usuario_Id,Inicio_del_viaje,Origen_Id\n"
        "1,2026-03-01 10:00:00,1\n"
        "2,2026-03-01 11:00:00.250,1\n"  # con fraccion de segundo
        "3,2026-03-02T09:30:00,2\n"  # con T entre fecha y hora
        "4,,2\n"  # sin fecha: queda vacia y el panel la cuenta como invalida
    )
    fechas = fe.cargar_salidas(ruta)["Inicio_del_viaje"]
    assert fechas.iloc[:3].tolist() == [
        pd.Timestamp("2026-03-01 10:00:00"),
        pd.Timestamp("2026-03-01 11:00:00.250"),
        pd.Timestamp("2026-03-02 09:30:00"),
    ]
    assert pd.isna(fechas.iloc[3])


def test_cargar_salidas_se_detiene_si_muchas_fechas_son_ilegibles(tmp_path):
    ruta = tmp_path / "combined.csv"
    encabezado = ["Inicio_del_viaje,Origen_Id"]
    # 2 de 100 en otro formato: mas del 1 %, se detiene en lugar de descartarlas en silencio
    ruta.write_text("\n".join(encabezado + ["2026-03-01 10:00:00,1"] * 98 + ["01/03/2026,1"] * 2))
    with pytest.raises(ValueError, match="ilegibles"):
        fe.cargar_salidas(ruta)
    # 1 de 200: se avisa, se descarta y la carga sigue
    ruta.write_text("\n".join(encabezado + ["2026-03-01 10:00:00,1"] * 199 + ["01/03/2026,1"]))
    assert fe.cargar_salidas(ruta)["Inicio_del_viaje"].isna().sum() == 1


def test_rezagos_y_medias_con_valores_conocidos():
    conteos = list(range(1, 61))  # el dia de indice t tiene t + 1 salidas
    df = fe.construir_features(hacer_panel({1: conteos})).set_index("fecha")
    t = 40
    fila = df.iloc[t]
    c = np.array(conteos, dtype=float)
    assert fila["objetivo"] == c[t]
    assert fila["lag_2"] == c[t - 2]
    assert [fila[f"lag_{k}"] for k in (7, 14, 21, 28)] == [c[t - k] for k in (7, 14, 21, 28)]
    assert fila["media_7d"] == pytest.approx(c[t - 8 : t - 1].mean())  # dias t-8 ... t-2
    assert fila["media_28d"] == pytest.approx(c[t - 29 : t - 1].mean())  # dias t-29 ... t-2
    assert fila["media_misma_dsem_4"] == pytest.approx(
        np.mean([c[t - k] for k in (7, 14, 21, 28)])
    )
    assert fila["baseline_semana_previa"] == c[t - 7]
    assert fila["baseline_persistencia"] == c[t - 2]


def test_historia_minima_para_ser_elegible():
    df = fe.construir_features(hacer_panel({1: [5] * 40}))
    # media_28d usa los dias t-29 ... t-2, asi que el primer dia elegible es el de indice 29
    assert df["sin_historia"].tolist() == [True] * 29 + [False] * 11
    assert df["elegible"].tolist() == [False] * 29 + [True] * 11


def test_estacion_inactiva_deja_de_ser_elegible_sin_usar_el_futuro():
    # 10 dias con 5 salidas y luego solo ceros: la ventana de 28 dias se vacia en t = 39
    df = fe.construir_features(hacer_panel({1: [5] * 10 + [0] * 60}))
    assert bool(df.loc[38, "elegible"]) and not bool(df.loc[38, "inactiva"])
    assert bool(df.loc[39, "inactiva"]) and not bool(df.loc[39, "elegible"])
    assert df.loc[39:, "inactiva"].all()


def test_rechaza_panel_con_huecos():
    panel = hacer_panel({1: [1] * 40})
    with pytest.raises(ValueError):
        fe.construir_features(panel.drop(index=20))


def test_banderas_de_festivo_del_dia_y_de_hace_una_semana():
    panel = hacer_panel({1: [3] * 12}, inicio="2026-04-01")
    df = fe.construir_features(panel).set_index("fecha")
    assert df.loc["2026-04-02", "festivo"] == 1 and df.loc["2026-04-03", "festivo"] == 1
    assert df.loc["2026-04-01", "festivo"] == 0
    assert df.loc["2026-04-09", "festivo_lag7"] == 1 and df.loc["2026-04-10", "festivo_lag7"] == 1
    assert df.loc["2026-04-08", "festivo_lag7"] == 0


def panel_aleatorio(dias: int = 100, semilla: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    return hacer_panel({e: rng.poisson(20 + 5 * e, dias).tolist() for e in (1, 2, 3)})


COLUMNAS_CAUSALES = [
    *fe.FEATURES,
    "baseline_semana_previa",
    "baseline_persistencia",
    "sin_historia",
    "inactiva",
    "elegible",
]


def test_features_no_usan_el_dia_anterior_ni_el_objetivo():
    base = panel_aleatorio()
    t = 70  # indice del dia objetivo
    fecha_objetivo = base["fecha"].iloc[t]
    alterado = base.copy()
    # cambiar todo lo que pasa desde D-1 en adelante, incluido el propio dia objetivo
    desde = alterado["fecha"] >= fecha_objetivo - pd.Timedelta(days=1)
    alterado.loc[desde, "salidas"] += 500
    a = fe.construir_features(base).set_index(["estacion", "fecha"])
    b = fe.construir_features(alterado).set_index(["estacion", "fecha"])
    for estacion in (1, 2, 3):
        pd.testing.assert_series_equal(
            a.loc[(estacion, fecha_objetivo), COLUMNAS_CAUSALES],
            b.loc[(estacion, fecha_objetivo), COLUMNAS_CAUSALES],
        )
    # el objetivo si cambia: la prueba detecta diferencias cuando las hay
    assert (a["objetivo"] != b["objetivo"]).any()


def test_features_responden_a_d_menos_2():
    base = panel_aleatorio()
    t = 70
    fecha_objetivo = base["fecha"].iloc[t]
    alterado = base.copy()
    alterado.loc[alterado["fecha"] == fecha_objetivo - pd.Timedelta(days=2), "salidas"] += 500
    a = fe.construir_features(base).set_index(["estacion", "fecha"])
    b = fe.construir_features(alterado).set_index(["estacion", "fecha"])
    for columna in ("lag_2", "media_7d", "media_28d", "baseline_persistencia"):
        assert (b.loc[(1, fecha_objetivo), columna] != a.loc[(1, fecha_objetivo), columna]).all()


def test_features_iguales_si_se_corta_el_panel_en_el_dia_objetivo():
    # lo que se calcula para un dia no cambia aunque no existan los datos posteriores
    completo = panel_aleatorio(dias=120)
    fecha_corte = completo["fecha"].iloc[80]
    recortado = completo[completo["fecha"] <= fecha_corte]
    a = fe.construir_features(completo).set_index(["estacion", "fecha"])
    b = fe.construir_features(recortado).set_index(["estacion", "fecha"])
    pd.testing.assert_frame_equal(a.loc[b.index, COLUMNAS_CAUSALES], b[COLUMNAS_CAUSALES])
