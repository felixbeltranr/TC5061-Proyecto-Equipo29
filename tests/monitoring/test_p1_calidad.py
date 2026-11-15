import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from mibici.features import construir_panel_diario
from mibici.monitoring import p1_calidad

INICIO, FIN = "2026-10-01", "2026-11-01"
VARIABLES_PANEL = [
    "objetivo",
    "lag_2",
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
    "media_7d",
    "media_28d",
    "media_misma_dsem_4",
]


def viajes(inicio: str, fin: str, estaciones=(7, 42), semilla: int = 29) -> pd.DataFrame:
    """Viajes sintéticos: cada estación tiene entre 1 y ~10 salidas al día, al azar.

    Cada viaje sale un minuto después del anterior de su estación y dura 10 minutos hacia la
    siguiente estación, así que ninguno repite inicio, fin, origen y destino.
    """
    rng = np.random.default_rng(semilla)
    filas = []
    for dia in pd.date_range(inicio, fin):
        for i, estacion in enumerate(estaciones):
            destino = estaciones[(i + 1) % len(estaciones)]
            for j in range(1 + int(rng.poisson(3))):
                salida = dia + pd.Timedelta(hours=8, minutes=j)
                filas.append((salida, salida + pd.Timedelta(minutes=10), estacion, destino))
    frame = pd.DataFrame(
        filas, columns=["Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"]
    )
    frame.insert(0, "Viaje_Id", np.arange(1, len(frame) + 1) + semilla * 1_000_000)
    return frame


@pytest.fixture(scope="module")
def viajes_octubre():
    return viajes("2026-10-01", "2026-10-31", semilla=30)


@pytest.fixture(scope="module")
def poblaciones(viajes_octubre):
    return p1_calidad.construir_poblaciones(
        viajes("2025-01-01", "2026-09-30"), viajes_octubre, INICIO, FIN
    )


def recargar(frame: pd.DataFrame, mascara, veces: int) -> pd.DataFrame:
    """Agrega ``veces`` copias de las filas de ``mascara`` con Viaje_Id nuevos (recarga del ETL)."""
    copias = pd.concat([frame.loc[mascara]] * veces, ignore_index=True)
    copias["Viaje_Id"] = np.arange(len(copias)) + 900_000_000
    return pd.concat([frame, copias], ignore_index=True)


def por_nombre(comprobaciones: list[dict], poblacion: str = "actual") -> dict[str, dict]:
    return {c["nombre"]: c for c in comprobaciones if c["poblacion"] == poblacion}


def test_caso_limpio_pasa_y_festivo_constante_es_advertencia(poblaciones, viajes_octubre):
    integridad = p1_calidad.comprobar_integridad(poblaciones, INICIO, FIN, viajes=viajes_octubre)

    assert [c["nombre"] for c in integridad if c["resultado"] == "fail"] == []
    actual = por_nombre(integridad)
    for nombre in (
        "viaje_id_unico",
        "viajes_con_inicio_y_origen_legibles",
        "viajes_repetidos_con_otro_viaje_id",
        "salidas_del_panel_igual_a_viajes_validos",
    ):
        assert actual[nombre]["resultado"] == "pass", nombre
    assert actual["llave_estacion_fecha_unica"]["resultado"] == "pass"
    assert actual["dia_semana_coherente_con_fecha"]["resultado"] == "pass"
    assert actual["festivo_coherente_con_fecha"]["resultado"] == "pass"
    for variable in VARIABLES_PANEL:
        assert actual[f"{variable}_consistente_con_panel"]["resultado"] == "pass"
    assert actual["cobertura_del_mes"]["valor_observado"] == {
        "dias_esperados": 31,
        "dias_con_filas_elegibles": 31,
        "dias_con_viajes": 31,
    }
    constantes = actual["columnas_constantes_o_duplicadas_explicables"]
    assert constantes["resultado"] == "warning"
    assert constantes["valor_observado"]["constantes"] == ["festivo", "festivo_lag7"]
    assert "no tiene festivos" in constantes["detalle"]["explicacion"]
    assert poblaciones.metadatos["actual"]["filas"] == 62
    assert poblaciones.metadatos["referencia"]["observado_fin"] == "2026-04-29"


def test_detecta_llave_duplicada(poblaciones):
    frame = pd.concat([poblaciones.actual, poblaciones.actual.iloc[[0]]], ignore_index=True)

    comprobacion = p1_calidad.comprobar_llave(frame, "actual")

    assert comprobacion["resultado"] == "fail"
    assert comprobacion["valor_observado"] == 1


def test_detecta_dia_semana_y_festivo_incoherentes(poblaciones):
    frame = poblaciones.actual.copy()
    frame.loc[0, "dia_semana"] = (frame.loc[0, "dia_semana"] + 1) % 7
    frame.loc[5, "festivo"] = 1  # octubre de 2026 no tiene festivos

    resultado = {c["nombre"]: c for c in p1_calidad.comprobar_calendario(frame, "actual")}

    assert resultado["dia_semana_coherente_con_fecha"]["resultado"] == "fail"
    assert resultado["dia_semana_coherente_con_fecha"]["valor_observado"] == 1
    assert resultado["festivo_coherente_con_fecha"]["valor_observado"] == 1
    assert resultado["festivo_lag7_coherente_con_fecha"]["resultado"] == "pass"


def test_detecta_rezago_y_media_inconsistentes(poblaciones):
    frame = poblaciones.actual.copy()
    frame.loc[3, "lag_7"] += 1
    frame.loc[4, "media_7d"] += 0.5

    resultado = por_nombre(
        p1_calidad.comprobar_consistencia_temporal(frame, poblaciones.panel, "actual")
    )

    assert resultado["lag_7_consistente_con_panel"]["valor_observado"] == 1
    assert resultado["media_7d_consistente_con_panel"]["valor_observado"] == 1
    fallas = {nombre for nombre, c in resultado.items() if c["resultado"] == "fail"}
    assert fallas == {"lag_7_consistente_con_panel", "media_7d_consistente_con_panel"}


def test_detecta_variable_que_usa_el_dia_anterior(poblaciones):
    """Si lag_2 tomara D−1 (aún no cerrado al emitir) en lugar de D−2, hay fuga y se detecta."""
    frame = poblaciones.actual.copy()
    matriz, estaciones, dia0 = p1_calidad._matriz_panel(poblaciones.panel)
    frame["lag_2"] = p1_calidad._salidas_previas(frame, matriz, estaciones, dia0, 1)

    resultado = por_nombre(
        p1_calidad.comprobar_consistencia_temporal(frame, poblaciones.panel, "actual")
    )

    assert resultado["lag_2_consistente_con_panel"]["resultado"] == "fail"
    assert resultado["lag_2_consistente_con_panel"]["valor_observado"] > 0


def test_detecta_mes_incompleto():
    pob = p1_calidad.construir_poblaciones(
        viajes("2025-01-01", "2026-09-30"), viajes("2026-10-01", "2026-10-20"), INICIO, FIN
    )

    cobertura = p1_calidad.comprobar_cobertura(
        pob.actual, pob.mes, pob.dias_con_viajes, INICIO, FIN
    )

    assert cobertura["resultado"] == "fail"
    assert cobertura["valor_observado"]["dias_con_filas_elegibles"] == 20
    assert cobertura["detalle"]["dias_sin_viajes"][0] == "2026-10-21"
    assert len(cobertura["detalle"]["dias_sin_filas_elegibles"]) == 11


def test_columna_constante_no_explicada_falla(poblaciones):
    frame = poblaciones.actual.copy()
    frame["lag_14"] = 3.0

    comprobacion = p1_calidad.comprobar_columnas_constantes(frame, INICIO, FIN)

    assert comprobacion["resultado"] == "fail"
    assert "lag_14" in comprobacion["detalle"]["no_explicadas"]


def buscar(pruebas: list[dict], metrica: str, columna: str) -> list[dict]:
    return [p for p in pruebas if p["metrica"] == metrica and p["columna"] == columna]


def fuera_de_rango(pruebas: list[dict], columna: str) -> tuple[dict, dict]:
    """Las dos pruebas de OutRangeValueCount de una columna: (conteo, proporción)."""
    pruebas = buscar(pruebas, "OutRangeValueCount", columna)
    conteo = next(p for p in pruebas if p["criterio"].startswith("conteo"))
    proporcion = next(p for p in pruebas if p["criterio"].startswith("proporción"))
    return conteo, proporcion


def test_evidently_detecta_negativo_y_advierte_fuera_de_rango(poblaciones, tmp_path):
    referencia = poblaciones.referencia
    actual = poblaciones.actual.copy()
    assert not {"lag_2", "lag_7", "lag_28"} & set(
        p1_calidad.detallar_fuera_de_rango(actual, p1_calidad.rangos_referencia(referencia))
    )
    actual.loc[0, "lag_2"] = -1
    actual.loc[1, "lag_28"] = referencia["lag_28"].max() + 1  # pico aislado, bajo el tope
    actual.loc[10:14, "lag_7"] = referencia["lag_7"].max() + 1  # 5 de 62 filas: 8 % > 5 %
    html_path, json_path = tmp_path / "q.html", tmp_path / "q.json"

    pruebas = p1_calidad.pruebas_evidently(
        p1_calidad.generar_reporte(referencia, actual, html_path, json_path)
    )

    assert html_path.stat().st_size > 1000
    assert json.loads(json_path.read_text(encoding="utf-8"))["tests"]
    (minimo,) = buscar(pruebas, "MinValue", "lag_2")
    assert minimo["resultado"] == "fail" and minimo["valor_observado"] == -1
    assert {p["resultado"] for p in buscar(pruebas, "MinValue", "lag_7")} == {"pass"}
    # Un solo valor fuera de rango advierte por conteo, sin llegar a la proporción crítica
    conteo, proporcion = fuera_de_rango(pruebas, "lag_28")
    assert conteo["resultado"] == "warning" and not conteo["critica"]
    assert conteo["valor_observado"] == 1
    assert proporcion["resultado"] == "pass" and proporcion["critica"]
    assert proporcion["valor_observado"] == pytest.approx(1 / len(actual))
    (tope,) = buscar(pruebas, "MaxValue", "lag_28")
    assert tope["resultado"] == "pass" and tope["critica"]
    # Más de UMBRAL_FUERA_DE_RANGO de las filas fuera de rango falla
    conteo, proporcion = fuera_de_rango(pruebas, "lag_7")
    assert conteo["valor_observado"] == 5
    assert proporcion["resultado"] == "fail"
    assert proporcion["valor_observado"] == pytest.approx(5 / len(actual))
    conteos = p1_calidad.comprobar_conteos(actual, "actual")
    assert conteos["resultado"] == "fail" and conteos["detalle"]["por_columna"] == {"lag_2": 1}


def test_valor_absurdo_a_escala_real_falla_y_se_lista_donde_esta(poblaciones, tmp_path):
    """Con más de 10,000 filas, un solo valor imposible en objetivo o lag_2 hace fallar el tope."""
    referencia, base = poblaciones.referencia, poblaciones.actual
    copias = 170  # 62 × 170 = 10,540 filas, del orden de octubre real (14,508)
    grande = pd.concat([base] * copias, ignore_index=True)
    grande["estacion"] += np.repeat(np.arange(copias), len(base)) * 1000
    grande.loc[0, ["objetivo", "lag_2"]] = 100_000.0
    rangos = p1_calidad.rangos_referencia(referencia)

    pruebas = p1_calidad.pruebas_evidently(
        p1_calidad.generar_reporte(referencia, grande, tmp_path / "g.html", tmp_path / "g.json")
    )
    detalle = p1_calidad.detallar_fuera_de_rango(grande, rangos)

    assert len(grande) >= 10_000
    for columna in ("objetivo", "lag_2"):
        (tope,) = buscar(pruebas, "MaxValue", columna)
        assert tope["resultado"] == "fail" and tope["critica"]
        assert tope["valor_observado"] == 100_000
        assert tope["criterio"] == f"valor ≤ {p1_calidad.FACTOR_MAXIMO * rangos[columna][1]}"
        conteo, proporcion = fuera_de_rango(pruebas, columna)
        assert conteo["resultado"] == "warning" and conteo["valor_observado"] == 1
        assert proporcion["resultado"] == "pass" and proporcion["valor_observado"] < 1e-4
        assert detalle[columna]["filas"] == 1
        assert detalle[columna]["filas_por_estacion"] == {str(int(grande.loc[0, "estacion"])): 1}
        assert detalle[columna]["dias"] == [str(grande.loc[0, "fecha"].date())]
    texto = p1_calidad._explicar_fuera_de_rango("lag_2", rangos["lag_2"], detalle["lag_2"])
    assert texto.startswith("1 fila (") and "en 1 estación (filas por estación: 7 (1))" in texto
    assert f"el {grande.loc[0, 'fecha'].date()}." in texto
    assert "tope crítico" not in texto  # 100,000 rebasa el tope: no se dice que queda debajo


def test_viaje_id_repetido_o_faltante_falla(viajes_octubre):
    limpio = p1_calidad.comprobar_viaje_id(viajes_octubre)
    alterado = viajes_octubre.copy()
    alterado.loc[1, "Viaje_Id"] = alterado.loc[0, "Viaje_Id"]
    alterado["Viaje_Id"] = alterado["Viaje_Id"].astype("float64")
    alterado.loc[2, "Viaje_Id"] = np.nan

    comprobacion = p1_calidad.comprobar_viaje_id(alterado)

    assert limpio["resultado"] == "pass"
    assert comprobacion["resultado"] == "fail" and comprobacion["valor_observado"] == 2
    assert comprobacion["detalle"]["sin_viaje_id"] == 1
    assert comprobacion["detalle"]["viaje_id_repetidos"] == 1
    assert len(comprobacion["detalle"]["ejemplos"]) == 2


def test_recarga_de_viajes_con_ids_nuevos_se_detecta(viajes_octubre, tmp_path):
    """Los viajes de una estación-día repetidos 20 veces con Viaje_Id nuevos no pasan en silencio.

    La llave (estacion, fecha) sigue única porque el panel cuenta por estación y día: lo ven la
    comprobación de viajes repetidos, el volumen diario y el tope MaxValue de objetivo.
    """
    historicos = viajes("2025-01-01", "2026-09-30")
    estacion_dia = (viajes_octubre["Origen_Id"] == 7) & (
        viajes_octubre["Inicio_del_viaje"].dt.normalize() == pd.Timestamp("2026-10-10")
    )
    k = int(estacion_dia.sum())
    alterado = recargar(viajes_octubre, estacion_dia, 20)
    pob = p1_calidad.construir_poblaciones(historicos, alterado, INICIO, FIN)
    tope_objetivo = p1_calidad.FACTOR_MAXIMO * pob.referencia["objetivo"].max()

    integridad = por_nombre(p1_calidad.comprobar_integridad(pob, INICIO, FIN, viajes=alterado))
    pruebas = p1_calidad.pruebas_evidently(
        p1_calidad.generar_reporte(
            pob.referencia, pob.actual, tmp_path / "r.html", tmp_path / "r.json"
        )
    )

    assert 21 * k > tope_objetivo
    assert integridad["llave_estacion_fecha_unica"]["resultado"] == "pass"  # estructural
    assert integridad["viaje_id_unico"]["resultado"] == "pass"
    repetidos = integridad["viajes_repetidos_con_otro_viaje_id"]
    assert repetidos["resultado"] == "fail" and repetidos["valor_observado"] == 20 * k
    assert repetidos["detalle"]["estacion_dia_mas_afectada"] == {
        "estacion": 7,
        "fecha": "2026-10-10",
        "salidas_de_mas": 20 * k,
        "salidas_en_panel": 21 * k,
    }
    assert repetidos["detalle"]["filas_de_mas_por_dia"] == {"2026-10-10": 20 * k}
    assert "(estación 7, 2026-10-10)" in repetidos["detalle"]["explicacion"]
    assert integridad["salidas_del_panel_igual_a_viajes_validos"]["resultado"] == "pass"
    volumen = integridad["volumen_diario_coherente_en_el_mes"]
    assert "2026-10-10" in [d["fecha"] for d in volumen["detalle"]["dias_fuera"]]
    (tope,) = buscar(pruebas, "MaxValue", "objetivo")
    assert tope["resultado"] == "fail" and tope["valor_observado"] == 21 * k


def test_dia_cargado_dos_veces_falla_como_viajes_repetidos(poblaciones, viajes_octubre):
    dia = viajes_octubre["Inicio_del_viaje"].dt.normalize() == pd.Timestamp("2026-10-10")
    alterado = recargar(viajes_octubre, dia, 1)

    comprobacion = p1_calidad.comprobar_viajes_repetidos(alterado, poblaciones.panel)
    limpio = p1_calidad.comprobar_viajes_repetidos(viajes_octubre, poblaciones.panel)

    # un día de 31 recargado pesa mucho más que UMBRAL_VIAJES_REPETIDOS de los viajes
    assert comprobacion["resultado"] == "fail" and comprobacion["critica"]
    assert comprobacion["valor_observado"] == int(dia.sum())
    assert comprobacion["detalle"]["grupos"] == int(dia.sum())
    assert comprobacion["detalle"]["filas_de_mas_por_dia"] == {"2026-10-10": int(dia.sum())}
    assert comprobacion["detalle"]["involucradas_cortas_con_origen_igual_a_destino"] == 0
    assert limpio["resultado"] == "pass" and limpio["valor_observado"] == 0


def test_pocos_viajes_de_segundos_con_origen_igual_a_destino_advierten():
    """Como en octubre real: mismo inicio, fin y estación, 2 s, con Viaje_Id consecutivos.

    Un par entre ~25,000 viajes queda por debajo de UMBRAL_VIAJES_REPETIDOS: advertencia.
    """
    alterado = viajes("2026-10-01", "2026-10-31", estaciones=tuple(range(1, 201)), semilla=30)
    inicio, estacion = alterado.loc[0, "Inicio_del_viaje"], alterado.loc[0, "Origen_Id"]
    alterado.loc[[0, 1], "Inicio_del_viaje"] = inicio
    alterado.loc[[0, 1], "Fin_del_viaje"] = inicio + pd.Timedelta(seconds=2)
    alterado.loc[[0, 1], ["Origen_Id", "Destino_Id"]] = estacion
    panel, _ = construir_panel_diario(alterado)

    comprobacion = p1_calidad.comprobar_viajes_repetidos(alterado, panel)

    assert 1 / len(alterado) < p1_calidad.UMBRAL_VIAJES_REPETIDOS
    assert comprobacion["resultado"] == "warning" and not comprobacion["critica"]
    assert comprobacion["valor_observado"] == 1
    assert comprobacion["detalle"]["involucradas_cortas_con_origen_igual_a_destino"] == 2
    assert sorted(e["Viaje_Id"] for e in comprobacion["detalle"]["ejemplos"]) == sorted(
        alterado.loc[[0, 1], "Viaje_Id"].tolist()
    )
    assert "2 filas involucradas duran 60 s o menos" in comprobacion["detalle"]["explicacion"]


def test_viajes_sin_origen_legible_advierten_y_por_encima_del_limite_fallan(viajes_octubre):
    pocos = viajes_octubre.copy()
    pocos["Origen_Id"] = pocos["Origen_Id"].astype("float64")
    pocos.loc[0, "Origen_Id"] = np.nan  # 1 de ~250 viajes: menos de 1 %
    muchos = pocos.copy()
    muchos.loc[: len(muchos) // 10, "Origen_Id"] = np.nan  # ~10 %

    advertencia = p1_calidad.comprobar_campos_legibles(pocos)
    falla = p1_calidad.comprobar_campos_legibles(muchos)

    assert advertencia["resultado"] == "warning" and advertencia["valor_observado"] == 1
    assert "no entran al panel" in advertencia["detalle"]["explicacion"]
    assert falla["resultado"] == "fail" and falla["critica"]
    assert p1_calidad.comprobar_campos_legibles(viajes_octubre)["resultado"] == "pass"


def test_conciliacion_detecta_salidas_que_no_estan_en_el_archivo(poblaciones, viajes_octubre):
    limpio = p1_calidad.comprobar_conciliacion(viajes_octubre, poblaciones.panel, INICIO, FIN)
    sin_un_viaje = viajes_octubre.iloc[1:]

    comprobacion = p1_calidad.comprobar_conciliacion(sin_un_viaje, poblaciones.panel, INICIO, FIN)

    assert limpio["resultado"] == "pass" and limpio["valor_observado"] == 0
    assert limpio["detalle"]["salidas_del_panel_en_el_mes"] == len(viajes_octubre)
    assert comprobacion["resultado"] == "fail" and comprobacion["valor_observado"] == 1


def panel_constante(salidas_mes: int = 100) -> pd.Series:
    """Salidas diarias de una estación: 100 en la referencia y ``salidas_mes`` en octubre."""
    ref = pd.Series(100, index=pd.date_range("2025-01-01", "2026-09-30"))
    mes = pd.Series(salidas_mes, index=pd.date_range("2026-10-01", "2026-10-31"))
    return pd.concat([ref, mes])


def como_panel(salidas: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"estacion": 7, "fecha": salidas.index, "salidas": salidas.to_numpy()})


def test_volumen_diario_advierte_dia_duplicado_e_incompleto():
    salidas = panel_constante()
    salidas[pd.Timestamp("2026-10-14")] = 200  # miércoles cargado dos veces
    salidas[pd.Timestamp("2026-10-20")] = 40  # martes incompleto

    comprobacion = p1_calidad.comprobar_volumen_diario(como_panel(salidas), INICIO, FIN)

    assert comprobacion["resultado"] == "warning" and not comprobacion["critica"]
    assert comprobacion["detalle"]["dias_fuera"] == [
        {"fecha": "2026-10-14", "viajes": 200, "mediana_mismo_dia_semana": 100.0, "razon": 2.0},
        {"fecha": "2026-10-20", "viajes": 40, "mediana_mismo_dia_semana": 100.0, "razon": 0.4},
    ]
    assert "2026-10-14 (200 viajes, 2.00)" in comprobacion["detalle"]["explicacion"]


def test_volumen_diario_no_confunde_cambio_de_nivel_con_recarga():
    """Un mes 50 % por encima de la referencia pasa: ese cambio lo mide el reporte de drift."""
    comprobacion = p1_calidad.comprobar_volumen_diario(
        como_panel(panel_constante(150)), INICIO, FIN
    )

    assert comprobacion["resultado"] == "pass"
    referencia = comprobacion["detalle"]["referencia_del_campeon"]
    assert referencia["p99"] == 100 and referencia["dias_del_mes_sobre_p99"] == 31


def test_cli_escribe_html_json_y_resumen(poblaciones, viajes_octubre, tmp_path):
    historicos, actuales = tmp_path / "historicos.parquet", tmp_path / "actuales.csv"
    viajes("2025-01-01", "2026-09-30").to_parquet(historicos, index=False)
    viajes_octubre.to_csv(actuales, index=False)
    panel_campeon = tmp_path / "panel.parquet"
    poblaciones.panel.loc[poblaciones.panel["fecha"] <= "2026-09-30"].to_parquet(panel_campeon)
    salida = tmp_path / "salida"

    resultado = CliRunner().invoke(
        p1_calidad.app,
        [
            "--historical-trips",
            str(historicos),
            "--current-trips",
            str(actuales),
            "--output",
            str(salida),
            "--panel-campeon",
            str(panel_campeon),
            "--estricto",
        ],
    )

    assert resultado.exit_code == 0, resultado.output
    assert (salida / "p1_data_quality.html").stat().st_size > 1000
    assert json.loads((salida / "p1_data_quality.json").read_text(encoding="utf-8"))["tests"]
    resumen = json.loads((salida / "p1_calidad_resumen.json").read_text(encoding="utf-8"))
    assert resumen["resultado_global"] == "warning"
    assert resumen["totales"]["evidently"]["fail"] == 0
    assert resumen["totales"]["integridad"]["fail"] == 0
    assert (
        por_nombre(resumen["integridad"], "referencia")["panel_igual_al_del_campeon"]["resultado"]
        == "pass"
    )
    assert {a["prueba"] for a in resumen["advertencias_explicadas"]} >= {
        "columnas_constantes_o_duplicadas_explicables"
    }
    assert all(a["explicacion"] for a in resumen["advertencias_explicadas"])
    assert len(resumen["entradas"]["current_trips"]["sha256"]) == 64
    actual = por_nombre(resumen["integridad"])
    assert actual["viaje_id_unico"]["resultado"] == "pass"
    assert actual["salidas_del_panel_igual_a_viajes_validos"]["detalle"][
        "salidas_del_panel_en_el_mes"
    ] == len(viajes_octubre)
    # media_28d tiene una fila fuera del rango de la referencia en estos datos sintéticos
    assert resumen["fuera_de_rango"]["media_28d"]["filas"] == 1
    assert (
        "media_28d" in resumen["rangos_referencia"] and "objetivo" in resumen["rangos_referencia"]
    )
    assert any(
        a["explicacion"].startswith("1 fila (")
        and "de media_28d fuera del rango" in a["explicacion"]
        for a in resumen["advertencias_explicadas"]
    )
