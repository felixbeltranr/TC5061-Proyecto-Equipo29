import json

import numpy as np
import pandas as pd
import pytest

from mibici.monitoring import p2_datasets as d2
from mibici.monitoring import p2_drift


def _nomenclatura(ruta, filas):
    pd.DataFrame(filas, columns=["id", "name", "latitude", "longitude", "dpcapacity"]).to_csv(
        ruta, index=False, encoding="utf-8"
    )
    return ruta


def test_tabla_estaciones_vigente_prefiere_la_nomenclatura_mas_reciente(tmp_path):
    reciente = _nomenclatura(
        tmp_path / "nomenclatura_2026_10.csv", [[1, "a", 0, 0, 20], [2, "b", 0, 0, 15]]
    )
    anterior = _nomenclatura(
        tmp_path / "nomenclatura_2026_09.csv", [[1, "a", 0, 0, 12], [3, "c", 0, 0, 9]]
    )
    estaciones, resumen = d2.tabla_estaciones_vigente([reciente, anterior])
    capacidad = dict(zip(estaciones["estacion"].astype(int), estaciones["capacidad"]))
    assert capacidad == {1: 20, 2: 15, 3: 9}
    assert resumen["estaciones_por_nomenclatura"] == {
        "nomenclatura_2026_10.csv": 2,
        "nomenclatura_2026_09.csv": 1,
    }


def test_tabla_estaciones_vigente_exige_alguna_nomenclatura():
    with pytest.raises(ValueError, match="al menos una"):
        d2.tabla_estaciones_vigente([])


def test_nomenclaturas_locales_de_la_mas_reciente_a_la_mas_antigua(tmp_path):
    for nombre in ("nomenclatura_2026_09.csv", "nomenclatura_2026_10.csv", "otro.csv"):
        (tmp_path / nombre).write_text("id\n", encoding="utf-8")
    rutas = d2.nomenclaturas_locales(tmp_path)
    assert [r.name for r in rutas] == ["nomenclatura_2026_10.csv", "nomenclatura_2026_09.csv"]
    with pytest.raises(FileNotFoundError):
        d2.nomenclaturas_locales(tmp_path / "vacio")


def _filas(estaciones=(1, 2), horas=("2026-09-30 22:00", "2026-10-01 06:00", "2026-10-31 23:00")):
    ts = pd.to_datetime(list(horas))
    filas = pd.DataFrame(
        [(e, t) for e in estaciones for t in ts], columns=["estacion", "ts_objetivo"]
    )
    for col in d2.COLUMNAS_FILAS:
        if col not in filas:
            filas[col] = 0.0
    filas["hora"] = filas["ts_objetivo"].dt.hour
    filas["objetivo"] = "segura"
    return filas


def test_en_periodo_deja_solo_el_mes_y_las_columnas_del_contrato():
    filas = _filas().assign(extra=1)
    actual = d2.en_periodo(filas, pd.Timestamp("2026-10-01"), pd.Timestamp("2026-11-01"))
    assert len(actual) == 4
    assert list(actual.columns) == list(d2.COLUMNAS_FILAS)
    assert actual["ts_objetivo"].min() == pd.Timestamp("2026-10-01 06:00")


def test_por_mes_indexa_por_mes_y_estacion():
    filas = _filas().assign(
        est_tasa_sin_bicis=0.1, est_tasa_sin_anclajes=0.1, est_neto_medio_hora=0
    )
    tabla = d2.por_mes(filas)
    assert tabla.index.names == ["mes", "estacion"]
    assert set(tabla.index.get_level_values("mes")) == {"2026-09", "2026-10"}
    assert list(tabla.columns) == list(d2.P2_NUMERICAL_FEATURES)


def test_est_por_version_cambia_solo_las_estadisticas_y_conserva_filas():
    filas = _filas()
    stats_a = (
        pd.DataFrame(
            {"estacion": [1], "est_tasa_sin_bicis": [0.1], "est_tasa_sin_anclajes": [0.2]}
        ),
        pd.DataFrame({"estacion": [1, 1], "hora": [6, 23], "est_neto_medio_hora": [0.5, -0.5]}),
    )
    stats_b = (
        pd.DataFrame(
            {
                "estacion": [1, 2],
                "est_tasa_sin_bicis": [0.3, 0.4],
                "est_tasa_sin_anclajes": [0.0, 0.1],
            }
        ),
        pd.DataFrame({"estacion": [2], "hora": [6], "est_neto_medio_hora": [1.0]}),
    )
    a = d2.con_estadisticas(filas, stats_a)
    b = d2.con_estadisticas(filas, stats_b)
    comunes = list(d2.COLUMNAS_FILAS)
    pd.testing.assert_frame_equal(a[comunes], filas[comunes])
    pd.testing.assert_frame_equal(b[comunes], filas[comunes])
    assert a.loc[a.estacion == 2, "est_tasa_sin_bicis"].isna().all()
    assert b.loc[b.estacion == 2, "est_tasa_sin_bicis"].eq(0.4).all()


def test_proporciones_incluye_las_tres_clases():
    shares = d2.proporciones(pd.Series(["segura", "segura", "sin_anclajes", "segura"]))
    assert shares == {"segura": 0.75, "sin_bicicletas": 0.0, "sin_anclajes": 0.25}


def test_drift_evidently_lee_solo_value_drift(tmp_path):
    reporte = {
        "metrics": [
            {"config": {"type": "evidently:metric_v2:DriftedColumnsCount"}, "value": {}},
            {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "capacidad",
                    "method": "Wasserstein distance (normed)",
                    "threshold": 0.1,
                },
                "value": 0.12,
            },
            {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "hora",
                    "method": "Jensen-Shannon distance",
                    "threshold": 0.1,
                },
                "value": 0.01,
            },
        ]
    }
    ruta = tmp_path / "reporte.json"
    ruta.write_text(json.dumps(reporte), encoding="utf-8")
    tabla = p2_drift.drift_evidently(ruta)
    assert tabla.index.tolist() == ["capacidad", "hora"]
    assert tabla["drift_evidently_reporte"].tolist() == [True, False]


def test_tabla_drift_p2_agrega_control_estacional_y_separa_nuevas():
    rng = np.random.default_rng(29)
    piezas = []
    for mes in ("2025-10", "2025-11", "2025-12"):
        f = _filas(estaciones=(1, 2), horas=pd.date_range(f"{mes}-01 06:00", periods=24, freq="h"))
        piezas.append(f)
    referencia = pd.concat(piezas, ignore_index=True)
    actual = _filas(
        estaciones=(1, 2, 3), horas=pd.date_range("2026-10-01 06:00", periods=24, freq="h")
    )
    for tabla in (referencia, actual):
        for col in d2.P2_NUMERICAL_FEATURES:
            tabla[col] = rng.normal(size=len(tabla))
    drift = p2_drift.tabla_drift_p2(referencia, actual, "2025-10")
    assert "distancia_vs_2025-10" in drift.columns
    assert set(drift.index) == set(d2.P2_NUMERICAL_FEATURES)
    assert drift["distancia_nuevas"].notna().all()
    grupos = p2_drift.proporciones_por_grupo(referencia, actual, "2025-10")
    assert set(grupos) == {"referencia", "2025-10", "actual", "actual_existentes", "actual_nuevas"}
