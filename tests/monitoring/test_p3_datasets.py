from dataclasses import replace
from functools import partial
import json

import pandas as pd
import pytest

from mibici.config import CONFIG_P3
from mibici.features_p3 import COLUMNAS
from mibici.monitoring import p3_drift
from mibici.monitoring.p3_datasets import (
    P3_FEATURES,
    build_p3_populations,
    leer_viajes_mes,
)

# Entrenamiento corto (marzo y abril de 2026) para que los datos sintéticos sean pequeños
CFG = replace(
    CONFIG_P3, fecha_inicio_entrenamiento="2026-03-01", fecha_fin_entrenamiento="2026-04-30"
)


def _viajes(inicio: str, fin: str, rutas: list[tuple[int, int, int]], dias=None) -> pd.DataFrame:
    """Un viaje de 10 minutos por día y por ruta (origen, destino, hora)."""
    fechas = pd.date_range(inicio, fin, freq="D")
    if dias is not None:
        fechas = fechas[:dias]
    filas = [
        (f + pd.Timedelta(hours=h), f + pd.Timedelta(hours=h, minutes=10), o, d)
        for f in fechas
        for o, d, h in rutas
    ]
    v = pd.DataFrame(filas, columns=COLUMNAS[1:])
    return v


# 1 y 2 se intercambian viajes mañana y tarde; 3 no tiene ubicación; 4 tiene pocos días
RUTAS_RED = [(1, 2, 8), (2, 1, 18), (1, 2, 22), (3, 1, 8), (1, 3, 18)]


def _combinado(ruta, inicio="2026-03-01", fin="2026-04-30"):
    v = pd.concat(
        [_viajes(inicio, fin, RUTAS_RED), _viajes(inicio, fin, [(4, 1, 8), (1, 4, 18)], dias=5)]
    )
    v.insert(0, "Viaje_Id", range(len(v)))
    v.to_parquet(ruta, index=False)
    return ruta


def _csv_mes(ruta, inicio="2026-10-01", fin="2026-10-31", extra=None):
    # 5 es una estación nueva que solo aparece en el mes actual
    v = _viajes(inicio, fin, RUTAS_RED + [(5, 1, 8), (1, 5, 18)])
    if extra is not None:
        v = pd.concat([v, extra])
    crudo = pd.DataFrame(
        {
            "Viaje_Id": range(len(v)),
            "Usuario_Id": 999,
            "Genero": "M",
            "Año_de_nacimiento": 1990,
            " Inicio_del_viaje": v["Inicio_del_viaje"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Fin_del_viaje": v["Fin_del_viaje"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Origen_Id": v["Origen_Id"].to_numpy(),
            "Destino_Id": v["Destino_Id"].to_numpy(),
        }
    )
    crudo.to_csv(ruta, index=False, encoding="cp1252")
    return ruta


def _nomenclatura(ruta):
    pd.DataFrame(
        {
            "id": [1, 2, 4, 5],
            "name": [f"(GDL-00{i}) Estación {i}" for i in [1, 2, 4, 5]],
            "latitude": 20.67,
            "longitude": -103.35,
        }
    ).to_csv(ruta, index=False)
    return ruta


@pytest.fixture
def archivos(tmp_path):
    return {
        "historical_trips_path": _combinado(tmp_path / "combinado.parquet"),
        "current_trips_path": _csv_mes(tmp_path / "datos_abiertos_2026_10.csv"),
        "nomenclature_path": _nomenclatura(tmp_path / "nomenclatura_2026_10.csv"),
        "cache_dir": tmp_path / "cache",
    }


def test_leer_viajes_mes_no_carga_datos_personales(tmp_path):
    viajes = leer_viajes_mes(_csv_mes(tmp_path / "datos_abiertos_2026_10.csv"))

    assert viajes.columns.tolist() == COLUMNAS


def test_perfiles_estacion_mes_de_referencia_y_actual(archivos):
    referencia, actual, meta = build_p3_populations(
        current_start="2026-10-01", current_end_exclusive="2026-11-01", cfg=CFG, **archivos
    )

    assert referencia.columns.tolist() == actual.columns.tolist() == list(P3_FEATURES)
    assert referencia.index.names == ["mes", "estacion"]
    assert referencia.index.tolist() == [
        ("2026-03", 1),
        ("2026-03", 2),
        ("2026-04", 1),
        ("2026-04", 2),
    ]
    assert actual.index.tolist() == [("2026-10", 1), ("2026-10", 2), ("2026-10", 5)]
    assert meta["unit_of_analysis"] == "station-month"
    assert meta["current_period"]["station_ids_not_in_reference"] == [5]
    marzo = meta["reference_period"]["eligibility_by_month"][0]
    assert marzo == {
        "mes": "2026-03",
        "estaciones_con_viajes": 4,
        "estaciones_con_dias_suficientes": 3,
        "estaciones_sin_ubicacion": 1,
        "estaciones_elegibles": 2,
    }


def test_rechaza_mes_actual_dentro_del_entrenamiento(archivos):
    with pytest.raises(ValueError, match="posterior al entrenamiento"):
        build_p3_populations(
            current_start="2026-04-01", current_end_exclusive="2026-05-01", cfg=CFG, **archivos
        )


def test_rechaza_viajes_fuera_del_mes_actual(archivos, tmp_path):
    fuera = _viajes("2026-11-01", "2026-11-01", [(1, 2, 8)])
    archivos["current_trips_path"] = _csv_mes(tmp_path / "otro.csv", extra=fuera)

    with pytest.raises(ValueError, match="empiezan fuera"):
        build_p3_populations(
            current_start="2026-10-01", current_end_exclusive="2026-11-01", cfg=CFG, **archivos
        )


def test_rechaza_dataset_sin_todos_los_meses_de_entrenamiento(archivos):
    cfg = replace(CFG, fecha_inicio_entrenamiento="2026-02-01")

    with pytest.raises(ValueError, match="faltan meses"):
        build_p3_populations(
            current_start="2026-10-01", current_end_exclusive="2026-11-01", cfg=cfg, **archivos
        )


def test_run_guarda_reporte_manifiesto_y_perfiles(archivos, tmp_path, monkeypatch):
    cache_dir = archivos.pop("cache_dir")
    monkeypatch.setattr(
        p3_drift,
        "build_p3_populations",
        partial(build_p3_populations, cfg=CFG, cache_dir=cache_dir),
    )
    salida = tmp_path / "reporte"

    meta = p3_drift.run(
        archivos["historical_trips_path"],
        archivos["current_trips_path"],
        salida,
        nomenclature_path=archivos["nomenclature_path"],
    )

    for nombre in meta["outputs"].values():
        assert (salida / nombre).is_file()
    guardado = json.loads((salida / "manifest.json").read_text(encoding="utf-8"))
    assert guardado["current_period"]["rows"] == 3
    assert pd.read_parquet(salida / "p3_perfiles_actual.parquet").shape == (3, 12)


def test_distancia_normada_en_desviaciones_de_la_referencia():
    ref = pd.Series([0.0, 1.0, 2.0, 3.0])

    assert p3_drift.distancia_normada(ref, ref) == 0
    assert p3_drift.distancia_normada(ref, ref + ref.std(ddof=0)) == pytest.approx(1)


def _perfiles(valores: dict[tuple[str, int], float]) -> pd.DataFrame:
    indice = pd.MultiIndex.from_tuples(list(valores), names=["mes", "estacion"])
    return pd.DataFrame({"x": list(valores.values())}, index=indice)


def test_umbral_calibrado_es_el_mes_que_mas_se_aleja_de_los_demas():
    referencia = _perfiles(
        {(m, e): v + (5 if m == "2026-02" else 0) for m in ["2026-01", "2026-02", "2026-03"]
         for e, v in [(1, 0.0), (2, 1.0), (3, 2.0)]}
    )

    umbrales = p3_drift.umbrales_calibrados(referencia)

    assert umbrales.loc["x", "mes_del_maximo"] == "2026-02"
    assert umbrales.loc["x", "umbral_calibrado"] > 1


def test_tabla_drift_separa_estaciones_existentes_y_nuevas():
    referencia = _perfiles({(m, e): float(e) for m in ["2026-01", "2026-02"] for e in [1, 2, 3]})
    actual = _perfiles({("2026-10", 1): 1.0, ("2026-10", 2): 2.0, ("2026-10", 9): 30.0})

    tabla = p3_drift.tabla_drift(referencia, actual)

    assert tabla.loc["x", "umbral_calibrado"] == 0
    assert tabla.loc["x", "distancia_existentes"] < tabla.loc["x", "distancia_nuevas"]
    assert tabla.loc["x", "media_nuevas"] == 30
    assert bool(tabla.loc["x", "drift_evidently"]) and bool(tabla.loc["x", "drift_calibrado"])
