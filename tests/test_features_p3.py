"""Pruebas de las variables de P3 con datos sintéticos (no necesitan los datos reales)."""

import json

import numpy as np
import pandas as pd
import pytest

from mibici.features_p3 import (
    COLUMNAS,
    COLUMNAS_PERSONALES,
    DatasetP3,
    cargar_dataset,
    elegibles,
    filtrar_periodo,
    flujos_horarios,
    guardar_dataset,
    leer_viajes,
    limpiar_viajes,
    nomenclatura_mas_reciente,
    perfil_estaciones,
    seleccionar_variables,
    sha256_archivo,
    version_manifiesto,
)
from mibici.modeling.train_p3 import Renumerado

FILAS = [
    # Viaje_Id, Usuario_Id, Genero, Año_de_nacimiento, inicio, fin, origen, destino
    (1, 10, "M", 1990, "2025-03-03 08:00:00", "2025-03-03 08:15:00", 1, 2),
    (1, 10, "M", 1990, "2025-03-03 08:00:00", "2025-03-03 08:15:00", 1, 2),  # duplicado
    (2, 11, "F", 1985, "2025-03-03 08:10:00", "2025-03-03 08:10:30", 1, 2),  # 30 s
    (3, 12, "M", 2000, "2025-03-03 18:05:00", "2025-03-03 18:20:00", 2, 1),
    (4, 13, "F", 1995, None, "2025-03-03 19:00:00", 2, 1),  # fecha inválida
    (5, 14, "M", 1999, "2025-04-05 12:00:00", "2025-04-05 12:30:00", 1, 2),  # sábado, abril
]
COLUMNAS_COMBINADO = [
    "Viaje_Id",
    "Usuario_Id",
    "Genero",
    "Año_de_nacimiento",
    "Inicio_del_viaje",
    "Fin_del_viaje",
    "Origen_Id",
    "Destino_Id",
    "origen_nombre",
]


def _combinado():
    """Mismas columnas que el dataset combinado de mibici.dataset (abreviado)."""
    df = pd.DataFrame([f + ("estación",) for f in FILAS], columns=COLUMNAS_COMBINADO)
    return df


@pytest.fixture(params=["csv", "parquet"])
def ruta_combinado(request, tmp_path):
    """El dataset combinado en CSV (main antes del PR #2) y en Parquet (después)."""
    df = _combinado()
    ruta = tmp_path / f"combined_dataset.{request.param}"
    if request.param == "csv":
        df.to_csv(ruta, index=False)
    else:
        pytest.importorskip("pyarrow")
        df = df.assign(**{c: pd.to_datetime(df[c]) for c in ["Inicio_del_viaje", "Fin_del_viaje"]})
        df.to_parquet(ruta, index=False)
    return ruta


def test_leer_viajes_no_carga_datos_personales(ruta_combinado):
    v = leer_viajes(ruta_combinado)
    assert list(v.columns) == COLUMNAS
    assert COLUMNAS_PERSONALES.isdisjoint(v.columns)


def test_leer_viajes_indica_como_generar_el_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match="mibici.dataset"):
        leer_viajes(tmp_path / "combined_dataset.parquet")


def test_limpiar_viajes_limpieza_minima(ruta_combinado):
    v, bit = limpiar_viajes(leer_viajes(ruta_combinado))
    assert sorted(v["Viaje_Id"]) == [1, 3, 5]
    tot = bit.drop(columns="mes").sum()
    assert tot.filas_crudas == 6
    assert tot.duplicados_exactos == 1
    assert tot.fechas_o_ids_invalidos == 1
    assert tot.duracion_menor_1min == 1
    assert tot.viajes_validos == 3


def test_limpiar_viajes_bitacora_por_mes(ruta_combinado):
    _, bit = limpiar_viajes(leer_viajes(ruta_combinado))
    por_mes = bit.set_index("mes").viajes_validos.to_dict()
    assert por_mes == {"2025_03": 2, "2025_04": 1, "sin_fecha": 0}


def test_flujos_horarios_cuenta_salidas_y_llegadas(ruta_combinado):
    fl = flujos_horarios(limpiar_viajes(leer_viajes(ruta_combinado))[0])
    assert fl.salidas.sum() == fl.llegadas.sum() == 3
    fila = fl[(fl.estacion == 1) & (fl.hora == pd.Timestamp("2025-03-03 08:00"))]
    assert fila.salidas.item() == 1 and fila.llegadas.item() == 0


def test_nomenclatura_mas_reciente(tmp_path):
    for nombre in ["nomenclatura_2025_12.csv", "nomenclatura_2026_09.csv", "otro.csv"]:
        (tmp_path / nombre).write_text("id,name\n")
    assert nomenclatura_mas_reciente(tmp_path).name == "nomenclatura_2026_09.csv"


def test_filtrar_periodo_incluye_el_ultimo_dia():
    fl = pd.DataFrame(
        {
            "estacion": [1, 1, 1],
            "salidas": [1, 1, 1],
            "llegadas": [0, 0, 0],
            "hora": pd.to_datetime(["2026-04-30 23:00", "2026-05-01 00:00", "2025-01-01 00:00"]),
        }
    )
    en = filtrar_periodo(fl, "2025-01-01", "2026-04-30")
    ev = filtrar_periodo(fl, "2026-05-01", "2026-09-30")
    assert len(en) == 2 and len(ev) == 1  # sin traslape entre periodos


def _flujos_sinteticos(dias=70):
    """Estación 1 se vacía en la mañana (residencial); estación 2 se llena (empleo)."""
    horas = pd.date_range("2025-01-06", periods=dias * 24, freq="h")
    filas = []
    for h in horas:
        manana, tarde = h.hour in (6, 7, 8, 9), h.hour in (17, 18, 19, 20)
        filas.append((1, h, 3 if manana else 1, 3 if tarde else 1))
        filas.append((2, h, 3 if tarde else 1, 3 if manana else 1))
    return pd.DataFrame(filas, columns=["estacion", "hora", "salidas", "llegadas"])


def test_perfil_estaciones_signo_del_flujo_neto():
    p = perfil_estaciones(_flujos_sinteticos())
    assert p.loc[1, "net_am"] < 0 < p.loc[1, "net_pm"]
    assert p.loc[2, "net_am"] > 0 > p.loc[2, "net_pm"]
    franjas = p[[c for c in p if c.startswith("dep_")]].iloc[0]
    assert ((franjas > 0) & (franjas < 1)).all()  # proporciones del día hábil
    assert franjas.sum() < 1  # las 10 y las 16 h quedan fuera de las franjas
    assert p.loc[1, "dias_activos"] == 70


def test_elegibles_exige_ubicacion_y_dias():
    p = perfil_estaciones(_flujos_sinteticos(dias=70))
    assert list(elegibles(p, ids_con_ubicacion=[1, 2], min_dias=60).index) == [1, 2]
    assert list(elegibles(p, ids_con_ubicacion=[1], min_dias=60).index) == [1]
    assert elegibles(p, ids_con_ubicacion=[1, 2], min_dias=71).empty


def test_seleccionar_variables_quita_la_redundante_de_menor_prioridad():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=200), rng.normal(size=200)
    df = pd.DataFrame({"a": a, "b": b, "c": a + b + rng.normal(scale=0.01, size=200)})
    assert seleccionar_variables(df, prioridad=["a", "b", "c"], umbral=5) == ["a", "b"]


def test_version_manifiesto_avisa_si_los_datos_no_coinciden(tmp_path):
    from loguru import logger

    crudo = tmp_path / "datos_abiertos_2025_01.csv"
    crudo.write_text("Viaje_Id\n1\n")
    man = tmp_path / "manifiesto.json"
    entrada = {"archivo": crudo.name, "sha256": sha256_archivo(crudo), "registros": 1}
    man.write_text(json.dumps({"version": "abc123def456", "archivos": [entrada]}))
    avisos = []
    id_ = logger.add(lambda m: avisos.append(m), level="WARNING")
    try:
        assert version_manifiesto(man, tmp_path) == "abc123def456"
        assert not avisos
        crudo.write_text("Viaje_Id\n2\n")  # el archivo ya no coincide con el manifiesto
        assert version_manifiesto(man, tmp_path) == "abc123def456"
        assert len(avisos) == 1
    finally:
        logger.remove(id_)


def test_guardar_y_cargar_dataset_conserva_los_perfiles(tmp_path):
    pytest.importorskip("pyarrow")
    nom = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["(GDL-001) A", "(ZPN-002) B"],
            "latitude": [20.6, 20.7],
            "longitude": [-103.3, -103.4],
        }
    )
    nom.to_csv(tmp_path / "nomenclatura_2026_09.csv", index=False)
    perfil = perfil_estaciones(_flujos_sinteticos())
    ds = DatasetP3(
        bitacora=pd.DataFrame({"mes": ["2025_01"], "viajes_validos": [10]}),
        nom=nom.set_index("id"),
        entrenamiento=perfil,
        evaluacion=perfil,
        variables=["net_am", "net_pm"],
        nuevas=pd.Index([2]),
        data_version="abc123def456",
        nomenclatura="nomenclatura_2026_09.csv",
    )
    guardar_dataset(ds, tmp_path / "interim")
    leido = cargar_dataset(tmp_path / "interim", raw_dir=tmp_path)
    pd.testing.assert_frame_equal(leido.entrenamiento, perfil)
    assert leido.variables == ["net_am", "net_pm"] and list(leido.nuevas) == [2]
    assert leido.data_version == "abc123def456"
    assert leido.nom.loc[2, "municipio"] == "ZPN"


def test_cargar_dataset_indica_como_generar_los_perfiles(tmp_path):
    with pytest.raises(FileNotFoundError, match="mibici.features_p3"):
        cargar_dataset(tmp_path)


def test_renumerado_ordena_zonas_de_poniente_a_oriente():
    from sklearn.cluster import KMeans

    X = np.array([[10, 0], [10.1, 0], [0, 0], [0.1, 0], [5, 0], [5.1, 0]])
    km = Renumerado(KMeans(n_clusters=3, n_init=5, random_state=0).fit(X), columna=0)
    assert list(km.labels_) == [2, 2, 0, 0, 1, 1]
    assert list(km.predict(np.array([[-1, 0], [11, 0]]))) == [0, 2]
