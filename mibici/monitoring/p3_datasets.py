"""Poblaciones de referencia y actual de P3 para el monitoreo de drift.

La unidad de análisis es la **estación-mes**: el perfil de uso de cada estación calculado con
los viajes de un solo mes calendario (las 12 variables que recibe el modelo registrado de P3,
``CONFIG_P3.prioridad_variables``, con ``features_p3.perfil_estaciones``).

- Referencia: un perfil por estación y por cada mes del periodo de entrenamiento de P3
  (``CONFIG_P3.fecha_inicio_entrenamiento`` a ``fecha_fin_entrenamiento``).
- Actual: el perfil de cada estación en el mes que se monitorea (por ejemplo, octubre de 2026).

Se compara mes contra mes, y no el perfil de 16 meses contra el de un mes, porque un perfil de
un solo mes es más ruidoso: comparar ventanas de distinto largo mostraría drift aunque el uso
no hubiera cambiado. Como un mes no alcanza los 60 días activos que pide el entrenamiento, aquí
se exigen ``MIN_DIAS_ACTIVOS_MES`` días con viajes en el mes, además de ubicación conocida.

Los viajes pasan por la misma limpieza de P3 (``features_p3.limpiar_viajes``) y nunca se leen
columnas con datos personales.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mibici.config import CONFIG, CONFIG_P3, P3Config
from mibici.dataset import _normalize_column_name
from mibici.features_p3 import (
    COLUMNAS,
    cargar_flujos,
    elegibles,
    filtrar_periodo,
    flujos_horarios,
    leer_nomenclatura,
    limpiar_viajes,
    nomenclatura_mas_reciente,
    perfil_estaciones,
)
from mibici.monitoring.contracts import validate_feature_frame

P3_FEATURES = tuple(CONFIG_P3.prioridad_variables)  # mismas 12 columnas y orden del modelo
P3_NUMERICAL_FEATURES = P3_FEATURES
P3_CATEGORICAL_FEATURES: tuple[str, ...] = ()
MIN_DIAS_ACTIVOS_MES = 15  # cerca de la mitad de un mes con viajes


def leer_viajes_mes(ruta: Path) -> pd.DataFrame:
    """Lee de un CSV mensual crudo solo las 5 columnas de P3 (sin datos personales).

    Los CSV del portal cambian de codificación y a veces traen espacios en los encabezados;
    los nombres se normalizan igual que en ``mibici.dataset``.
    """
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el CSV del mes actual: {ruta}")

    def usar(columna: str) -> bool:
        return _normalize_column_name(columna) in COLUMNAS

    try:
        viajes = pd.read_csv(ruta, usecols=usar, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        viajes = pd.read_csv(ruta, usecols=usar, dtype=str, encoding="cp1252")
    viajes = viajes.rename(columns=_normalize_column_name)
    faltantes = sorted(set(COLUMNAS) - set(viajes.columns))
    if faltantes:
        raise ValueError(f"A {ruta.name} le faltan columnas de P3: {faltantes}")
    return viajes[COLUMNAS]


def perfiles_mensuales(
    flujos: pd.DataFrame,
    meses: pd.PeriodIndex,
    ids_con_ubicacion,
    min_dias: int = MIN_DIAS_ACTIVOS_MES,
    cfg: P3Config = CONFIG_P3,
) -> tuple[pd.DataFrame, list[dict]]:
    """Perfil de uso de cada estación elegible en cada mes, indexado por (mes, estacion)."""
    perfiles, resumen = [], []
    for mes in meses:
        fl = filtrar_periodo(
            flujos, str(mes.start_time.date()), str(mes.end_time.normalize().date())
        )
        perfil = perfil_estaciones(fl, cfg.franjas)
        con_dias = perfil[perfil.dias_activos >= min_dias]
        ok = elegibles(perfil, ids_con_ubicacion, min_dias)
        resumen.append(
            {
                "mes": str(mes),
                "estaciones_con_viajes": len(perfil),
                "estaciones_con_dias_suficientes": len(con_dias),
                "estaciones_sin_ubicacion": int((~con_dias.index.isin(ids_con_ubicacion)).sum()),
                "estaciones_elegibles": len(ok),
            }
        )
        perfiles.append(
            ok[list(P3_FEATURES)].assign(mes=str(mes)).set_index("mes", append=True).swaplevel()
        )
    if not perfiles:
        raise ValueError("No hay meses que perfilar.")
    return pd.concat(perfiles), resumen


def flujos_del_mes_actual(
    ruta: Path, inicio: pd.Timestamp, fin_excl: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flujos horarios del mes actual, su bitácora de limpieza y los viajes válidos."""
    validos, bitacora = limpiar_viajes(leer_viajes_mes(ruta))
    if validos.empty:
        raise ValueError(f"{ruta.name} no tiene viajes válidos.")
    fuera = (validos["Inicio_del_viaje"] < inicio) | (validos["Inicio_del_viaje"] >= fin_excl)
    if fuera.any():
        raise ValueError(
            f"{int(fuera.sum())} viajes de {ruta.name} empiezan fuera de "
            f"[{inicio.date()}, {fin_excl.date()})."
        )
    return flujos_horarios(validos), bitacora, validos


def build_p3_populations(
    historical_trips_path: Path,
    current_trips_path: Path,
    current_start: str,
    current_end_exclusive: str,
    nomenclature_path: Path | None = None,
    min_dias: int = MIN_DIAS_ACTIVOS_MES,
    cfg: P3Config = CONFIG_P3,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Perfiles estación-mes de referencia y actual, más los metadatos del monitoreo.

    ``historical_trips_path`` es el dataset combinado de ``python -m mibici.dataset`` (el mismo
    que usa ``features_p3``); su resumen horario se reutiliza de la caché de P3 si no cambió.
    ``current_trips_path`` es el CSV crudo del mes actual. Ambos DataFrames regresan indexados
    por (mes, estacion) con las 12 columnas de ``P3_FEATURES``.
    """
    inicio = pd.Timestamp(current_start)
    fin_excl = pd.Timestamp(current_end_exclusive)
    if inicio.day != 1 or fin_excl != inicio + pd.offsets.MonthBegin():
        raise ValueError("El periodo actual de P3 debe ser exactamente un mes calendario.")
    mes_actual = pd.Period(inicio, freq="M")
    meses_ref = pd.period_range(
        cfg.fecha_inicio_entrenamiento, cfg.fecha_fin_entrenamiento, freq="M"
    )
    if mes_actual <= meses_ref[-1]:
        raise ValueError(
            f"El mes actual ({mes_actual}) debe ser posterior al entrenamiento de P3 "
            f"({meses_ref[0]} a {meses_ref[-1]})."
        )

    cache_dir = cache_dir or CONFIG.interim_data_dir / cfg.cache_subdir
    flujos_hist, bitacora_hist, huella_hist = cargar_flujos(historical_trips_path, cache_dir)
    con_viajes = set(bitacora_hist.loc[bitacora_hist.viajes_validos > 0, "mes"])
    faltantes = [str(m) for m in meses_ref if m.strftime("%Y_%m") not in con_viajes]
    if faltantes:
        raise ValueError(f"Al dataset combinado le faltan meses de entrenamiento: {faltantes}")

    flujos_act, bitacora_act, validos_act = flujos_del_mes_actual(
        current_trips_path, inicio, fin_excl
    )

    ruta_nom = nomenclature_path or nomenclatura_mas_reciente()
    nom = leer_nomenclatura(ruta_nom)
    referencia, resumen_ref = perfiles_mensuales(flujos_hist, meses_ref, nom.index, min_dias, cfg)
    actual, resumen_act = perfiles_mensuales(
        flujos_act, pd.PeriodIndex([mes_actual]), nom.index, min_dias, cfg
    )

    calidad_ref = validate_feature_frame(
        referencia.reset_index(drop=True),
        "P3 referencia",
        P3_FEATURES,
        P3_NUMERICAL_FEATURES,
        P3_CATEGORICAL_FEATURES,
    )
    calidad_act = validate_feature_frame(
        actual.reset_index(drop=True),
        "P3 actual",
        P3_FEATURES,
        P3_NUMERICAL_FEATURES,
        P3_CATEGORICAL_FEATURES,
    )

    estaciones_ref = set(referencia.index.get_level_values("estacion"))
    estaciones_act = actual.index.get_level_values("estacion")
    nuevas = sorted(int(e) for e in set(estaciones_act) - estaciones_ref)
    metadata = {
        "problem": "P3",
        "unit_of_analysis": "station-month",
        "min_active_days_per_month": min_dias,
        "reference_period": {
            "configured_start_inclusive": cfg.fecha_inicio_entrenamiento,
            "configured_end_inclusive": cfg.fecha_fin_entrenamiento,
            "months": [str(m) for m in meses_ref],
            "stations": len(estaciones_ref),
            **calidad_ref,
            "eligibility_by_month": resumen_ref,
        },
        "current_period": {
            "configured_start_inclusive": str(inicio.date()),
            "configured_end_exclusive": str(fin_excl.date()),
            "observed_start": str(validos_act["Inicio_del_viaje"].min()),
            "observed_end": str(validos_act["Inicio_del_viaje"].max()),
            "stations": int(estaciones_act.nunique()),
            "stations_not_in_reference": len(nuevas),
            "station_ids_not_in_reference": nuevas,
            **calidad_act,
            "eligibility_by_month": resumen_act,
            "cleaning_log": bitacora_act.to_dict(orient="records"),
        },
        "input_trip_data": {
            "historical_sha256": huella_hist,
            "nomenclature": Path(ruta_nom).name,
        },
        "feature_roles": {
            "categorical": list(P3_CATEGORICAL_FEATURES),
            "numerical": list(P3_NUMERICAL_FEATURES),
        },
        "exclusions": ["mes", "estacion"],
        "notes": [
            (
                "Reference and current rows are station-month usage profiles computed with the "
                "same P3 cleaning and feature code; months are compared with months so that "
                "the aggregation window does not create artificial drift."
            ),
            (
                f"A station-month is eligible with a known location and at least {min_dias} "
                "active days in the month (training uses 60 days over 16 months)."
            ),
            "Rows repeat stations across months and are not statistically independent.",
        ],
    }
    return referencia, actual, metadata
