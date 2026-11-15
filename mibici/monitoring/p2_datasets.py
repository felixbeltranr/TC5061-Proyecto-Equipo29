"""Poblaciones de referencia y actual de P2 para el monitoreo de drift y el A/B testing.

La unidad es la fila estación-hora que recibe el modelo: las 18 variables de
``features_p2.FEATURES`` con horizonte de 1 hora y horas objetivo de 6 a 23.

- Referencia: las filas de train del campeón (fecha objetivo del 2025-01-01 al 2026-04-30), del
  panel con que se entrenó y con las estadísticas por estación que guardó su run
  (``data/stats_estacion.csv`` y ``data/stats_estacion_hora.csv``).
- Actual: las filas del mes monitoreado (por ejemplo, octubre de 2026). El panel se arma con todos
  los viajes del dataset combinado más los del mes, así los rezagos de hasta cuatro semanas son
  los mismos que vería el modelo en operación. La capacidad de cada estación sale de la
  nomenclatura más reciente que la incluye.

Las variables ``est_*`` dependen de la versión del modelo: cada versión usa las estadísticas de su
propio train y una estación que no estaba ahí queda con faltantes en esas tres columnas.
Los viajes pasan por la limpieza de siempre (``features_p3.limpiar_viajes``) y nunca se leen
columnas con datos personales.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile

import mlflow
from mlflow import MlflowClient
import pandas as pd

from mibici import features_p2 as fp
from mibici.config import CONFIG, CONFIG_P2
from mibici.dataset import _read_dataset
from mibici.features_p3 import limpiar_viajes
from mibici.modeling import train_p2 as t2
from mibici.monitoring.contracts import validate_feature_frame
from mibici.monitoring.p1_drift import file_sha256
from mibici.monitoring.p3_datasets import leer_viajes_mes

P2_FEATURES = tuple(fp.FEATURES)
ESTADISTICAS = ("est_tasa_sin_bicis", "est_tasa_sin_anclajes", "est_neto_medio_hora")
P2_CATEGORICAL_FEATURES = ("hora", "dia_semana", "fin_semana")
P2_NUMERICAL_FEATURES = tuple(f for f in P2_FEATURES if f not in P2_CATEGORICAL_FEATURES)
CLAVES = ("estacion", "ts_objetivo")
COLUMNAS_FILAS = (*CLAVES, *(f for f in P2_FEATURES if f not in ESTADISTICAS), "objetivo")
ARCHIVOS_ESTADISTICAS = ("data/stats_estacion.csv", "data/stats_estacion_hora.csv")


def version_por_alias(
    alias: str,
    tracking_uri: str,
    nombre: str = CONFIG_P2.modelo_registrado,
) -> tuple[str, str]:
    """Versión y run de origen a los que apunta ``alias`` en el Model Registry."""
    cliente = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    version = cliente.get_model_version_by_alias(nombre, alias)
    return str(version.version), version.run_id


def estadisticas_del_run(run_id: str, tracking_uri: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estadísticas por estación y por estación y hora que guardó el run de una versión."""
    with tempfile.TemporaryDirectory() as carpeta:
        tablas = [
            pd.read_csv(
                mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path=archivo,
                    dst_path=carpeta,
                    tracking_uri=tracking_uri,
                )
            )
            for archivo in ARCHIVOS_ESTADISTICAS
        ]
    return tablas[0], tablas[1]


def con_estadisticas(
    filas: pd.DataFrame, estadisticas: tuple[pd.DataFrame, pd.DataFrame]
) -> pd.DataFrame:
    """Filas con las ``est_*`` de una versión; conserva el orden de las filas."""
    return fp.unir_stats(filas, *estadisticas)


def tabla_estaciones_vigente(rutas: list[Path]) -> tuple[pd.DataFrame, dict]:
    """Capacidad de cada estación según la nomenclatura más reciente que la incluye.

    ``rutas`` va de la nomenclatura más reciente a la más antigua.
    """
    if not rutas:
        raise ValueError("Se necesita al menos una nomenclatura.")
    tablas = [fp.tabla_estaciones(_read_dataset(ruta)).assign(origen=ruta.name) for ruta in rutas]
    estaciones = pd.concat(tablas, ignore_index=True).drop_duplicates("estacion", keep="first")
    resumen = {
        "nomenclaturas": [ruta.name for ruta in rutas],
        "estaciones": len(estaciones),
        "estaciones_por_nomenclatura": {
            str(k): int(v) for k, v in estaciones["origen"].value_counts().items()
        },
    }
    return estaciones.drop(columns="origen").reset_index(drop=True), resumen


def nomenclaturas_locales(raw_dir: Path = CONFIG.raw_data_dir) -> list[Path]:
    """Nomenclaturas mensuales de ``raw_dir``, de la más reciente a la más antigua."""
    rutas = sorted(raw_dir.glob("nomenclatura_*.csv"), reverse=True)
    if not rutas:
        raise FileNotFoundError(f"No hay nomenclatura en {raw_dir}.")
    return rutas


def viajes_del_mes(
    ruta: Path, inicio: pd.Timestamp, fin_excl: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Viajes válidos y bitácora de limpieza del CSV del mes; todos deben empezar en el mes."""
    validos, bitacora = limpiar_viajes(leer_viajes_mes(ruta))
    if validos.empty:
        raise ValueError(f"{ruta.name} no tiene viajes válidos.")
    fuera = (validos["Inicio_del_viaje"] < inicio) | (validos["Inicio_del_viaje"] >= fin_excl)
    if fuera.any():
        raise ValueError(
            f"{int(fuera.sum())} viajes de {ruta.name} empiezan fuera de "
            f"[{inicio.date()}, {fin_excl.date()})."
        )
    return validos, bitacora


def panel_actual(
    historicos: Path,
    actual_csv: Path,
    nomenclaturas: list[Path],
    inicio: pd.Timestamp,
    fin_excl: pd.Timestamp,
    cache_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    """Panel estación-hora con todos los viajes históricos más los del mes actual.

    Se guarda en ``cache_dir`` con la huella de las entradas en el nombre y se reutiliza si
    ninguna cambió.
    """
    huellas = {ruta.name: file_sha256(ruta) for ruta in (historicos, actual_csv, *nomenclaturas)}
    clave = sha256(json.dumps(huellas, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    ruta_panel = cache_dir / f"p2_panel_{inicio:%Y_%m}_{clave}.parquet"
    ruta_resumen = ruta_panel.with_suffix(".json")
    if ruta_panel.is_file() and ruta_resumen.is_file():
        return pd.read_parquet(ruta_panel), json.loads(ruta_resumen.read_text(encoding="utf-8"))

    hist, _ = fp.cargar_viajes(historicos)
    if (hist["Inicio_del_viaje"] >= inicio).any():
        raise ValueError(f"El dataset histórico ya trae viajes desde {inicio.date()}.")
    mes, bitacora_mes = viajes_del_mes(actual_csv, inicio, fin_excl)
    estaciones, resumen_estaciones = tabla_estaciones_vigente(nomenclaturas)
    panel, resumen = fp.construir_panel(
        pd.concat([hist, mes], ignore_index=True),
        estaciones,
        fin=str(fin_excl - pd.Timedelta(hours=1)),
    )
    resumen = resumen | {
        "limpieza_mes_actual": {
            str(k): int(v) for k, v in bitacora_mes.drop(columns="mes").sum().items()
        },
        "viajes_validos_mes_actual": len(mes),
        "estaciones_nomenclatura": resumen_estaciones,
        "huellas_sha256": huellas,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(ruta_panel, index=False)
    ruta_resumen.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    return panel, resumen


def filas_del_modelo(panel: pd.DataFrame, nivel: float = CONFIG_P2.nivel_inicial) -> pd.DataFrame:
    """Filas con features y objetivo al horizonte del modelo publicado, como en train_p2."""
    return fp.construir_features(fp.agregar_ocupacion(panel, nivel), CONFIG_P2.horizonte_horas)


def en_periodo(filas: pd.DataFrame, inicio: pd.Timestamp, fin_excl: pd.Timestamp) -> pd.DataFrame:
    """Filas cuya fecha objetivo cae en [inicio, fin_excl), solo con ``COLUMNAS_FILAS``."""
    t = filas["ts_objetivo"]
    return filas.loc[(t >= inicio) & (t < fin_excl), list(COLUMNAS_FILAS)].reset_index(drop=True)


def poblacion_referencia(
    panel_entrenamiento: pd.DataFrame, estadisticas: tuple[pd.DataFrame, pd.DataFrame]
) -> pd.DataFrame:
    """Filas de train del campeón (cortes de ``train_p2.CORTES``) con sus estadísticas."""
    train = t2.particionar(filas_del_modelo(panel_entrenamiento))["train"]
    return con_estadisticas(train[list(COLUMNAS_FILAS)].reset_index(drop=True), estadisticas)


def proporciones(objetivo: pd.Series) -> dict[str, float]:
    """Proporción de cada clase de la etiqueta proxy, con las tres clases siempre presentes."""
    shares = objetivo.value_counts(normalize=True).reindex(fp.CLASES, fill_value=0.0)
    return {str(k): float(v) for k, v in shares.items()}


def por_mes(filas: pd.DataFrame) -> pd.DataFrame:
    """Variables numéricas indexadas por (mes, estacion) para calibrar umbrales mes a mes."""
    indice = pd.MultiIndex.from_arrays(
        [filas["ts_objetivo"].dt.to_period("M").astype(str), filas["estacion"]],
        names=["mes", "estacion"],
    )
    return filas[list(P2_NUMERICAL_FEATURES)].set_axis(indice)


def resumen_calidad(filas: pd.DataFrame, nombre: str) -> dict:
    """Contrato de columnas y calidad básica de una población: filas, nulos y duplicados."""
    calidad = validate_feature_frame(
        filas[list(P2_FEATURES)],
        nombre,
        P2_FEATURES,
        P2_NUMERICAL_FEATURES,
        P2_CATEGORICAL_FEATURES,
    )
    return calidad | {
        "stations": int(filas["estacion"].nunique()),
        "duplicated_station_hours": int(filas.duplicated(list(CLAVES)).sum()),
        "first_target_hour": str(filas["ts_objetivo"].min()),
        "last_target_hour": str(filas["ts_objetivo"].max()),
        "proxy_label_share": proporciones(filas["objetivo"]),
    }
