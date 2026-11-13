"""Reporte de drift de P3 con Evidently: perfiles estación-mes de referencia contra el mes actual.

Uso desde la raíz del repositorio (después de ``python -m mibici.dataset``)::

    python -m mibici.monitoring.p3_drift --download-current
    python -m mibici.monitoring --question p3 --download-current

Guarda en ``reports/monitoring/AAAA_MM/p3``: el reporte HTML y JSON de Evidently, el
manifiesto con periodos, elegibilidad y huellas de los archivos, la tabla de drift con umbrales
calibrados (``p3_drift_calibrado.csv``) y los perfiles de referencia y actual en Parquet (sin
datos personales) para el A/B testing de P3.

Umbrales calibrados: Evidently marca drift en una variable numérica cuando la distancia de
Wasserstein normalizada (dividida entre la desviación estándar de la referencia) pasa de 0.1.
Con perfiles mensuales ese umbral salta por la variación normal entre meses (calendario,
vacaciones, fines de semana). Por eso cada mes del entrenamiento se compara contra los demás y
el umbral calibrado de cada variable es la mayor distancia observada: el mes actual solo se
marca si se aleja más de lo que se alejó cualquier mes del entrenamiento.

Además la distancia se separa por estaciones existentes (ya estaban en la referencia) y nuevas,
para distinguir un cambio de comportamiento de un cambio en la composición de la red.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from mibici.config import CONFIG
from mibici.monitoring.evidently_report import generate_report
from mibici.monitoring.p1_drift import file_sha256, month_bounds
from mibici.monitoring.p3_datasets import (
    MIN_DIAS_ACTIVOS_MES,
    P3_CATEGORICAL_FEATURES,
    P3_NUMERICAL_FEATURES,
    build_p3_populations,
)

UMBRAL_EVIDENTLY = 0.1  # umbral por omisión de Evidently para la distancia de Wasserstein normada


def distancia_normada(referencia: pd.Series, actual: pd.Series) -> float:
    """Distancia de Wasserstein entre dos muestras dividida entre la desviación de la referencia.

    Es la misma medida que usa Evidently para columnas numéricas ("Wasserstein distance
    (normed)"). Los faltantes se ignoran; si una muestra queda vacía regresa NaN.
    """
    referencia, actual = referencia.dropna(), actual.dropna()
    if referencia.empty or actual.empty:
        return float("nan")
    distancia = wasserstein_distance(referencia, actual)
    desviacion = float(np.std(referencia))
    if desviacion == 0:
        return 0.0 if distancia == 0 else float("inf")
    return float(distancia / desviacion)


def umbrales_calibrados(referencia: pd.DataFrame) -> pd.DataFrame:
    """Mayor distancia normada de cada variable entre un mes de la referencia y los demás.

    ``referencia`` está indexada por (mes, estacion). Regresa, por variable, el umbral calibrado
    y el mes en que se alcanzó.
    """
    meses = referencia.index.get_level_values("mes")
    if meses.nunique() < 2:
        raise ValueError("Se necesitan al menos dos meses de referencia para calibrar.")
    distancias = pd.DataFrame(
        {
            mes: {
                col: distancia_normada(referencia.loc[meses != mes, col], grupo[col])
                for col in referencia.columns
            }
            for mes, grupo in referencia.groupby(level="mes")
        }
    ).T
    return pd.DataFrame(
        {"umbral_calibrado": distancias.max(), "mes_del_maximo": distancias.idxmax()}
    )


def tabla_drift(referencia: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Drift por variable del mes actual con el umbral de Evidently y con el calibrado.

    Las columnas ``distancia_existentes`` y ``distancia_nuevas`` repiten la medida solo con las
    estaciones que ya estaban en la referencia y solo con las que no.
    """
    ya_estaban = actual.index.get_level_values("estacion").isin(
        referencia.index.get_level_values("estacion")
    )
    tabla = umbrales_calibrados(referencia)
    for nombre, filas in (
        ("distancia", slice(None)),
        ("distancia_existentes", ya_estaban),
        ("distancia_nuevas", ~ya_estaban),
    ):
        grupo = actual[filas]
        tabla[nombre] = [
            distancia_normada(referencia[col], grupo[col]) if len(grupo) else np.nan
            for col in tabla.index
        ]
    tabla["drift_evidently"] = tabla["distancia"] > UMBRAL_EVIDENTLY
    tabla["drift_calibrado"] = tabla["distancia"] > tabla["umbral_calibrado"]
    tabla["media_referencia"] = referencia.mean()
    tabla["media_existentes"] = actual[ya_estaban].mean()
    tabla["media_nuevas"] = actual[~ya_estaban].mean()
    return tabla.rename_axis("variable")


def run(
    historical_trips_path: Path,
    current_trips_path: Path,
    output_dir: Path,
    current_start: str = "2026-10-01",
    current_end_exclusive: str = "2026-11-01",
    download_current: bool = False,
    nomenclature_path: Path | None = None,
    min_dias: int = MIN_DIAS_ACTIVOS_MES,
) -> dict:
    """Genera el reporte de drift de P3 y regresa su manifiesto."""
    year, month = month_bounds(current_start, current_end_exclusive)
    if not historical_trips_path.is_file():
        raise FileNotFoundError(
            f"No existe el dataset combinado {historical_trips_path}. "
            "Genéralo con: python -m mibici.dataset"
        )
    if not current_trips_path.is_file():
        if not download_current:
            raise FileNotFoundError(
                f"No existe el CSV del mes actual {current_trips_path}. "
                "Usa --download-current para descargarlo."
            )
        from mibici.dataset import download_month

        descargado = download_month(year, month, output_dir=current_trips_path.parent)
        if descargado != current_trips_path:
            raise ValueError(f"Se descargó {descargado} en lugar de {current_trips_path}.")

    referencia, actual, metadata = build_p3_populations(
        historical_trips_path,
        current_trips_path,
        current_start,
        current_end_exclusive,
        nomenclature_path=nomenclature_path,
        min_dias=min_dias,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "p3_data_drift.html"
    json_path = output_dir / "p3_data_drift.json"
    generate_report(
        referencia.reset_index(drop=True),
        actual.reset_index(drop=True),
        html_path,
        json_path,
        P3_NUMERICAL_FEATURES,
        P3_CATEGORICAL_FEATURES,
    )
    referencia.to_parquet(output_dir / "p3_perfiles_referencia.parquet")
    actual.to_parquet(output_dir / "p3_perfiles_actual.parquet")
    tabla = tabla_drift(referencia, actual)
    tabla.to_csv(output_dir / "p3_drift_calibrado.csv")

    metadata.update(
        {
            "runtime": {
                "python": platform.python_version(),
                "evidently": version("evidently"),
            },
            "input_files": {
                "historical_trips": {
                    "path": str(historical_trips_path),
                    "sha256": metadata["input_trip_data"]["historical_sha256"],
                },
                "current_trips": {
                    "path": str(current_trips_path),
                    "sha256": file_sha256(current_trips_path),
                },
            },
            "outputs": {
                "evidently_html": html_path.name,
                "evidently_json": json_path.name,
                "reference_profiles": "p3_perfiles_referencia.parquet",
                "current_profiles": "p3_perfiles_actual.parquet",
                "calibrated_drift_table": "p3_drift_calibrado.csv",
            },
            "drift_summary": {
                "evidently_threshold": UMBRAL_EVIDENTLY,
                "drifted_with_evidently_threshold": tabla.index[tabla.drift_evidently].tolist(),
                "drifted_with_calibrated_threshold": tabla.index[tabla.drift_calibrado].tolist(),
                "calibration": (
                    "Per-variable threshold = largest normed Wasserstein distance between one "
                    "reference month and the remaining reference months."
                ),
            },
            "interpretation": (
                "Drift is a feature-distribution signal, not proof of model-quality loss "
                "or an automatic retraining decision. Confirm it with the A/B evaluation."
            ),
        }
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera el reporte de drift de P3 (perfiles estación-mes) con Evidently."
    )
    parser.add_argument(
        "--historical-trips",
        type=Path,
        default=CONFIG.processed_data_dir / CONFIG.output_filename,
        help="Dataset combinado de python -m mibici.dataset (incluye el entrenamiento de P3).",
    )
    parser.add_argument(
        "--current-trips",
        type=Path,
        default=None,
        help="CSV crudo del mes actual (por omisión data/raw/datos_abiertos_AAAA_MM.csv).",
    )
    parser.add_argument("--nomenclature", type=Path, default=None)
    parser.add_argument("--min-dias", type=int, default=MIN_DIAS_ACTIVOS_MES)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--current-start", default="2026-10-01")
    parser.add_argument("--current-end-exclusive", default="2026-11-01")
    parser.add_argument(
        "--download-current",
        action="store_true",
        help="Descarga el CSV del mes actual si no existe.",
    )
    args = parser.parse_args()

    year, month = month_bounds(args.current_start, args.current_end_exclusive)
    output = args.output or CONFIG.reports_dir / "monitoring" / f"{year}_{month:02d}" / "p3"
    current = args.current_trips or CONFIG.raw_data_dir / f"datos_abiertos_{year}_{month:02d}.csv"
    metadata = run(
        args.historical_trips,
        current,
        output,
        args.current_start,
        args.current_end_exclusive,
        args.download_current,
        args.nomenclature,
        args.min_dias,
    )
    print(
        f"Reporte de drift de P3 en {output}: "
        f"{metadata['reference_period']['rows']} filas de referencia, "
        f"{metadata['current_period']['rows']} del mes actual."
    )


if __name__ == "__main__":
    main()
