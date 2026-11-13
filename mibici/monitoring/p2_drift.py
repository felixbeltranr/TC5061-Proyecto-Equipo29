"""Reporte de drift de P2 con Evidently: filas de train del campeón contra el mes actual.

Uso desde la raíz del repositorio (después de ``python -m mibici.dataset`` y del panel de
``python -m mibici.features_p2``)::

    python -m mibici.monitoring.p2_drift --download-current
    python -m mibici.monitoring --question p2 --download-current

El campeón se busca en el Model Registry de ``MLFLOW_TRACKING_URI`` (o de
``CONFIG_P2.tracking_uri`` si no está definida), porque de su run salen las estadísticas
``est_*`` de la referencia y del mes actual.

Guarda en ``reports/monitoring/AAAA_MM/p2``: el reporte HTML y JSON de Evidently, la tabla de
drift con umbrales calibrados (``p2_drift_calibrado.csv``), las filas del mes actual sin
``est_*`` (``p2_filas_actual.parquet``, para el A/B testing), dos figuras y el manifiesto.

Las variables de calendario (``hora``, ``dia_semana`` y ``fin_semana``) se declaran categóricas;
su drift depende de cómo cae el mes en la semana y no se interpreta. Para las 15 numéricas el
umbral calibrado de cada variable es la mayor distancia de Wasserstein normada entre un mes del
entrenamiento y los demás (``p3_drift.umbrales_calibrados``), y la distancia se separa por
estaciones que ya estaban en el train del campeón y estaciones nuevas. Como control estacional se
reporta también la distancia contra el mismo mes del año anterior.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from mibici.config import CONFIG, CONFIG_P2
from mibici.monitoring import p2_datasets as d2
from mibici.monitoring.evidently_report import generate_report
from mibici.monitoring.p1_drift import file_sha256, month_bounds
from mibici.monitoring.p3_drift import UMBRAL_EVIDENTLY, distancia_normada, tabla_drift


def tracking_uri_por_omision() -> str:
    """``MLFLOW_TRACKING_URI`` si está definida; si no, el almacén local de P2."""
    return os.environ.get("MLFLOW_TRACKING_URI") or CONFIG_P2.tracking_uri


def drift_evidently(json_path: Path) -> pd.DataFrame:
    """Puntaje, método y umbral de Evidently por variable, leídos de su reporte JSON."""
    metricas = json.loads(json_path.read_text(encoding="utf-8"))["metrics"]
    filas = [
        {
            "variable": m["config"]["column"],
            "metodo_evidently": m["config"]["method"],
            "puntaje_evidently": float(m["value"]),
            "umbral_evidently": float(m["config"]["threshold"]),
        }
        for m in metricas
        if m["config"]["type"].endswith(":ValueDrift")
    ]
    tabla = pd.DataFrame(filas).set_index("variable")
    tabla["drift_evidently_reporte"] = tabla["puntaje_evidently"] >= tabla["umbral_evidently"]
    return tabla


def tabla_drift_p2(
    referencia: pd.DataFrame, actual: pd.DataFrame, mes_control: str
) -> pd.DataFrame:
    """Tabla de ``p3_drift.tabla_drift`` más la distancia contra el mes de control."""
    ref, act = d2.por_mes(referencia), d2.por_mes(actual)
    tabla = tabla_drift(ref, act)
    control = ref.xs(mes_control, level="mes")
    tabla[f"distancia_vs_{mes_control}"] = [
        distancia_normada(control[col], act[col]) for col in tabla.index
    ]
    return tabla


def proporciones_por_grupo(
    referencia: pd.DataFrame, actual: pd.DataFrame, mes_control: str
) -> dict[str, dict[str, float]]:
    """Proporción de clases de la etiqueta proxy en referencia, control y mes actual."""
    ya_estaban = actual["estacion"].isin(referencia["estacion"].unique())
    mes = referencia["ts_objetivo"].dt.to_period("M").astype(str)
    return {
        "referencia": d2.proporciones(referencia["objetivo"]),
        mes_control: d2.proporciones(referencia.loc[mes == mes_control, "objetivo"]),
        "actual": d2.proporciones(actual["objetivo"]),
        "actual_existentes": d2.proporciones(actual.loc[ya_estaban, "objetivo"]),
        "actual_nuevas": d2.proporciones(actual.loc[~ya_estaban, "objetivo"]),
    }


def figura_distancias(tabla: pd.DataFrame, titulo: str, ruta: Path) -> Path:
    """Distancia del mes actual por variable contra el umbral calibrado y el de Evidently."""
    orden = tabla.sort_values("distancia")
    fig, ax = plt.subplots(figsize=(8, 6))
    y = range(len(orden))
    ax.barh(y, orden["distancia"], color="#4C72B0", label="Mes actual, todas las estaciones")
    ax.scatter(orden["umbral_calibrado"], y, color="#C44E52", marker="|", s=250, zorder=3)
    ax.scatter([], [], color="#C44E52", marker="|", s=120, label="Umbral calibrado")
    ax.axvline(UMBRAL_EVIDENTLY, color="gray", linestyle="--", label="Umbral de Evidently (0.1)")
    ax.set_yticks(list(y), orden.index)
    ax.set_xlabel("Distancia de Wasserstein normada")
    ax.set_title(titulo)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    return ruta


def figura_clases(proporciones: dict[str, dict[str, float]], titulo: str, ruta: Path) -> Path:
    """Proporción de sin_bicicletas y sin_anclajes por grupo de filas."""
    tabla = pd.DataFrame(proporciones).T[["sin_bicicletas", "sin_anclajes"]] * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    tabla.plot.bar(ax=ax, color=["#DD8452", "#55A868"], rot=0)
    ax.set_ylabel("% de filas estación-hora")
    ax.set_title(titulo)
    for contenedor in ax.containers:
        ax.bar_label(contenedor, fmt="%.1f", fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    return ruta


def run(
    historical_trips_path: Path,
    current_trips_path: Path,
    output_dir: Path,
    current_start: str = "2026-10-01",
    current_end_exclusive: str = "2026-11-01",
    download_current: bool = False,
    nomenclature_path: Path | None = None,
    tracking_uri: str | None = None,
    training_panel_path: Path = CONFIG_P2.panel_path,
) -> dict:
    """Genera el reporte de drift de P2 y regresa su manifiesto."""
    year, month = month_bounds(current_start, current_end_exclusive)
    inicio, fin_excl = pd.Timestamp(current_start), pd.Timestamp(current_end_exclusive)
    if inicio < pd.Timestamp(CONFIG_P2.fecha_fin_prueba):
        raise ValueError(f"El mes actual debe empezar en {CONFIG_P2.fecha_fin_prueba} o después.")
    for ruta, comando in (
        (historical_trips_path, "python -m mibici.dataset"),
        (training_panel_path, "python -m mibici.features_p2"),
    ):
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe {ruta}. Genéralo con: {comando}")
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

    tracking_uri = tracking_uri or tracking_uri_por_omision()
    version_a, run_a = d2.version_por_alias(CONFIG_P2.alias_campeon, tracking_uri)
    estadisticas_a = d2.estadisticas_del_run(run_a, tracking_uri)
    nomenclaturas = [nomenclature_path] if nomenclature_path else d2.nomenclaturas_locales()

    referencia = d2.poblacion_referencia(pd.read_parquet(training_panel_path), estadisticas_a)
    panel, resumen_panel = d2.panel_actual(
        historical_trips_path,
        current_trips_path,
        nomenclaturas,
        inicio,
        fin_excl,
        CONFIG.interim_data_dir / "monitoring",
    )
    filas_actual = d2.en_periodo(d2.filas_del_modelo(panel), inicio, fin_excl)
    del panel
    actual = d2.con_estadisticas(filas_actual, estadisticas_a)
    mes_control = str(pd.Period(inicio, freq="M") - 12)

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "p2_data_drift.html"
    json_path = output_dir / "p2_data_drift.json"
    generate_report(
        referencia[list(d2.P2_FEATURES)],
        actual[list(d2.P2_FEATURES)],
        html_path,
        json_path,
        d2.P2_NUMERICAL_FEATURES,
        d2.P2_CATEGORICAL_FEATURES,
    )
    tabla = tabla_drift_p2(referencia, actual, mes_control)
    evidently = drift_evidently(json_path)
    tabla = tabla.join(evidently[["metodo_evidently", "puntaje_evidently"]], how="left")
    tabla.to_csv(output_dir / "p2_drift_calibrado.csv")
    evidently.to_csv(output_dir / "p2_drift_evidently.csv")
    filas_actual.to_parquet(output_dir / "p2_filas_actual.parquet", index=False)
    clases = proporciones_por_grupo(referencia, actual, mes_control)
    etiqueta_mes = f"{year}-{month:02d}"
    figuras = [
        figura_distancias(
            tabla,
            f"P2: drift de {etiqueta_mes} contra el train del campeón",
            output_dir / "p2_drift_distancias.png",
        ),
        figura_clases(
            clases,
            "P2: proporción de clases de riesgo (etiqueta proxy)",
            output_dir / "p2_drift_clases.png",
        ),
    ]

    estaciones_ref = set(referencia["estacion"].unique())
    estaciones_act = set(actual["estacion"].unique())
    metadata = {
        "problem": "P2",
        "unit_of_analysis": "station-hour (target hour 6-23, horizon 1 h)",
        "champion": {
            "model": CONFIG_P2.modelo_registrado,
            "alias": CONFIG_P2.alias_campeon,
            "version": version_a,
            "run_id": run_a,
        },
        "reference_period": {
            "target_start_inclusive": CONFIG_P2.fecha_inicio_entrenamiento,
            "target_end_exclusive": CONFIG_P2.fecha_inicio_prueba,
            **d2.resumen_calidad(referencia, "P2 referencia"),
        },
        "current_period": {
            "target_start_inclusive": current_start,
            "target_end_exclusive": current_end_exclusive,
            **d2.resumen_calidad(actual, "P2 actual"),
            "stations_not_in_reference": len(estaciones_act - estaciones_ref),
            "reference_stations_without_rows": len(estaciones_ref - estaciones_act),
            "panel": resumen_panel,
        },
        "seasonal_control_month": mes_control,
        "proxy_label_share": clases,
        "runtime": {
            "python": platform.python_version(),
            "evidently": version("evidently"),
            "pandas": version("pandas"),
        },
        "input_files": {
            "historical_trips": {
                "path": str(historical_trips_path),
                "sha256": resumen_panel["huellas_sha256"][historical_trips_path.name],
            },
            "current_trips": {
                "path": str(current_trips_path),
                "sha256": file_sha256(current_trips_path),
            },
            "training_panel": {
                "path": str(training_panel_path),
                "sha256": file_sha256(training_panel_path),
            },
            "nomenclatures": {
                ruta.name: resumen_panel["huellas_sha256"][ruta.name] for ruta in nomenclaturas
            },
        },
        "outputs": {
            "evidently_html": html_path.name,
            "evidently_json": json_path.name,
            "calibrated_drift_table": "p2_drift_calibrado.csv",
            "evidently_drift_table": "p2_drift_evidently.csv",
            "current_rows": "p2_filas_actual.parquet",
            "figures": [ruta.name for ruta in figuras],
        },
        "drift_summary": {
            "evidently_drifted": evidently.index[evidently.drift_evidently_reporte].tolist(),
            "evidently_threshold": UMBRAL_EVIDENTLY,
            "drifted_with_calibrated_threshold": tabla.index[tabla.drift_calibrado].tolist(),
            "calibration": (
                "Per-variable threshold = largest normed Wasserstein distance between one "
                "reference month and the remaining reference months (numerical features only)."
            ),
        },
        "interpretation": (
            "Drift is a feature-distribution signal, not proof of model-quality loss "
            "or an automatic retraining decision. Confirm it with the A/B evaluation."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera el reporte de drift de P2 (filas estación-hora) con Evidently."
    )
    parser.add_argument(
        "--historical-trips",
        type=Path,
        default=CONFIG.processed_data_dir / CONFIG.output_filename,
        help="Dataset combinado de python -m mibici.dataset (termina antes del mes actual).",
    )
    parser.add_argument(
        "--current-trips",
        type=Path,
        default=None,
        help="CSV crudo del mes actual (por omisión data/raw/datos_abiertos_AAAA_MM.csv).",
    )
    parser.add_argument(
        "--nomenclature",
        type=Path,
        default=None,
        help="Nomenclatura única; por omisión todas las de data/raw, la más reciente primero.",
    )
    parser.add_argument("--training-panel", type=Path, default=CONFIG_P2.panel_path)
    parser.add_argument("--tracking-uri", default=None)
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
    output = args.output or CONFIG.reports_dir / "monitoring" / f"{year}_{month:02d}" / "p2"
    current = args.current_trips or CONFIG.raw_data_dir / f"datos_abiertos_{year}_{month:02d}.csv"
    metadata = run(
        args.historical_trips,
        current,
        output,
        args.current_start,
        args.current_end_exclusive,
        args.download_current,
        args.nomenclature,
        args.tracking_uri,
        args.training_panel,
    )
    print(
        f"Reporte de drift de P2 en {output}: "
        f"{metadata['reference_period']['rows']} filas de referencia, "
        f"{metadata['current_period']['rows']} del mes actual."
    )


if __name__ == "__main__":
    main()
