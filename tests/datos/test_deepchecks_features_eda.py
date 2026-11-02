"""DeepChecks sobre entradas y features EDA del dataset combinado real.

Requiere el entorno Python 3.11 de requirements-deepchecks.txt. No carga IDs ni datos
personales, no entrena y no inventa una partición train/test para medir drift.
"""

import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

pytest.importorskip("deepchecks", reason="requiere el entorno DeepChecks (ver README)")

from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import MixedDataTypes, PercentOfNulls

RAIZ = Path(__file__).resolve().parents[2]
INTERIM = RAIZ / "data" / "interim"
ENTRADA = INTERIM / "features_eda.parquet"
REPORTES = RAIZ / "reports" / "deepchecks"
SEMILLA = 29
MUESTRA_POR_GRUPO = 2_000
COLUMNAS = [
    "origen_dpcapacidad",
    "destino_dpcapacidad",
    "Año_de_nacimiento",
    "origen_latitud",
    "origen_longitud",
    "destino_latitud",
    "destino_longitud",
    "Genero",
]
NUMERICAS = [column for column in COLUMNAS if column != "Genero"]


@pytest.fixture(scope="module")
def datos_eda() -> pd.DataFrame:
    if not ENTRADA.is_file():
        pytest.skip(
            "No existe data/interim/features_eda.parquet. "
            "Generalo con 'python -m mibici.features_eda' en el entorno principal."
        )

    parquet = pq.ParquetFile(ENTRADA)
    faltantes = sorted(set(COLUMNAS) - set(parquet.schema_arrow.names))
    if faltantes:
        pytest.fail(f"Faltan columnas EDA requeridas en el dataset combinado: {faltantes}")

    rng = np.random.default_rng(SEMILLA)
    muestras = []
    for row_group in range(parquet.metadata.num_row_groups):
        grupo = parquet.read_row_group(row_group, columns=COLUMNAS).to_pandas()
        indices = rng.choice(len(grupo), size=min(MUESTRA_POR_GRUPO, len(grupo)), replace=False)
        muestras.append(grupo.iloc[indices])
    return pd.concat(muestras, ignore_index=True)


def _guardar(resultado, nombre: str) -> None:
    REPORTES.mkdir(parents=True, exist_ok=True)
    ruta = REPORTES / nombre
    ruta.unlink(missing_ok=True)
    resultado.save_as_html(str(ruta), as_widget=False)


def _suite_integridad(nombre: str, data: pd.DataFrame, categoricas: list[str]):
    resultado = Suite(
        nombre,
        MixedDataTypes(),
        PercentOfNulls(),
    ).run(Dataset(data, cat_features=categoricas))
    assert len(resultado.results) == 2
    return resultado


def test_integridad_de_entradas_eda(datos_eda):
    """Revisa tipos mixtos y cuantifica nulos que los imputadores EDA están diseñados a tratar."""
    for column in NUMERICAS:
        assert pd.api.types.is_numeric_dtype(datos_eda[column]), column

    resultado = _suite_integridad("Integridad entradas EDA", datos_eda, ["Genero"])
    _guardar(resultado, "features_eda_entradas.html")
    assert "Mixed Data Types" in {r.get_header() for r in resultado.results}
    assert resultado.passed(fail_if_warning=True, fail_if_check_not_run=True)
    nulls = datos_eda.isna().sum().astype(int).to_dict()
    (REPORTES / "features_eda_resumen.json").write_text(
        json.dumps(
            {
                "scope": "sampled_real_combined_dataset_no_ids_or_user_columns",
                "input": ENTRADA.name,
                "seed": SEMILLA,
                "sample_rows": len(datos_eda),
                "sampled_parquet_row_groups": pq.ParquetFile(ENTRADA).metadata.num_row_groups,
                "null_counts": nulls,
                "null_ratios": {
                    column: float(count / len(datos_eda)) for column, count in nulls.items()
                },
                "drift": "not_measured_no_semantic_train_test_split_for_shared_eda",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_transformacion_eda_genera_features_finitas_sin_nulos(datos_eda):
    """Ejecuta la transformación real sin importar la configuración del proyecto."""
    ruta_features = RAIZ / "mibici" / "features_eda.py"
    create_features = runpy.run_path(str(ruta_features))["create_features"]
    transformer = create_features()
    transformed = transformer.fit_transform(datos_eda)
    names = transformer.get_feature_names_out()
    features = pd.DataFrame(transformed, columns=names)

    assert features.shape[0] == len(datos_eda)
    assert features.shape[1] > 0
    assert np.isfinite(features.to_numpy(dtype=float)).all()
    assert not features.isna().any().any()

    integrity = _suite_integridad("Integridad features EDA", features, [])
    _guardar(integrity, "features_eda_transformadas.html")
    assert integrity.passed(fail_if_warning=True, fail_if_check_not_run=True)
    result = integrity.results[-1]
    assert result.get_header() == "PercentOfNulls"
    assert result.value.to_numpy().max() == 0
