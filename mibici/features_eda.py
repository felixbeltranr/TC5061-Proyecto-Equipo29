import argparse
from pathlib import Path
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PowerTransformer

ROOT = Path(__file__).resolve().parents[1]
EDA_COLUMNS = [
    "origen_dpcapacidad",
    "destino_dpcapacidad",
    "Año_de_nacimiento",
    "origen_latitud",
    "origen_longitud",
    "destino_latitud",
    "destino_longitud",
    "Genero",
]


def create_features():
    # Transformadores para las columnas del DataFrame, para poder generar un modelo de predicción con regresión logística.
    # NUMÉRICAS:
    numericas_power_pipeline = Pipeline(
        steps=[
            ("impMediana", SimpleImputer(strategy="median")),
            ("powerNum", PowerTransformer(method="yeo-johnson", standardize=True)),
        ]
    )

    numericas_power_nombres = ["origen_dpcapacidad", "destino_dpcapacidad"]

    numericas_pipeline = Pipeline(
        steps=[("impMediana", SimpleImputer(strategy="median")), ("escalaNum", MinMaxScaler())]
    )

    numericas_pipeline_nombres = [
        "Año_de_nacimiento",
        "origen_latitud",
        "origen_longitud",
        "destino_latitud",
        "destino_longitud",
    ]

    # BINARIAS:
    catBin_pipeline = Pipeline(
        steps=[
            ("impModa", SimpleImputer(strategy="most_frequent")),
            ("codBinaria", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
        ]
    )

    catBin_pipeline_nombres = ["Genero"]

    columnasTransformer = ColumnTransformer(
        transformers=[
            ("numPower", numericas_power_pipeline, numericas_power_nombres),
            ("numpipe", numericas_pipeline, numericas_pipeline_nombres),
            ("catBin", catBin_pipeline, catBin_pipeline_nombres),
        ],
        remainder="drop",
    )

    return columnasTransformer


def export_features_eda(
    entrada: Path = ROOT / "data" / "processed" / "combined_dataset.parquet",
    salida: Path = ROOT / "data" / "interim" / "features_eda.parquet",
) -> Path:
    """Exporta solo las columnas de entrada de EDA a data/interim, conservando grupos Parquet."""
    entrada = Path(entrada)
    salida = Path(salida)
    parquet = pq.ParquetFile(entrada)
    faltantes = sorted(set(EDA_COLUMNS) - set(parquet.schema_arrow.names))
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas para EDA: {faltantes}")

    schema = pa.schema(
        [parquet.schema_arrow.field(column) for column in EDA_COLUMNS],
        metadata=parquet.schema_arrow.metadata,
    )
    salida.parent.mkdir(parents=True, exist_ok=True)
    temporal = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=salida.parent,
            prefix=f".{salida.name}.",
            suffix=".tmp",
            delete=False,
        ) as archivo:
            temporal = Path(archivo.name)
        with pq.ParquetWriter(temporal, schema, compression="zstd") as writer:
            for row_group in range(parquet.metadata.num_row_groups):
                writer.write_table(parquet.read_row_group(row_group, columns=EDA_COLUMNS))
        temporal.replace(salida)
    finally:
        if temporal is not None:
            temporal.unlink(missing_ok=True)
    return salida


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entrada", type=Path, default=ROOT / "data" / "processed" / "combined_dataset.parquet"
    )
    parser.add_argument(
        "--salida", type=Path, default=ROOT / "data" / "interim" / "features_eda.parquet"
    )
    args = parser.parse_args(argv)
    salida = export_features_eda(args.entrada, args.salida)
    print(f"Features EDA guardadas en {salida}")


if __name__ == "__main__":
    main()
