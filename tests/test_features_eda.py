import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from mibici.features_eda import EDA_COLUMNS, create_features, export_features_eda


def _sample():
    return pd.DataFrame(
        {
            "origen_dpcapacidad": [10.0, 20.0, np.nan, 40.0],
            "destino_dpcapacidad": [12.0, 18.0, 24.0, 32.0],
            "Año_de_nacimiento": [1980.0, 1990.0, np.nan, 2000.0],
            "origen_latitud": [20.1, 20.2, 20.3, 20.4],
            "origen_longitud": [-103.1, -103.2, -103.3, -103.4],
            "destino_latitud": [20.5, 20.6, 20.7, 20.8],
            "destino_longitud": [-103.5, -103.6, -103.7, -103.8],
            "Genero": ["M", "F", None, "M"],
        }
    )


def test_create_features_imputes_scales_and_encodes_known_data():
    data = _sample()
    transformer = create_features()

    transformed = transformer.fit_transform(data)

    assert transformed.shape == (len(data), 8)
    assert np.isfinite(transformed).all()
    scaled_columns = transformed[:, 2:7]
    np.testing.assert_allclose(scaled_columns.min(axis=0), 0.0)
    np.testing.assert_allclose(scaled_columns.max(axis=0), 1.0)


def test_create_features_accepts_missing_values_and_unseen_gender():
    train = _sample()
    test = train.iloc[[0, 1]].copy()
    test.loc[test.index[0], "Genero"] = "X"
    test.loc[test.index[1], "Año_de_nacimiento"] = np.nan
    transformer = create_features().fit(train)

    with pytest.warns(UserWarning, match="unknown categories"):
        transformed = transformer.transform(test)

    assert transformed.shape == (2, 8)
    assert np.isfinite(transformed).all()
    # Unknown categories are ignored by the configured OneHotEncoder.
    assert transformed[0, -1] == 0


def test_export_features_eda_writes_only_selected_columns_and_preserves_row_groups(tmp_path):
    source = tmp_path / "combined.parquet"
    output = tmp_path / "interim" / "features_eda.parquet"
    raw = _sample().assign(Viaje_Id=range(4), Usuario_Id=[10, 11, 12, 13])
    raw.to_parquet(source, index=False, row_group_size=2)

    written = export_features_eda(source, output)

    exported = pd.read_parquet(written)
    parquet = pq.ParquetFile(written)
    assert exported.columns.tolist() == EDA_COLUMNS
    assert len(exported) == len(raw)
    assert parquet.metadata.num_row_groups == 2
    pd.testing.assert_frame_equal(exported, raw[EDA_COLUMNS])


def test_export_features_eda_rejects_missing_columns(tmp_path):
    source = tmp_path / "incomplete.parquet"
    output = tmp_path / "features_eda.parquet"
    pd.DataFrame({"Genero": ["M"]}).to_parquet(source, index=False)

    with pytest.raises(ValueError, match="Faltan columnas requeridas para EDA"):
        export_features_eda(source, output)

    assert not output.exists()
