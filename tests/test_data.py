import pandas as pd
import pytest

from mibici import dataset


def _viajes():
    return pd.DataFrame(
        {
            "Inicio_del_viaje": ["2025-01-01 00:00:44", "2025-01-01 00:04:11"],
            "Fin_del_viaje": ["2025-01-01 00:11:29", "2025-01-01 00:15:07"],
            "Origen_Id": ["211", "35"],
            "Destino_Id": ["395", "260"],
        }
    )


def _nomenclatura():
    return pd.DataFrame(
        {
            "id": [35, 211, 260, 395],
            "name": ["a", "b", "c", "d"],
            "latitude": [20.6, 20.7, 20.8, 20.9],
            "longitude": [-103.3, -103.4, -103.5, -103.6],
            "dpcapacity": [12, 15, 20, 24],
        }
    )


def test_normaliza_nombres_de_columnas(tmp_path):
    ruta = tmp_path / "datos_abiertos_2025_01.csv"
    ruta.write_bytes(
        " Viaje_Id ,Inicio del viaje,A\xf1o_de_nacimiento\n1,2025-01-01,1990\n".encode("cp1252")
    )
    columnas = list(dataset._read_dataset(ruta).columns)
    assert columnas == ["Viaje_Id", "Inicio_del_viaje", "Año_de_nacimiento"]


def test_combinado_tiene_fechas_y_estaciones():
    combinado = dataset.merge_dataset_define_nomenclature(
        {"datos_abiertos_2025_01": _viajes()}, _nomenclatura()
    )
    assert pd.api.types.is_datetime64_any_dtype(combinado["Inicio_del_viaje"])
    assert pd.api.types.is_datetime64_any_dtype(combinado["Fin_del_viaje"])
    assert combinado["origen_nombre"].tolist() == ["b", "a"]
    assert combinado["destino_dpcapacidad"].tolist() == [24, 20]


def test_main_crea_carpeta_y_guarda_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dataset,
        "download_and_load_datasets",
        lambda *args: {"datos_abiertos_2025_01": _viajes()},
    )
    monkeypatch.setattr(
        dataset, "download_and_load_latest_nomenclature", lambda *args: _nomenclatura()
    )
    salida = tmp_path / "processed"
    dataset.main(output_raw_dir=tmp_path / "raw", output_processed_dir=salida)
    guardado = pd.read_parquet(salida / dataset.CONFIG.output_filename)
    assert len(guardado) == 2
    assert pd.api.types.is_datetime64_any_dtype(guardado["Inicio_del_viaje"])


def test_filtra_dataset_por_rango_inclusivo_y_ordenado():
    urls = [
        "https://example.test/datos_abiertos_2024_12.csv",
        "https://example.test/datos_abiertos_2025_01.csv",
        "https://example.test/datos_abiertos_2025_03.csv",
        "https://example.test/datos_abiertos_2025_04.csv",
    ]

    filtradas = dataset._filter_dataset_urls(urls, 2025, 1, 2025, 3)

    assert [url.rsplit("/", 1)[-1] for url in filtradas] == [
        "datos_abiertos_2025_01.csv",
        "datos_abiertos_2025_03.csv",
    ]


@pytest.mark.parametrize(
    "bounds",
    [
        (2025, 1, 2025, None),
        (2025, 0, 2025, 12),
        (2025, 10, 2025, 1),
    ],
)
def test_filtra_dataset_rechaza_rangos_invalidos(bounds):
    with pytest.raises(ValueError):
        dataset._filter_dataset_urls([], *bounds)


def test_preprocess_dataframe_limpia_sin_mutar_entrada():
    original = pd.DataFrame(
        {
            "origen_nombre": ["Centro", "Norte", "Sur", None, "Centro"],
            "destino_nombre": ["Norte", "Sur", "Centro", "Norte", "Norte"],
            "Usuario_Id": [10, 11, 12, 13, 10],
            "Año_de_nacimiento": [1990.0, None, 1890.0, 1985.0, 1990.0],
            "Genero": ["M", None, "F", "F", "M"],
        }
    )
    antes = original.copy(deep=True)

    resultado = dataset.preprocess_dataframe(original)

    assert len(resultado) == 2
    assert "Usuario_Id" not in resultado
    assert resultado["Año_de_nacimiento"].notna().all()
    assert set(resultado["Genero"]) == {"M", "Desconocido"}
    pd.testing.assert_frame_equal(original, antes)
