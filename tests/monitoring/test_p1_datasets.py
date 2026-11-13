import pandas as pd
import pytest

from types import SimpleNamespace

from mibici.monitoring import p1_datasets
from mibici.monitoring.p1_datasets import build_p1_populations


def trip_frame(start: str, end: str, station: int) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame(
        {
            "Inicio_del_viaje": dates + pd.Timedelta(hours=8),
            "Origen_Id": station,
        }
    )


def test_build_p1_populations_uses_configured_train_and_current_month(tmp_path):
    history = tmp_path / "historical.parquet"
    current = tmp_path / "current.csv"
    trip_frame("2025-01-01", "2026-09-30", 42).to_parquet(history, index=False)
    trip_frame("2026-10-01", "2026-10-31", 42).to_csv(current, index=False)

    reference, actual, metadata = build_p1_populations(
        history,
        current,
        "2026-10-01",
        "2026-11-01",
    )

    assert list(reference.columns) == list(actual.columns)
    assert metadata["reference_period"]["observed_end"] == "2026-04-29"
    assert metadata["current_period"]["observed_start"] == "2026-10-01"
    assert metadata["current_period"]["observed_end"] == "2026-10-31"


def test_build_p1_populations_rejects_wrong_current_month(tmp_path):
    history = tmp_path / "historical.parquet"
    current = tmp_path / "current.csv"
    trip_frame("2025-01-01", "2026-09-30", 42).to_parquet(history, index=False)
    trip_frame("2026-09-01", "2026-09-30", 42).to_csv(current, index=False)

    with pytest.raises(ValueError, match="Current-period trips must fall"):
        build_p1_populations(history, current, "2026-10-01", "2026-11-01")


def test_build_p1_populations_rejects_current_period_already_in_history(tmp_path):
    history = tmp_path / "historical.parquet"
    current = tmp_path / "current.csv"
    trip_frame("2025-01-01", "2026-10-31", 42).to_parquet(history, index=False)
    trip_frame("2026-10-01", "2026-10-31", 42).to_csv(current, index=False)

    with pytest.raises(ValueError, match="must end before the current-period start"):
        build_p1_populations(history, current, "2026-10-01", "2026-11-01")


def test_prepare_historical_trips_downloads_missing_months_and_writes_p1_columns(
    tmp_path, monkeypatch
):
    raw_dir = tmp_path / "raw"
    monkeypatch.setattr(p1_datasets, "CONFIG", SimpleNamespace(raw_data_dir=raw_dir))

    def download(year, month):
        raw_dir.mkdir(parents=True, exist_ok=True)
        trips = trip_frame(f"{year}-{month:02d}-01", f"{year}-{month:02d}-02", 42)
        path = raw_dir / f"datos_abiertos_{year}_{month:02d}.csv"
        trips.to_csv(path, index=False)
        return path

    monkeypatch.setattr(p1_datasets.dataset, "download_month", download)
    output = tmp_path / "interim" / "history.parquet"

    result = p1_datasets.prepare_historical_trips(
        "2025-03-01",
        output,
        start_year=2025,
        start_month=1,
    )

    history = pd.read_parquet(result)
    assert list(history.columns) == ["Inicio_del_viaje", "Origen_Id"]
    assert len(history) == 4
    assert history["Inicio_del_viaje"].max().month == 2
