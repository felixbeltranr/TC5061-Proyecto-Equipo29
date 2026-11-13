from __future__ import annotations

from calendar import monthrange
from datetime import date
from pathlib import Path

import pandas as pd

from mibici import dataset, features
from mibici.config import CONFIG, CONFIG_P1
from mibici.features import FEATURES
from mibici.monitoring.contracts import validate_feature_frame

P1_CATEGORICAL_FEATURES = ("dia_semana", "festivo", "festivo_lag7")
P1_NUMERICAL_FEATURES = tuple(
    feature for feature in FEATURES if feature not in P1_CATEGORICAL_FEATURES
)


def prepare_historical_trips(
    current_start: str,
    output_path: Path,
    start_year: int = CONFIG.download_data_start_year,
    start_month: int = CONFIG.download_data_start_month,
) -> Path:
    """Download/load raw months before current_start into a P1-only historical Parquet."""
    first = date(start_year, start_month, 1)
    current = date.fromisoformat(current_start)
    if current.day != 1:
        raise ValueError("The current period must start on the first day of a month.")
    if (first.year, first.month) >= (current.year, current.month):
        raise ValueError("The configured historical start must precede the current period.")

    previous_month = date(current.year, current.month, 1)
    if previous_month.month == 1:
        last = date(previous_month.year - 1, 12, 1)
    else:
        last = date(previous_month.year, previous_month.month - 1, 1)

    months = []
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        months.append((year, month))
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1

    trips = []
    for year, month in months:
        source = CONFIG.raw_data_dir / f"datos_abiertos_{year}_{month:02d}.csv"
        if not source.is_file():
            dataset.download_month(year, month)
        frame = features.cargar_salidas(source)
        dates = frame["Inicio_del_viaje"].dropna().dt.normalize()
        if (
            dates.empty
            or dates.min().date() < date(year, month, 1)
            or dates.max().date() > date(year, month, monthrange(year, month)[1])
        ):
            raise ValueError(f"Trip dates in {source.name} do not match its month.")
        trips.append(frame)

    history = pd.concat(trips, ignore_index=True)
    history = history.loc[:, ["Inicio_del_viaje", "Origen_Id"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(output_path, index=False)
    return output_path


def build_p1_populations(
    historical_trips_path: Path,
    current_trips_path: Path,
    current_start: str,
    current_end_exclusive: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build P1 reference/current features using the existing feature pipeline.

    The historical file must end before the current period. The current CSV is appended
    in memory so the normal lag/rolling-window logic can construct October features.
    """
    current_start_ts = pd.Timestamp(current_start)
    current_end_ts = pd.Timestamp(current_end_exclusive)
    if current_start_ts >= current_end_ts:
        raise ValueError("The current period start must be before its exclusive end.")
    if current_start_ts.day != 1 or current_end_ts != current_start_ts + pd.offsets.MonthBegin():
        raise ValueError("This P1 example expects exactly one complete calendar month.")

    historical = features.cargar_salidas(historical_trips_path)
    current = features.cargar_salidas(current_trips_path)
    historical_dates = historical["Inicio_del_viaje"].dropna().dt.normalize()
    if historical_dates.empty:
        raise ValueError("The historical file has no parseable trip dates.")
    if (historical_dates >= current_start_ts).any():
        raise ValueError(
            "The historical trips file must end before the current-period start; "
            "otherwise current observations may be duplicated."
        )

    current_dates = current["Inicio_del_viaje"].dropna().dt.normalize()
    if current_dates.empty:
        raise ValueError("The current-period file has no parseable trip dates.")
    if (current_dates < current_start_ts).any() or (current_dates >= current_end_ts).any():
        raise ValueError(
            f"Current-period trips must fall in [{current_start_ts.date()}, "
            f"{current_end_ts.date()})."
        )

    trips = pd.concat([historical, current], ignore_index=True)
    panel, _ = features.construir_panel_diario(
        trips,
        inicio=pd.Timestamp(CONFIG_P1.train_inicio),
        fin=current_end_ts - pd.Timedelta(days=1),
    )
    engineered = features.construir_features(panel)
    eligible = engineered.loc[engineered["elegible"]].copy()

    reference_end = features.ultimo_dia_conocido(pd.Timestamp(CONFIG_P1.test_inicio))
    reference_start = pd.Timestamp(CONFIG_P1.train_inicio)
    reference_mask = eligible["fecha"].between(reference_start, reference_end)
    current_period_mask = (engineered["fecha"] >= current_start_ts) & (
        engineered["fecha"] < current_end_ts
    )
    current_all = engineered.loc[current_period_mask]
    current_mask = (eligible["fecha"] >= current_start_ts) & (eligible["fecha"] < current_end_ts)

    reference = eligible.loc[reference_mask, FEATURES].copy()
    current_features = eligible.loc[current_mask, FEATURES].copy()
    reference_quality = validate_feature_frame(
        reference,
        "P1 reference",
        FEATURES,
        P1_NUMERICAL_FEATURES,
        P1_CATEGORICAL_FEATURES,
    )
    current_quality = validate_feature_frame(
        current_features,
        "P1 current",
        FEATURES,
        P1_NUMERICAL_FEATURES,
        P1_CATEGORICAL_FEATURES,
    )

    actual_reference_dates = eligible.loc[reference_mask, "fecha"]
    actual_current_dates = eligible.loc[current_mask, "fecha"]
    metadata = {
        "problem": "P1",
        "unit_of_analysis": "station-day",
        "reference_period": {
            "configured_start_inclusive": str(reference_start.date()),
            "configured_end_inclusive": str(reference_end.date()),
            "observed_start": str(actual_reference_dates.min().date()),
            "observed_end": str(actual_reference_dates.max().date()),
            **reference_quality,
        },
        "current_period": {
            "configured_start_inclusive": str(current_start_ts.date()),
            "configured_end_exclusive": str(current_end_ts.date()),
            "observed_start": str(actual_current_dates.min().date()),
            "observed_end": str(actual_current_dates.max().date()),
            **current_quality,
            "rows_before_eligibility_filter": len(current_all),
            "stations_before_eligibility_filter": int(current_all["estacion"].nunique()),
            "rows_without_sufficient_history": int(current_all["sin_historia"].sum()),
            "rows_inactive": int(current_all["inactiva"].sum()),
            "eligible_stations": int(eligible.loc[current_mask, "estacion"].nunique()),
        },
        "input_trip_data": {
            "historical": {
                "rows": len(historical),
                "unparseable_start_dates": int(historical["Inicio_del_viaje"].isna().sum()),
                "observed_start": str(historical_dates.min().date()),
                "observed_end": str(historical_dates.max().date()),
            },
            "current": {
                "rows": len(current),
                "unparseable_start_dates": int(current["Inicio_del_viaje"].isna().sum()),
                "observed_start": str(current_dates.min().date()),
                "observed_end": str(current_dates.max().date()),
            },
        },
        "feature_roles": {
            "categorical": list(P1_CATEGORICAL_FEATURES),
            "numerical": list(P1_NUMERICAL_FEATURES),
        },
        "exclusions": ["estacion", "fecha", "objetivo"],
        "notes": [
            (
                "The reference uses the configured P1 training period and its existing "
                "two-day label-availability embargo before the May 2026 evaluation window."
            ),
            (
                "Current-period features are calculated by the original P1 feature builder; "
                "current labels and row identifiers are excluded from the drift comparison."
            ),
            "Rows repeat stations and dates and are not statistically independent.",
            (
                "Stations without sufficient P1 history are counted in metadata but excluded "
                "from the feature-drift comparison."
            ),
        ],
    }
    return reference, current_features, metadata
