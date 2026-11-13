from __future__ import annotations

import argparse
from datetime import date, timedelta
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform

from mibici.config import CONFIG
from mibici.monitoring.evidently_report import generate_report
from mibici.monitoring.p1_datasets import (
    P1_CATEGORICAL_FEATURES,
    P1_NUMERICAL_FEATURES,
    build_p1_populations,
)


def month_bounds(start: str, end_exclusive: str) -> tuple[int, int]:
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end_exclusive)
    if start_date.day != 1 or end_date != (start_date.replace(day=28) + timedelta(days=4)).replace(
        day=1
    ):
        raise ValueError("Current period must be exactly one calendar month.")
    return start_date.year, start_date.month


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    historical_trips_path: Path,
    current_trips_path: Path,
    output_dir: Path,
    current_start: str = "2026-10-01",
    current_end_exclusive: str = "2026-11-01",
    download_current: bool = False,
) -> dict:
    current_year, current_month = month_bounds(current_start, current_end_exclusive)
    if not historical_trips_path.is_file():
        raise FileNotFoundError(
            f"Historical trips file not found: {historical_trips_path}. "
            "Prepare it first with the documented historical-data command."
        )
    if not current_trips_path.is_file():
        if not download_current:
            raise FileNotFoundError(
                f"Current trips file not found: {current_trips_path}. "
                "Use --download-current to download the matching month."
            )
        from mibici.dataset import download_month

        downloaded = download_month(
            current_year, current_month, output_dir=current_trips_path.parent
        )
        if downloaded != current_trips_path:
            raise ValueError(
                f"Downloaded path {downloaded} does not match expected path {current_trips_path}."
            )

    reference, current, metadata = build_p1_populations(
        historical_trips_path,
        current_trips_path,
        current_start,
        current_end_exclusive,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "p1_data_drift.html"
    json_path = output_dir / "p1_data_drift.json"
    generate_report(
        reference,
        current,
        html_path,
        json_path,
        P1_NUMERICAL_FEATURES,
        P1_CATEGORICAL_FEATURES,
    )

    metadata.update(
        {
            "runtime": {
                "python": platform.python_version(),
                "evidently": version("evidently"),
            },
            "input_files": {
                "historical_trips": {
                    "path": str(historical_trips_path),
                    "sha256": file_sha256(historical_trips_path),
                },
                "current_trips": {
                    "path": str(current_trips_path),
                    "sha256": file_sha256(current_trips_path),
                },
            },
            "outputs": {
                "evidently_html": html_path.name,
                "evidently_json": json_path.name,
            },
            "interpretation": (
                "Drift is a feature-distribution signal, not proof of model-quality loss "
                "or an automatic retraining decision. Confirm with a separately designed "
                "labeled evaluation."
            ),
        }
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the P1 Evidently drift example for October 2026."
    )
    parser.add_argument(
        "--historical-trips",
        type=Path,
        default=CONFIG.processed_data_dir / CONFIG.output_filename,
        help="Combined historical trips through September 2026 (Parquet or CSV).",
    )
    parser.add_argument(
        "--current-trips",
        type=Path,
        default=None,
        help="Raw monthly CSV for the current period (default: matching YYYY_MM file).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for HTML, JSON, and input/data manifest.",
    )
    parser.add_argument("--current-start", default="2026-10-01")
    parser.add_argument("--current-end-exclusive", default="2026-11-01")
    parser.add_argument(
        "--download-current",
        action="store_true",
        help="Download the current monthly CSV if it is missing.",
    )
    args = parser.parse_args()

    current_year, current_month = month_bounds(args.current_start, args.current_end_exclusive)
    output = args.output or (
        CONFIG.reports_dir / "monitoring" / f"{current_year}_{current_month:02d}" / "p1"
    )
    current_trips = args.current_trips or (
        CONFIG.raw_data_dir / f"datos_abiertos_{current_year}_{current_month:02d}.csv"
    )
    metadata = run(
        args.historical_trips,
        current_trips,
        output,
        args.current_start,
        args.current_end_exclusive,
        args.download_current,
    )
    print(
        f"P1 drift report saved to {output}; "
        f"{metadata['reference_period']['rows']} reference rows, "
        f"{metadata['current_period']['rows']} current rows."
    )


if __name__ == "__main__":
    main()
