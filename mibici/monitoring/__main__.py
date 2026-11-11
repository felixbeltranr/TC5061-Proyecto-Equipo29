from __future__ import annotations

import argparse
import json
from pathlib import Path

from mibici.config import CONFIG
from mibici.monitoring.p1_datasets import prepare_historical_trips
from mibici.monitoring.p1_drift import month_bounds
from mibici.monitoring.p1_drift import run as run_p1

QUESTIONS = ("p1", "p2", "p3")


def run(
    question: str,
    current_start: str,
    current_end_exclusive: str,
    output_dir: Path,
    historical_trips: Path,
    current_trips: Path,
    download_current: bool = False,
    prepare_history: bool = False,
) -> dict:
    selected = question.lower()
    if selected != "all" and selected not in QUESTIONS:
        raise ValueError(f"question must be one of all, {', '.join(QUESTIONS)}.")
    if selected not in ("all", "p1"):
        raise ValueError(f"No monitoring adapter is registered for {selected.upper()}.")
    month_bounds(current_start, current_end_exclusive)
    targets = QUESTIONS if selected == "all" else (selected,)
    results = {}
    question_output_dirs = {name: output_dir / name for name in targets}
    for question_output_dir in question_output_dirs.values():
        question_output_dir.mkdir(parents=True, exist_ok=True)

    if "p1" in targets:
        p1_history = historical_trips
        if prepare_history:
            year, month = month_bounds(current_start, current_end_exclusive)
            p1_history = (
                CONFIG.interim_data_dir
                / "monitoring"
                / (f"p1_historical_before_{year}_{month:02d}.parquet")
            )
            prepare_historical_trips(current_start, p1_history)
        run_p1(
            p1_history,
            current_trips,
            question_output_dirs["p1"],
            current_start,
            current_end_exclusive,
            download_current,
        )
        results["p1"] = {
            "status": "completed",
            "report_dir": "p1",
        }

    summary = {
        "question": selected,
        "current_period": {
            "start_inclusive": current_start,
            "end_exclusive": current_end_exclusive,
        },
        "questions": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "monitoring_run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate data-monitoring reports for one or all project questions."
    )
    parser.add_argument(
        "--question",
        choices=("all", *QUESTIONS),
        default="all",
        help="all runs registered adapters; select a project question.",
    )
    parser.add_argument("--current-start", default="2026-10-01")
    parser.add_argument("--current-end-exclusive", default="2026-11-01")
    parser.add_argument(
        "--historical-trips",
        type=Path,
        default=CONFIG.processed_data_dir / CONFIG.output_filename,
        help="Existing historical combined dataset, normally ending before the current month.",
    )
    parser.add_argument(
        "--current-trips",
        type=Path,
        default=None,
        help="Current monthly trip CSV. Defaults to data/raw/datos_abiertos_YYYY_MM.csv.",
    )
    parser.add_argument(
        "--download-current",
        action="store_true",
        help="Download the current monthly CSV if it is missing.",
    )
    parser.add_argument(
        "--prepare-history",
        action="store_true",
        help="Ensure raw historical months exist and build a separate P1-only history Parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: reports/monitoring/YYYY_MM).",
    )
    args = parser.parse_args()

    if args.question not in ("all", "p1"):
        parser.error(f"No monitoring adapter is registered for {args.question.upper()}.")

    year, month = month_bounds(args.current_start, args.current_end_exclusive)
    output = args.output or CONFIG.reports_dir / "monitoring" / f"{year}_{month:02d}"
    current = args.current_trips or CONFIG.raw_data_dir / f"datos_abiertos_{year}_{month:02d}.csv"
    summary = run(
        args.question,
        args.current_start,
        args.current_end_exclusive,
        output,
        args.historical_trips,
        current,
        args.download_current,
        args.prepare_history,
    )
    for name, result in summary["questions"].items():
        print(f"{name.upper()}: {result['status']}")
    print(f"Monitoring reports saved to {output}")


if __name__ == "__main__":
    main()
