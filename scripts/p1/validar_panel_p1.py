"""Contrato del panel diario P1 y dos checks explícitos de DeepChecks.

No importa mibici, no entrena, no modifica entradas y no contacta MLflow.
Valida el panel previo a las features; no mide drift ni calidad del modelo.
Corre en el entorno de DeepChecks (Python 3.11, requirements-deepchecks.txt).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["DISABLE_DEEPCHECKS_ANONYMOUS_TELEMETRY"] = "1"
os.environ["DISABLE_LATEST_VERSION_CHECK"] = "1"
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

SPEC = {
    "p1": {
        "columns": ["estacion", "fecha", "salidas", "primer_dia"],
        "time": "fecha",
        "step": "1D",
        "numbers": ["estacion", "salidas"],
        "counts": ["salidas"],
    }
}


def contract(df: pd.DataFrame, pipeline: str) -> dict:
    """Valida todas las filas; no corrige, imputa ni descarta datos silenciosamente."""
    if pipeline not in SPEC:
        raise ValueError("pipeline debe ser p1")
    s = SPEC[pipeline]
    failures = []
    missing = sorted(set(s["columns"]) - set(df.columns))
    result = {"pipeline": pipeline, "rows": len(df), "failures": failures}
    if missing:
        failures.append({"rule": "required_columns", "missing": missing})
        return result
    if df.empty:
        failures.append({"rule": "nonempty"})
        return result
    data = df[s["columns"]]
    nulls = {str(c): int(n) for c, n in data.isna().sum().items() if n}
    if nulls:
        failures.append({"rule": "no_nulls_in_raw_panel", "counts": nulls})
    badtypes = [c for c in s["numbers"] if not is_numeric_dtype(data[c])]
    datecols = [s["time"], "primer_dia"]
    badtypes += [c for c in datecols if not is_datetime64_any_dtype(data[c])]
    if badtypes:
        failures.append({"rule": "column_types", "columns": badtypes})
    if nulls or badtypes:
        return result
    if any(data[c].dt.tz is not None for c in datecols):
        failures.append({"rule": "local_naive_timestamps"})
        return result
    nums = data[s["numbers"]].to_numpy(dtype=float)
    if not np.isfinite(nums).all():
        failures.append({"rule": "finite_numbers"})
        return result
    if not np.equal(data["estacion"], np.floor(data["estacion"])).all():
        failures.append({"rule": "integer_station_id"})
    for c in s["counts"]:
        if not ((data[c] >= 0) & (data[c] == np.floor(data[c]))).all():
            failures.append({"rule": "nonnegative_integer_count", "column": c})
    key = ["estacion", s["time"]]
    duplicated = int(data.duplicated(key).sum())
    if duplicated:
        failures.append({"rule": "unique_station_timestamp", "extra_rows": duplicated})
    ordered = data.sort_values(key)
    groups = ordered.groupby("estacion", sort=False)
    diffs = groups[s["time"]].diff().dropna()
    if not diffs.eq(pd.Timedelta(s["step"])).all():
        failures.append({"rule": "consecutive_timestamps_per_station"})
    first = groups[s["time"]].transform("min")
    if not ordered["primer_dia"].eq(first).all():
        failures.append({"rule": "primer_dia_matches_first_row"})
    if not data["fecha"].eq(data["fecha"].dt.normalize()).all():
        failures.append({"rule": "daily_midnight_timestamps"})
    result.update(
        {
            "stations": int(data["estacion"].nunique()),
            "min_timestamp": str(data[s["time"]].min()),
            "max_timestamp": str(data[s["time"]].max()),
            "duplicate_keys": duplicated,
        }
    )
    return result


def deepchecks_integrity(df: pd.DataFrame, pipeline: str, output: Path) -> dict:
    """Full-data checks: zero nulls and zero duplicate station/time keys.

    Repeated counts alone are legitimate and are NOT classified as duplicates.
    No default correlation/constant-feature gate is applied to these raw panels.
    """
    os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib")
    os.environ["XDG_CACHE_HOME"] = str(output / ".cache")
    from deepchecks.tabular import Dataset, Suite
    from deepchecks.tabular.checks import DataDuplicates, PercentOfNulls

    s = SPEC[pipeline]
    data = df[s["columns"]].copy()
    checks = Suite(
        f"Integridad panel {pipeline.upper()}",
        PercentOfNulls(n_samples=len(data)).add_condition_percent_of_nulls_not_greater_than(0),
        DataDuplicates(
            columns=["estacion", s["time"]], n_samples=len(data)
        ).add_condition_ratio_less_or_equal(max_ratio=0),
    )
    suite_result = checks.run(Dataset(data, cat_features=[]))
    html = output / f"{pipeline}_integridad_panel.html"
    suite_result.save_as_html(str(html), as_widget=False)
    ran = [r for r in suite_result.results if hasattr(r, "conditions_results")]
    conditions = [
        {"check": r.get_header(), "condition": c.name, "status": c.category.name}
        for r in ran
        for c in r.conditions_results
    ]
    passed = bool(suite_result.passed(fail_if_warning=True, fail_if_check_not_run=True))
    passed = passed and len(ran) == 2 and (len(conditions) == 2)
    return {
        "passed": passed,
        "checks_run": len(ran),
        "conditions": conditions,
        "html": html.name,
        "rows_checked": len(data),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", default="p1", choices=["p1"])
    parser.add_argument("--input", required=True, type=Path, help="Panel Parquet ya generado")
    parser.add_argument("--output", required=True, type=Path, help="Carpeta NUEVA, fuera del repo")
    parser.add_argument(
        "--repo", required=True, type=Path, help="Raíz del repositorio (sólo se lee)"
    )
    args = parser.parse_args(argv)
    try:
        source = args.input.resolve(strict=True)
        repo = args.repo.resolve(strict=True)
        output = args.output.resolve()
        if not source.is_file() or source.suffix.lower() != ".parquet":
            raise ValueError("La entrada debe ser un archivo Parquet existente")
        if not repo.is_dir():
            raise ValueError("--repo debe ser un directorio existente")
        if output.is_relative_to(repo) or output == source.parent:
            raise ValueError("La salida debe estar fuera del repo y separada de la entrada")
        if output.exists():
            raise ValueError("--output ya existe; use una carpeta nueva para no sobreescribir")
        import pyarrow.parquet as pq

        columns = SPEC[args.pipeline]["columns"]
        available = pq.read_schema(source).names
        missing = sorted(set(columns) - set(available))
        if missing:
            raise ValueError(f"Faltan columnas: {missing}")
        data = pd.read_parquet(source, columns=columns)
        provenance = {
            "input_name": source.name,
            "sha256": sha256(source),
            "input_bytes": source.stat().st_size,
        }
        report = {
            "scope": "panel previo a features; sin drift/modelo",
            "provenance": provenance,
            "contract": contract(data, args.pipeline),
        }
        output.mkdir(parents=True, exist_ok=False)
        if report["contract"]["failures"]:
            report["deepchecks"] = {"passed": False, "not_run": "Contrato de panel no satisfecho"}
        else:
            report["deepchecks"] = deepchecks_integrity(data, args.pipeline, output)
        report["passed"] = not report["contract"]["failures"] and report["deepchecks"]["passed"]
        (output / "resultado.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["passed"] else 1
    except Exception as exc:  # noqa: BLE001 - la CLI informa cualquier error con exit 2
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
