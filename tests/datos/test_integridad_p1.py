"""Pruebas de validar_panel_p1.py con paneles fabricados; no son resultados P1 reales."""

import os
import sys

import pytest

if sys.version_info[:2] != (3, 11):
    pytest.skip(
        "Validaciones P1 requieren el entorno DeepChecks Python 3.11; "
        "este SKIP no acredita calidad de datos ni ejecución de checks",
        allow_module_level=True,
    )
os.environ["DISABLE_DEEPCHECKS_ANONYMOUS_TELEMETRY"] = "1"
os.environ["DISABLE_LATEST_VERSION_CHECK"] = "1"
pytest.importorskip(
    "deepchecks",
    reason="Falta DeepChecks en este intérprete; ejecutar con el entorno 3.11 separado",
)

import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/p1"))
from validar_panel_p1 import contract, deepchecks_integrity, main


def sample(pipeline):
    t = pd.date_range("2026-01-01", periods=48, freq="D")
    rng = np.random.default_rng(29)
    rows = []
    for station in (1, 2):
        df = pd.DataFrame(
            {
                "estacion": station,
                "fecha": t,
                "salidas": rng.integers(0, 30, len(t)),
                "primer_dia": t[0],
            }
        )
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


@pytest.mark.parametrize("pipeline", ["p1"])
def test_valid_contract(pipeline):
    assert contract(sample(pipeline), pipeline)["failures"] == []


@pytest.mark.parametrize("pipeline", ["p1"])
@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("empty", "nonempty"),
        ("missing", "required_columns"),
        ("null", "no_nulls_in_raw_panel"),
        ("infinity", "finite_numbers"),
        ("negative", "nonnegative_integer_count"),
        ("fractional", "nonnegative_integer_count"),
        ("duplicate", "unique_station_timestamp"),
        ("gap", "consecutive_timestamps_per_station"),
        ("first_day", "primer_dia_matches_first_row"),
        ("station", "integer_station_id"),
        ("text_dates", "column_types"),
        ("tz_dates", "local_naive_timestamps"),
    ],
)
def test_common_negative_controls(pipeline, mutation, expected):
    df = sample(pipeline)
    time = "fecha"
    if mutation == "empty":
        df = df.iloc[:0]
    elif mutation == "missing":
        df = df.drop(columns="salidas")
    elif mutation == "null":
        df.loc[0, "salidas"] = np.nan
    elif mutation == "infinity":
        df["salidas"] = df["salidas"].astype(float)
        df.loc[0, "salidas"] = np.inf
    elif mutation == "negative":
        df.loc[0, "salidas"] = -1
    elif mutation == "fractional":
        df["salidas"] = df["salidas"].astype(float)
        df.loc[0, "salidas"] = 0.5
    elif mutation == "duplicate":
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    elif mutation == "gap":
        df = df.drop(index=10)
    elif mutation == "first_day":
        df.loc[0, "primer_dia"] += pd.Timedelta(days=1)
    elif mutation == "station":
        df["estacion"] = df["estacion"].astype(float)
        df.loc[0, "estacion"] = 1.5
    elif mutation == "text_dates":
        df[time] = df[time].astype(str)
    elif mutation == "tz_dates":
        df[time] = df[time].dt.tz_localize("UTC")
    assert expected in {f["rule"] for f in contract(df, pipeline)["failures"]}


def test_p1_requires_midnight():
    df = sample("p1")
    df["fecha"] += pd.Timedelta(hours=1)
    assert "daily_midnight_timestamps" in {f["rule"] for f in contract(df, "p1")["failures"]}


def test_repeated_counts_are_allowed():
    df = sample("p1").assign(salidas=0)
    assert contract(df, "p1")["failures"] == []


@pytest.mark.parametrize("pipeline", ["p1"])
def test_deepchecks_original_library_smoke(pipeline, tmp_path):
    result = deepchecks_integrity(sample(pipeline), pipeline, tmp_path)
    assert result["passed"] and result["checks_run"] == 2
    assert len(result["conditions"]) == 2
    assert all(c["status"] == "PASS" for c in result["conditions"])
    assert (tmp_path / result["html"]).stat().st_size > 10000


@pytest.mark.parametrize("mutation", ["duplicate", "null"])
def test_deepchecks_negative_directly(mutation, tmp_path):
    df = sample("p1")
    if mutation == "duplicate":
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    else:
        df.loc[0, "salidas"] = np.nan
    result = deepchecks_integrity(df, "p1", tmp_path)
    assert not result["passed"]
    assert any(c["status"] != "PASS" for c in result["conditions"])


@pytest.mark.parametrize("pipeline", ["p1"])
def test_cli_parquet_success(pipeline, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / f"{pipeline}.parquet"
    sample(pipeline).to_parquet(source, index=False)
    before = source.read_bytes()
    output = tmp_path / "reports"
    assert (
        main(
            [
                "--pipeline",
                pipeline,
                "--input",
                str(source),
                "--output",
                str(output),
                "--repo",
                str(repo),
            ]
        )
        == 0
    )
    report = json.loads((output / "resultado.json").read_text())
    assert report["passed"]
    assert source.read_bytes() == before
    assert set(repo.iterdir()) == {source}


def test_cli_fail_not_skip(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    output = tmp_path / "reports"
    assert (
        main(
            [
                "--pipeline",
                "p1",
                "--input",
                str(repo / "absent.parquet"),
                "--output",
                str(output),
                "--repo",
                str(repo),
            ]
        )
        == 2
    )
    assert not output.exists()


def test_cli_refuses_output_inside_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "p1.parquet"
    sample("p1").to_parquet(source, index=False)
    output = repo / "reports"
    assert (
        main(
            [
                "--pipeline",
                "p1",
                "--input",
                str(source),
                "--output",
                str(output),
                "--repo",
                str(repo),
            ]
        )
        == 2
    )
    assert not output.exists()


def test_cli_bad_data_cannot_pass(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "p1.parquet"
    df = sample("p1")
    df.loc[0, "salidas"] = -1
    df.to_parquet(source, index=False)
    output = tmp_path / "reports"
    assert (
        main(
            [
                "--pipeline",
                "p1",
                "--input",
                str(source),
                "--output",
                str(output),
                "--repo",
                str(repo),
            ]
        )
        == 1
    )
    report = json.loads((output / "resultado.json").read_text())
    assert not report["passed"] and "not_run" in report["deepchecks"]
