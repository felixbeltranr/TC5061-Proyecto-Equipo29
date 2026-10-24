from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

from mibici.config import CONFIG_P1, CONFIG_P2, CONFIG_P3


def _normalize_empty_argument(value: str) -> str:
    if value.strip() in {"", '""', "''"}:
        return ""
    return value


def _parse_hyperparameters(value: str) -> dict[str, Any]:
    value = _normalize_empty_argument(value)
    if not value:
        return {}
    if value.lstrip().startswith("{"):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("Hyperparameters must be a JSON object")
        return parsed

    parsed = {}
    for item in value.split(","):
        key, separator, raw_value = item.partition("=")
        if not separator or not key.strip() or not raw_value.strip():
            raise ValueError(
                "Hyperparameters must be comma-separated key=value pairs or a JSON object"
            )
        raw_value = raw_value.strip()
        if re.fullmatch(r"-?\d+", raw_value):
            parsed[key.strip()] = int(raw_value)
        else:
            try:
                parsed[key.strip()] = float(raw_value)
            except ValueError:
                parsed[key.strip()] = raw_value
    return parsed


def _run_p1(model: str, hyperparameters: dict[str, Any]) -> None:
    from mibici import features
    from mibici.modeling import train_p1

    if model not in ("auto", "poisson", "hgb_poisson"):
        raise ValueError("P1 model must be auto, poisson, or hgb_poisson")
    features.main()
    encoded_parameters = json.dumps(hyperparameters, separators=(",", ":"))
    train_p1.main(["--model", model, "--hyperparameters", encoded_parameters])


def _run_p2(model: str, hyperparameters: dict[str, Any]) -> None:
    from mibici import features_p2
    from mibici.modeling import train_p2

    if model != CONFIG_P2.model:
        raise ValueError(f"P2 currently supports only model={CONFIG_P2.model}")
    features_p2.main()
    encoded_parameters = json.dumps(hyperparameters, separators=(",", ":"))
    train_p2.main(["--model", model, "--hyperparameters", encoded_parameters])


def _run_p3(model: str, hyperparameters: dict[str, Any]) -> None:
    from mibici import features_p3
    from mibici.modeling import train_p3

    if model != CONFIG_P3.model:
        raise ValueError(f"P3 currently supports only model={CONFIG_P3.model}")
    unsupported = set(hyperparameters).difference({"k", "n_init"})
    if unsupported:
        raise ValueError(f"Unsupported P3 hyperparameters: {sorted(unsupported)}")

    k_value = hyperparameters.get("k")
    if k_value is not None and (
        isinstance(k_value, bool) or not isinstance(k_value, int) or k_value < 2
    ):
        raise ValueError("P3 hyperparameter k must be an integer greater than or equal to 2")
    n_init = hyperparameters.get("n_init")
    if n_init is not None and (
        isinstance(n_init, bool) or not isinstance(n_init, int) or n_init < 1
    ):
        raise ValueError("P3 hyperparameter n_init must be a positive integer")

    features_p3.main()
    train_p3.main(
        k=[k_value] if k_value is not None else None,
        model=model,
        n_init=n_init,
    )


def run_train(
    question: str, model: str = "", hyperparameters: dict[str, Any] | None = None
) -> None:
    """Run feature preparation and configured training for one project question."""
    params = hyperparameters or {}
    if not isinstance(params, dict):
        raise TypeError("Hyperparameters must be a mapping")

    project_run_id = os.environ.pop("MLFLOW_RUN_ID", None)
    try:
        if question == "p1":
            _run_p1(model or CONFIG_P1.model, params)
        elif question == "p2":
            _run_p2(model or CONFIG_P2.model, params)
        elif question == "p3":
            _run_p3(model or CONFIG_P3.model, params)
        elif question == "all":
            _run_p1(CONFIG_P1.model, params)
            _run_p2(CONFIG_P2.model, params)
            _run_p3(CONFIG_P3.model, params)
        else:
            raise ValueError("question must be one of: p1, p2, p3, all")
    finally:
        if project_run_id is not None:
            os.environ["MLFLOW_RUN_ID"] = project_run_id


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MiBici MLflow Project entry point runner")
    subparsers = parser.add_subparsers(dest="entry_point", required=True)
    subparsers.add_parser("ingest", help="download and combine configured MiBici data")

    train_parser = subparsers.add_parser("train", help="prepare features and train one question")
    train_parser.add_argument("--question", choices=("p1", "p2", "p3", "all"), required=True)
    train_parser.add_argument("--model", default="")
    train_parser.add_argument("--hyperparameters", default="")

    args = parser.parse_args(argv)
    if args.entry_point == "ingest":
        from mibici.dataset import main as ingest

        ingest()
        return

    run_train(
        args.question,
        model=_normalize_empty_argument(args.model),
        hyperparameters=_parse_hyperparameters(args.hyperparameters),
    )


if __name__ == "__main__":
    main()
