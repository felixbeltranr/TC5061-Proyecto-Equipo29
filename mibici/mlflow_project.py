from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any

from mibici.config import CONFIG_P1, CONFIG_P2, CONFIG_P3, PROJ_ROOT


def _project_tracking_uri() -> str:
    """Resolve deliberate project configuration independently of the caller's cwd.

    MLflow injects MLFLOW_TRACKING_URI into every project subprocess, so that variable
    is evidence about the outer run, not an explicit project configuration choice.
    """
    configured = os.environ.get("MIBICI_TRACKING_URI", "").strip()
    if not configured:
        return f"sqlite:///{(PROJ_ROOT / 'mlflow.db').resolve().as_posix()}"
    prefix = "sqlite:///"
    if configured.startswith(prefix):
        database, separator, query = configured[len(prefix) :].partition("?")
        if not database or database == ":memory:":
            raise ValueError("MIBICI_TRACKING_URI requiere un archivo SQLite persistente")
        path = Path(database)
        if not path.is_absolute():
            path = PROJ_ROOT / path
        return f"{prefix}{path.resolve().as_posix()}" + (separator + query if separator else "")
    if configured.startswith("sqlite:"):
        raise ValueError("MIBICI_TRACKING_URI SQLite debe usar sqlite:///ruta")
    return configured


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

    if model not in ("auto", *train_p1.CANDIDATOS):
        raise ValueError(f"P1 model must be auto or one of: {', '.join(train_p1.CANDIDATOS)}")
    features.main()
    encoded_parameters = json.dumps(hyperparameters, separators=(",", ":"))
    args = ["--model", model, "--hyperparameters", encoded_parameters]
    # run_train validates the outer project run before propagating the canonical backend.
    if project_tracking_uri := os.environ.get("MLFLOW_TRACKING_URI"):
        args.extend(["--tracking-uri", project_tracking_uri])
    train_p1.main(args)


def _run_p2(model: str, hyperparameters: dict[str, Any]) -> None:
    from mibici import features_p2
    from mibici.modeling import train_p2

    if model != CONFIG_P2.model:
        raise ValueError(f"P2 currently supports only model={CONFIG_P2.model}")
    features_p2.main()
    encoded_parameters = json.dumps(hyperparameters, separators=(",", ":"))
    args = ["--model", model, "--hyperparameters", encoded_parameters]
    if project_tracking_uri := os.environ.get("MLFLOW_TRACKING_URI"):
        args.extend(["--tracking-uri", project_tracking_uri])
    train_p2.main(args)


def _run_p3(model: str, hyperparameters: dict[str, Any]) -> None:
    from mibici import features_p3
    from mibici.modeling import train_p3
    from mibici.modeling.modelos_p3 import validar_hiperparametros

    hyperparameters = dict(hyperparameters)
    if model == "seleccion":
        # flujo completo: baseline, búsquedas, candidatas, campeón y Model Registry
        if hyperparameters:
            raise ValueError("model=seleccion no recibe hiperparámetros; se toman de CONFIG_P3")
        features_p3.main()
        train_p3.main(
            model="seleccion",
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", CONFIG_P3.tracking_uri),
        )
        return
    # buscar=1 lanza la búsqueda de la rejilla de CONFIG_P3 con runs anidados
    buscar = bool(hyperparameters.pop("buscar", False))
    validar_hiperparametros(model, hyperparameters)
    if buscar and (model == "kmeans_geo" or hyperparameters):
        raise ValueError("buscar=1 solo aplica a kmeans_uso o gmm_uso, sin otros hiperparámetros")
    features_p3.main()
    k = hyperparameters.get("k")
    n_components = hyperparameters.get("n_components")
    train_p3.main(
        k=[k] if k is not None else None,
        model=model,
        n_init=hyperparameters.get("n_init"),
        n_components=[n_components] if n_components is not None else None,
        covariance_type=hyperparameters.get("covariance_type"),
        buscar_=buscar,
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", CONFIG_P3.tracking_uri),
    )


def run_train(
    question: str, model: str = "", hyperparameters: dict[str, Any] | None = None
) -> None:
    """Run feature preparation and configured training for one project question."""
    params = {} if hyperparameters is None else hyperparameters
    model = _normalize_empty_argument(model)
    if not isinstance(params, dict):
        raise TypeError("Hyperparameters must be a mapping")

    if question == "all" and (model or params):
        raise ValueError("question=all no admite model ni hiperparámetros compartidos")

    tracking_uri = _project_tracking_uri()
    project_run_id = os.environ.get("MLFLOW_RUN_ID")
    previous_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if project_run_id is not None and previous_tracking_uri != tracking_uri:
        raise ValueError(
            "El run exterior usa otra URI o una ruta SQLite relativa ambigua. "
            "Antes de mlflow run fija MIBICI_TRACKING_URI con una URI absoluta y "
            "MLFLOW_TRACKING_URI con exactamente el mismo valor."
        )
    os.environ.pop("MLFLOW_RUN_ID", None)
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    try:
        if question == "p1":
            _run_p1(model or CONFIG_P1.model, params)
        elif question == "p2":
            _run_p2(model or CONFIG_P2.model, params)
        elif question == "p3":
            _run_p3(model or CONFIG_P3.model, params)
        elif question == "all":
            _run_p1("auto", {})
            _run_p2(CONFIG_P2.model, {})
            _run_p3("seleccion", {})
        else:
            raise ValueError("question must be one of: p1, p2, p3, all")
    finally:
        if previous_tracking_uri is None:
            os.environ.pop("MLFLOW_TRACKING_URI", None)
        else:
            os.environ["MLFLOW_TRACKING_URI"] = previous_tracking_uri
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
        from mibici.manifiesto_p2 import main as crear_manifiesto

        ingest()
        crear_manifiesto()
        return

    run_train(
        args.question,
        model=_normalize_empty_argument(args.model),
        hyperparameters=_parse_hyperparameters(args.hyperparameters),
    )


if __name__ == "__main__":
    main()
