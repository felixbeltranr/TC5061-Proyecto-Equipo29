"""Drift descriptivo P1 en DeepChecks 3.11; exit 0 NO significa calidad aprobada.

Lee exportaciones verificadas del auxiliar, no mibici/MLflow. Separa claves,
nulos y poblaciones. No añade umbrales de aprobación o reentrenamiento.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ["DISABLE_DEEPCHECKS_ANONYMOUS_TELEMETRY"] = "1"
os.environ["DISABLE_LATEST_VERSION_CHECK"] = "1"
from exportar_features_p1 import BASE_REVISION, CATEGORICAL, SOURCE_SHA256
import numpy as np
import pandas as pd
from validar_panel_p1 import sha256

EXPECTED_FEATURES = {
    "p1": [
        "dia_semana",
        "festivo",
        "festivo_lag7",
        "lag_2",
        "lag_7",
        "lag_14",
        "lag_21",
        "lag_28",
        "media_7d",
        "media_28d",
        "media_misma_dsem_4",
    ]
}


def load_exports(folder):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("reviewed_base") != BASE_REVISION
        or manifest.get("source_files_sha256") != SOURCE_SHA256
        or manifest.get("pipeline") not in CATEGORICAL
        or (manifest.get("status") != "EXPORTED_NOT_MODEL_VALIDATION")
        or (manifest.get("mode") not in ["native", "isolated-source"])
    ):
        raise ValueError("Commit/pipeline del manifiesto no corresponde a esta revisión")
    pipeline = manifest["pipeline"]
    key = ["estacion", "fecha"]
    features = manifest["features"]
    if (
        features != EXPECTED_FEATURES[pipeline]
        or manifest["key"] != key
        or set(features)
        & {"estacion", "fecha", "ts", "ts_objetivo", "objetivo", "salidas", "clase_actual"}
        or (manifest["categorical"] != CATEGORICAL[pipeline])
    ):
        raise ValueError("Esquema del manifiesto inválido: llaves/labels no son features")
    tables = {}
    for name in ["train", "test"]:
        meta = manifest["tables"][name]
        if meta["file"] != f"{name}.parquet":
            raise ValueError("Nombre de exportación no admitido")
        path = folder / meta["file"]
        if sha256(path) != meta["sha256"]:
            raise ValueError(f"Hash distinto para {name}")
        df = pd.read_parquet(path, columns=key + features)
        if (
            len(df) != meta["rows"]
            or df.empty
            or df[key].isna().any().any()
            or df.duplicated(key).any()
        ):
            raise ValueError(f"Conteo o claves inválidas en {name}")
        if not pd.api.types.is_datetime64_any_dtype(df[key[1]]):
            raise ValueError("La llave temporal debe ser datetime")
        for feature in features:
            if not pd.api.types.is_numeric_dtype(df[feature]):
                raise ValueError(f"Feature no numérica en el contrato actual: {feature}")
            non_null = df[feature].dropna().to_numpy(dtype=float)
            if not np.isfinite(non_null).all():
                raise ValueError(f"Infinitos en {name}.{feature}; no son nulos legítimos")
        tables[name] = df
    train, test = (tables["train"], tables["test"])
    if not train[key].merge(test[key], on=key).empty:
        raise ValueError("Claves temporales compartidas entre train/test")
    if train[key[1]].max() >= test[key[1]].min():
        raise ValueError("Separación temporal inválida")
    return (manifest, tables)


def measure(folder, output, repo, sample_size=100000, seed=29, populations="both"):
    folder, output, repo = (
        folder.resolve(strict=True),
        output.resolve(),
        repo.resolve(strict=True),
    )
    if output.exists() or output.is_relative_to(repo) or output.is_relative_to(folder):
        raise ValueError("Salida debe ser nueva, externa al repo y a las exportaciones")
    if sample_size < 10:
        raise ValueError("sample-size debe ser al menos 10")
    manifest, tables = load_exports(folder)
    output.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib")
    os.environ["XDG_CACHE_HOME"] = str(output / ".cache")
    from deepchecks.tabular import Dataset
    from deepchecks.tabular.checks import FeatureDrift

    features, cats = (manifest["features"], manifest["categorical"])
    common = set(tables["train"].estacion) & set(tables["test"].estacion)
    report = {
        "scope": "DESCRIPTIVE_DRIFT_NOT_MODEL_QUALITY",
        "pipeline": manifest["pipeline"],
        "export_mode": manifest["mode"],
        "source_head": manifest["source_head"],
        "input_sha256": manifest["input_sha256"],
        "export_manifest_sha256": sha256(folder / "manifest.json"),
        "numeric_method": "KS",
        "categorical_method": "cramers_v",
        "seed": seed,
        "maximum_sample_per_period": sample_size,
        "thresholds": None,
        "quality_pass": None,
        "populations": {},
    }
    names = ["all", "common"] if populations == "both" else [populations]
    complete = True
    for population in names:
        frames = {
            k: d if population == "all" else d[d.estacion.isin(common)] for k, d in tables.items()
        }
        item = {
            "rows": {k: len(d) for k, d in frames.items()},
            "stations": {k: int(d.estacion.nunique()) for k, d in frames.items()},
            "samples_used_per_period": {k: min(sample_size, len(d)) for k, d in frames.items()},
            "nulls_full_population": {
                k: {
                    c: {
                        "count": int(d[c].isna().sum()),
                        "ratio": float(d[c].isna().mean()) if len(d) else None,
                    }
                    for c in features
                }
                for k, d in frames.items()
            },
            "scores": {},
            "missing_scores": [],
        }
        report["populations"][population] = item
        if any(len(d) < 10 for d in frames.values()):
            item["status"] = "NOT_RUN_INSUFFICIENT_ROWS"
            complete = False
            continue
        result = FeatureDrift(
            numerical_drift_method="KS",
            categorical_drift_method="cramers_v",
            n_samples=sample_size,
            random_state=seed,
            ignore_na=False,
        ).run(
            Dataset(frames["train"][features], cat_features=cats, dataset_name="train"),
            Dataset(frames["test"][features], cat_features=cats, dataset_name="test"),
        )
        if not hasattr(result, "value") or not isinstance(result.value, dict):
            item["status"] = "CHECK_DID_NOT_RETURN_SCORES"
            complete = False
            continue
        for feature in features:
            entry = result.value.get(feature, {})
            score = entry.get("Drift score")
            if score is None or not math.isfinite(float(score)) or (not 0 <= float(score) <= 1):
                item["missing_scores"].append(feature)
            else:
                item["scores"][feature] = {
                    "score": float(score),
                    "method": "cramers_v" if feature in cats else "KS",
                }
        item["status"] = (
            "COMPLETE_DESCRIPTIVE" if not item["missing_scores"] else "INCOMPLETE_SCORES"
        )
        complete &= not item["missing_scores"]
        html = output / f"{manifest['pipeline']}_drift_{population}.html"
        result.save_as_html(str(html), as_widget=False)
        item["html"] = html.name
    train_stations, test_stations = (set(tables["train"].estacion), set(tables["test"].estacion))
    report["station_populations"] = {
        "common": len(common),
        "test_only": len(test_stations - train_stations),
        "train_only": len(train_stations - test_stations),
    }
    report["computation_complete"] = bool(complete)
    report["interpretation"] = (
        "Scores describen diferencias de distribución; no hay umbral de aprobación. KS numérico ignora nulos; Cramér categórico los conserva como categoría. Tasas completas separadas. Filas temporales dependientes y mezcla de estaciones pueden explicar cambios; no causalidad."
    )
    (output / "drift_resumen.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exports", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--sample-size", type=int, default=100000)
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--populations", choices=["all", "common", "both"], default="both")
    a = p.parse_args(argv)
    try:
        result = measure(a.exports, a.output, a.repo, a.sample_size, a.seed, a.populations)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0 if result["computation_complete"] else 1
    except Exception as exc:  # noqa: BLE001 - la CLI informa cualquier error con exit 2
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
