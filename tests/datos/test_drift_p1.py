"""Pruebas de exportar_features_p1.py y medir_drift_p1.py con datos sintéticos.

Las fuentes originales del paquete no se modifican.
"""

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
import shutil
import subprocess

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/p1"))
from exportar_features_p1 import BASE_REVISION, export, load_functions
from medir_drift_p1 import load_exports, measure
from validar_panel_p1 import sha256

# Tupla con nombre: ruff (objetivo 3.14) quitaría los paréntesis de un except con dos tipos,
# sintaxis que Python 3.11 no acepta.
ERRORES_GIT = (OSError, subprocess.CalledProcessError)


def git_toplevel(path):
    """Raíz del checkout git que contiene path, o None sin git o fuera de un checkout."""
    try:
        salida = subprocess.run(
            ["git", "-c", f"safe.directory={path.resolve().as_posix()}", "-C", str(path)]
            + ["rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except ERRORES_GIT:
        return None
    return Path(salida).resolve()


@pytest.fixture(scope="module")
def ruta_repo():
    path = Path(os.environ.get("MIBICI_REPO", str(Path(__file__).resolve().parents[2])))
    if not path.is_dir():
        raise pytest.UsageError("Defina MIBICI_REPO con un checkout de sólo lectura")
    return path


@pytest.fixture(scope="module")
def repo(ruta_repo):
    """Sólo para las pruebas que exportan: el exportador registra el commit con git."""
    if git_toplevel(ruta_repo) != ruta_repo.resolve():
        pytest.skip("El exportador requiere git y una raíz de checkout con su propio .git")
    return ruta_repo


def panel(pipeline):
    rng = np.random.default_rng(29)
    pieces = []
    for station, start in [(1, "2026-01-01"), (2, "2026-01-01"), (3, "2026-05-01")]:
        dates = pd.date_range(start, "2026-06-30", freq="D")
        df = pd.DataFrame(
            {
                "estacion": station,
                "fecha": dates,
                "salidas": rng.poisson(10 + station * 5, len(dates)),
                "primer_dia": dates[0],
            }
        )
        pieces.append(df)
    return pd.concat(pieces, ignore_index=True)


@pytest.fixture(scope="module")
def exports(tmp_path_factory, repo):
    root = tmp_path_factory.mktemp("exports_synthetic")
    result = {}
    for pipeline in ["p1"]:
        source = root / f"{pipeline}.parquet"
        panel(pipeline).to_parquet(source, index=False)
        before = sha256(source)
        output = root / f"{pipeline}_exports"
        manifest = export(repo, source, output, pipeline, mode="isolated-source")
        assert sha256(source) == before
        result[pipeline] = (output, manifest)
    return result


@pytest.mark.parametrize("pipeline,count", [("p1", 11)])
def test_export_uses_original_features_and_no_labels(exports, pipeline, count):
    folder, m = exports[pipeline]
    _, tables = load_exports(folder)
    assert m["reviewed_base"] == BASE_REVISION and m["mode"] == "isolated-source"
    assert len(m["features"]) == count
    assert not m["tracking_or_training_called"] and (not m["label_exported"])
    for name in ("train", "test"):
        assert set(tables[name].columns) == set(m["key"] + m["features"])
        # El archivo completo, no sólo las columnas que lee load_exports.
        assert pq.read_schema(folder / f"{name}.parquet").names == m["key"] + m["features"]
    assert m["embargo_days"] == 2
    assert m["mlflow_modules_introduced"] == []
    assert m["station_populations"] == {"common": 2, "test_only": 1, "train_only": 0}


def test_p1_preserves_embargo(exports):
    m = exports["p1"][1]
    assert m["tables"]["train"]["max_time"] == "2026-04-29 00:00:00"
    assert m["tables"]["test"]["min_time"] == "2026-05-01 00:00:00"
    assert all(n == 0 for part in m["tables"].values() for n in part["nulls"].values())


@pytest.mark.parametrize("pipeline", ["p1"])
def test_descriptive_drift_has_scores_not_quality_pass(exports, repo, tmp_path, pipeline):
    r = measure(exports[pipeline][0], tmp_path / pipeline, repo, sample_size=500)
    assert r["computation_complete"]
    assert r["quality_pass"] is None and r["thresholds"] is None
    for p in r["populations"].values():
        assert p["status"] == "COMPLETE_DESCRIPTIVE"
        assert len(p["scores"]) == len(exports[pipeline][1]["features"])
        assert {s["method"] for s in p["scores"].values()} == {"KS", "cramers_v"}
        assert (tmp_path / pipeline / p["html"]).stat().st_size > 10000


def copy_exports(exports, tmp_path):
    dest = tmp_path / "copy"
    shutil.copytree(exports["p1"][0], dest)
    return dest


def save_changed(folder, name, df):
    p = folder / f"{name}.parquet"
    df.to_parquet(p, index=False)
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["tables"][name]["sha256"] = sha256(p)
    manifest["tables"][name]["rows"] = len(df)
    (folder / "manifest.json").write_text(json.dumps(manifest))


def test_detects_tampered_bytes(exports, tmp_path):
    folder = copy_exports(exports, tmp_path)
    with (folder / "train.parquet").open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="Hash distinto"):
        load_exports(folder)


def test_detects_duplicate_keys(exports, tmp_path):
    folder = copy_exports(exports, tmp_path)
    df = pd.read_parquet(folder / "train.parquet")
    save_changed(folder, "train", pd.concat([df, df.iloc[[0]]], ignore_index=True))
    with pytest.raises(ValueError, match="claves"):
        load_exports(folder)


def test_detects_cross_split_keys(exports, tmp_path):
    folder = copy_exports(exports, tmp_path)
    train, test = [pd.read_parquet(folder / f"{x}.parquet") for x in ["train", "test"]]
    save_changed(folder, "test", pd.concat([test, train.iloc[[0]]], ignore_index=True))
    with pytest.raises(ValueError, match="compartidas"):
        load_exports(folder)


def test_rejects_infinity_not_null(exports, tmp_path):
    folder = copy_exports(exports, tmp_path)
    df = pd.read_parquet(folder / "train.parquet")
    df.loc[0, "lag_2"] = np.inf
    save_changed(folder, "train", df)
    with pytest.raises(ValueError, match="Infinitos"):
        load_exports(folder)


def test_all_null_feature_cannot_be_complete(exports, repo, tmp_path):
    folder = copy_exports(exports, tmp_path)
    for name in ["train", "test"]:
        df = pd.read_parquet(folder / f"{name}.parquet")
        df["lag_2"] = np.nan
        save_changed(folder, name, df)
    result = measure(folder, tmp_path / "report", repo, sample_size=500, populations="all")
    assert not result["computation_complete"]
    assert "lag_2" in result["populations"]["all"]["missing_scores"]


def test_no_common_stations_is_explicit_not_run(exports, repo, tmp_path):
    folder = copy_exports(exports, tmp_path)
    df = pd.read_parquet(folder / "test.parquet")
    df["estacion"] += 100
    save_changed(folder, "test", df)
    result = measure(folder, tmp_path / "report", repo, sample_size=500, populations="common")
    assert not result["computation_complete"]
    assert result["populations"]["common"]["status"] == "NOT_RUN_INSUFFICIENT_ROWS"


def test_native_rejects_wrong_python_without_installing(ruta_repo):
    if sys.version_info[:2] == (3, 14):
        return
    with pytest.raises(RuntimeError, match="Python 3.14"):
        load_functions(ruta_repo, "p1", "native")


def test_refuses_output_inside_repository(exports, repo):
    with pytest.raises(ValueError, match="externa"):
        measure(exports["p1"][0], repo / "forbidden_report", repo)


def test_detects_source_hash_mismatch_before_export(exports, repo, tmp_path, monkeypatch):
    import exportar_features_p1 as exporter

    monkeypatch.setattr(exporter, "normalized_sha256", lambda _: "0" * 64)
    panel_file = tmp_path / "panel.parquet"
    panel("p1").to_parquet(panel_file, index=False)
    with pytest.raises(ValueError, match="Fuente distinta"):
        export(repo, panel_file, tmp_path / "rejected", "p1", mode="isolated-source")
    assert not (tmp_path / "rejected").exists()


def test_config_fingerprint_covers_only_p1config(ruta_repo, tmp_path):
    import exportar_features_p1 as exporter

    copia = tmp_path / "repo"
    (copia / "mibici").mkdir(parents=True)
    original = (ruta_repo / "mibici" / "config.py").read_text(encoding="utf-8")
    destino = copia / "mibici" / "config.py"
    clave = next(k for k in exporter.SOURCE_SHA256 if k.startswith("mibici/config.py::"))
    destino.write_text(original, encoding="utf-8")
    assert exporter.source_fingerprint(copia, clave) == exporter.SOURCE_SHA256[clave]
    # Un cambio fuera de P1Config (por ejemplo, de P2 o P3) no altera la huella.
    destino.write_text(original + "\nOTRA_PREGUNTA = 1\n", encoding="utf-8")
    assert exporter.source_fingerprint(copia, clave) == exporter.SOURCE_SHA256[clave]
    # Tampoco un salto de página u otro separador Unicode en un comentario ajeno.
    destino.write_text("# \x0c \u2028 \x85\n" + original, encoding="utf-8")
    assert exporter.source_fingerprint(copia, clave) == exporter.SOURCE_SHA256[clave]
    # Una segunda asignación de CONFIG_P1 se rechaza en vez de pasar inadvertida.
    destino.write_text(original + "\nCONFIG_P1 = P1Config(hora_emision=20)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactamente una vez"):
        exporter.source_fingerprint(copia, clave)
    # Un cambio dentro de P1Config sí la altera.
    assert "hora_emision: int = 23" in original
    destino.write_text(
        original.replace("hora_emision: int = 23", "hora_emision: int = 22"), encoding="utf-8"
    )
    assert exporter.source_fingerprint(copia, clave) != exporter.SOURCE_SHA256[clave]
    # También la altera un cambio en la instancia CONFIG_P1.
    assert "CONFIG_P1 = P1Config()" in original
    destino.write_text(
        original.replace(
            "CONFIG_P1 = P1Config()", 'CONFIG_P1 = P1Config(test_inicio="2026-06-01")'
        ),
        encoding="utf-8",
    )
    assert exporter.source_fingerprint(copia, clave) != exporter.SOURCE_SHA256[clave]


def copy_sources(ruta_repo, copia):
    for rel in ("mibici/config.py", "mibici/features.py", "mibici/modeling/train_p1.py"):
        (copia / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ruta_repo / rel, copia / rel)


def test_export_requires_its_own_git_checkout(ruta_repo, tmp_path):
    # Una copia sin .git (dentro o fuera de otro repositorio) no puede tomar un commit ajeno.
    copia = tmp_path / "sin_git"
    copy_sources(ruta_repo, copia)
    panel_file = tmp_path / "panel.parquet"
    panel("p1").to_parquet(panel_file, index=False)
    with pytest.raises(ValueError, match="raíz de un checkout git"):
        export(copia, panel_file, tmp_path / "out", "p1", mode="isolated-source")
    assert not (tmp_path / "out").exists()


def test_rejects_config_p1_instance_with_arguments(ruta_repo, tmp_path):
    copia = tmp_path / "repo"
    copy_sources(ruta_repo, copia)
    destino = copia / "mibici" / "config.py"
    texto = destino.read_text(encoding="utf-8")
    destino.write_text(
        texto.replace("CONFIG_P1 = P1Config()", 'CONFIG_P1 = P1Config(test_inicio="2026-06-01")'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="CONFIG_P1"):
        load_functions(copia, "p1", "isolated-source")


def test_rejects_decorated_original_functions(tmp_path):
    from exportar_features_p1 import original_functions

    fuente = tmp_path / "fuente.py"
    fuente.write_text(
        "def deco(*a, **k):\n    return lambda f: f\n\n\n"
        "@deco(\n    estricto=True\n)\n# comentario\n\ndef objetivo():\n    return 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="decoradas"):
        original_functions(fuente, ["objetivo"], {})


def test_detects_wrong_production_hashes_in_manifest(exports, tmp_path):
    folder = copy_exports(exports, tmp_path)
    m = json.loads((folder / "manifest.json").read_text())
    m["source_files_sha256"]["mibici/features.py"] = "0" * 64
    (folder / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="Commit/pipeline"):
        load_exports(folder)


def test_preexisting_mlflow_import_does_not_trigger_false_failure(repo, tmp_path, monkeypatch):
    from types import ModuleType

    monkeypatch.setitem(sys.modules, "mlflow", ModuleType("mlflow"))
    source = tmp_path / "panel.parquet"
    panel("p1").to_parquet(source, index=False)
    result = export(repo, source, tmp_path / "exports", "p1", mode="isolated-source")
    assert result["mlflow_modules_introduced"] == []
    assert result["tracking_or_training_called"] is False
