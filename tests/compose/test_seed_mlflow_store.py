import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "compose"))
import seed_mlflow_store as seed


def _make_store(root: Path, p2_location: str = "mlflow-artifacts:/p2/artifacts"):
    model_rows = [
        (
            "mibici-equipo29-p1-demanda",
            "champion",
            "run-p1",
            "mlflow-artifacts:/p1/artifacts",
        ),
        ("mibici-equipo29-p2-riesgo", "champion", "run-p2", p2_location),
        (
            "mibici-equipo29-p3-grupos",
            "champion",
            "run-p3",
            "mlflow-artifacts:/p3/artifacts",
        ),
    ]
    artifact_root = root / "mlartifacts"
    artifact_root.mkdir(parents=True)
    for _, _, _, location in model_rows:
        if location.startswith("mlflow-artifacts:/"):
            relative = location.removeprefix("mlflow-artifacts:/").lstrip("/")
            directory = artifact_root / relative
            directory.mkdir(parents=True)
            (directory / "MLmodel").write_text("test model", encoding="utf-8")
            if "p1/" in relative:
                (directory / "model.skops").write_bytes(b"p1 test model")

    database = root / "mlflow.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE registered_model_aliases (
                name TEXT NOT NULL,
                alias TEXT NOT NULL,
                version TEXT NOT NULL
            );
            CREATE TABLE model_versions (
                name TEXT NOT NULL,
                version INTEGER NOT NULL,
                run_id TEXT NOT NULL,
                storage_location TEXT NOT NULL,
                status TEXT NOT NULL
            );
            """
        )
        for name, alias, run_id, location in model_rows:
            connection.execute(
                "INSERT INTO registered_model_aliases VALUES (?, ?, ?)",
                (name, alias, "1"),
            )
            connection.execute(
                "INSERT INTO model_versions VALUES (?, ?, ?, ?, ?)",
                (name, 1, run_id, location, "READY"),
            )

    p1_model = {
        "nombre": model_rows[0][0],
        "alias": model_rows[0][1],
        "version": "1",
        "run_id": model_rows[0][2],
        "storage_location": model_rows[0][3],
        "model_skops_sha256": hashlib.sha256(b"p1 test model").hexdigest(),
    }
    manifest = root / "p1_manifest.json"
    manifest.write_text(json.dumps({"modelo": p1_model}), encoding="utf-8")
    return database, artifact_root, manifest


def _configure(monkeypatch, source: Path, destination: Path, manifest: Path):
    monkeypatch.setattr(seed, "MODEL_STORE_DIR", source)
    monkeypatch.setattr(seed, "BACKEND_DIR", destination / "backend")
    monkeypatch.setattr(seed, "ARTIFACTS_DIR", destination / "artifacts")
    monkeypatch.setattr(seed, "P1_MANIFEST", manifest)


def test_seed_copies_and_reuses_a_valid_three_model_store(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    _, _, manifest = _make_store(source)
    destination = tmp_path / "destination"
    _configure(monkeypatch, source, destination, manifest)

    assert seed.main() == 0
    assert (destination / "backend" / "mlflow.db").is_file()
    assert (destination / "artifacts" / "p1" / "artifacts" / "model.skops").is_file()
    assert seed.main() == 0


def test_seed_rejects_artifacts_outside_mlflow_artifact_store(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    _, _, manifest = _make_store(source, p2_location="file:///local/p2")
    destination = tmp_path / "destination"
    _configure(monkeypatch, source, destination, manifest)

    assert seed.main() == 1
    assert not (destination / "backend" / "mlflow.db").exists()


def test_seed_does_not_overwrite_a_partial_destination(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    _, _, manifest = _make_store(source)
    destination = tmp_path / "destination"
    _configure(monkeypatch, source, destination, manifest)
    seed.BACKEND_DIR.mkdir(parents=True)
    (seed.BACKEND_DIR / "mlflow.db").write_bytes(b"partial database")

    assert seed.main() == 1
    assert (seed.BACKEND_DIR / "mlflow.db").read_bytes() == b"partial database"
