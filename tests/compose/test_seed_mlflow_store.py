import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "compose"))
import seed_mlflow_store as seed

P1 = "mibici-equipo29-p1-demanda"
P2 = "mibici-equipo29-p2-riesgo"
P3 = "mibici-equipo29-p3-grupos"
# Los modelos con manifiesto llevan model.skops; su contenido fija el SHA256 esperado.
CON_MANIFIESTO = {P1: b"p1 test model", P3: b"p3 test model"}


def _make_store(root: Path, p2_location: str = "mlflow-artifacts:/p2/artifacts"):
    model_rows = [
        (P1, "champion", "run-p1", "mlflow-artifacts:/p1/artifacts"),
        (P2, "champion", "run-p2", p2_location),
        (P3, "champion", "run-p3", "mlflow-artifacts:/p3/artifacts"),
    ]
    artifact_root = root / "mlartifacts"
    artifact_root.mkdir(parents=True)
    for name, _, _, location in model_rows:
        if location.startswith("mlflow-artifacts:/"):
            relative = location.removeprefix("mlflow-artifacts:/").lstrip("/")
            directory = artifact_root / relative
            directory.mkdir(parents=True)
            (directory / "MLmodel").write_text("test model", encoding="utf-8")
            if name in CON_MANIFIESTO:
                (directory / "model.skops").write_bytes(CON_MANIFIESTO[name])

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

    manifests = {}
    for name, alias, run_id, location in model_rows:
        if name not in CON_MANIFIESTO:
            continue
        modelo = {
            "nombre": name,
            "alias": alias,
            "version": "1",
            "run_id": run_id,
            "storage_location": location,
            "model_skops_sha256": hashlib.sha256(CON_MANIFIESTO[name]).hexdigest(),
        }
        ruta = root / f"{name}_manifest.json"
        ruta.write_text(json.dumps({"modelo": modelo}), encoding="utf-8")
        manifests[name] = ruta
    return database, artifact_root, manifests


def _configure(monkeypatch, source: Path, destination: Path, manifests: dict):
    monkeypatch.setattr(seed, "MODEL_STORE_DIR", source)
    monkeypatch.setattr(seed, "BACKEND_DIR", destination / "backend")
    monkeypatch.setattr(seed, "ARTIFACTS_DIR", destination / "artifacts")
    monkeypatch.setattr(seed, "MANIFESTS", manifests)


def test_seed_copies_and_reuses_a_valid_three_model_store(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    _, _, manifests = _make_store(source)
    destination = tmp_path / "destination"
    _configure(monkeypatch, source, destination, manifests)

    assert seed.main() == 0
    assert (destination / "backend" / "mlflow.db").is_file()
    assert (destination / "artifacts" / "p1" / "artifacts" / "model.skops").is_file()
    assert (destination / "artifacts" / "p3" / "artifacts" / "model.skops").is_file()
    assert seed.main() == 0


def test_seed_rejects_artifacts_outside_mlflow_artifact_store(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    _, _, manifests = _make_store(source, p2_location="file:///local/p2")
    destination = tmp_path / "destination"
    _configure(monkeypatch, source, destination, manifests)

    assert seed.main() == 1
    assert not (destination / "backend" / "mlflow.db").exists()


def test_seed_does_not_overwrite_a_partial_destination(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    _, _, manifests = _make_store(source)
    destination = tmp_path / "destination"
    _configure(monkeypatch, source, destination, manifests)
    seed.BACKEND_DIR.mkdir(parents=True)
    (seed.BACKEND_DIR / "mlflow.db").write_bytes(b"partial database")

    assert seed.main() == 1
    assert (seed.BACKEND_DIR / "mlflow.db").read_bytes() == b"partial database"


def test_seed_rejects_a_model_whose_artifact_does_not_match_its_manifest(tmp_path, monkeypatch):
    """Un almacen con otro campeon de P3 tiene la misma estructura: solo cambia la huella."""
    source = tmp_path / "source"
    source.mkdir()
    _, artifact_root, manifests = _make_store(source)
    (artifact_root / "p3" / "artifacts" / "model.skops").write_bytes(b"otro modelo de p3")
    destination = tmp_path / "destination"
    _configure(monkeypatch, source, destination, manifests)

    assert seed.main() == 1
    assert not (destination / "backend" / "mlflow.db").exists()


def test_seed_rejects_a_model_registered_under_another_run(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    _, _, manifests = _make_store(source)
    manifiesto = json.loads(manifests[P3].read_text(encoding="utf-8"))
    manifiesto["modelo"]["run_id"] = "otro-run"
    manifests[P3].write_text(json.dumps(manifiesto), encoding="utf-8")
    destination = tmp_path / "destination"
    _configure(monkeypatch, source, destination, manifests)

    assert seed.main() == 1
    assert not (destination / "backend" / "mlflow.db").exists()


def test_seed_requires_the_serialized_model_of_a_model_with_manifest(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    _, artifact_root, manifests = _make_store(source)
    destino_skops = artifact_root / "p3" / "artifacts" / "model.skops"
    destino_skops.rename(destino_skops.with_name("model.skops.bak"))
    destination = tmp_path / "destination"
    _configure(monkeypatch, source, destination, manifests)

    assert seed.main() == 1
