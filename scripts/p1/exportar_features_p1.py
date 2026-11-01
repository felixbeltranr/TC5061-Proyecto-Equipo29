"""Exportador privado P1: funciones originales, sin entrenamiento ni MLflow.

native requiere Python 3.14 e importa features/configuración de mibici. En ambos modos
particionar se compila por AST para no importar train_p1, que importa MLflow; isolated-source
compila así también las funciones puras de features: NO es importación del paquete completo
ni validación del entorno principal. Las exportaciones son datos privados.
"""

from __future__ import annotations

import argparse
import ast
from datetime import UTC, date, datetime
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tokenize
from types import SimpleNamespace

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
import numpy as np
import pandas as pd
from validar_panel_p1 import SPEC, contract, sha256

BASE_REVISION = "d2bec0502daaed12762da790ae6d29523124ad32"
# config.py lo comparten P1, P2 y P3: sólo se fijan la clase P1Config y la instancia CONFIG_P1.
SOURCE_SHA256 = {
    "mibici/config.py::P1Config,CONFIG_P1": "c2420646acac270783243aff51c7e65d076725f068ec4b8c7fcf6ec88f096f6d",
    "mibici/features.py": "5f4ed26b6945a9ed3c22719f6b707f4373b94e647433a03552ec5f334f9cbefd",
    "mibici/modeling/train_p1.py": "5882a96a132a5cce0887739a9d05bd259c6e936f52d0ae8fd90bf13ad66d5b10",
}
CATEGORICAL = {"p1": ["dia_semana", "festivo", "festivo_lag7"]}
# Tupla con nombre: ruff (objetivo 3.14) quitaría los paréntesis de un except con dos tipos,
# sintaxis que Python 3.11 no acepta.
ERRORES_GIT = (OSError, subprocess.CalledProcessError)


def literal_field(repo, class_name, field):
    tree = ast.parse((repo / "mibici/config.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(
        n.value
        for n in cls.body
        if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name)
        and (n.target.id == field)
    )
    return ast.literal_eval(node)


def assignment(path, name):
    for n in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in n.targets
        ):
            return ast.literal_eval(n.value)
    raise ValueError(f"Asignación literal ausente: {name}")


def original_functions(path, names, namespace):
    """Compile exact selected function ASTs; never execute module imports/main."""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    decoradas = {
        n.name
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.decorator_list and n.name in names
    }
    if decoradas:
        raise ValueError(f"Funciones decoradas no admitidas en {path.name}: {sorted(decoradas)}")
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    selected = []
    for i, token in enumerate(tokens):
        if token.type != tokenize.NAME or token.string != "def" or token.start[1] != 0:
            continue
        if tokens[i + 1].string not in names:
            continue
        start = token.start[0]
        depth, entered, end = (0, False, None)
        for later in tokens[i + 1 :]:
            if later.type == tokenize.INDENT:
                depth += 1
                entered = True
            elif later.type == tokenize.DEDENT:
                depth -= 1
                if entered and depth == 0:
                    end = later.start[0] - 1
                    break
        if end is None:
            raise ValueError("No se localizó el límite de una función original")
        block = "".join(lines[start - 1 : end])
        node = ast.parse(block).body[0]
        ast.increment_lineno(node, start - 1)
        selected.append(node)
    if {n.name for n in selected} != set(names):
        raise ValueError(f"Funciones ausentes en {path.name}")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    # Sólo funciones puras elegidas por AST de fuentes con hash verificado.
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return namespace


def load_functions(repo: Path, pipeline: str, mode: str):
    class_name = "P1Config"
    cfg_names = [
        "hora_emision",
        "retraso_conteo_min",
        "train_inicio",
        "val_inicio",
        "test_inicio",
        "test_fin",
    ]
    instancia = top_level(
        ast.parse((repo / "mibici/config.py").read_text(encoding="utf-8")), "CONFIG_P1"
    ).value
    if not (
        isinstance(instancia, ast.Call)
        and isinstance(instancia.func, ast.Name)
        and instancia.func.id == class_name
        and not instancia.args
        and not instancia.keywords
    ):
        raise ValueError(
            "CONFIG_P1 debe ser P1Config() sin argumentos: se usan los valores de la clase"
        )
    values = {k: literal_field(repo, class_name, k) for k in cfg_names}
    cuts = {k: values[k] for k in ["train_inicio", "val_inicio", "test_inicio", "test_fin"]}
    feature_path = repo / "mibici" / "features.py"
    if mode == "native":
        if sys.version_info[:2] != (3, 14):
            raise RuntimeError(
                "native requiere el entorno principal Python 3.14; no instalar nada automáticamente"
            )
        sys.path.insert(0, str(repo))
        feature = importlib.import_module("mibici.features")
        if Path(feature.__file__).resolve() != feature_path:
            raise RuntimeError(
                "Se importó otra copia de mibici; detener para evitar evidencia ambigua"
            )
        for name, module in list(sys.modules.items()):
            filename = getattr(module, "__file__", None)
            if (name == "mibici" or name.startswith("mibici.")) and filename:
                path = Path(filename).resolve()
                if not path.is_relative_to(repo):
                    raise RuntimeError(f"Módulo de otra copia: {name}: {path}")
                rel = path.relative_to(repo).as_posix()
                for clave, esperado in SOURCE_SHA256.items():
                    if clave.partition("::")[0] == rel and (
                        source_fingerprint(repo, clave) != esperado
                    ):
                        raise RuntimeError(f"Fuente importada distinta a la revisada: {clave}")
    else:
        ns = {"np": np, "pd": pd, "date": date}
        ns["FEATURES"] = assignment(feature_path, "FEATURES")
        ns.update(
            {
                "HORA_EMISION": values["hora_emision"],
                "RETRASO_CONTEO_MIN": values["retraso_conteo_min"],
            }
        )
        for key in ["LAGS_SEMANALES", "VENTANA_CORTA", "VENTANA_LARGA"]:
            ns[key] = assignment(feature_path, key)
        names = [
            "emision",
            "disponible_desde",
            "_lag_minimo",
            "ultimo_dia_conocido",
            "pascua",
            "_lunes_n",
            "festivos",
            "construir_features",
        ]
        original_functions(feature_path, names, ns)
        ns["LAG_MINIMO"] = ns["_lag_minimo"]()
        feature = SimpleNamespace(**ns)
    ns = {"pd": pd, "CORTES": cuts, "fe": feature}
    original_functions(repo / "mibici/modeling" / f"train_{pipeline}.py", ["particionar"], ns)
    return (feature, ns["particionar"], cuts)


def prepare(panel, pipeline, feature, partition):
    if pipeline != "p1":
        raise ValueError("Este auxiliar es exclusivamente P1")
    data = feature.construir_features(panel)
    parts = partition(data.loc[data["elegible"]])
    return (parts["train"], parts["test"])


def normalized_sha256(path):
    import hashlib

    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def top_level(tree, nombre):
    """Única clase, función o asignación de primer nivel con ese nombre."""
    encontrados = []
    for n in tree.body:
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)):
            nombres = [n.name]
        elif isinstance(n, ast.Assign):
            nombres = [t.id for t in n.targets if isinstance(t, ast.Name)]
        elif isinstance(n, (ast.AnnAssign, ast.AugAssign)):
            nombres = [n.target.id] if isinstance(n.target, ast.Name) else []
        else:
            nombres = []
        if nombre in nombres:
            encontrados.append(n)
    if len(encontrados) != 1:
        raise ValueError(f"{nombre} debe definirse exactamente una vez en el primer nivel")
    return encontrados[0]


def source_fingerprint(repo, clave):
    """SHA256 con LF del archivo o, con «archivo::A,B», sólo de esas definiciones."""
    import hashlib

    rel, _, nombres = clave.partition("::")
    if not nombres:
        return normalized_sha256(repo / rel)
    source = (repo / rel).read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    # Cortar sólo en \n, como cuenta las líneas el AST (splitlines también corta en \x0c, etc.).
    tree, lines, partes = ast.parse(source), source.split("\n"), []
    for nombre in nombres.split(","):
        nodo = top_level(tree, nombre)
        decoradores = [d.lineno for d in getattr(nodo, "decorator_list", [])]
        inicio = min([nodo.lineno, *decoradores])
        partes.append("\n".join(lines[inicio - 1 : nodo.end_lineno]) + "\n")
    return hashlib.sha256("".join(partes).encode("utf-8")).hexdigest()


def export(repo, source, output, pipeline, mode="native"):
    repo, source, output = (
        repo.resolve(strict=True),
        source.resolve(strict=True),
        output.resolve(),
    )
    if output.is_relative_to(repo) or output == source.parent or output.exists():
        raise ValueError("Use una carpeta nueva fuera del repo y separada de la entrada")
    git = ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), "rev-parse"]
    try:
        raiz = Path(subprocess.check_output([*git, "--show-toplevel"], text=True).strip())
        head = subprocess.check_output([*git, "HEAD"], text=True).strip()
    except ERRORES_GIT as exc:
        raise ValueError("--repo debe ser la raíz de un checkout git con su .git") from exc
    # Sin esto, una copia sin .git dentro de otro repositorio tomaría el commit de ese otro.
    if raiz.resolve() != repo:
        raise ValueError("--repo debe ser la raíz de un checkout git con su .git")
    source_hashes = {}
    for rel, expected in SOURCE_SHA256.items():
        actual = source_fingerprint(repo, rel)
        if actual != expected:
            raise ValueError(
                f"Fuente distinta a la base revisada: {rel}; revisar antes de cambiar hashes"
            )
        source_hashes[rel] = actual
    data = pd.read_parquet(source, columns=SPEC[pipeline]["columns"])
    checked = contract(data, pipeline)
    if checked["failures"]:
        raise ValueError(f"Contrato del panel íntegro no satisfecho: {checked['failures']}")
    mlflow_before = {
        name for name in sys.modules if name == "mlflow" or name.startswith("mlflow.")
    }
    f, partition, cuts = load_functions(repo, pipeline, mode)
    input_rows = len(data)
    train, test = prepare(data, pipeline, f, partition)
    del data
    key = ["estacion", "fecha"]
    columns = key + list(f.FEATURES)
    train, test = (train[columns].copy(), test[columns].copy())
    if train.duplicated(key).any() or test.duplicated(key).any():
        raise ValueError("Claves duplicadas en las exportaciones")
    if not train[key].merge(test[key], on=key).empty:
        raise ValueError("Hay claves compartidas entre entrenamiento y prueba")
    if not train[key[1]].max() < test[key[1]].min():
        raise ValueError("Orden temporal entre conjuntos inválido")
    if test[key[1]].min() - train[key[1]].max() < pd.Timedelta(days=f.LAG_MINIMO):
        raise ValueError("Embargo entre entrenamiento y prueba menor al rezago mínimo")
    mlflow_after = {name for name in sys.modules if name == "mlflow" or name.startswith("mlflow.")}
    mlflow_introduced = sorted(mlflow_after - mlflow_before)
    if mlflow_introduced:
        raise RuntimeError("El exportador introdujo módulos MLflow; no continuar")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "EXPORTED_NOT_MODEL_VALIDATION",
        "pipeline": pipeline,
        "mode": mode,
        "python": sys.version.split()[0],
        "created_utc": datetime.now(UTC).isoformat(),
        "source_head": head,
        "reviewed_base": BASE_REVISION,
        "source_hash_normalization": "CRLF_to_LF",
        "source_files_sha256": source_hashes,
        "input_name": source.name,
        "input_sha256": sha256(source),
        "input_rows": input_rows,
        "features": list(f.FEATURES),
        "categorical": CATEGORICAL[pipeline],
        "key": key,
        "cuts": cuts,
        "embargo_days": int(f.LAG_MINIMO),
        "label_exported": False,
        "tracking_or_training_called": False,
        "mlflow_modules_introduced": mlflow_introduced,
        "tables": {},
    }
    for name, df in [("train", train), ("test", test)]:
        p = output / f"{name}.parquet"
        df.to_parquet(p, index=False)
        manifest["tables"][name] = {
            "file": p.name,
            "sha256": sha256(p),
            "rows": len(df),
            "stations": int(df["estacion"].nunique()),
            "min_time": str(df[key[1]].min()),
            "max_time": str(df[key[1]].max()),
            "nulls": {c: int(n) for c, n in df[f.FEATURES].isna().sum().items()},
        }
    manifest["station_populations"] = {
        "common": len(set(train.estacion) & set(test.estacion)),
        "test_only": len(set(test.estacion) - set(train.estacion)),
        "train_only": len(set(train.estacion) - set(test.estacion)),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--pipeline", default="p1", choices=["p1"])
    p.add_argument("--mode", choices=["native", "isolated-source"], default="native")
    a = p.parse_args(argv)
    try:
        m = export(a.repo, a.input, a.output, a.pipeline, a.mode)
        print(json.dumps(m, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - la CLI informa cualquier error con exit 2
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
