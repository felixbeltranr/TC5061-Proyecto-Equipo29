"""Experimento base de P2 (clasificacion de riesgo por estacion) registrado en MLflow.

Ejemplo:
    python -m mibici.modeling.train_p2 --niveles 0.5 0.3 0.7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from mibici import features_p2 as fp
from mibici.config import CONFIG, CONFIG_P2, PROJ_ROOT

EXPERIMENTO = CONFIG_P2.experimento
RIESGO = ["sin_bicicletas", "sin_anclajes"]
FAMILIA_HGB = "hist_gradient_boosting"
CORTES = {
    "train_inicio": CONFIG_P2.fecha_inicio_entrenamiento,
    "val_inicio": CONFIG_P2.fecha_inicio_validacion,
    "test_inicio": CONFIG_P2.fecha_inicio_prueba,
    "test_fin": CONFIG_P2.fecha_fin_prueba,
}


def rango_meses(inicio: str, fin_excluido: str) -> str:
    """Rango AAAA-MM..AAAA-MM de un intervalo de fechas con fin excluido."""
    ultimo = pd.Timestamp(fin_excluido) - pd.Timedelta(days=1)
    return f"{pd.Timestamp(inicio):%Y-%m}..{ultimo:%Y-%m}"


# rango descargado, el mismo de mibici.dataset
RANGO_DATOS = (
    f"{CONFIG.download_data_start_year}-{CONFIG.download_data_start_month:02d}.."
    f"{CONFIG.download_data_end_year}-{CONFIG.download_data_end_month:02d}"
)


def git_commit() -> tuple[str, bool]:
    """Commit actual y si hay cambios sin guardar, para etiquetar el run."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, text=True
        ).strip()
        sucio = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJ_ROOT, text=True
            ).strip()
        )
        return sha, sucio
    except subprocess.CalledProcessError, FileNotFoundError:
        return "desconocido", True


def macro_f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Macro F1 con las tres clases fijas; una clase sin aciertos ni apariciones aporta 0.

    Es la misma cuenta para elegir hiperparametros en validacion y para las metricas de test.
    """
    return float(f1_score(y_true, y_pred, labels=fp.CLASES, average="macro", zero_division=0))


def metricas(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    proba: np.ndarray | None,
    clases: list[str],
    prefijo: str,
) -> dict[str, float]:
    """Macro F1, precision, recall y F1 por clase; PR AUC de las clases de riesgo si hay probas.

    El prefijo indica el conjunto (test, test_nuevas) para nombrar las metricas en MLflow.
    """
    m = {f"{prefijo}_macro_f1": macro_f1(y_true, y_pred)}
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=fp.CLASES, zero_division=0)
    for i, c in enumerate(fp.CLASES):
        m[f"{prefijo}_precision_{c}"] = p[i]
        m[f"{prefijo}_recall_{c}"] = r[i]
        m[f"{prefijo}_f1_{c}"] = f[i]
    if proba is not None:
        for c in RIESGO:
            y_bin = (np.asarray(y_true) == c).astype(int)
            if y_bin.sum():
                m[f"{prefijo}_pr_auc_{c}"] = average_precision_score(
                    y_bin, proba[:, clases.index(c)]
                )
    return {k: float(v) for k, v in m.items()}


def figura_confusion(y_true: ArrayLike, y_pred: ArrayLike, titulo: str, ruta: Path) -> Path:
    """Guarda la matriz de confusion en ruta y la regresa para registrarla como artefacto."""
    cm = confusion_matrix(y_true, y_pred, labels=fp.CLASES)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=fp.CLASES).plot(
        ax=ax, values_format="d", colorbar=False
    )
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    return ruta


def particionar(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Cortes temporales por fecha objetivo: fit, val (ajuste), train completo y test.

    Falla si alguna particion queda vacia, igual que train_p1.particionar.
    """
    t = df["ts_objetivo"]
    c = {k: pd.Timestamp(v) for k, v in CORTES.items()}
    partes = {
        "fit": df[(t >= c["train_inicio"]) & (t < c["val_inicio"])],
        "val": df[(t >= c["val_inicio"]) & (t < c["test_inicio"])],
        "train": df[(t >= c["train_inicio"]) & (t < c["test_inicio"])],
        "test": df[(t >= c["test_inicio"]) & (t < c["test_fin"])],
    }
    vacias = [k for k, v in partes.items() if v.empty]
    if vacias:
        raise ValueError(f"Particiones sin filas elegibles: {vacias}; revisa el rango del panel")
    return partes


def preparar(nivel: float, panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Etiqueta el panel con el nivel inicial dado, arma features y particiona."""
    df = fp.construir_features(fp.agregar_ocupacion(panel, nivel))
    return particionar(df)


def ajustar(
    train: pd.DataFrame, params: dict, semilla: int, balanceado: bool = True
) -> HistGradientBoostingClassifier:
    """Entrena HistGradientBoosting; con balanceado usa pesos de clase balanceados."""
    modelo = HistGradientBoostingClassifier(
        class_weight="balanced" if balanceado else None,
        random_state=semilla,
        early_stopping=False,
        **params,
    )
    modelo.fit(train[fp.FEATURES], train["objetivo"])
    return modelo


def afinar(
    partes: dict, semilla: int, rejilla: tuple[dict, ...] | list[dict] | None = None
) -> tuple[dict, list[dict]]:
    """Elige hiperparametros por macro F1 en validacion, entrenando solo con fit.

    Por defecto recorre CONFIG_P2.rejilla; con una rejilla de una sola configuracion solo
    la evalua. Regresa la configuracion elegida y la rejilla con su val_macro_f1.
    """
    por_est, por_hora = fp.stats_estacion(partes["fit"])
    fit = fp.unir_stats(partes["fit"], por_est, por_hora)
    val = fp.unir_stats(partes["val"], por_est, por_hora)
    resultados = []
    for params in CONFIG_P2.rejilla if rejilla is None else rejilla:
        modelo = ajustar(fit, params, semilla)
        f1 = macro_f1(val["objetivo"], modelo.predict(val[fp.FEATURES]))
        resultados.append({**params, "val_macro_f1": float(f1)})
        logger.info("Rejilla {} -> val_macro_f1 {:.4f}", params, f1)
    mejor = max(resultados, key=lambda r: r["val_macro_f1"])
    return {k: v for k, v in mejor.items() if k != "val_macro_f1"}, resultados


def predecir(
    modelo: HistGradientBoostingClassifier, test: pd.DataFrame
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Clases del modelo, probabilidades y clase predicha (la de mayor probabilidad) en test."""
    clases = list(modelo.classes_)
    proba = modelo.predict_proba(test[fp.FEATURES])
    pred = np.array(clases, dtype=object)[proba.argmax(1)]
    return clases, proba, pred


def metricas_test_y_nuevas(
    test: pd.DataFrame,
    nuevas: pd.Series,
    pred: np.ndarray,
    proba: np.ndarray | None = None,
    clases: list[str] = fp.CLASES,
) -> dict[str, float]:
    """Metricas en todo test (test_*) y solo en las estaciones nuevas (test_nuevas_*).

    nuevas es la mascara de filas de estaciones que empiezan en el periodo de prueba.
    """
    mascara = nuevas.to_numpy()
    m = metricas(test["objetivo"], pred, proba, clases, "test")
    return m | metricas(
        test.loc[nuevas, "objetivo"],
        pred[mascara],
        None if proba is None else proba[mascara],
        clases,
        "test_nuevas",
    )


def guardar_importancia(
    modelo: HistGradientBoostingClassifier,
    test: pd.DataFrame,
    semilla: int,
    figs: Path,
    sufijo: str,
) -> list[Path]:
    """Importancia por permutacion en una muestra de test; guarda el CSV y la figura."""
    muestra = test.sample(min(CONFIG_P2.muestra_importancia, len(test)), random_state=semilla)
    imp = permutation_importance(
        modelo,
        muestra[fp.FEATURES],
        muestra["objetivo"],
        scoring="f1_macro",
        n_repeats=CONFIG_P2.repeticiones_importancia,
        random_state=semilla,
        n_jobs=1,
    )
    imp_df = pd.DataFrame(
        {
            "feature": fp.FEATURES,
            "importancia_media": imp.importances_mean,
            "importancia_std": imp.importances_std,
        }
    ).sort_values("importancia_media", ascending=False)
    ruta_imp = figs / f"p2_importancia_permutacion_{sufijo}.csv"
    imp_df.to_csv(ruta_imp, index=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    imp_df.iloc[::-1].plot.barh(
        x="feature", y="importancia_media", xerr="importancia_std", ax=ax, legend=False
    )
    ax.set_xlabel("Caida en macro F1 (test, muestra)")
    ax.set_title(f"Importancia por permutacion (inicio {sufijo})")
    fig.tight_layout()
    ruta_fig = figs / f"p2_importancia_permutacion_{sufijo}.png"
    fig.savefig(ruta_fig, dpi=120)
    plt.close(fig)
    return [ruta_imp, ruta_fig]


def etiquetas_modelo(modelo: str, familia: str, etapa: str) -> dict[str, str]:
    """Etiquetas de un run: modelo, model_family y stage_of_work (baseline, search o candidate)."""
    return {"modelo": modelo, "model_family": familia, "stage_of_work": etapa}


def registrar(
    nombre: str, tags: dict, params: dict, mets: dict[str, float], artefactos: list[Path]
) -> None:
    """Crea un run en MLflow con nombre descriptivo, etiquetas, parametros, metricas y artefactos."""
    with mlflow.start_run(run_name=nombre):
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        mlflow.log_metrics(mets)
        for a in artefactos:
            mlflow.log_artifact(str(a))


def parametros_comunes(
    nivel: float,
    semilla: int,
    manifiesto: dict,
    resumen: dict,
    n_train: int,
    n_test: int,
    n_nuevas: int,
) -> dict:
    """Parametros que comparten todos los runs de un nivel inicial."""
    return {
        "nivel_inicial": nivel,
        "umbral_bajo": CONFIG_P2.umbral_bajo,
        "umbral_alto": CONFIG_P2.umbral_alto,
        "horizonte_horas": CONFIG_P2.horizonte_horas,
        "horas_objetivo": f"{min(CONFIG_P2.horas_etiqueta)}-{max(CONFIG_P2.horas_etiqueta)}",
        "hora_inicio_dia_servicio": CONFIG_P2.hora_inicio_servicio,
        "train_rango": rango_meses(CORTES["train_inicio"], CORTES["test_inicio"]),
        "val_rango": rango_meses(CORTES["val_inicio"], CORTES["test_inicio"]),
        "test_rango": rango_meses(CORTES["test_inicio"], CORTES["test_fin"]),
        "semilla": semilla,
        "data_version": manifiesto["version"],
        "n_train": n_train,
        "n_test": n_test,
        "estaciones_modeladas": resumen["estaciones_modeladas"],
        "estaciones_sin_nomenclatura": resumen["estaciones_sin_nomenclatura"],
        "estaciones_capacidad_cero": resumen["estaciones_capacidad_cero"],
        "estaciones_nuevas_test": n_nuevas,
    }


def correr_nivel(
    nivel: float,
    panel: pd.DataFrame,
    args: argparse.Namespace,
    manifiesto: dict,
    resumen: dict,
    params_hgb: dict,
    rejilla: list[dict],
    tags_base: dict,
    sin_pesos: bool = False,
) -> dict[str, dict[str, float]]:
    """Corre baselines y HGB para un nivel inicial y registra cada modelo como run."""
    partes = preparar(nivel, panel)
    por_est, por_hora = fp.stats_estacion(partes["train"])
    train = fp.unir_stats(partes["train"], por_est, por_hora)
    test = fp.unir_stats(partes["test"], por_est, por_hora)
    nuevas = test["primer_dia"] >= pd.Timestamp(CORTES["test_inicio"])
    sufijo = f"ini{round(nivel * 100)}"
    figs = Path(args.figuras)
    figs.mkdir(parents=True, exist_ok=True)
    n_nuevas = int(test.loc[nuevas, "estacion"].nunique())

    params_comunes = parametros_comunes(
        nivel, args.semilla, manifiesto, resumen, len(train), len(test), n_nuevas
    )
    dist = test["objetivo"].value_counts(normalize=True)
    dist_m = {f"test_prop_{c}": float(dist.get(c, 0)) for c in fp.CLASES}
    salida = {}

    # baseline 1: clase mayoritaria del train
    mayoritaria = train["objetivo"].mode()[0]
    pred = np.full(len(test), mayoritaria, dtype=object)
    m = metricas_test_y_nuevas(test, nuevas, pred)
    registrar(
        f"p2_baseline_mayoritaria_{sufijo}",
        tags_base | etiquetas_modelo("baseline_mayoritaria", "baseline_mayoritaria", "baseline"),
        params_comunes | {"modelo": "clase_mayoritaria", "clase": mayoritaria},
        m | dist_m,
        [],
    )
    salida["baseline_mayoritaria"] = m | dist_m

    # baseline 2: misma clase a la misma hora la semana previa
    pred = test["baseline_semana"].fillna(mayoritaria).to_numpy()
    m = metricas_test_y_nuevas(test, nuevas, pred)
    fig = figura_confusion(
        test["objetivo"],
        pred,
        f"Baseline semana previa (inicio {sufijo})",
        figs / f"p2_cm_baseline_semana_{sufijo}.png",
    )
    registrar(
        f"p2_baseline_semana_previa_{sufijo}",
        tags_base
        | etiquetas_modelo("baseline_semana_previa", "baseline_semana_previa", "baseline"),
        params_comunes | {"modelo": "clase_misma_hora_semana_previa"},
        m | dist_m,
        [fig],
    )
    salida["baseline_semana_previa"] = m | dist_m

    # baseline 3: persistencia, la clase estimada de la hora actual se repite en t+1
    pred = test["clase_actual"].to_numpy()
    m = metricas_test_y_nuevas(test, nuevas, pred)
    fig = figura_confusion(
        test["objetivo"],
        pred,
        f"Baseline persistencia (inicio {sufijo})",
        figs / f"p2_cm_baseline_persistencia_{sufijo}.png",
    )
    registrar(
        f"p2_baseline_persistencia_{sufijo}",
        tags_base | etiquetas_modelo("baseline_persistencia", "baseline_persistencia", "baseline"),
        params_comunes | {"modelo": "clase_actual_se_repite"},
        m | dist_m,
        [fig],
    )
    salida["baseline_persistencia"] = m | dist_m

    # candidato: HistGradientBoosting con pesos balanceados
    modelo = ajustar(train, params_hgb, args.semilla)
    clases, proba, pred = predecir(modelo, test)
    m = metricas_test_y_nuevas(test, nuevas, pred, proba, clases)
    if rejilla:
        m["val_macro_f1"] = max(r["val_macro_f1"] for r in rejilla)
    artefactos = [
        figura_confusion(
            test["objetivo"], pred, f"HGB test (inicio {sufijo})", figs / f"p2_cm_hgb_{sufijo}.png"
        ),
        figura_confusion(
            test.loc[nuevas, "objetivo"],
            pred[nuevas.to_numpy()],
            f"HGB estaciones nuevas (inicio {sufijo})",
            figs / f"p2_cm_hgb_nuevas_{sufijo}.png",
        ),
    ]
    artefactos += guardar_importancia(modelo, test, args.semilla, figs, sufijo)
    if rejilla:
        ruta_rej = figs / "p2_rejilla_validacion.json"
        ruta_rej.write_text(json.dumps(rejilla, indent=2))
        artefactos.append(ruta_rej)
    registrar(
        f"p2_hgb_balanceado_{sufijo}",
        tags_base | etiquetas_modelo("hist_gradient_boosting", FAMILIA_HGB, "candidate"),
        params_comunes
        | {
            "modelo": "HistGradientBoostingClassifier",
            "class_weight": "balanced",
            "features": ",".join(fp.FEATURES),
        }
        | params_hgb,
        m | dist_m,
        artefactos + [Path(args.manifiesto), Path(args.panel).with_suffix(".json")],
    )
    salida["hgb"] = m | dist_m

    if sin_pesos:
        # misma configuracion sin pesos de clase, para ver el efecto del balanceo
        modelo = ajustar(train, params_hgb, args.semilla, balanceado=False)
        clases, proba, pred = predecir(modelo, test)
        m = metricas_test_y_nuevas(test, nuevas, pred, proba, clases)
        fig = figura_confusion(
            test["objetivo"],
            pred,
            f"HGB sin pesos (inicio {sufijo})",
            figs / f"p2_cm_hgb_sin_pesos_{sufijo}.png",
        )
        registrar(
            f"p2_hgb_sin_pesos_{sufijo}",
            tags_base
            | etiquetas_modelo("hist_gradient_boosting_sin_pesos", FAMILIA_HGB, "candidate"),
            params_comunes
            | {"modelo": "HistGradientBoostingClassifier", "class_weight": "ninguno"}
            | params_hgb,
            m | dist_m,
            [fig],
        )
        salida["hgb_sin_pesos"] = m | dist_m
    return salida


def main(argv: list[str] | None = None) -> None:
    """Punto de entrada: python -m mibici.modeling.train_p2 --niveles 0.5 0.3 0.7."""
    ap = argparse.ArgumentParser(description="Experimento base P2 en MLflow")
    ap.add_argument("--panel", default=str(CONFIG_P2.panel_path))
    ap.add_argument("--model", choices=(CONFIG_P2.model,), default=CONFIG_P2.model)
    ap.add_argument(
        "--hyperparameters",
        default="{}",
        help=(
            "objeto JSON con hiperparametros del HGB. Se mezcla con la primera configuracion de "
            "CONFIG_P2.rejilla (lo no indicado toma ese valor), no hay busqueda en la rejilla y "
            "solo esa configuracion se evalua en validacion. Tiene prioridad sobre --sin-afinar"
        ),
    )
    ap.add_argument(
        "--niveles",
        type=float,
        nargs="+",
        default=[CONFIG_P2.nivel_inicial],
        help="niveles iniciales de ocupacion; el primero se usa para afinar",
    )
    ap.add_argument("--semilla", type=int, default=CONFIG_P2.seed)
    ap.add_argument("--tracking-uri", default=CONFIG_P2.tracking_uri)
    ap.add_argument(
        "--figuras",
        default=str(CONFIG_P2.corridas_dir),
        help="carpeta de figuras, CSV y p2_resultados.json (por defecto ignorada por git)",
    )
    ap.add_argument(
        "--artefactos",
        default=str(CONFIG_P2.artefactos_dir),
        help="carpeta de artefactos de MLflow; solo aplica si el experimento aun no existe",
    )
    ap.add_argument("--manifiesto", default=str(CONFIG_P2.manifiesto_path))
    ap.add_argument(
        "--sin-afinar", action="store_true", help="usa la primera configuracion de la rejilla"
    )
    args = ap.parse_args(argv)
    try:
        hyperparameters = json.loads(args.hyperparameters)
    except json.JSONDecodeError as exc:
        ap.error(f"--hyperparameters must be valid JSON: {exc}")
    if not isinstance(hyperparameters, dict):
        ap.error("--hyperparameters must be a JSON object")
    first_grid = CONFIG_P2.rejilla[0]
    unsupported = set(hyperparameters).difference(first_grid)
    if unsupported:
        ap.error(f"Unsupported {args.model} hyperparameters: {sorted(unsupported)}")

    mlflow.set_tracking_uri(args.tracking_uri)
    if mlflow.get_experiment_by_name(CONFIG_P2.experimento) is None:
        mlflow.create_experiment(
            CONFIG_P2.experimento, artifact_location=Path(args.artefactos).resolve().as_uri()
        )
    mlflow.set_experiment(CONFIG_P2.experimento)

    manifiesto = json.loads(Path(args.manifiesto).read_text())
    resumen = json.loads(Path(args.panel).with_suffix(".json").read_text())
    panel = pd.read_parquet(args.panel)
    sha, sucio = git_commit()
    tags_base = {
        "question": CONFIG_P2.question,
        "team": CONFIG_P2.team,
        "author": CONFIG_P2.author,
        "data_range": RANGO_DATOS,
        "seed": str(args.semilla),
        "data_version": manifiesto["version"],
        "mlflow.source.git.commit": sha,
        "git_commit": sha,
        "git_dirty": str(sucio),
    }

    if hyperparameters:
        # sin busqueda: se evalua en validacion solo esta configuracion, para registrar su
        # val_macro_f1 y dejar la rejilla de un solo elemento
        configuracion = first_grid | hyperparameters
        params_hgb, rejilla = afinar(
            preparar(args.niveles[0], panel), args.semilla, [configuracion]
        )
    elif args.sin_afinar:
        params_hgb, rejilla = CONFIG_P2.rejilla[0], []
    else:
        params_hgb, rejilla = afinar(preparar(args.niveles[0], panel), args.semilla)
    logger.info("Parametros elegidos: {}", params_hgb)

    todo = {}
    for i, nivel in enumerate(args.niveles):
        todo[str(nivel)] = correr_nivel(
            nivel,
            panel,
            args,
            manifiesto,
            resumen,
            params_hgb,
            rejilla if i == 0 else [],
            tags_base,
            sin_pesos=i == 0,
        )
    ruta = Path(args.figuras) / "p2_resultados.json"
    ruta.write_text(
        json.dumps({"params_hgb": params_hgb, "rejilla": rejilla, "resultados": todo}, indent=2)
    )
    logger.info("Resultados en {}", ruta)


if __name__ == "__main__":
    main()
