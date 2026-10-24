"""Experimento base de P2 (clasificacion de riesgo por estacion) registrado en MLflow.

Ejemplo:
    python -m mibici.modeling.train_p2 --niveles 0.5 0.3 0.7
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import subprocess
import time

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mibici import features_p2 as fp
from mibici.config import CONFIG, CONFIG_P2, PROJ_ROOT

EXPERIMENTO = CONFIG_P2.experimento
RIESGO = ["sin_bicicletas", "sin_anclajes"]
FAMILIA_HGB = "hist_gradient_boosting"
FAMILIA_LOGISTICA = "logistic_regression"
FAMILIA_BOSQUE = "random_forest"
CATEGORICAS = ["hora", "dia_semana"]  # enteros que la logistica codifica uno a uno
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


def sufijo_nivel(nivel: float) -> str:
    """Sufijo de los nombres de run y de archivo de un nivel inicial, por ejemplo ini50."""
    return f"ini{round(nivel * 100)}"


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


def ajustar_logistica(train: pd.DataFrame, params: dict, semilla: int) -> Pipeline:
    """Regresion logistica balanceada: hora y dia de la semana en one-hot, el resto imputado
    con la mediana (hay faltantes en los rezagos largos) y escalado."""
    numericas = [c for c in fp.FEATURES if c not in CATEGORICAS]
    preproceso = ColumnTransformer(
        [
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
            (
                "numericas",
                make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
                ),
                numericas,
            ),
        ]
    )
    modelo = make_pipeline(
        preproceso, LogisticRegression(class_weight="balanced", random_state=semilla, **params)
    )
    return modelo.fit(train[fp.FEATURES], train["objetivo"])


def ajustar_random_forest(train: pd.DataFrame, params: dict, semilla: int) -> Pipeline:
    """RandomForest balanceado por arbol; max_samples acota el costo y se recorta a len(train)."""
    params = {**params, "max_samples": min(params["max_samples"], len(train))}
    modelo = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        RandomForestClassifier(
            class_weight="balanced_subsample", n_jobs=-1, random_state=semilla, **params
        ),
    )
    return modelo.fit(train[fp.FEATURES], train["objetivo"])


def medir_ajuste(ajustador, train: pd.DataFrame) -> tuple[object, float, float]:
    """Ajusta con ajustador(train) y regresa el modelo, los segundos de ajuste y su tamano en MB.

    El tamano es el del pickle en memoria; sirve para comparar el costo de las candidatas.
    """
    inicio = time.perf_counter()
    modelo = ajustador(train)
    segundos = time.perf_counter() - inicio
    return modelo, segundos, len(pickle.dumps(modelo)) / 1e6


def afinar(
    partes: dict,
    semilla: int,
    rejilla: tuple[dict, ...] | list[dict] | None = None,
    nombre_padre: str | None = None,
    tags: dict | None = None,
) -> tuple[dict, list[dict]]:
    """Elige hiperparametros por macro F1 en validacion, entrenando solo con fit.

    Por defecto recorre CONFIG_P2.rejilla; con una rejilla de una sola configuracion solo
    la evalua. Regresa la configuracion elegida y la rejilla con su val_macro_f1.
    Con nombre_padre registra la busqueda en MLflow como corridas anidadas: un run padre con
    la mejor combinacion y un hijo por combinacion (parametros, metricas de validacion y
    fit_seconds). tags son las etiquetas de todos esos runs.
    """
    por_est, por_hora = fp.stats_estacion(partes["fit"])
    fit = fp.unir_stats(partes["fit"], por_est, por_hora)
    val = fp.unir_stats(partes["val"], por_est, por_hora)
    resultados = []
    registrando = nombre_padre is not None
    with mlflow.start_run(run_name=nombre_padre) if registrando else nullcontext():
        if registrando:
            mlflow.set_tags(tags or {})
        for i, params in enumerate(CONFIG_P2.rejilla if rejilla is None else rejilla, start=1):
            inicio = time.perf_counter()
            modelo = ajustar(fit, params, semilla)
            segundos = time.perf_counter() - inicio
            clases, proba, pred = predecir(modelo, val)
            mets = metricas(val["objetivo"], pred, proba, clases, "val")
            f1 = mets["val_macro_f1"]
            resultados.append({**params, "val_macro_f1": f1})
            logger.info("Rejilla {} -> val_macro_f1 {:.4f}", params, f1)
            if registrando:
                registrar(
                    f"{nombre_padre}_c{i}",
                    tags or {},
                    {"combinacion": i, "semilla": semilla} | params,
                    mets | {"fit_seconds": segundos},
                    [],
                    anidado=True,
                )
        mejor = max(resultados, key=lambda r: r["val_macro_f1"])
        elegida = {k: v for k, v in mejor.items() if k != "val_macro_f1"}
        if registrando:
            mlflow.log_params({f"mejor_{k}": v for k, v in elegida.items()})
            mlflow.log_metric("val_macro_f1", mejor["val_macro_f1"])
            mlflow.log_dict(resultados, "p2_rejilla_validacion.json")
    return elegida, resultados


def predecir(modelo, test: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
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
    nombre: str,
    tags: dict,
    params: dict,
    mets: dict[str, float],
    artefactos: list[Path],
    anidado: bool = False,
) -> None:
    """Crea un run en MLflow con nombre descriptivo, etiquetas, parametros, metricas y artefactos.

    Con anidado el run cuelga del run activo (por ejemplo, el padre de una busqueda).
    """
    with mlflow.start_run(run_name=nombre, nested=anidado):
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


@dataclass
class ContextoNivel:
    """Datos de un nivel inicial: particiones con sus estadisticas y lo comun a todos los runs.

    fit y val llevan las estadisticas de fit (validacion de cada candidata); train y test, las
    de train (modelo final y prueba). nuevas marca las filas de estaciones nuevas en test.
    """

    sufijo: str
    figs: Path
    tags_base: dict
    params_comunes: dict
    dist: dict
    fit: pd.DataFrame
    val: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    nuevas: pd.Series


def preparar_nivel(
    nivel: float,
    panel: pd.DataFrame,
    args: argparse.Namespace,
    manifiesto: dict,
    resumen: dict,
    tags_base: dict,
) -> ContextoNivel:
    """Etiqueta el panel con el nivel inicial y arma las particiones y los parametros comunes."""
    partes = preparar(nivel, panel)
    est_fit, hora_fit = fp.stats_estacion(partes["fit"])
    est_train, hora_train = fp.stats_estacion(partes["train"])
    train = fp.unir_stats(partes["train"], est_train, hora_train)
    test = fp.unir_stats(partes["test"], est_train, hora_train)
    nuevas = test["primer_dia"] >= pd.Timestamp(CORTES["test_inicio"])
    figs = Path(args.figuras)
    figs.mkdir(parents=True, exist_ok=True)
    dist = test["objetivo"].value_counts(normalize=True)
    return ContextoNivel(
        sufijo=sufijo_nivel(nivel),
        figs=figs,
        tags_base=tags_base,
        params_comunes=parametros_comunes(
            nivel,
            args.semilla,
            manifiesto,
            resumen,
            len(train),
            len(test),
            int(test.loc[nuevas, "estacion"].nunique()),
        ),
        dist={f"test_prop_{c}": float(dist.get(c, 0)) for c in fp.CLASES},
        fit=fp.unir_stats(partes["fit"], est_fit, hora_fit),
        val=fp.unir_stats(partes["val"], est_fit, hora_fit),
        train=train,
        test=test,
        nuevas=nuevas,
    )


def predicciones_baseline(df: pd.DataFrame, mayoritaria: str) -> dict[str, np.ndarray]:
    """Clase predicha por cada baseline: mayoritaria, misma hora la semana previa y persistencia."""
    return {
        "mayoritaria": np.full(len(df), mayoritaria, dtype=object),
        "semana_previa": df["baseline_semana"].fillna(mayoritaria).to_numpy(),
        "persistencia": df["clase_actual"].to_numpy(),
    }


def correr_baselines(ctx: ContextoNivel) -> dict[str, dict[str, float]]:
    """Registra las tres baselines con metricas de validacion y de prueba."""
    # la mayoritaria de validacion sale de fit, la de prueba de train
    mayoritaria_fit = ctx.fit["objetivo"].mode()[0]
    mayoritaria = ctx.train["objetivo"].mode()[0]
    pred_val = predicciones_baseline(ctx.val, mayoritaria_fit)
    pred_test = predicciones_baseline(ctx.test, mayoritaria)
    # clave: (nombre del run, valor del parametro modelo, titulo y archivo de la figura)
    detalle = {
        "mayoritaria": ("mayoritaria", "clase_mayoritaria", None, None),
        "semana_previa": (
            "semana_previa",
            "clase_misma_hora_semana_previa",
            "Baseline semana previa",
            f"p2_cm_baseline_semana_{ctx.sufijo}.png",
        ),
        "persistencia": (
            "persistencia",
            "clase_actual_se_repite",
            "Baseline persistencia",
            f"p2_cm_baseline_persistencia_{ctx.sufijo}.png",
        ),
    }
    salida = {}
    for clave, (nombre, modelo, titulo, archivo) in detalle.items():
        m = metricas(ctx.val["objetivo"], pred_val[clave], None, fp.CLASES, "val")
        m |= metricas_test_y_nuevas(ctx.test, ctx.nuevas, pred_test[clave])
        params = {"modelo": modelo} | ({"clase": mayoritaria} if clave == "mayoritaria" else {})
        artefactos = []
        if titulo:
            artefactos.append(
                figura_confusion(
                    ctx.test["objetivo"],
                    pred_test[clave],
                    f"{titulo} (inicio {ctx.sufijo})",
                    ctx.figs / archivo,
                )
            )
        registrar(
            f"p2_baseline_{nombre}_{ctx.sufijo}",
            ctx.tags_base
            | etiquetas_modelo(f"baseline_{nombre}", f"baseline_{nombre}", "baseline"),
            ctx.params_comunes | params,
            m | ctx.dist,
            artefactos,
        )
        salida[f"baseline_{nombre}"] = m | ctx.dist
    return salida


def correr_candidata(
    ctx: ContextoNivel,
    clave: str,
    nombre_run: str,
    etiquetas: dict[str, str],
    params: dict,
    ajustador,
    mejor_baseline: tuple[str, float],
    con_nuevas: bool = True,
    extras=None,
) -> dict[str, float]:
    """Ajusta una candidata dos veces y registra su run con validacion, prueba y costo.

    Con fit se ajusta para las metricas val_*; con train completo, para las test_* y el costo
    (fit_seconds y model_size_mb son los de este segundo ajuste). mejor_baseline es el nombre y
    el macro F1 de prueba de la mejor baseline por validacion, para medir la mejora.
    extras(modelo) regresa archivos adicionales a registrar como artefactos.
    """
    modelo_val, _, _ = medir_ajuste(ajustador, ctx.fit)
    clases, proba, pred = predecir(modelo_val, ctx.val)
    m = metricas(ctx.val["objetivo"], pred, proba, clases, "val")
    modelo, segundos, mb = medir_ajuste(ajustador, ctx.train)
    clases, proba, pred = predecir(modelo, ctx.test)
    m |= metricas_test_y_nuevas(ctx.test, ctx.nuevas, pred, proba, clases)
    m |= {
        "fit_seconds": segundos,
        "model_size_mb": mb,
        "test_mejora_macro_f1_vs_mejor_baseline": m["test_macro_f1"] - mejor_baseline[1],
    }
    sufijo = ctx.sufijo
    artefactos = [
        figura_confusion(
            ctx.test["objetivo"],
            pred,
            f"{etiquetas['modelo']} test (inicio {sufijo})",
            ctx.figs / f"p2_cm_{clave}_{sufijo}.png",
        )
    ]
    if con_nuevas:
        artefactos.append(
            figura_confusion(
                ctx.test.loc[ctx.nuevas, "objetivo"],
                pred[ctx.nuevas.to_numpy()],
                f"{etiquetas['modelo']} estaciones nuevas (inicio {sufijo})",
                ctx.figs / f"p2_cm_{clave}_nuevas_{sufijo}.png",
            )
        )
    if extras:
        artefactos += extras(modelo)
    registrar(
        nombre_run,
        ctx.tags_base | etiquetas,
        ctx.params_comunes | params | {"mejor_baseline": mejor_baseline[0]},
        m | ctx.dist,
        artefactos,
    )
    return m | ctx.dist


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
    """Corre baselines y candidatas para un nivel inicial y registra cada modelo como run.

    Candidatas: C1 regresion logistica balanceada, C2 HGB balanceado con los hiperparametros
    de la busqueda y C3 RandomForest balanceado. Con sin_pesos se agrega el HGB sin pesos de
    clase para ver el efecto del balanceo. Los parametros de C1 y C3 salen de CONFIG_P2.
    """
    ctx = preparar_nivel(nivel, panel, args, manifiesto, resumen, tags_base)
    salida = correr_baselines(ctx)
    # mejor baseline por macro F1 de validacion; su macro F1 de prueba es la referencia de mejora
    nombre_mejor = max(salida, key=lambda n: salida[n]["val_macro_f1"])
    mejor = (nombre_mejor.removeprefix("baseline_"), salida[nombre_mejor]["test_macro_f1"])
    semilla, sufijo = args.semilla, ctx.sufijo
    common = {"ctx": ctx, "mejor_baseline": mejor}

    salida["logistica"] = correr_candidata(
        **common,
        clave="logistica",
        nombre_run=f"p2_logistica_balanceada_{sufijo}",
        etiquetas=etiquetas_modelo(
            "regresion_logistica_balanceada", FAMILIA_LOGISTICA, "candidate"
        ),
        params={"modelo": "LogisticRegression", "class_weight": "balanced"}
        | CONFIG_P2.params_logistica,
        ajustador=lambda df: ajustar_logistica(df, CONFIG_P2.params_logistica, semilla),
    )

    def extras_hgb(modelo) -> list[Path]:
        archivos = guardar_importancia(modelo, ctx.test, semilla, ctx.figs, sufijo)
        if rejilla:
            ruta_rej = ctx.figs / "p2_rejilla_validacion.json"
            ruta_rej.write_text(json.dumps(rejilla, indent=2))
            archivos.append(ruta_rej)
        return archivos + [Path(args.manifiesto), Path(args.panel).with_suffix(".json")]

    salida["hgb"] = correr_candidata(
        **common,
        clave="hgb",
        nombre_run=f"p2_hgb_balanceado_{sufijo}",
        etiquetas=etiquetas_modelo("hist_gradient_boosting", FAMILIA_HGB, "candidate"),
        params={
            "modelo": "HistGradientBoostingClassifier",
            "class_weight": "balanced",
            "features": ",".join(fp.FEATURES),
        }
        | params_hgb,
        ajustador=lambda df: ajustar(df, params_hgb, semilla),
        extras=extras_hgb,
    )
    salida["random_forest"] = correr_candidata(
        **common,
        clave="random_forest",
        nombre_run=f"p2_random_forest_balanceado_{sufijo}",
        etiquetas=etiquetas_modelo("random_forest_balanceado", FAMILIA_BOSQUE, "candidate"),
        params={"modelo": "RandomForestClassifier", "class_weight": "balanced_subsample"}
        | CONFIG_P2.params_random_forest,
        ajustador=lambda df: ajustar_random_forest(df, CONFIG_P2.params_random_forest, semilla),
    )
    if sin_pesos:
        # misma configuracion sin pesos de clase, para ver el efecto del balanceo
        salida["hgb_sin_pesos"] = correr_candidata(
            **common,
            clave="hgb_sin_pesos",
            nombre_run=f"p2_hgb_sin_pesos_{sufijo}",
            etiquetas=etiquetas_modelo(
                "hist_gradient_boosting_sin_pesos", FAMILIA_HGB, "candidate"
            ),
            params={"modelo": "HistGradientBoostingClassifier", "class_weight": "ninguno"}
            | params_hgb,
            ajustador=lambda df: ajustar(df, params_hgb, semilla, balanceado=False),
            con_nuevas=False,
        )
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
            "objeto JSON con hiperparametros del HGB (C2). Se mezcla con la primera configuracion de "
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

    nombre_busqueda = f"p2_busqueda_{sufijo_nivel(args.niveles[0])}"
    tags_busqueda = tags_base | etiquetas_modelo(
        "hist_gradient_boosting_busqueda", FAMILIA_HGB, "search"
    )
    if hyperparameters:
        # sin busqueda: se evalua en validacion solo esta configuracion, para registrar su
        # val_macro_f1 y dejar la rejilla de un solo elemento
        configuracion = first_grid | hyperparameters
        params_hgb, rejilla = afinar(
            preparar(args.niveles[0], panel),
            args.semilla,
            [configuracion],
            nombre_padre=nombre_busqueda,
            tags=tags_busqueda,
        )
    elif args.sin_afinar:
        params_hgb, rejilla = CONFIG_P2.rejilla[0], []
    else:
        params_hgb, rejilla = afinar(
            preparar(args.niveles[0], panel),
            args.semilla,
            nombre_padre=nombre_busqueda,
            tags=tags_busqueda,
        )
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
