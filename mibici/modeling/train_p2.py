"""Experimento de P2 (clasificacion de riesgo por estacion) registrado en MLflow.

Ejemplo:
    python -m mibici.modeling.train_p2 --niveles 0.5 0.3 0.7

Cada candidata se registra con su modelo (sklearn, formato skops), firma y ejemplo de entrada.
Para predecir, un consumidor necesita:
- Las columnas de fp.FEATURES, en ese orden (estan en la firma del modelo y en
  config/config_p2.json del run). hora, dia_semana y fin_semana son del instante objetivo (t+1).
- Las tres ultimas, est_tasa_sin_bicis, est_tasa_sin_anclajes y est_neto_medio_hora, salen de
  las estadisticas por estacion y por estacion y hora calculadas con train, que el run guarda en
  data/stats_estacion.csv y data/stats_estacion_hora.csv; se unen con fp.unir_stats.
- Las estaciones que no estaban en train quedan sin esas tres columnas (valores faltantes).
La salida es la clase de riesgo (segura, sin_bicicletas o sin_anclajes); las probabilidades
estan en el modelo sklearn (mlflow.sklearn.load_model y predict_proba, columnas en classes_).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from importlib.metadata import version
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
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import skops.io as sio

from mibici import features_p2 as fp
from mibici import features_pipeline, registro
from mibici.config import CONFIG, CONFIG_P2, PROJ_ROOT

EXPERIMENTO = CONFIG_P2.experimento
RIESGO = ["sin_bicicletas", "sin_anclajes"]
FAMILIA_HGB = "hist_gradient_boosting"
FAMILIA_LOGISTICA = "logistic_regression"
FAMILIA_BOSQUE = "random_forest"
CATEGORICAS = ["hora", "dia_semana"]  # enteros que la logistica codifica uno a uno
MUESTRA_DATASET = 2000  # filas de muestra de cada dataset registrado con log_input
PREFIJOS_CONFIABLES = ("sklearn.", "numpy.")  # unicos tipos que skops puede cargar sin avisar
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


def figura_pr(
    y_true: ArrayLike, proba: np.ndarray, clases: list[str], titulo: str, ruta: Path
) -> Path:
    """Curvas precision-recall de las clases de riesgo (una contra el resto) guardadas en ruta.

    Una clase de riesgo que no aparece en y_true no tiene curva y se omite, como en metricas.
    """
    y_true = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in RIESGO:
        y_bin = (y_true == c).astype(int)
        if not y_bin.sum():
            continue
        columna = proba[:, clases.index(c)]
        precision, recall, _ = precision_recall_curve(y_bin, columna)
        ap = average_precision_score(y_bin, columna)
        ax.plot(recall, precision, label=f"{c} (PR AUC {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(titulo)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    return ruta


def heatmap_error(
    test: pd.DataFrame, y_pred: ArrayLike, volumen: pd.Series, ruta: Path, top_n: int
) -> tuple[Path, pd.DataFrame]:
    """Tasa de error por estacion y hora objetivo en prueba.

    Guarda en ruta la figura de las top_n estaciones con mas volumen (salidas mas llegadas),
    ordenadas por error medio, y junto a ella el CSV con la matriz de todas las estaciones.
    """
    errores = test[["estacion", "hora"]].assign(error=np.asarray(y_pred) != test["objetivo"])
    matriz = errores.pivot_table(index="estacion", columns="hora", values="error", aggfunc="mean")
    matriz.to_csv(ruta.with_suffix(".csv"))
    top = volumen[volumen.index.isin(matriz.index)].nlargest(top_n).index
    vista = matriz.loc[top]
    vista = vista.loc[vista.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(vista.to_numpy(), aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(vista.shape[1]), vista.columns)
    ax.set_yticks(range(vista.shape[0]), vista.index)
    ax.set_xlabel("Hora objetivo")
    ax.set_ylabel("Estacion")
    ax.set_title(f"Tasa de error en prueba, top {len(vista)} estaciones por volumen")
    fig.colorbar(im, ax=ax, label="Tasa de error")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    return ruta, vista


def volumen_estacion(df: pd.DataFrame) -> pd.Series:
    """Salidas mas llegadas por estacion."""
    return (df["salidas"] + df["llegadas"]).groupby(df["estacion"]).sum()


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
    numericas_pipeline = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
    )
    modelo = features_pipeline.make_tabular_pipeline(
        LogisticRegression(class_weight="balanced", random_state=semilla, **params),
        numerical_features=numericas,
        categorical_features=CATEGORICAS,
        numerical_transformer=numericas_pipeline,
        categorical_transformer=OneHotEncoder(handle_unknown="ignore"),
    )
    return modelo.fit(train[fp.FEATURES], train["objetivo"])


def ajustar_random_forest(train: pd.DataFrame, params: dict, semilla: int) -> Pipeline:
    """RandomForest balanceado por arbol; max_samples acota el costo y se recorta a len(train)."""
    params = {**params, "max_samples": min(params["max_samples"], len(train))}
    modelo = features_pipeline.make_tabular_pipeline(
        RandomForestClassifier(
            class_weight="balanced_subsample", n_jobs=-1, random_state=semilla, **params
        ),
        numerical_features=fp.FEATURES,
        numerical_transformer=SimpleImputer(strategy="median", keep_empty_features=True),
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
            mlflow.log_dict(resultados, "config/p2_rejilla_validacion.json")
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
    artefactos: list[tuple[Path, str]],
    anidado: bool = False,
    dentro: Callable[[], None] | None = None,
) -> str:
    """Crea un run en MLflow con nombre descriptivo, etiquetas, parametros, metricas y artefactos.

    artefactos son pares (archivo, subcarpeta del run). Con anidado el run cuelga del run activo
    (por ejemplo, el padre de una busqueda). dentro, si se da, se ejecuta con el run abierto
    (para registrar el modelo, los datasets y otros objetos). Regresa el id del run.
    """
    with mlflow.start_run(run_name=nombre, nested=anidado) as run:
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        mlflow.log_metrics(mets)
        for ruta, carpeta in artefactos:
            mlflow.log_artifact(str(ruta), carpeta)
        if dentro:
            dentro()
    return run.info.run_id


def tipos_confiables(modelo) -> list[str]:
    """Tipos que skops marca como no confiables en el modelo, solo si son de sklearn o numpy.

    Se pasan a log_model como skops_trusted_types para que el modelo se pueda cargar. Si hay
    un tipo de otro paquete se rechaza en lugar de confiar en el sin revisarlo.
    """
    tipos = sio.get_untrusted_types(data=sio.dumps(modelo))
    ajenos = [t for t in tipos if not t.startswith(PREFIJOS_CONFIABLES)]
    if ajenos:
        raise ValueError(f"Tipos no confiables en el modelo: {ajenos}")
    return sorted(tipos)


def requisitos_modelo() -> list[str]:
    """Versiones instaladas de lo que necesita cargar el modelo (evita inferirlas en cada log)."""
    paquetes = ("scikit-learn", "pandas", "numpy", "skops", "cloudpickle")
    return [f"{p}=={version(p)}" for p in paquetes]


def registrar_modelo(modelo, ejemplo: pd.DataFrame):
    """Guarda el modelo con firma (features de entrada y clase_riesgo de salida) y ejemplo.

    Regresa la informacion del modelo registrado (incluye model_uri).
    """
    entrada = ejemplo[fp.FEATURES].astype("float64")  # enteros y flotantes llegan como double
    salida = pd.Series(modelo.predict(entrada), name="clase_riesgo")
    return mlflow.sklearn.log_model(
        modelo,
        name="model",
        signature=infer_signature(entrada, salida),
        input_example=entrada,
        serialization_format="skops",
        skops_trusted_types=tipos_confiables(modelo),
        pip_requirements=requisitos_modelo(),
    )


def registrar_datasets(ctx: ContextoNivel, semilla: int) -> None:
    """Registra train, val y test como entradas del run con una muestra pequena de cada uno.

    log_input guarda esquema, perfil y digest de la muestra, no los datos; n_filas lleva el
    tamano real de cada particion y data_version la version del manifiesto.
    """
    for contexto, df in (("training", ctx.train), ("validation", ctx.val), ("testing", ctx.test)):
        muestra = df[[*fp.FEATURES, "objetivo"]].sample(
            min(MUESTRA_DATASET, len(df)), random_state=semilla
        )
        dataset = mlflow.data.from_pandas(
            muestra, targets="objetivo", name=f"p2_{contexto}_{ctx.sufijo}"
        )
        mlflow.log_input(
            dataset,
            context=contexto,
            tags={"data_version": ctx.data_version, "n_filas": str(len(df))},
        )


def config_run(ctx: ContextoNivel, params_modelo: dict) -> dict:
    """Configuracion del run para config/config_p2.json: cortes, umbrales, features y modelo."""
    return {
        "question": CONFIG_P2.question,
        "nivel_inicial": ctx.params_comunes["nivel_inicial"],
        "cortes": dict(CORTES),
        "umbral_bajo": CONFIG_P2.umbral_bajo,
        "umbral_alto": CONFIG_P2.umbral_alto,
        "horizonte_horas": CONFIG_P2.horizonte_horas,
        "horas_objetivo": list(CONFIG_P2.horas_etiqueta),
        "semilla": ctx.params_comunes["semilla"],
        "data_version": ctx.data_version,
        "features": fp.FEATURES,
        "clases": fp.CLASES,
        "modelo": params_modelo,
    }


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
    data_version: str
    ruta_manifiesto: Path
    ruta_resumen: Path
    stats_estacion: pd.DataFrame
    stats_hora: pd.DataFrame
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
        data_version=manifiesto["version"],
        ruta_manifiesto=Path(args.manifiesto),
        ruta_resumen=Path(args.panel).with_suffix(".json"),
        stats_estacion=est_train,
        stats_hora=hora_train,
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
            figura = figura_confusion(
                ctx.test["objetivo"],
                pred_test[clave],
                f"{titulo} (inicio {ctx.sufijo})",
                ctx.figs / archivo,
            )
            artefactos.append((figura, "plots"))
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
    semilla: int,
    con_nuevas: bool = True,
    extras: Callable[[object], list[tuple[Path, str]]] | None = None,
    run_ids: dict | None = None,
) -> dict[str, float]:
    """Ajusta una candidata dos veces y registra su run con validacion, prueba y costo.

    Con fit se ajusta para las metricas val_*; con train completo, para las test_* y el costo
    (fit_seconds y model_size_mb son los de este segundo ajuste). mejor_baseline es el nombre y
    el macro F1 de prueba de la mejor baseline por validacion, para medir la mejora.
    extras(modelo) regresa pares (archivo, subcarpeta) adicionales a registrar como artefactos.
    El run guarda el modelo con firma y ejemplo, los datasets de entrada, las estadisticas por
    estacion y hora que el modelo necesita (data/), la configuracion (config/) y las graficas
    (plots/). Si se da run_ids, se llena run_ids[clave] con el id del run y el uri del modelo.
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
    sufijo, test, titulo = ctx.sufijo, ctx.test["objetivo"], etiquetas["modelo"]
    figuras = [
        figura_confusion(
            test,
            pred,
            f"{titulo} test (inicio {sufijo})",
            ctx.figs / f"p2_cm_{clave}_{sufijo}.png",
        )
    ]
    if con_nuevas:
        figuras.append(
            figura_confusion(
                test[ctx.nuevas],
                pred[ctx.nuevas.to_numpy()],
                f"{titulo} estaciones nuevas (inicio {sufijo})",
                ctx.figs / f"p2_cm_{clave}_nuevas_{sufijo}.png",
            )
        )
    figuras.append(
        figura_pr(
            test,
            proba,
            clases,
            f"Precision-recall de riesgo, {titulo} (inicio {sufijo})",
            ctx.figs / f"p2_pr_{clave}_{sufijo}.png",
        )
    )
    artefactos = [(f, "plots") for f in figuras]
    artefactos += [(ctx.ruta_manifiesto, "data"), (ctx.ruta_resumen, "data")]
    if extras:
        artefactos += extras(modelo)
    ejemplo = ctx.train.sample(5, random_state=semilla)
    salida = {}

    def dentro_del_run() -> None:
        info = registrar_modelo(modelo, ejemplo)
        salida["model_uri"] = info.model_uri
        registrar_datasets(ctx, semilla)
        # estadisticas de train: con ellas un consumidor rearma est_* con fp.unir_stats
        mlflow.log_text(ctx.stats_estacion.to_csv(index=False), "data/stats_estacion.csv")
        mlflow.log_text(ctx.stats_hora.to_csv(index=False), "data/stats_estacion_hora.csv")
        mlflow.log_dict(config_run(ctx, params), "config/config_p2.json")

    salida["run_id"] = registrar(
        nombre_run,
        ctx.tags_base | etiquetas,
        ctx.params_comunes | params | {"mejor_baseline": mejor_baseline[0]},
        m | ctx.dist,
        artefactos,
        dentro=dentro_del_run,
    )
    if run_ids is not None:
        run_ids[clave] = salida
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
    run_ids: dict | None = None,
) -> dict[str, dict[str, float]]:
    """Corre baselines y candidatas para un nivel inicial y registra cada modelo como run.

    Candidatas: C1 regresion logistica balanceada, C2 HGB balanceado con los hiperparametros
    de la busqueda y C3 RandomForest balanceado. Con sin_pesos se agrega el HGB sin pesos de
    clase para ver el efecto del balanceo. Los parametros de C1 y C3 salen de CONFIG_P2.
    Si se da run_ids, se llena con el id del run y el uri del modelo de cada candidata.
    """
    ctx = preparar_nivel(nivel, panel, args, manifiesto, resumen, tags_base)
    salida = correr_baselines(ctx)
    # mejor baseline por macro F1 de validacion; su macro F1 de prueba es la referencia de mejora
    nombre_mejor = max(salida, key=lambda n: salida[n]["val_macro_f1"])
    mejor = (nombre_mejor.removeprefix("baseline_"), salida[nombre_mejor]["test_macro_f1"])
    semilla, sufijo = args.semilla, ctx.sufijo
    common = {"ctx": ctx, "mejor_baseline": mejor, "semilla": semilla, "run_ids": run_ids}

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

    def extras_hgb(modelo) -> list[tuple[Path, str]]:
        ruta_csv, ruta_png = guardar_importancia(modelo, ctx.test, semilla, ctx.figs, sufijo)
        archivos = [(ruta_png, "plots"), (ruta_csv, "data")]
        if rejilla:
            ruta_rej = ctx.figs / "p2_rejilla_validacion.json"
            ruta_rej.write_text(json.dumps(rejilla, indent=2))
            archivos.append((ruta_rej, "config"))
        if nivel == CONFIG_P2.nivel_inicial:
            _, _, pred = predecir(modelo, ctx.test)
            ruta_heat, _ = heatmap_error(
                ctx.test,
                pred,
                volumen_estacion(ctx.train),
                ctx.figs / f"p2_heatmap_error_{sufijo}.png",
                CONFIG_P2.heatmap_top_estaciones,
            )
            archivos += [(ruta_heat, "plots"), (ruta_heat.with_suffix(".csv"), "data")]
        return archivos

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


def registrar_ganadora(args: argparse.Namespace, todo: dict, run_ids: dict) -> dict:
    """Elige al campeon con el primer nivel, lo registra con alias y comprueba que se carga.

    Regresa un resumen (clave, run, version, promedios del criterio y ranking) para el JSON.
    """
    nivel = str(args.niveles[0])
    campeon = registro.seleccionar_campeon(todo[nivel], CONFIG_P2.candidatas_campeon)
    origen = run_ids[nivel][campeon["clave"]]
    ejemplo = pd.DataFrame([dict.fromkeys(fp.FEATURES, 0.0)])
    version = registro.registrar_campeon(
        CONFIG_P2.modelo_registrado,
        CONFIG_P2.alias_campeon,
        origen["run_id"],
        origen["model_uri"],
        campeon,
        {"experimento": CONFIG_P2.experimento, "nivel_inicial": nivel},
        ejemplo,
    )
    # prueba de humo: carga por alias y predice una fila de ejemplo
    clases = registro.verificar_campeon(
        CONFIG_P2.modelo_registrado, CONFIG_P2.alias_campeon, ejemplo
    )
    logger.info(
        "Campeon {} registrado como {} v{} (alias {}), prediccion de ejemplo {}",
        campeon["clave"],
        CONFIG_P2.modelo_registrado,
        version.version,
        CONFIG_P2.alias_campeon,
        clases,
    )
    return {
        "clave": campeon["clave"],
        "nivel_referencia": nivel,
        "run_id": origen["run_id"],
        "model_uri": origen["model_uri"],
        "modelo_registrado": CONFIG_P2.modelo_registrado,
        "alias": CONFIG_P2.alias_campeon,
        "version": int(version.version),
        "criterio": registro.CRITERIO_CAMPEON,
        "conjunto": campeon["conjunto"],
        "ranking": campeon["ranking"],
    }


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
    ap.add_argument(
        "--registrar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "registra al campeon en el Model Registry (CONFIG_P2.modelo_registrado, alias "
            "CONFIG_P2.alias_campeon) con la regla de registro.py sobre el primer nivel; "
            "--no-registrar lo omite"
        ),
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

    todo, run_ids = {}, {}
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
            run_ids=run_ids.setdefault(str(nivel), {}),
        )
    salida = {"params_hgb": params_hgb, "rejilla": rejilla, "resultados": todo}
    if args.registrar:
        salida["campeon"] = registrar_ganadora(args, todo, run_ids)
    ruta = Path(args.figuras) / "p2_resultados.json"
    ruta.write_text(json.dumps(salida, indent=2))
    logger.info("Resultados en {}", ruta)


if __name__ == "__main__":
    main()
