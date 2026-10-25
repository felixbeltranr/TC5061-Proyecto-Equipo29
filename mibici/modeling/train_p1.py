"""Experimento base de P1 (demanda diaria de salidas por estacion) registrado en MLflow.

Ejemplo:
    python -m mibici.features
    python -m mibici.modeling.train_p1

Baselines: misma estacion y mismo dia de la semana previa (D-7) y persistencia del ultimo
dia completo (D-2). Candidatos: regresion de Poisson regularizada, gradient boosting con
perdida de Poisson y gradient boosting con perdida absoluta (misma rejilla que el de Poisson,
solo cambia la perdida). Los hiperparametros y el modelo seleccionado se eligen solo con la
validacion; la prueba se evalua una vez, con el modelo reentrenado con todas las filas
elegibles de train, hasta el ultimo objetivo disponible al iniciar la prueba.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from importlib.metadata import version
import json
from pathlib import Path
import time
import warnings

from loguru import logger
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
import skops.io as sio

from mibici import features as fe
from mibici import features_pipeline, registro_p1, tracking
from mibici.config import CONFIG, CONFIG_P1

# Cortes temporales de CONFIG_P1 (los mismos meses que train_p2.CORTES)
CORTES = {
    "train_inicio": CONFIG_P1.train_inicio,
    "val_inicio": CONFIG_P1.val_inicio,
    "test_inicio": CONFIG_P1.test_inicio,
    "test_fin": CONFIG_P1.test_fin,
}
CONTEOS = [
    "lag_2",
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
    "media_7d",
    "media_28d",
    "media_misma_dsem_4",
]
BINARIAS = ["festivo", "festivo_lag7"]
ALPHAS_POISSON = list(CONFIG_P1.alphas_poisson)
REJILLA_HGB = [dict(c) for c in CONFIG_P1.rejilla_hgb]
SEGMENTOS = ("baja", "media", "alta", "nuevas")
CANDIDATOS = ("poisson", "hgb_poisson", "hgb_absoluto")
# Una demanda diaria no puede ser negativa. La elegibilidad se decide exclusivamente con
# predicciones de validacion; no se recortan predicciones ni se reajusta ningun modelo.
POLITICA_DOMINIO = "predicciones de validacion numericas, finitas y mayores o iguales a cero"
# Perdida de cada candidato HGB. La absoluta estima la mediana condicional y optimiza
# directamente el MAE con el que se elige; la de Poisson estima la media de un conteo.
PERDIDAS_HGB = {"hgb_poisson": "poisson", "hgb_absoluto": "absolute_error"}
FEATURES_BASELINE = {"semana_previa": ["lag_7"], "persistencia": ["lag_2"]}
FAMILIAS = {
    "semana_previa": "baseline_semana_previa",
    "persistencia": "baseline_persistencia",
    "poisson": "poisson_regression",
    "hgb_poisson": "hist_gradient_boosting",
    "hgb_absoluto": "hist_gradient_boosting",
}
MUESTRA_DATASET = 2000
PREFIJOS_CONFIABLES = ("sklearn.", "numpy.")
# HGB guarda internamente un functools.partial de sklearn.utils.validation.check_array
TIPOS_CONFIABLES = {"functools.partial"}
# modelo -> (nombre del run en MLflow, etiqueta del modelo)
NOMBRES_RUN = {
    "semana_previa": ("p1_baseline_semana_previa", "baseline_semana_previa"),
    "persistencia": ("p1_baseline_persistencia", "baseline_persistencia"),
    "poisson": ("p1_poisson_regularizado", "poisson_regularizado"),
    "hgb_poisson": ("p1_hgb_poisson", "hist_gradient_boosting_poisson"),
    "hgb_absoluto": ("p1_hgb_absoluto", "hist_gradient_boosting_absoluto"),
}


def etiquetas_modelo(modelo: str, familia: str, etapa: str) -> dict[str, str]:
    """Etiquetas para distinguir baseline, busqueda y candidata final."""
    return {"modelo": modelo, "model_family": familia, "stage_of_work": etapa}


def tipos_confiables(modelo) -> list[str]:
    """Tipos que skops puede cargar; solo se aceptan implementaciones sklearn/numpy."""
    tipos = sio.get_untrusted_types(data=sio.dumps(modelo))
    ajenos = [
        tipo
        for tipo in tipos
        if tipo not in TIPOS_CONFIABLES and not tipo.startswith(PREFIJOS_CONFIABLES)
    ]
    if ajenos:
        raise ValueError(f"Tipos no confiables en el modelo: {ajenos}")
    return sorted(tipos)


def requisitos_modelo() -> list[str]:
    """Dependencias exactas necesarias para cargar los modelos P1."""
    paquetes = ("scikit-learn", "pandas", "numpy", "skops", "cloudpickle")
    return [f"{paquete}=={version(paquete)}" for paquete in paquetes]


def registrar_modelo(modelo, ejemplo: pd.DataFrame):
    """Guarda un candidato con skops, firma de regresion y ejemplo de entrada."""
    # MLflow aplica la firma al cargar el pyfunc. Declarar las variables numericas como double
    # permite que un consumidor mande enteros o valores faltantes sin una firma fragil.
    entrada = ejemplo[fe.FEATURES].astype("float64")
    salida = pd.Series(np.asarray(modelo.predict(entrada), dtype=float), name="demanda_salidas")
    return mlflow.sklearn.log_model(
        modelo,
        name="model",
        signature=infer_signature(entrada, salida),
        input_example=entrada,
        serialization_format="skops",
        skops_trusted_types=tipos_confiables(modelo),
        pip_requirements=requisitos_modelo(),
    )


def registrar_datasets(partes: dict[str, pd.DataFrame], semilla: int, data_version: str) -> None:
    """Registra muestras de fit/validacion/train/prueba como entradas, no como artefactos."""
    contextos = {
        "fit": "training",
        "val": "validation",
        "train": "training-final",
        "test": "testing",
    }
    for nombre, contexto in contextos.items():
        df = partes[nombre]
        muestra = df[[*fe.FEATURES, "objetivo"]].sample(
            min(MUESTRA_DATASET, len(df)), random_state=semilla
        )
        dataset = mlflow.data.from_pandas(muestra, targets="objetivo", name=f"p1_{nombre}")
        mlflow.log_input(
            dataset,
            context=contexto,
            tags={"data_version": data_version, "n_filas": str(len(df))},
        )


def metricas(y_real: np.ndarray, y_pred: np.ndarray, prefijo: str) -> dict[str, float]:
    """MAE, RMSE, sesgo (pronostico menos real) y WAPE; prefijo indica el conjunto.

    WAPE es la suma de errores absolutos entre la suma de salidas reales (razon, no %).
    """
    y = np.asarray(y_real, dtype=float)
    err = np.asarray(y_pred, dtype=float) - y
    m = {
        f"{prefijo}_mae": np.abs(err).mean(),
        f"{prefijo}_rmse": np.sqrt((err**2).mean()),
        f"{prefijo}_bias": err.mean(),
        f"{prefijo}_n": len(y),
    }
    if y.sum() > 0:
        m[f"{prefijo}_wape"] = np.abs(err).sum() / y.sum()
    return {k: float(v) for k, v in m.items()}


def segmentos(train: pd.DataFrame, filas: pd.DataFrame) -> pd.Series:
    """Segmento de volumen de cada fila: terciles del promedio diario en train, o nuevas.

    Los terciles se calculan solo con train; las estaciones sin filas en train son nuevas.
    """
    nivel = train.groupby("estacion")["objetivo"].mean()
    corte_bajo, corte_alto = nivel.quantile([1 / 3, 2 / 3])
    n = filas["estacion"].map(nivel)
    etiqueta = np.select(
        [n.isna(), n <= corte_bajo, n <= corte_alto], ["nuevas", "baja", "media"], "alta"
    )
    return pd.Series(etiqueta, index=filas.index)


def metricas_segmentos(
    filas: pd.DataFrame, pred: np.ndarray, segmento: pd.Series, prefijo: str
) -> dict[str, float]:
    """MAE y WAPE por segmento de volumen, para ver lo que el promedio esconde."""
    salida: dict[str, float] = {}
    for s in SEGMENTOS:
        mascara = (segmento == s).to_numpy()
        if mascara.any():
            m = metricas(filas["objetivo"].to_numpy()[mascara], pred[mascara], prefijo)
            salida[f"{prefijo}_mae_{s}"] = m[f"{prefijo}_mae"]
            if f"{prefijo}_wape" in m:
                salida[f"{prefijo}_wape_{s}"] = m[f"{prefijo}_wape"]
            salida[f"{prefijo}_n_{s}"] = float(mascara.sum())
    return salida


def metricas_mensuales(filas: pd.DataFrame, pred: np.ndarray, prefijo: str) -> dict[str, float]:
    """MAE de cada mes calendario (``<prefijo>_mae_AAAA_MM``) y su desviacion entre meses.

    Es un diagnostico de estabilidad del modelo ya elegido: se calcula despues de la seleccion
    y no interviene en ella ni en la eleccion de hiperparametros.
    """
    error = np.abs(np.asarray(pred, dtype=float) - filas["objetivo"].to_numpy(dtype=float))
    mes = pd.to_datetime(filas["fecha"]).dt.strftime("%Y_%m").to_numpy()
    por_mes = pd.Series(error).groupby(mes).mean()
    salida = {f"{prefijo}_mae_{m}": float(v) for m, v in por_mes.items()}
    salida[f"{prefijo}_mae_mensual_desv"] = float(por_mes.std(ddof=0))
    return salida


def particionar(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Cortes temporales por fecha objetivo: fit, val (ajuste), train completo y test.

    Un modelo que se usa desde el periodo que empieza en S solo puede entrenar con dias
    cuyo conteo ya existe al emitir el primer pronostico de S, es decir hasta S menos
    LAG_MINIMO dias; ese es el embargo que evita usar etiquetas todavia no disponibles.
    """
    c = {k: pd.Timestamp(v) for k, v in CORTES.items()}
    f = df["fecha"]
    ultimo_fit = fe.ultimo_dia_conocido(c["val_inicio"])
    ultimo_train = fe.ultimo_dia_conocido(c["test_inicio"])
    partes = {
        "fit": df[(f >= c["train_inicio"]) & (f <= ultimo_fit)],
        # La seleccion tambien debe usar etiquetas disponibles al emitir el primer test.
        "val": df[(f >= c["val_inicio"]) & (f <= ultimo_train)],
        "train": df[(f >= c["train_inicio"]) & (f <= ultimo_train)],
        "test": df[(f >= c["test_inicio"]) & (f < c["test_fin"])],
    }
    vacias = [k for k, v in partes.items() if v.empty]
    if vacias:
        raise ValueError(f"Particiones sin filas elegibles: {vacias}; revisa el rango del panel")
    return partes


def pipeline_poisson(alpha: float) -> Pipeline:
    """Regresion de Poisson (enlace log) sobre conteos en log1p, dia de la semana y festivos."""
    conteos = Pipeline(
        [
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("escala", StandardScaler()),
        ]
    )
    return features_pipeline.make_tabular_pipeline(
        PoissonRegressor(alpha=alpha, max_iter=2000),
        numerical_features=CONTEOS,
        categorical_features=["dia_semana"],
        passthrough_features=BINARIAS,
        numerical_transformer=conteos,
        categorical_transformer=OneHotEncoder(categories=[list(range(7))], drop="first"),
    )


def ajustar_poisson(entrena: pd.DataFrame, alpha: float) -> Pipeline:
    """Entrena el pipeline de Poisson; falla si el optimizador no converge."""
    modelo = pipeline_poisson(alpha)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        modelo.fit(entrena[fe.FEATURES], entrena["objetivo"])
    return modelo


def ajustar_hgb(
    entrena: pd.DataFrame, params: dict, semilla: int, perdida: str = "poisson"
) -> HistGradientBoostingRegressor:
    """Entrena gradient boosting con la perdida indicada (el dia de la semana va como categoria)."""
    modelo = HistGradientBoostingRegressor(
        loss=perdida,
        categorical_features=["dia_semana"],
        early_stopping=False,
        random_state=semilla,
        **params,
    )
    modelo.fit(entrena[fe.FEATURES], entrena["objetivo"])
    return modelo


def ajustar_candidato(candidato: str, entrena: pd.DataFrame, params: dict, semilla: int):
    """Entrena un candidato de P1 con sus hiperparametros (mismo camino en afinar y en finales)."""
    if candidato == "poisson":
        return ajustar_poisson(entrena, params["alpha"])
    # posicional: las pruebas espian ajustar_hgb con la firma (entrena, *args)
    return ajustar_hgb(entrena, params, semilla, PERDIDAS_HGB[candidato])


def mae(y_real: pd.Series, y_pred: np.ndarray) -> float:
    """Error absoluto medio."""
    return float(np.abs(np.asarray(y_pred, dtype=float) - y_real.to_numpy()).mean())


def afinar(
    partes: dict,
    semilla: int,
    completo: bool = True,
    model: str = CONFIG_P1.model,
    hyperparameters: dict | None = None,
    tags_base: dict[str, str] | None = None,
) -> tuple[dict, dict, dict]:
    """Elige alpha de Poisson y parametros de cada HGB por MAE de validacion, solo con fit.

    Los dos HGB recorren la misma rejilla, asi que entre ellos solo cambia la perdida.
    Devuelve los parametros elegidos por candidato, la rejilla completa con su val_mae y las
    predicciones de validacion del mejor de cada candidato.
    """
    fit, val = partes["fit"], partes["val"]
    rejilla_hgb = REJILLA_HGB if completo else REJILLA_HGB[:1]
    configs = {
        "poisson": [{"alpha": a} for a in (ALPHAS_POISSON if completo else ALPHAS_POISSON[:1])],
        "hgb_poisson": rejilla_hgb,
        "hgb_absoluto": rejilla_hgb,
    }
    if model not in ("auto", *CANDIDATOS):
        raise ValueError(f"Unsupported P1 model: {model}")
    hyperparameters = hyperparameters or {}
    if hyperparameters and model == "auto":
        raise ValueError(
            "Choose 'poisson', 'hgb_poisson' or 'hgb_absoluto' to override hyperparameters"
        )
    params_hgb = {"learning_rate", "max_leaf_nodes", "max_iter", "l2_regularization"}
    allowed = {"poisson": {"alpha"}, "hgb_poisson": params_hgb, "hgb_absoluto": params_hgb}
    if hyperparameters and not set(hyperparameters).issubset(allowed[model]):
        invalid = sorted(set(hyperparameters).difference(allowed[model]))
        raise ValueError(f"Unsupported {model} hyperparameters: {invalid}")
    if model != "auto":
        base = configs[model][0]
        configs = {model: [{**base, **hyperparameters}]}

    rejilla: dict[str, list[dict]] = {}
    elegidos: dict[str, dict] = {}
    pred_val: dict[str, np.ndarray] = {}
    for candidato, opciones in configs.items():
        nombre_padre = f"p1_busqueda_{candidato}"
        contexto = (
            mlflow.start_run(run_name=nombre_padre) if tags_base is not None else nullcontext()
        )
        with contexto:
            tags_busqueda = etiquetas_modelo(candidato, FAMILIAS[candidato], "search")
            if tags_base is not None:
                mlflow.set_tags(tags_base | tags_busqueda)
                mlflow.log_params(
                    {
                        "candidato": candidato,
                        "n_combinaciones": len(opciones),
                        "semilla": semilla,
                        "features": ",".join(fe.FEATURES),
                    }
                )
            resultados, predicciones = [], []
            for indice, params in enumerate(opciones, start=1):
                inicio = time.perf_counter()
                modelo = ajustar_candidato(candidato, fit, params, semilla)
                segundos = time.perf_counter() - inicio
                pred = modelo.predict(val[fe.FEATURES])
                resultado = {
                    **params,
                    "val_mae": mae(val["objetivo"], pred),
                }
                resultados.append(resultado)
                predicciones.append(pred)
                if tags_base is not None:
                    tracking.registrar(
                        f"{nombre_padre}_c{indice}",
                        tags_base | tags_busqueda,
                        {"combinacion": indice, "semilla": semilla} | params,
                        {"val_mae": resultado["val_mae"], "fit_seconds": segundos},
                        [],
                        anidado=True,
                    )
            mejor = min(range(len(resultados)), key=lambda i: resultados[i]["val_mae"])
            rejilla[candidato] = resultados
            elegidos[candidato] = opciones[mejor]
            pred_val[candidato] = predicciones[mejor]
            if tags_base is not None:
                mlflow.log_params({f"mejor_{k}": v for k, v in opciones[mejor].items()})
                mlflow.log_metric("val_mae", resultados[mejor]["val_mae"])
                mlflow.log_dict(resultados, f"config/p1_rejilla_{candidato}.json")
            logger.info("Rejilla {}: elegido {}", candidato, resultados[mejor])
    return elegidos, rejilla, pred_val


def diagnostico_dominio(predicciones: np.ndarray) -> dict[str, float | int | bool | str | None]:
    """Evalua en validacion el dominio de demanda sin alterar las predicciones.

    Una candidata sigue apareciendo en runs, metricas y artefactos aunque resulte inelegible.
    Esta funcion no recorta negativos ni consulta el conjunto de prueba.
    """
    valores = np.asarray(predicciones, dtype=float).reshape(-1)
    finitos = np.isfinite(valores)
    negativos = finitos & (valores < 0)
    minimo = float(valores[finitos].min()) if finitos.any() else None
    motivos = []
    if not len(valores):
        motivos.append("sin_predicciones_validacion")
    if (~finitos).any():
        motivos.append("predicciones_no_finitas_validacion")
    if negativos.any():
        motivos.append("predicciones_negativas_validacion")
    return {
        "eligible": not motivos,
        "reason": "eligible" if not motivos else ",".join(motivos),
        "prediction_min": minimo,
        "negative_n": int(negativos.sum()),
        "nonfinite_n": int((~finitos).sum()),
        "n": len(valores),
    }


def seleccionar(val_mae: dict[str, float], elegibilidad: dict[str, bool]) -> str:
    """Elige por MAE de validacion entre baseline y candidatas de dominio elegible."""
    mejor = "semana_previa"
    for candidato in CANDIDATOS:
        if (
            candidato in val_mae
            and elegibilidad.get(candidato, False)
            and val_mae[candidato] < val_mae[mejor]
        ):
            mejor = candidato
    return mejor


def rango_meses(inicio: str, fin_exclusivo: str) -> str:
    """Texto AAAA-MM..AAAA-MM de un periodo [inicio, fin_exclusivo)."""
    ultimo = pd.Timestamp(fin_exclusivo) - pd.Timedelta(days=1)
    return f"{inicio[:7]}..{ultimo:%Y-%m}"


def figura_agregado(filas: pd.DataFrame, pred: np.ndarray, titulo: str, ruta: Path) -> Path:
    """Guarda el total diario del sistema (real contra pronostico) y regresa la ruta."""
    diario = filas.assign(pronostico=pred).groupby("fecha")[["objetivo", "pronostico"]].sum()
    fig = Figure(figsize=(10, 4))
    ax = fig.subplots()
    ax.plot(diario.index, diario["objetivo"], label="real", lw=1)
    ax.plot(diario.index, diario["pronostico"], label="pronostico", lw=1)
    ax.axvline(pd.Timestamp(CORTES["test_inicio"]), color="gray", ls="--", lw=1)
    ax.set_ylabel("Salidas por dia (suma de estaciones elegibles)")
    ax.set_title(titulo)
    ax.legend()
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=120)
    return ruta


# Colores de la grafica de residuales: rampa secuencial de un solo tono para la densidad,
# barras en el mismo tono y lineas de referencia en tinta neutra.
RAMPA_DENSIDAD = ("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
COLOR_BARRAS = "#2a78d6"
TINTA, TINTA_SECUNDARIA, FONDO = "#0b0b0b", "#52514e", "#fcfcfb"


def figura_residuales(filas: pd.DataFrame, pred: np.ndarray, titulo: str, ruta: Path) -> Path:
    """Guarda los residuales de prueba (pronostico menos real) y regresa la ruta.

    A la izquierda, densidad de residuales contra el pronostico (detecta errores que crecen
    con el nivel de demanda); a la derecha, su distribucion con el sesgo medio marcado.
    """
    pronostico = np.asarray(pred, dtype=float)
    residuo = pronostico - filas["objetivo"].to_numpy(dtype=float)
    fig = Figure(figsize=(11, 4.2), facecolor=FONDO)
    ax_densidad, ax_hist = fig.subplots(1, 2, gridspec_kw={"width_ratios": [3, 2]})
    rampa = LinearSegmentedColormap.from_list("densidad", RAMPA_DENSIDAD)
    densidad = ax_densidad.hexbin(
        pronostico, residuo, gridsize=50, mincnt=1, bins="log", cmap=rampa, linewidths=0
    )
    fig.colorbar(densidad, ax=ax_densidad, label="filas estacion-dia (escala log)")
    ax_densidad.axhline(0, color=TINTA_SECUNDARIA, lw=1.5)
    ax_densidad.set_xlabel("Pronostico (salidas del dia)")
    ax_densidad.set_ylabel("Residual = pronostico - real")
    bajo, alto = np.percentile(residuo, [0.5, 99.5])
    ax_hist.hist(
        np.clip(residuo, bajo, alto), bins=60, color=COLOR_BARRAS, edgecolor=FONDO, linewidth=0.5
    )
    ax_hist.axvline(0, color=TINTA_SECUNDARIA, lw=1.5, label="cero")
    ax_hist.axvline(
        residuo.mean(), color=TINTA, lw=1.5, ls="--", label=f"sesgo medio {residuo.mean():.2f}"
    )
    ax_hist.set_xlabel("Residual (colas fuera de p0.5-p99.5 en los bordes)")
    ax_hist.set_ylabel("Filas estacion-dia")
    ax_hist.legend(frameon=False, labelcolor=TINTA, loc="upper left")
    for ax in (ax_densidad, ax_hist):
        ax.set_facecolor(FONDO)
        ax.grid(color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    fig.suptitle(titulo, color=TINTA)
    fig.tight_layout()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=120, facecolor=FONDO)
    return ruta


def coeficientes_poisson(modelo: Pipeline) -> pd.DataFrame:
    """Coeficientes del Poisson y su efecto multiplicativo exp(coef) sobre la media."""
    nombres = modelo[:-1].get_feature_names_out()
    coef = modelo[-1].coef_
    return pd.DataFrame(
        {"variable": nombres, "coeficiente": coef, "efecto_multiplicativo": np.exp(coef)}
    ).sort_values("coeficiente", key=abs, ascending=False)


def config_run(params_modelo: dict, data_version: str, semilla: int) -> dict:
    """Configuracion reproducible que cada candidata guarda bajo ``config/``."""
    return {
        "question": "P1",
        "cortes": dict(CORTES),
        "semilla": semilla,
        "data_version": data_version,
        "features": list(fe.FEATURES),
        "objetivo": "salidas por estacion y dia",
        "momento_pronostico": "23:00 de D-1",
        "modelo": params_modelo,
    }


def predecir(
    partes: dict,
    semilla: int,
    completo: bool,
    model: str = CONFIG_P1.model,
    hyperparameters: dict | None = None,
    tags_base: dict[str, str] | None = None,
) -> dict:
    """Afina en validacion y devuelve las predicciones de val y test de baselines y candidatos.

    En val los candidatos estan entrenados solo con fit; en test, reentrenados con train.
    """
    val, train, test = partes["val"], partes["train"], partes["test"]
    elegidos, rejilla, pred_val_cand = afinar(
        partes,
        semilla,
        completo=completo,
        model=model,
        hyperparameters=hyperparameters,
        tags_base=tags_base,
    )
    logger.info("Parametros elegidos: {}", elegidos)
    pred_val = {
        "semana_previa": val["baseline_semana_previa"].to_numpy(),
        "persistencia": val["baseline_persistencia"].to_numpy(),
    } | pred_val_cand
    pred_test = {
        "semana_previa": test["baseline_semana_previa"].to_numpy(),
        "persistencia": test["baseline_persistencia"].to_numpy(),
    }
    modelos = {}
    costos = {}
    for candidato in CANDIDATOS:
        if candidato not in elegidos:
            continue
        inicio = time.perf_counter()
        modelos[candidato] = ajustar_candidato(candidato, train, elegidos[candidato], semilla)
        costos[candidato] = {
            "fit_seconds": time.perf_counter() - inicio,
            "model_size_mb": len(sio.dumps(modelos[candidato])) / (1024**2),
        }
        pred_test[candidato] = modelos[candidato].predict(test[fe.FEATURES])
    return {
        "elegidos": elegidos,
        "rejilla": rejilla,
        "pred_val": pred_val,
        "pred_test": pred_test,
        "modelo_poisson": modelos.get("poisson"),
        "modelos": modelos,
        "costos": costos,
    }


def parametros_comunes(
    partes: dict, df: pd.DataFrame, resumen: dict, semilla: int, data_version: str
) -> dict:
    """Parametros que describen la ventana de datos y los cortes; iguales en los runs finales."""
    train, test = partes["train"], partes["test"]
    nuevas = segmentos(train, test) == "nuevas"
    return {
        "unidad": "estacion-dia",
        "horizonte_dias": 1,
        "hora_emision": fe.HORA_EMISION,
        "lag_minimo_dias": fe.LAG_MINIMO,
        "lags_semanales": ",".join(map(str, fe.LAGS_SEMANALES)),
        "train_rango": rango_meses(CORTES["train_inicio"], CORTES["test_inicio"]),
        "val_rango": rango_meses(CORTES["val_inicio"], CORTES["test_inicio"]),
        "test_rango": rango_meses(CORTES["test_inicio"], CORTES["test_fin"]),
        "ultimo_objetivo_fit": str(fe.ultimo_dia_conocido(CORTES["val_inicio"]).date()),
        "ultimo_objetivo_val": str(fe.ultimo_dia_conocido(CORTES["test_inicio"]).date()),
        "ultimo_objetivo_train": str(fe.ultimo_dia_conocido(CORTES["test_inicio"]).date()),
        "semilla": semilla,
        "data_version": data_version,
        "n_fit": len(partes["fit"]),
        "n_val": len(partes["val"]),
        "n_train": len(train),
        "n_test": len(test),
        "estaciones_panel": resumen["estaciones"],
        "estaciones_train": train["estacion"].nunique(),
        "estaciones_test": test["estacion"].nunique(),
        "estaciones_nuevas_test": test.loc[nuevas, "estacion"].nunique(),
        "filas_panel": len(df),
        "filas_elegibles": int(df["elegible"].sum()),
        "filas_sin_historia": int(df["sin_historia"].sum()),
        "filas_inactivas": int(df["inactiva"].sum()),
    }


def registrar_ganadora(
    seleccionado: str,
    metricas_modelo: dict[str, dict[str, float]],
    modelos_logueados: dict[str, dict[str, str]],
    elegibilidad: dict[str, dict[str, float | int | bool | str | None]],
    partes: dict[str, pd.DataFrame],
    semilla: int,
) -> dict:
    """Promueve exactamente la candidata elegida en validacion, si es desplegable.

    La politica vigente permite que gane ``semana_previa``. Esa baseline no tiene un modelo
    serializado, por lo que en ese caso se informa la decision pendiente en lugar de promover
    silenciosamente otra candidata.
    """
    if seleccionado in CANDIDATOS and not elegibilidad.get(seleccionado, {}).get(
        "eligible", False
    ):
        return {
            "status": "not_registered",
            "reason": "selected_candidate_ineligible_nonnegative_domain",
            "selected": seleccionado,
        }
    if seleccionado not in modelos_logueados:
        return {
            "status": "not_registered",
            "reason": "selected_baseline_has_no_deployable_model",
            "selected": seleccionado,
        }

    origen = modelos_logueados[seleccionado]
    ejemplo = partes["train"].sample(5, random_state=semilla)[fe.FEATURES].astype("float64")
    diagnostico = elegibilidad[seleccionado]
    version, predicciones = registro_p1.registrar_campeon(
        origen["run_id"],
        origen["model_uri"],
        seleccionado,
        metricas_modelo[seleccionado]["val_mae"],
        {
            "experimento": tracking.EXPERIMENTO,
            "politica_dominio": POLITICA_DOMINIO,
            "val_pred_min": str(diagnostico["prediction_min"]),
            "val_pred_negative_n": str(diagnostico["negative_n"]),
            "val_pred_nonfinite_n": str(diagnostico["nonfinite_n"]),
        },
        ejemplo,
    )
    logger.info(
        "Campeon P1 {} validado como {} v{} y promovido a {}; prediccion {}",
        seleccionado,
        CONFIG_P1.modelo_registrado,
        version.version,
        CONFIG_P1.alias_campeon,
        predicciones,
    )
    return {
        "status": "registered",
        "selected": seleccionado,
        "run_id": origen["run_id"],
        "model_uri": origen["model_uri"],
        "registered_model": CONFIG_P1.modelo_registrado,
        "alias": CONFIG_P1.alias_campeon,
        "version": int(version.version),
        "criterion": registro_p1.CRITERIO_CAMPEON,
        "validation_mae": metricas_modelo[seleccionado]["val_mae"],
        "validation_domain": diagnostico,
        "smoke_predictions": predicciones,
    }


def resolver_registro(
    model_solicitado: str,
    no_registrar: bool,
    seleccionado: str,
    metricas_modelo: dict[str, dict[str, float]],
    modelos_logueados: dict[str, dict[str, str]],
    elegibilidad: dict[str, dict[str, float | int | bool | str | None]],
    partes: dict[str, pd.DataFrame],
    semilla: int,
) -> dict:
    """Aplica los candados de gobernanza antes de tocar el alias ``champion``."""
    if no_registrar:
        return {"status": "disabled", "reason": "--no-registrar", "selected": seleccionado}
    if model_solicitado != "auto":
        return {
            "status": "not_registered",
            "reason": "promotion_requires_model_auto",
            "selected": seleccionado,
        }
    if not any(elegibilidad.get(candidato, {}).get("eligible", False) for candidato in CANDIDATOS):
        return {
            "status": "not_registered",
            "reason": "no_eligible_candidate_nonnegative_domain",
            "selected": seleccionado,
        }
    return registrar_ganadora(
        seleccionado,
        metricas_modelo,
        modelos_logueados,
        elegibilidad,
        partes,
        semilla,
    )


def main(argv: list[str] | None = None) -> dict:
    """Punto de entrada: python -m mibici.modeling.train_p1."""
    ap = argparse.ArgumentParser(description="Experimento base P1 en MLflow")
    ap.add_argument("--panel", default=str(CONFIG_P1.panel_path))
    ap.add_argument("--semilla", type=int, default=CONFIG_P1.semilla)
    ap.add_argument("--tracking-uri", default=tracking.TRACKING_URI)
    ap.add_argument("--artefactos", default=str(CONFIG_P1.artefactos_dir))
    ap.add_argument("--figuras", default=str(CONFIG.reports_dir / "figures"))
    ap.add_argument("--manifiesto", default=str(tracking.MANIFIESTO))
    ap.add_argument("--autor", default=None, help="por defecto, git config user.name")
    ap.add_argument(
        "--no-registrar",
        action="store_true",
        help="omite el Model Registry; por omision se promueve el ganador desplegable",
    )
    ap.add_argument(
        "--model",
        choices=("auto", *CANDIDATOS),
        default=CONFIG_P1.model,
        help="auto compara los modelos configurados; un modelo específico restringe la comparación",
    )
    ap.add_argument("--hyperparameters", default="{}")
    ap.add_argument(
        "--sin-afinar", action="store_true", help="usa la primera configuracion de cada rejilla"
    )
    args = ap.parse_args(argv)
    try:
        hyperparameters = json.loads(args.hyperparameters)
    except json.JSONDecodeError as exc:
        ap.error(f"--hyperparameters must be valid JSON: {exc}")
    if not isinstance(hyperparameters, dict):
        ap.error("--hyperparameters must be a JSON object")

    tracking.configurar(args.tracking_uri, Path(args.artefactos))
    data_version = tracking.version_datos(Path(args.manifiesto))
    ruta_resumen = Path(args.panel).with_suffix(".json")
    resumen = json.loads(ruta_resumen.read_text())
    df = fe.construir_features(pd.read_parquet(args.panel))
    partes = particionar(df[df["elegible"]])
    val, test = partes["val"], partes["test"]
    figs = Path(args.figuras)
    figs.mkdir(parents=True, exist_ok=True)
    rango_datos = f"{resumen['primer_dia_panel'][:7]}..{resumen['ultimo_dia_panel'][:7]}"
    tags_base = tracking.etiquetas_base(
        "P1", args.autor or tracking.autor_git(), args.semilla, rango_datos, data_version
    )

    r = predecir(
        partes,
        args.semilla,
        completo=not args.sin_afinar,
        model=args.model,
        hyperparameters=hyperparameters,
        tags_base=tags_base,
    )
    metricas_modelo = {}
    elegibilidad = {}
    # La elegibilidad y la seleccion quedan cerradas con validacion antes de calcular
    # cualquier metrica o diagnostico de prueba.
    for nombre, pred in r["pred_val"].items():
        m = metricas(val["objetivo"], pred, "val")
        if nombre in CANDIDATOS:
            diagnostico = diagnostico_dominio(pred)
            elegibilidad[nombre] = diagnostico
            m |= {
                "val_pred_negative_n": diagnostico["negative_n"],
                "val_pred_nonfinite_n": diagnostico["nonfinite_n"],
                "val_domain_eligible": float(diagnostico["eligible"]),
            }
            if diagnostico["prediction_min"] is not None:
                m["val_pred_min"] = diagnostico["prediction_min"]
        metricas_modelo[nombre] = m
    seleccionado = seleccionar(
        {k: v["val_mae"] for k, v in metricas_modelo.items()},
        {k: bool(v["eligible"]) for k, v in elegibilidad.items()},
    )
    logger.info("Modelo seleccionado en validacion: {}", seleccionado)

    segmento_test = segmentos(partes["train"], test)
    # Prueba se observa solo despues de cerrar seleccion y elegibilidad.
    for nombre, pred in r["pred_test"].items():
        m = metricas(test["objetivo"], pred, "test")
        m |= metricas_segmentos(test, pred, segmento_test, "test")
        m |= metricas_mensuales(test, pred, "test")
        if nombre in r["costos"]:
            m |= r["costos"][nombre]
        metricas_modelo[nombre] |= m
    for nombre in CANDIDATOS:
        if nombre in metricas_modelo:
            metricas_modelo[nombre]["val_mejora_mae_vs_semana_previa"] = (
                metricas_modelo["semana_previa"]["val_mae"] - metricas_modelo[nombre]["val_mae"]
            )
            metricas_modelo[nombre]["test_mejora_mae_vs_semana_previa"] = (
                metricas_modelo["semana_previa"]["test_mae"] - metricas_modelo[nombre]["test_mae"]
            )

    params_comunes = parametros_comunes(partes, df, resumen, args.semilla, data_version)
    ruta_rejilla = figs / "p1_rejilla_validacion.json"
    ruta_rejilla.write_text(json.dumps(r["rejilla"], indent=2))
    ruta_coef = None
    if r["modelo_poisson"] is not None:
        ruta_coef = figs / "p1_poisson_coeficientes.csv"
        coeficientes_poisson(r["modelo_poisson"]).to_csv(ruta_coef, index=False)

    # La figura cubre validacion y prueba. Solo las candidatas entrenadas guardan modelo,
    # datasets y configuracion; los baselines siguen siendo referencias no desplegables.
    val_test = pd.concat([val, test])
    runs = {}
    modelos_logueados = {}

    def contenido_candidata(candidato: str, parametros: dict):
        def registrar_contenido() -> None:
            info = registrar_modelo(
                r["modelos"][candidato], partes["train"].sample(5, random_state=args.semilla)
            )
            modelos_logueados[candidato] = {"model_uri": info.model_uri}
            registrar_datasets(partes, args.semilla, data_version)
            mlflow.log_dict(
                config_run(parametros, data_version, args.semilla), "config/config_p1.json"
            )

        return registrar_contenido

    for nombre, (run_name, etiqueta) in NOMBRES_RUN.items():
        if nombre in CANDIDATOS and nombre not in r["elegidos"]:
            continue
        pred = np.concatenate([r["pred_val"][nombre], r["pred_test"][nombre]])
        figura = figura_agregado(
            val_test,
            pred,
            f"P1 {nombre}: total diario real y pronostico",
            figs / f"p1_agregado_{nombre}.png",
        )
        artefactos: list[Path | tuple[Path, str]] = [(figura, "plots")]
        usadas = FEATURES_BASELINE.get(nombre, fe.FEATURES)
        params_modelo = {"modelo": etiqueta, "features": ",".join(usadas)}
        dentro = None
        if nombre in CANDIDATOS:
            params_modelo |= r["elegidos"][nombre]
            residuales = figura_residuales(
                test,
                r["pred_test"][nombre],
                f"P1 {nombre}: residuales en prueba",
                figs / f"p1_residuales_{nombre}.png",
            )
            artefactos += [
                (residuales, "plots"),
                (ruta_rejilla, "config"),
                (Path(args.manifiesto), "data"),
                (ruta_resumen, "data"),
            ]

            dentro = contenido_candidata(nombre, dict(params_modelo))
        if nombre == "poisson" and ruta_coef is not None:
            artefactos.append((ruta_coef, "data"))
        etapa = "candidate" if nombre in CANDIDATOS else "baseline"
        tags = (
            tags_base
            | etiquetas_modelo(etiqueta, FAMILIAS[nombre], etapa)
            | {
                "seleccionado_validacion": "si" if nombre == seleccionado else "no",
            }
        )
        if nombre in CANDIDATOS:
            diagnostico = elegibilidad[nombre]
            tags |= {
                "elegible_dominio_val": "si" if diagnostico["eligible"] else "no",
                "motivo_elegibilidad_val": str(diagnostico["reason"]),
                "politica_dominio": POLITICA_DOMINIO,
            }
        run_id = tracking.registrar(
            run_name,
            tags,
            params_comunes | params_modelo,
            metricas_modelo[nombre],
            artefactos,
            dentro=dentro,
        )
        runs[nombre] = run_id
        if nombre in modelos_logueados:
            modelos_logueados[nombre]["run_id"] = run_id

    registry = resolver_registro(
        args.model,
        args.no_registrar,
        seleccionado,
        metricas_modelo,
        modelos_logueados,
        elegibilidad,
        partes,
        args.semilla,
    )
    salida = {
        "seleccionado": seleccionado,
        "politica_dominio": POLITICA_DOMINIO,
        "elegibilidad_validacion": elegibilidad,
        "elegidos": r["elegidos"],
        "rejilla": r["rejilla"],
        "resultados": metricas_modelo,
        "runs": runs,
        "modelos": modelos_logueados,
        "registry": registry,
    }
    ruta = figs / "p1_resultados.json"
    ruta.write_text(json.dumps(salida, indent=2))
    logger.info("Resultados en {}", ruta)
    return salida


if __name__ == "__main__":
    main()
