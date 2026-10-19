"""Experimento base de P1 (demanda diaria de salidas por estacion) registrado en MLflow.

Ejemplo:
    python -m mibici.features
    python -m mibici.modeling.train_p1

Baselines: misma estacion y mismo dia de la semana previa (D-7) y persistencia del ultimo
dia completo (D-2). Candidatos: regresion de Poisson regularizada y gradient boosting con
perdida de Poisson. Los hiperparametros y el modelo seleccionado se eligen solo con la
validacion; la prueba se evalua una vez, con el modelo reentrenado con todas las filas
elegibles de train, hasta el ultimo objetivo disponible al iniciar la prueba.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

from loguru import logger
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from mibici import features as fe
from mibici import tracking
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
CANDIDATOS = ("poisson", "hgb_poisson")
FEATURES_BASELINE = {"semana_previa": ["lag_7"], "persistencia": ["lag_2"]}
# modelo -> (nombre del run en MLflow, etiqueta del modelo)
NOMBRES_RUN = {
    "semana_previa": ("p1_baseline_semana_previa", "baseline_semana_previa"),
    "persistencia": ("p1_baseline_persistencia", "baseline_persistencia"),
    "poisson": ("p1_poisson_regularizado", "poisson_regularizado"),
    "hgb_poisson": ("p1_hgb_poisson", "hist_gradient_boosting_poisson"),
}


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
    prep = ColumnTransformer(
        [
            ("conteos", conteos, CONTEOS),
            ("dia", OneHotEncoder(categories=[list(range(7))], drop="first"), ["dia_semana"]),
            ("binarias", "passthrough", BINARIAS),
        ]
    )
    return Pipeline([("prep", prep), ("modelo", PoissonRegressor(alpha=alpha, max_iter=2000))])


def ajustar_poisson(entrena: pd.DataFrame, alpha: float) -> Pipeline:
    """Entrena el pipeline de Poisson; falla si el optimizador no converge."""
    modelo = pipeline_poisson(alpha)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        modelo.fit(entrena[fe.FEATURES], entrena["objetivo"])
    return modelo


def ajustar_hgb(
    entrena: pd.DataFrame, params: dict, semilla: int
) -> HistGradientBoostingRegressor:
    """Entrena gradient boosting con perdida de Poisson (el dia de la semana va como categoria)."""
    modelo = HistGradientBoostingRegressor(
        loss="poisson",
        categorical_features=["dia_semana"],
        early_stopping=False,
        random_state=semilla,
        **params,
    )
    modelo.fit(entrena[fe.FEATURES], entrena["objetivo"])
    return modelo


def mae(y_real: pd.Series, y_pred: np.ndarray) -> float:
    """Error absoluto medio."""
    return float(np.abs(np.asarray(y_pred, dtype=float) - y_real.to_numpy()).mean())


def afinar(partes: dict, semilla: int, completo: bool = True) -> tuple[dict, dict, dict]:
    """Elige alpha de Poisson y parametros de HGB por MAE de validacion, entrenando solo con fit.

    Devuelve los parametros elegidos por candidato, la rejilla completa con su val_mae y las
    predicciones de validacion del mejor de cada candidato.
    """
    fit, val = partes["fit"], partes["val"]
    configs = {
        "poisson": [{"alpha": a} for a in (ALPHAS_POISSON if completo else ALPHAS_POISSON[:1])],
        "hgb_poisson": REJILLA_HGB if completo else REJILLA_HGB[:1],
    }
    rejilla: dict[str, list[dict]] = {}
    elegidos: dict[str, dict] = {}
    pred_val: dict[str, np.ndarray] = {}
    for candidato, opciones in configs.items():
        resultados, predicciones = [], []
        for params in opciones:
            if candidato == "poisson":
                modelo = ajustar_poisson(fit, params["alpha"])
            else:
                modelo = ajustar_hgb(fit, params, semilla)
            pred = modelo.predict(val[fe.FEATURES])
            resultados.append({**params, "val_mae": mae(val["objetivo"], pred)})
            predicciones.append(pred)
        mejor = min(range(len(resultados)), key=lambda i: resultados[i]["val_mae"])
        rejilla[candidato] = resultados
        elegidos[candidato] = opciones[mejor]
        pred_val[candidato] = predicciones[mejor]
        logger.info("Rejilla {}: elegido {}", candidato, resultados[mejor])
    return elegidos, rejilla, pred_val


def seleccionar(val_mae: dict[str, float]) -> str:
    """Modelo elegido: semana previa, salvo que un candidato tenga menor MAE de validacion."""
    mejor = "semana_previa"
    for candidato in CANDIDATOS:
        if val_mae[candidato] < val_mae[mejor]:
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


def coeficientes_poisson(modelo: Pipeline) -> pd.DataFrame:
    """Coeficientes del Poisson y su efecto multiplicativo exp(coef) sobre la media."""
    nombres = modelo[:-1].get_feature_names_out()
    coef = modelo[-1].coef_
    return pd.DataFrame(
        {"variable": nombres, "coeficiente": coef, "efecto_multiplicativo": np.exp(coef)}
    ).sort_values("coeficiente", key=abs, ascending=False)


def predecir(partes: dict, semilla: int, completo: bool) -> dict:
    """Afina en validacion y devuelve las predicciones de val y test de los cuatro modelos.

    En val los candidatos estan entrenados solo con fit; en test, reentrenados con train.
    """
    val, train, test = partes["val"], partes["train"], partes["test"]
    elegidos, rejilla, pred_val_cand = afinar(partes, semilla, completo=completo)
    logger.info("Parametros elegidos: {}", elegidos)
    pred_val = {
        "semana_previa": val["baseline_semana_previa"].to_numpy(),
        "persistencia": val["baseline_persistencia"].to_numpy(),
    } | pred_val_cand
    modelo_poisson = ajustar_poisson(train, elegidos["poisson"]["alpha"])
    modelo_hgb = ajustar_hgb(train, elegidos["hgb_poisson"], semilla)
    pred_test = {
        "semana_previa": test["baseline_semana_previa"].to_numpy(),
        "persistencia": test["baseline_persistencia"].to_numpy(),
        "poisson": modelo_poisson.predict(test[fe.FEATURES]),
        "hgb_poisson": modelo_hgb.predict(test[fe.FEATURES]),
    }
    return {
        "elegidos": elegidos,
        "rejilla": rejilla,
        "pred_val": pred_val,
        "pred_test": pred_test,
        "modelo_poisson": modelo_poisson,
    }


def parametros_comunes(
    partes: dict, df: pd.DataFrame, resumen: dict, semilla: int, data_version: str
) -> dict:
    """Parametros que describen la ventana de datos y los cortes; iguales en los cuatro runs."""
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
        "--sin-afinar", action="store_true", help="usa la primera configuracion de cada rejilla"
    )
    args = ap.parse_args(argv)

    tracking.configurar(args.tracking_uri, Path(args.artefactos))
    data_version = tracking.version_datos(Path(args.manifiesto))
    ruta_resumen = Path(args.panel).with_suffix(".json")
    resumen = json.loads(ruta_resumen.read_text())
    df = fe.construir_features(pd.read_parquet(args.panel))
    partes = particionar(df[df["elegible"]])
    val, test = partes["val"], partes["test"]
    figs = Path(args.figuras)
    figs.mkdir(parents=True, exist_ok=True)

    r = predecir(partes, args.semilla, completo=not args.sin_afinar)
    segmento_test = segmentos(partes["train"], test)
    metricas_modelo = {}
    for nombre, pred in r["pred_test"].items():
        m = metricas(val["objetivo"], r["pred_val"][nombre], "val")
        m |= metricas(test["objetivo"], pred, "test")
        m |= metricas_segmentos(test, pred, segmento_test, "test")
        metricas_modelo[nombre] = m
    seleccionado = seleccionar({k: v["val_mae"] for k, v in metricas_modelo.items()})
    logger.info("Modelo seleccionado en validacion: {}", seleccionado)

    rango_datos = f"{resumen['primer_dia_panel'][:7]}..{resumen['ultimo_dia_panel'][:7]}"
    tags_base = tracking.etiquetas_base(
        "P1", args.autor or tracking.autor_git(), args.semilla, rango_datos, data_version
    )
    params_comunes = parametros_comunes(partes, df, resumen, args.semilla, data_version)
    ruta_rejilla = figs / "p1_rejilla_validacion.json"
    ruta_rejilla.write_text(json.dumps(r["rejilla"], indent=2))
    ruta_coef = figs / "p1_poisson_coeficientes.csv"
    coeficientes_poisson(r["modelo_poisson"]).to_csv(ruta_coef, index=False)

    # la figura cubre validacion y prueba de cada modelo
    val_test = pd.concat([val, test])
    for nombre, (run_name, etiqueta) in NOMBRES_RUN.items():
        pred = np.concatenate([r["pred_val"][nombre], r["pred_test"][nombre]])
        artefactos = [
            figura_agregado(
                val_test,
                pred,
                f"P1 {nombre}: total diario real y pronostico",
                figs / f"p1_agregado_{nombre}.png",
            )
        ]
        usadas = FEATURES_BASELINE.get(nombre, fe.FEATURES)
        params_modelo = {"modelo": etiqueta, "features": ",".join(usadas)}
        if nombre in CANDIDATOS:
            params_modelo |= r["elegidos"][nombre]
            artefactos += [ruta_rejilla, Path(args.manifiesto), ruta_resumen]
        if nombre == "poisson":
            artefactos.append(ruta_coef)
        tags = tags_base | {
            "modelo": etiqueta,
            "seleccionado_validacion": "si" if nombre == seleccionado else "no",
        }
        tracking.registrar(
            run_name, tags, params_comunes | params_modelo, metricas_modelo[nombre], artefactos
        )

    salida = {
        "seleccionado": seleccionado,
        "elegidos": r["elegidos"],
        "rejilla": r["rejilla"],
        "resultados": metricas_modelo,
    }
    ruta = figs / "p1_resultados.json"
    ruta.write_text(json.dumps(salida, indent=2))
    logger.info("Resultados en {}", ruta)
    return salida


if __name__ == "__main__":
    main()
