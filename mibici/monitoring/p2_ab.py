"""A/B testing de P2: campeón (A) contra la versión reentrenada (B) en el mes actual.

Regla de decisión, fijada antes de correr la comparación:

- La métrica principal es el recall medio de las clases de riesgo (sin_bicicletas y
  sin_anclajes), la misma con que se eligió al campeón (``mibici.registro``).
- B reemplaza a A si su recall medio sube al menos ``EMPATE_RECALL`` y el límite inferior del
  intervalo de confianza del 95 % de la diferencia B - A es mayor que cero.
- Si la diferencia de recall medio está por debajo de ``EMPATE_RECALL`` en valor absoluto, hay
  empate y decide el PR AUC medio de las clases de riesgo: B reemplaza a A solo si su diferencia
  es positiva y el límite inferior de su intervalo es mayor que cero.
- En cualquier otro caso se mantiene A.

Los intervalos salen de un bootstrap pareado por estación: se remuestrean estaciones con
reemplazo y las dos versiones se califican con las mismas filas.

- **A**: la versión con alias ``champion`` de ``CONFIG_P2.modelo_registrado``.
- **B**: la versión con alias ``challenger``; con ``--registrar`` se entrena antes el mismo HGB
  balanceado (hiperparámetros leídos del run de A) con fecha objetivo de
  ``CONFIG_P2.fecha_inicio_entrenamiento`` a ``CONFIG_P2.fecha_fin_entrenamiento_retador``
  (excluida) y se registra como nueva versión. ``champion`` nunca se mueve aquí.

Las dos versiones califican las mismas filas estación-hora del mes actual; solo cambian las tres
``est_*``, que cada versión toma de las estadísticas de su propio train. Además de la tabla
general se reportan tres subgrupos: estaciones vistas por A, nuevas para A pero vistas por B y
nuevas para ambas.

Uso desde la raíz del repositorio (después del reporte de drift de P2)::

    python -m mibici.monitoring.p2_ab --registrar --tracking-uri sqlite:///ruta/mlflow.db
    python -m mibici.monitoring.p2_ab --tracking-uri sqlite:///ruta/mlflow.db
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path

from loguru import logger
import mlflow
from mlflow import MlflowClient
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from mibici import features_p2 as fp
from mibici import registro
from mibici.config import CONFIG, CONFIG_P2
from mibici.modeling import train_p2 as t2
from mibici.monitoring import p2_datasets as d2
from mibici.monitoring.p1_drift import file_sha256, month_bounds
from mibici.registro import EMPATE_RECALL
from mibici.tracking import configurar_experimento, etiquetas_base

PROMOVER = "promover B"
MANTENER = "mantener A"


def decidir(
    delta_recall: float,
    ic_recall: tuple[float, float],
    delta_pr_auc: float,
    ic_pr_auc: tuple[float, float],
    empate: float = EMPATE_RECALL,
) -> dict:
    """Aplica la regla de decisión del módulo a las diferencias B - A y sus intervalos."""
    if abs(delta_recall) < empate:
        criterio = "pr_auc_medio_riesgo"
        promover = delta_pr_auc > 0 and ic_pr_auc[0] > 0
    else:
        criterio = "recall_medio_riesgo"
        promover = delta_recall >= empate and ic_recall[0] > 0
    return {
        "decision": PROMOVER if promover else MANTENER,
        "criterio": criterio,
        "empate_recall": empate,
        "delta_recall_medio_riesgo": delta_recall,
        "ic95_recall_medio_riesgo": list(ic_recall),
        "delta_pr_auc_medio_riesgo": delta_pr_auc,
        "ic95_pr_auc_medio_riesgo": list(ic_pr_auc),
    }


CLASES = tuple(fp.CLASES)
N_BOOTSTRAP = 1000
METRICAS_MENOR_ES_MEJOR = ("brier_multiclase",)


def indices_de_clase(etiquetas: np.ndarray) -> np.ndarray:
    """Posición de cada etiqueta en ``CLASES``; falla si aparece una clase desconocida."""
    posiciones = pd.Index(CLASES).get_indexer(np.asarray(etiquetas, dtype=object))
    if (posiciones < 0).any():
        raise ValueError("Hay etiquetas fuera de las tres clases de P2.")
    return posiciones


def matrices_por_estacion(
    codigos: np.ndarray, y: np.ndarray, pred: np.ndarray, n_estaciones: int
) -> np.ndarray:
    """Matriz de confusión (real por predicha, en el orden de ``CLASES``) de cada estación."""
    plano = codigos * 9 + indices_de_clase(y) * 3 + indices_de_clase(pred)
    return np.bincount(plano, minlength=n_estaciones * 9).reshape(n_estaciones, 3, 3)


def _dividir(numerador: np.ndarray, denominador: np.ndarray) -> np.ndarray:
    return np.divide(
        numerador,
        denominador,
        out=np.zeros_like(numerador, dtype=float),
        where=denominador > 0,
    )


def metricas_de_confusion(cm: np.ndarray) -> dict[str, float]:
    """Recall, precisión y F1 por clase, macro F1 y recall medio de riesgo de una matriz 3x3.

    Una clase sin casos aporta 0, igual que ``train_p2.macro_f1`` con ``zero_division=0``.
    """
    aciertos = np.diag(cm).astype(float)
    recall = _dividir(aciertos, cm.sum(axis=1).astype(float))
    precision = _dividir(aciertos, cm.sum(axis=0).astype(float))
    f1 = _dividir(2 * precision * recall, precision + recall)
    salida = {"macro_f1": float(f1.mean())}
    for i, clase in enumerate(CLASES):
        salida[f"recall_{clase}"] = float(recall[i])
        salida[f"precision_{clase}"] = float(precision[i])
        salida[f"f1_{clase}"] = float(f1[i])
    salida["recall_medio_riesgo"] = float(
        np.mean([salida[f"recall_{c}"] for c in registro.RIESGO])
    )
    return salida


@dataclass(frozen=True)
class OrdenAP:
    """Filas ordenadas por puntaje para calcular el PR AUC con pesos por estación."""

    positivos: np.ndarray
    codigos: np.ndarray
    cortes: np.ndarray


def preparar_ap(puntaje: np.ndarray, positivos: np.ndarray, codigos: np.ndarray) -> OrdenAP:
    orden = np.argsort(-puntaje, kind="stable")
    ordenado = puntaje[orden]
    cortes = np.r_[np.flatnonzero(np.diff(ordenado)), len(ordenado) - 1]
    return OrdenAP(positivos[orden].astype(float), codigos[orden], cortes)


def ap_ponderado(orden: OrdenAP, pesos_estacion: np.ndarray) -> float:
    """Average precision (PR AUC) con el peso de la estación de cada fila.

    Con pesos 1 coincide con ``sklearn.metrics.average_precision_score``.
    """
    pesos = pesos_estacion[orden.codigos].astype(float)
    vp = np.cumsum(pesos * orden.positivos)[orden.cortes]
    fp_ = np.cumsum(pesos * (1 - orden.positivos))[orden.cortes]
    if vp[-1] == 0:
        return float("nan")
    precision = _dividir(vp, vp + fp_)
    recall = vp / vp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def brier_multiclase(proba: np.ndarray, y_idx: np.ndarray) -> np.ndarray:
    """Brier multiclase por fila: suma sobre las tres clases de (p_k - 1[y = k])^2, entre 0 y 2."""
    uno = np.zeros_like(proba)
    uno[np.arange(len(y_idx)), y_idx] = 1.0
    return ((proba - uno) ** 2).sum(axis=1)


def proba_en_orden(proba: np.ndarray, clases_modelo: list[str]) -> np.ndarray:
    """Columnas de ``predict_proba`` reordenadas a ``CLASES``."""
    return proba[:, [list(clases_modelo).index(c) for c in CLASES]]


@dataclass(frozen=True)
class Calificacion:
    """Lo que se necesita de una versión para calificarla y remuestrearla por estación."""

    matrices: np.ndarray
    ap: dict[str, OrdenAP]
    brier_suma: np.ndarray
    proba: np.ndarray

    def metricas(self, pesos: np.ndarray, filas_por_estacion: np.ndarray) -> dict[str, float]:
        cm = np.tensordot(pesos, self.matrices, axes=1)
        salida = metricas_de_confusion(cm)
        for clase in registro.RIESGO:
            salida[f"pr_auc_{clase}"] = ap_ponderado(self.ap[clase], pesos)
        salida["pr_auc_medio_riesgo"] = float(
            np.mean([salida[f"pr_auc_{c}"] for c in registro.RIESGO])
        )
        salida["brier_multiclase"] = float(pesos @ self.brier_suma / (pesos @ filas_por_estacion))
        return salida


def calificar(
    codigos: np.ndarray, y: np.ndarray, pred: np.ndarray, proba: np.ndarray, n: int
) -> Calificacion:
    y_idx = indices_de_clase(y)
    return Calificacion(
        matrices=matrices_por_estacion(codigos, y, pred, n),
        ap={
            c: preparar_ap(proba[:, CLASES.index(c)], y_idx == CLASES.index(c), codigos)
            for c in registro.RIESGO
        },
        brier_suma=np.bincount(codigos, weights=brier_multiclase(proba, y_idx), minlength=n),
        proba=proba,
    )


def comparar(
    estaciones: pd.Series,
    y: pd.Series,
    pred_a: np.ndarray,
    proba_a: np.ndarray,
    pred_b: np.ndarray,
    proba_b: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = CONFIG_P2.seed,
) -> pd.DataFrame:
    """Tabla A/B en las mismas filas: A, B, B - A y su IC del 95 % por bootstrap de estaciones.

    ``proba_a`` y ``proba_b`` van en el orden de ``CLASES``. En cada remuestreo se toman
    estaciones con reemplazo y las dos versiones se califican con las mismas estaciones.
    """
    codigos, unicas = pd.factorize(estaciones, sort=True)
    n = len(unicas)
    y = np.asarray(y, dtype=object)
    a = calificar(codigos, y, pred_a, proba_a, n)
    b = calificar(codigos, y, pred_b, proba_b, n)
    filas_por_estacion = np.bincount(codigos, minlength=n).astype(float)
    uno = np.ones(n)
    puntual_a, puntual_b = a.metricas(uno, filas_por_estacion), b.metricas(uno, filas_por_estacion)
    rng = np.random.default_rng(seed)
    diferencias = {m: [] for m in puntual_a}
    for _ in range(n_bootstrap):
        pesos = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        m_a, m_b = a.metricas(pesos, filas_por_estacion), b.metricas(pesos, filas_por_estacion)
        for m, lista in diferencias.items():
            lista.append(m_b[m] - m_a[m])

    filas = []
    for m in puntual_a:
        muestras = np.asarray(diferencias[m], dtype=float)
        muestras = muestras[np.isfinite(muestras)]
        bajo, alto = np.percentile(muestras, [2.5, 97.5]) if len(muestras) else (np.nan, np.nan)
        filas.append(
            {
                "metrica": m,
                "mejor": "menor" if m in METRICAS_MENOR_ES_MEJOR else "mayor",
                "a": puntual_a[m],
                "b": puntual_b[m],
                "b_menos_a": puntual_b[m] - puntual_a[m],
                "ic95_bajo": float(bajo),
                "ic95_alto": float(alto),
            }
        )
    y_idx = indices_de_clase(y)
    for clase in registro.RIESGO:
        positivos = y_idx == CLASES.index(clase)
        if positivos.all() or not positivos.any():
            continue
        roc = {
            v: float(roc_auc_score(positivos, p[:, CLASES.index(clase)]))
            for v, p in (("a", proba_a), ("b", proba_b))
        }
        filas.append(
            {
                "metrica": f"roc_auc_{clase}",
                "mejor": "mayor",
                "a": roc["a"],
                "b": roc["b"],
                "b_menos_a": roc["b"] - roc["a"],
                "ic95_bajo": np.nan,
                "ic95_alto": np.nan,
            }
        )
    tabla = pd.DataFrame(filas)
    tabla.insert(1, "filas", len(y))
    tabla.insert(2, "estaciones", n)
    return tabla


def decidir_tabla(tabla: pd.DataFrame) -> dict:
    """Aplica ``decidir`` a las filas de recall medio y PR AUC medio de riesgo de ``comparar``."""
    fila = tabla.set_index("metrica")
    recall, pr_auc = fila.loc["recall_medio_riesgo"], fila.loc["pr_auc_medio_riesgo"]
    return decidir(
        float(recall.b_menos_a),
        (float(recall.ic95_bajo), float(recall.ic95_alto)),
        float(pr_auc.b_menos_a),
        (float(pr_auc.ic95_bajo), float(pr_auc.ic95_alto)),
    )


def subgrupos(
    estaciones: pd.Series, entrenadas_a: set[int], entrenadas_b: set[int]
) -> dict[str, np.ndarray]:
    """Máscaras de filas por subgrupo de estaciones según los train de A y de B."""
    en_a = estaciones.isin(entrenadas_a).to_numpy()
    en_b = estaciones.isin(entrenadas_b).to_numpy()
    return {
        "todas": np.ones(len(estaciones), dtype=bool),
        "vistas_por_a": en_a,
        "nuevas_para_a": ~en_a & en_b,
        "nuevas_para_ambos": ~en_a & ~en_b,
    }


def filas_por_version(
    filas: pd.DataFrame,
    estadisticas: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    """Las mismas filas con las ``est_*`` de cada versión, en el orden de ``FEATURES``.

    Todas las columnas que no son ``est_*`` quedan idénticas entre versiones.
    """
    salida = {}
    for version, stats in estadisticas.items():
        x = d2.con_estadisticas(filas, stats)
        if not x[list(d2.CLAVES)].equals(filas[list(d2.CLAVES)]):
            raise ValueError("Unir las estadísticas cambió el orden de las filas.")
        salida[version] = x
    return salida


def parametros_del_campeon(run_id: str, tracking_uri: str) -> dict[str, float | int]:
    """Hiperparámetros del HGB con que se entrenó el campeón, leídos de su run."""
    params = MlflowClient(tracking_uri=tracking_uri).get_run(run_id).data.params
    return {
        "learning_rate": float(params["learning_rate"]),
        "max_leaf_nodes": int(params["max_leaf_nodes"]),
        "max_iter": int(params["max_iter"]),
        "l2_regularization": float(params["l2_regularization"]),
    }


def cortes_retador(fin_mes_actual: str) -> dict[str, str]:
    """Cortes de ``train_p2.particionar`` para B: train hasta el corte de ``CONFIG_P2``."""
    return {
        "train_inicio": CONFIG_P2.fecha_inicio_entrenamiento,
        "val_inicio": CONFIG_P2.fecha_inicio_prueba,
        "test_inicio": CONFIG_P2.fecha_fin_entrenamiento_retador,
        "test_fin": fin_mes_actual,
    }


def entrenar_retador(
    filas: pd.DataFrame, params: dict, semilla: int, cortes: dict[str, str]
) -> tuple[HistGradientBoostingClassifier, pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame]]:
    """Mismo HGB balanceado del campeón entrenado con el train de B y sus estadísticas."""
    columnas = [c for c in (*fp.FEATURES, "estacion", "ts_objetivo", "objetivo") if c in filas]
    train = t2.particionar(filas[columnas], cortes)["train"].reset_index(drop=True)
    estadisticas = fp.stats_estacion(train)
    train = fp.unir_stats(train, *estadisticas)
    return t2.ajustar(train, params, semilla), train, estadisticas


def manifiesto_retador(raw_dir: Path, fin_excl: str, nomenclaturas: list[Path]) -> dict:
    """Manifiesto (sha256 y registros) de los CSV mensuales que ve B y de las nomenclaturas.

    No se escribe en ``references/``: el manifiesto versionado del equipo no cambia.
    """
    ultimo = pd.Period(pd.Timestamp(fin_excl) - pd.Timedelta(days=1), freq="M")
    meses = pd.period_range(CONFIG_P2.fecha_inicio_entrenamiento, ultimo, freq="M")
    archivos = [raw_dir / f"datos_abiertos_{m.year}_{m.month:02d}.csv" for m in meses]
    faltantes = [ruta.name for ruta in archivos if not ruta.is_file()]
    if faltantes:
        raise FileNotFoundError(f"Faltan CSV mensuales para el manifiesto de B: {faltantes}")
    return fp.manifiesto_datos([*archivos, *nomenclaturas])


def registrar_version_b(
    modelo: HistGradientBoostingClassifier,
    train: pd.DataFrame,
    estadisticas: tuple[pd.DataFrame, pd.DataFrame],
    params: dict,
    semilla: int,
    cortes: dict[str, str],
    manifiesto: dict,
    version_a: str,
    resumen_panel: dict,
) -> str:
    """Registra B en un run y como nueva versión del modelo de P2 con el alias ``challenger``."""
    fin = cortes["test_inicio"]
    rango = t2.rango_meses(cortes["train_inicio"], fin)
    nombre = f"p2_hgb_balanceado_{t2.sufijo_nivel(CONFIG_P2.nivel_inicial)}_retador_{rango[-7:]}"
    with mlflow.start_run(run_name=nombre) as run:
        mlflow.set_tags(
            etiquetas_base("P2", CONFIG_P2.author, semilla, rango, manifiesto["version"])
            | t2.etiquetas_modelo("hist_gradient_boosting", t2.FAMILIA_HGB, "retraining")
            | {"origen": "mibici/monitoring/p2_ab.py", "version_base": version_a}
        )
        mlflow.log_params(
            {
                "modelo": "HistGradientBoostingClassifier",
                "class_weight": "balanced",
                "features": ",".join(fp.FEATURES),
                **params,
                "nivel_inicial": CONFIG_P2.nivel_inicial,
                "umbral_bajo": CONFIG_P2.umbral_bajo,
                "umbral_alto": CONFIG_P2.umbral_alto,
                "horizonte_horas": CONFIG_P2.horizonte_horas,
                "horas_objetivo": f"{min(CONFIG_P2.horas_etiqueta)}-{max(CONFIG_P2.horas_etiqueta)}",
                "train_rango": rango,
                "fecha_fin_entrenamiento_excluida": fin,
                "semilla": semilla,
                "data_version": manifiesto["version"],
                "n_train": len(train),
                "estaciones_train": int(train["estacion"].nunique()),
            }
        )
        info = t2.registrar_modelo(modelo, train.sample(5, random_state=semilla))
        mlflow.log_text(estadisticas[0].to_csv(index=False), "data/stats_estacion.csv")
        mlflow.log_text(estadisticas[1].to_csv(index=False), "data/stats_estacion_hora.csv")
        mlflow.log_dict(manifiesto, "data/manifiesto_datos_retador.json")
        mlflow.log_dict(resumen_panel, "data/p2_panel_monitoreo.json")
        mlflow.log_dict(
            {
                "question": CONFIG_P2.question,
                "nivel_inicial": CONFIG_P2.nivel_inicial,
                "cortes": cortes,
                "umbral_bajo": CONFIG_P2.umbral_bajo,
                "umbral_alto": CONFIG_P2.umbral_alto,
                "horizonte_horas": CONFIG_P2.horizonte_horas,
                "horas_objetivo": list(CONFIG_P2.horas_etiqueta),
                "semilla": semilla,
                "data_version": manifiesto["version"],
                "features": fp.FEATURES,
                "clases": fp.CLASES,
                "modelo": params,
            },
            "config/config_p2.json",
        )
        run_id = run.info.run_id

    version = registro.registrar_retador(
        CONFIG_P2.modelo_registrado,
        CONFIG_P2.alias_retador,
        CONFIG_P2.alias_campeon,
        run_id,
        info.model_uri,
        (
            f"Retador de P2: el HGB balanceado de la versión {version_a} reentrenado con fecha "
            f"objetivo de {cortes['train_inicio']} a {fin} (excluida). Se compara contra "
            "champion con mibici/monitoring/p2_ab.py; champion no se mueve aquí."
        ),
        {
            "experimento": CONFIG_P2.experimento,
            "nivel_inicial": str(CONFIG_P2.nivel_inicial),
            "version_base": version_a,
            "fecha_fin_entrenamiento_excluida": fin,
            "origen": "mibici/monitoring/p2_ab.py",
        },
        pd.DataFrame([dict.fromkeys(fp.FEATURES, 0.0)]),
    )
    return str(version.version)


def cargar_version(version: str, tracking_uri: str) -> HistGradientBoostingClassifier:
    """Carga una versión inmutable del Registry en formato sklearn (con ``predict_proba``)."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    return mlflow.sklearn.load_model(f"models:/{CONFIG_P2.modelo_registrado}/{version}")


def tabla_markdown(tabla: pd.DataFrame) -> str:
    """Tabla A/B en Markdown con tres decimales y el IC del 95 % de B - A."""
    lineas = ["| Métrica | A | B | B - A [IC 95 %] |", "|---|---|---|---|"]
    for r in tabla.itertuples():
        ic = "" if np.isnan(r.ic95_bajo) else f" [{r.ic95_bajo:+.3f}, {r.ic95_alto:+.3f}]"
        lineas.append(f"| {r.metrica} | {r.a:.3f} | {r.b:.3f} | {r.b_menos_a:+.3f}{ic} |")
    return "\n".join(lineas) + "\n"


def main(argv: list[str] | None = None) -> None:
    """Punto de entrada: python -m mibici.monitoring.p2_ab [--registrar]."""
    ap = argparse.ArgumentParser(description="A/B testing de P2: champion contra challenger")
    ap.add_argument(
        "--historical-trips",
        type=Path,
        default=CONFIG.processed_data_dir / CONFIG.output_filename,
    )
    ap.add_argument("--current-trips", type=Path, default=None)
    ap.add_argument("--current-start", default="2026-10-01")
    ap.add_argument("--current-end-exclusive", default="2026-11-01")
    ap.add_argument("--tracking-uri", default=None)
    ap.add_argument(
        "--registrar",
        action="store_true",
        help="entrena B y lo registra como nueva versión con el alias challenger",
    )
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--semilla", type=int, default=CONFIG_P2.seed)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)

    year, month = month_bounds(args.current_start, args.current_end_exclusive)
    inicio, fin_excl = pd.Timestamp(args.current_start), pd.Timestamp(args.current_end_exclusive)
    if pd.Timestamp(CONFIG_P2.fecha_fin_entrenamiento_retador) > inicio:
        raise ValueError("B debe entrenarse solo con fechas objetivo anteriores al mes actual.")
    output = args.output or CONFIG.reports_dir / "monitoring" / f"{year}_{month:02d}" / "p2"
    current = args.current_trips or CONFIG.raw_data_dir / f"datos_abiertos_{year}_{month:02d}.csv"
    tracking_uri = args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    tracking_uri = tracking_uri or CONFIG_P2.tracking_uri
    configurar_experimento(
        tracking_uri, CONFIG_P2.experimento, CONFIG_P2.artefactos_dir.resolve().as_uri()
    )

    version_a, run_a = d2.version_por_alias(CONFIG_P2.alias_campeon, tracking_uri)
    nomenclaturas = d2.nomenclaturas_locales()
    panel, resumen_panel = d2.panel_actual(
        args.historical_trips,
        current,
        nomenclaturas,
        inicio,
        fin_excl,
        CONFIG.interim_data_dir / "monitoring",
    )
    filas = d2.filas_del_modelo(panel)
    del panel
    actual = d2.en_periodo(filas, inicio, fin_excl)
    if args.registrar:
        params = parametros_del_campeon(run_a, tracking_uri)
        cortes = cortes_retador(args.current_end_exclusive)
        modelo_b, train_b, stats_b = entrenar_retador(filas, params, args.semilla, cortes)
        del filas
        manifiesto = manifiesto_retador(CONFIG.raw_data_dir, cortes["test_inicio"], nomenclaturas)
        version_nueva = registrar_version_b(
            modelo_b,
            train_b,
            stats_b,
            params,
            args.semilla,
            cortes,
            manifiesto,
            version_a,
            resumen_panel,
        )
        logger.success(f"B registrado como v{version_nueva} (@{CONFIG_P2.alias_retador})")
        del modelo_b, train_b
    else:
        del filas
    version_b, run_b = d2.version_por_alias(CONFIG_P2.alias_retador, tracking_uri)

    estadisticas = {
        "a": d2.estadisticas_del_run(run_a, tracking_uri),
        "b": d2.estadisticas_del_run(run_b, tracking_uri),
    }
    x = filas_por_version(actual, estadisticas)
    modelos = {
        "a": cargar_version(version_a, tracking_uri),
        "b": cargar_version(version_b, tracking_uri),
    }
    predicciones = {}
    for v, modelo in modelos.items():
        clases, proba, pred = t2.predecir(modelo, x[v])
        predicciones[v] = (pred, proba_en_orden(proba, clases))

    entrenadas = {v: set(stats[0]["estacion"].astype(int)) for v, stats in estadisticas.items()}
    mascaras = subgrupos(actual["estacion"], entrenadas["a"], entrenadas["b"])
    tablas = []
    for nombre, mascara in mascaras.items():
        if mascara.sum() == 0 or actual.loc[mascara, "estacion"].nunique() < 2:
            logger.info(f"Subgrupo {nombre}: sin estaciones suficientes, se omite")
            continue
        tabla = comparar(
            actual.loc[mascara, "estacion"],
            actual.loc[mascara, "objetivo"],
            predicciones["a"][0][mascara],
            predicciones["a"][1][mascara],
            predicciones["b"][0][mascara],
            predicciones["b"][1][mascara],
            args.n_bootstrap,
            args.semilla,
        )
        tablas.append(tabla.assign(subgrupo=nombre))
    resultados = pd.concat(tablas, ignore_index=True)
    general = resultados[resultados.subgrupo == "todas"].drop(columns="subgrupo")
    resumen = decidir_tabla(general) | {
        "periodo_actual": f"{args.current_start}..{args.current_end_exclusive} (fin excluido)",
        "modelo_a": f"{CONFIG_P2.modelo_registrado} v{version_a} (@{CONFIG_P2.alias_campeon})",
        "modelo_b": f"{CONFIG_P2.modelo_registrado} v{version_b} (@{CONFIG_P2.alias_retador})",
        "run_a": run_a,
        "run_b": run_b,
        "filas": len(actual),
        "estaciones": int(actual["estacion"].nunique()),
        "filas_por_subgrupo": {k: int(m.sum()) for k, m in mascaras.items()},
        "estaciones_por_subgrupo": {
            k: int(actual.loc[m, "estacion"].nunique()) for k, m in mascaras.items()
        },
        "proporcion_etiqueta_proxy": d2.proporciones(actual["objetivo"]),
        "proporcion_predicha": {
            v: d2.proporciones(pd.Series(p[0])) for v, p in predicciones.items()
        },
        "n_bootstrap": args.n_bootstrap,
        "semilla": args.semilla,
        "csv_actual_sha256": file_sha256(current),
    }

    output.mkdir(parents=True, exist_ok=True)
    archivos = {
        "csv": output / "p2_ab_resultados.csv",
        "md": output / "p2_ab_resultados.md",
        "json": output / "p2_ab_resumen.json",
    }
    resultados.to_csv(archivos["csv"], index=False)
    archivos["md"].write_text(
        "\n".join(
            f"### {nombre}\n\n{tabla_markdown(resultados[resultados.subgrupo == nombre])}"
            for nombre in resultados.subgrupo.unique()
        ),
        encoding="utf-8",
    )
    archivos["json"].write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with mlflow.start_run(run_name=f"p2_ab_{year}_{month:02d}"):
        mlflow.set_tags(
            etiquetas_base(
                "P2",
                CONFIG_P2.author,
                args.semilla,
                f"{year}-{month:02d}",
                resumen_panel["huellas_sha256"][current.name][:12],
            )
            | {
                "stage_of_work": "ab_test",
                "origen": "mibici/monitoring/p2_ab.py",
                "ab_decision": resumen["decision"],
                "version_a": version_a,
                "version_b": version_b,
            }
        )
        mlflow.log_metrics(
            {
                f"ab_{r.metrica}_{v}": float(getattr(r, v))
                for r in general.itertuples()
                for v in ("a", "b", "b_menos_a", "ic95_bajo", "ic95_alto")
                if np.isfinite(getattr(r, v))
            }
        )
        for ruta in archivos.values():
            mlflow.log_artifact(str(ruta), "ab_test")
    cliente = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    cliente.set_model_version_tag(
        CONFIG_P2.modelo_registrado, version_b, "ab_decision", resumen["decision"]
    )
    cliente.set_model_version_tag(
        CONFIG_P2.modelo_registrado, version_b, "ab_periodo", resumen["periodo_actual"]
    )
    logger.info("A/B de P2:\n" + general.round(4).to_string(index=False))
    logger.success(f"Decisión: {resumen['decision']} ({resumen['criterio']}) | {output}")


if __name__ == "__main__":
    main()
