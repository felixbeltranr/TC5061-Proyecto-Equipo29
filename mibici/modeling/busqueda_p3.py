"""Validación interna por estaciones y búsqueda de hiperparámetros de la pregunta 3 (P3).

Todo usa solo el periodo de entrenamiento (enero 2025 a abril 2026); mayo a septiembre de 2026 no
interviene en ninguna decisión.

Validación por estaciones completas
    ``particiones`` divide las estaciones de entrenamiento en ``CONFIG_P3.n_particiones`` grupos
    (KFold reproducible): en cada partición se entrena con el 80 % de las estaciones y se valida
    con el 20 % reservado. Las particiones son las mismas para todos los modelos, incluido el
    baseline. En cada partición, la selección por VIF y el escalado del espacio de uso donde se
    califica se aprenden solo con el 80 %.

Métricas de validación (promedio de las particiones)
    ``val_silhouette_uso`` (y su desviación estándar), ``val_davies_bouldin_uso`` y
    ``val_calinski_harabasz_uso``, sobre las estaciones reservadas en el espacio de uso.
    Una partición es inválida si sus estaciones reservadas caen en un solo grupo (no hay
    silueta). ``val_particiones_validas`` cuenta las válidas y, si falta alguna, las métricas
    de validación quedan en NaN: la configuración se reporta pero no entra al ranking.

Estabilidad
    ``val_ari_estabilidad``: los modelos de las particiones (cada uno con su 80 % y su semilla)
    asignan grupo a todas las estaciones de entrenamiento; es el ARI promedio de todos los pares
    (10 con 5 particiones). Recoge la variación por datos y por semilla. Es comparativo: 0 es
    coincidencia por azar y 1 es coincidencia total.

Ranking (``ordenar``)
    Mayor silueta de validación; entre las que quedan a menos de ``CONFIG_P3.empate_silueta`` de
    la mejor, mayor ARI; luego menor Davies-Bouldin; luego menor complejidad.
"""

from dataclasses import replace
from itertools import combinations, product
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from mibici.config import CONFIG_P3, P3Config
from mibici.features_p3 import DatasetP3, a_km
from mibici.modeling.modelos_p3 import (
    COVARIANZAS,
    SeleccionVIF,
    construir_modelo,
    hiperparametros_efectivos,
)


def particiones(
    estaciones: pd.Index, cfg: P3Config = CONFIG_P3
) -> list[tuple[pd.Index, pd.Index]]:
    """Pares (entrenamiento, validación) de estaciones; la misma semilla da las mismas particiones."""
    kf = KFold(n_splits=cfg.n_particiones, shuffle=True, random_state=cfg.seed)
    return [(estaciones[a], estaciones[b]) for a, b in kf.split(np.arange(len(estaciones)))]


def _espacio_uso_particion(en: pd.DataFrame, cfg: P3Config) -> tuple[SeleccionVIF, StandardScaler]:
    """Selección por VIF y escalado aprendidos solo con las estaciones de entrenamiento."""
    entrada = list(cfg.prioridad_variables)
    vif = SeleccionVIF(cfg.prioridad_variables, cfg.umbral_vif).fit(en[entrada])
    return vif, StandardScaler().fit(vif.transform(en[entrada]))


def _metricas_validacion(X: np.ndarray, etiquetas: np.ndarray) -> dict[str, float]:
    """Métricas internas sobre las estaciones reservadas; NaN si cayeron en un solo grupo."""
    if len(set(etiquetas)) < 2:
        return dict.fromkeys(("silhouette", "davies_bouldin", "calinski_harabasz"), np.nan)
    return {
        "silhouette": float(silhouette_score(X, etiquetas)),
        "davies_bouldin": float(davies_bouldin_score(X, etiquetas)),
        "calinski_harabasz": float(calinski_harabasz_score(X, etiquetas)),
    }


def _ajustar_y_asignar(ds: DatasetP3, model: str, params: dict, en_idx: pd.Index, cfg: P3Config):
    """Entrena con las estaciones ``en_idx`` y regresa una función que asigna grupo a estaciones."""
    perfiles = ds.entrenamiento
    if model == "kmeans_geo":
        p = hiperparametros_efectivos(model, params, cfg)
        km = KMeans(n_clusters=p["k"], n_init=p["n_init"], random_state=cfg.seed)
        km.fit(a_km(ds.nom, en_idx, cfg.lat_ref))
        return lambda idx: km.predict(a_km(ds.nom, idx, cfg.lat_ref))
    entrada = list(cfg.prioridad_variables)
    modelo = construir_modelo(model, params, cfg).fit(perfiles.loc[en_idx, entrada])
    return lambda idx: modelo.predict(perfiles.loc[idx, entrada])


def evaluar_validacion(
    ds: DatasetP3, model: str, params: dict, cfg: P3Config = CONFIG_P3
) -> dict[str, float]:
    """Métricas de validación por estaciones y estabilidad (ARI) de una configuración."""
    todas = ds.entrenamiento.index
    por_particion, asignaciones = [], []
    inicio = time.perf_counter()
    for i, (en_idx, va_idx) in enumerate(particiones(todas, cfg)):
        cfg_i = replace(cfg, seed=cfg.seed + i)  # una semilla por partición
        asignar = _ajustar_y_asignar(ds, model, params, en_idx, cfg_i)
        vif, escala = _espacio_uso_particion(ds.entrenamiento.loc[en_idx], cfg)
        X_va = escala.transform(
            vif.transform(ds.entrenamiento.loc[va_idx, list(cfg.prioridad_variables)])
        )
        por_particion.append(_metricas_validacion(X_va, asignar(va_idx)))
        asignaciones.append(asignar(todas))
    segundos = time.perf_counter() - inicio
    m = pd.DataFrame(por_particion)
    validas = int(m.silhouette.notna().sum())
    if validas < len(m):  # regla explícita: solo se promedian configuraciones completas
        m = m.assign(silhouette=np.nan, davies_bouldin=np.nan, calinski_harabasz=np.nan)
    ari = [adjusted_rand_score(a, b) for a, b in combinations(asignaciones, 2)]
    return {
        "val_silhouette_uso": float(m.silhouette.mean()),
        "val_silhouette_uso_std": float(m.silhouette.std(ddof=0)),
        "val_davies_bouldin_uso": float(m.davies_bouldin.mean()),
        "val_calinski_harabasz_uso": float(m.calinski_harabasz.mean()),
        "val_ari_estabilidad": float(np.mean(ari)),
        "val_particiones_validas": float(validas),
        "val_fit_seconds": segundos,
    }


def complejidad(model: str, params: dict, cfg: P3Config = CONFIG_P3) -> tuple[int, int]:
    """Orden de complejidad: número de grupos y, en GMM, el tipo de covarianza (spherical < full)."""
    p = hiperparametros_efectivos(model, params, cfg)
    if model == "gmm_uso":
        orden = ("spherical", "diag", "tied", "full")
        return p["n_components"], orden.index(p["covariance_type"]) + 1
    return p["k"], 0


def ordenar(resultados: list[dict], cfg: P3Config = CONFIG_P3) -> list[dict]:
    """Ranking de configuraciones con el criterio acordado (ver docstring del módulo).

    Cada resultado lleva ``model``, ``params`` y las métricas de ``evaluar_validacion``. Regresa
    una copia ordenada de mejor a peor con ``posicion`` (1 = mejor) y ``empate_silueta`` (si
    quedó a menos de ``cfg.empate_silueta`` de la mejor silueta).
    """
    validos = [r for r in resultados if not np.isnan(r["val_silhouette_uso"])]
    if not validos:
        raise ValueError("Ninguna configuración tiene silueta de validación")
    mejor = max(r["val_silhouette_uso"] for r in validos)

    def clave(r):
        empatada = mejor - r["val_silhouette_uso"] < cfg.empate_silueta
        return (
            0 if empatada else 1,
            -r["val_ari_estabilidad"] if empatada else -r["val_silhouette_uso"],
            r["val_davies_bouldin_uso"],
            complejidad(r["model"], r["params"], cfg),
        )

    ordenados = sorted(validos, key=clave) + [r for r in resultados if r not in validos]
    return [
        {
            **r,
            "posicion": i,
            "empate_silueta": bool(mejor - r["val_silhouette_uso"] < cfg.empate_silueta),
        }
        for i, r in enumerate(ordenados, start=1)
    ]


def rejilla(model: str, cfg: P3Config = CONFIG_P3) -> list[dict]:
    """Combinaciones de la búsqueda de ``model`` según ``cfg``."""
    if model == "kmeans_uso":
        return [{"k": k} for k in cfg.k_busqueda]
    if model == "gmm_uso":
        covs = [c for c in cfg.covarianzas_busqueda if c in COVARIANZAS]
        return [
            {"n_components": n, "covariance_type": c}
            for n, c in product(cfg.componentes_busqueda, covs)
        ]
    raise ValueError(f"No hay búsqueda para {model}; opciones: kmeans_uso, gmm_uso")
