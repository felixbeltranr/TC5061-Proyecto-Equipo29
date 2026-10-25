"""Modelos de uso de la pregunta 3 (P3) como pipelines de scikit-learn.

Cada candidata es un ``Pipeline`` que recibe las 12 variables candidatas de uso
(``CONFIG_P3.prioridad_variables``) y hace por dentro todo lo que aprende de los datos:

1. ``SeleccionVIF``: conserva las variables con VIF < ``CONFIG_P3.umbral_vif`` según su prioridad.
2. ``StandardScaler``: estandariza las variables conservadas.
3. ``GruposOrdenados``: agrupa (KMeans o GaussianMixture) y numera los grupos de menor a mayor
   uso (``log_trips_day`` estandarizado), para que "grupo 0" signifique lo mismo en cualquier
   corrida.

Así la selección de variables y el escalado se aprenden solo con las estaciones con que se
entrena el pipeline (por ejemplo, el 80 % de una partición de validación) y el modelo
registrado recibe siempre las mismas 12 columnas.

El baseline geográfico (KMeans sobre lat/lon) no está aquí: vive en ``train_p3.py``.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mibici.config import CONFIG_P3, P3Config
from mibici.features_p3 import seleccionar_variables

# hiperparámetros que acepta cada modelo (el baseline geográfico se entrena en train_p3.py)
HIPERPARAMETROS = {
    "kmeans_geo": ("k", "n_init"),
    "kmeans_uso": ("k", "n_init"),
    "gmm_uso": ("n_components", "covariance_type", "n_init"),
}
COVARIANZAS = ("full", "tied", "diag", "spherical")


class SeleccionVIF(TransformerMixin, BaseEstimator):
    """Conserva las variables con VIF menor a ``umbral``, quitando primero las de menor prioridad.

    Aprende ``variables_`` en ``fit`` y en ``transform`` regresa solo esas columnas.
    """

    def __init__(self, prioridad=CONFIG_P3.prioridad_variables, umbral=CONFIG_P3.umbral_vif):
        self.prioridad = prioridad
        self.umbral = umbral

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.variables_ = seleccionar_variables(X, list(self.prioridad), self.umbral)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(X)[self.variables_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.variables_, dtype=object)


class GruposOrdenados(ClusterMixin, BaseEstimator):
    """Envuelve un KMeans o GaussianMixture y numera sus grupos por una columna de sus centros.

    Con ``columna=0`` (``log_trips_day`` tras la selección por VIF) el grupo 0 es el de menor uso.
    """

    def __init__(self, estimador=None, columna: int = 0):
        self.estimador = estimador
        self.columna = columna

    def fit(self, X, y=None):
        self.estimador_ = clone(self.estimador).fit(X)
        centros = getattr(self.estimador_, "cluster_centers_", None)
        if centros is None:
            centros = self.estimador_.means_
        self.mapa_ = np.argsort(np.argsort(np.asarray(centros)[:, self.columna]))
        self.labels_ = self.predict(X)
        return self

    def predict(self, X) -> np.ndarray:
        return self.mapa_[self.estimador_.predict(X)]


def validar_hiperparametros(model: str, params: dict) -> None:
    """Revisa nombres y valores de los hiperparámetros de ``model``; lanza ValueError si fallan."""
    if model not in HIPERPARAMETROS:
        raise ValueError(f"Modelo de P3 no soportado: {model}. Opciones: {list(HIPERPARAMETROS)}")
    sobran = sorted(set(params) - set(HIPERPARAMETROS[model]))
    if sobran:
        raise ValueError(f"Hiperparámetros no soportados por {model}: {sobran}")
    for nombre, minimo in (("k", 2), ("n_components", 2), ("n_init", 1)):
        valor = params.get(nombre)
        if valor is not None and (
            isinstance(valor, bool) or not isinstance(valor, int) or valor < minimo
        ):
            raise ValueError(f"{nombre} debe ser un entero mayor o igual a {minimo}")
    cov = params.get("covariance_type")
    if cov is not None and cov not in COVARIANZAS:
        raise ValueError(f"covariance_type debe ser uno de {COVARIANZAS}")


def hiperparametros_efectivos(model: str, params: dict, cfg: P3Config = CONFIG_P3) -> dict:
    """Hiperparámetros completos de ``model``: los indicados y, para el resto, los de ``cfg``."""
    base = {"n_init": cfg.n_init}
    if model in ("kmeans_geo", "kmeans_uso"):
        base["k"] = cfg.k_baseline
    else:
        base |= {"n_components": cfg.k_baseline, "covariance_type": cfg.covarianza_gmm}
    return base | dict(params)


def construir_modelo(
    model: str, params: dict | None = None, cfg: P3Config = CONFIG_P3
) -> Pipeline:
    """Pipeline sin entrenar de una candidata de uso (``kmeans_uso`` o ``gmm_uso``).

    Los hiperparámetros que no se indiquen se toman de ``cfg``.
    """
    validar_hiperparametros(model, dict(params or {}))
    p = hiperparametros_efectivos(model, params or {}, cfg)
    if model == "kmeans_uso":
        grupos = KMeans(n_clusters=p["k"], n_init=p["n_init"], random_state=cfg.seed)
    elif model == "gmm_uso":
        grupos = GaussianMixture(
            n_components=p["n_components"],
            covariance_type=p["covariance_type"],
            n_init=p["n_init"],
            random_state=cfg.seed,
        )
    else:
        raise ValueError(f"{model} no es un modelo de uso; el baseline se entrena en train_p3.py")
    return Pipeline(
        [
            ("vif", SeleccionVIF(cfg.prioridad_variables, cfg.umbral_vif)),
            ("escala", StandardScaler().set_output(transform="pandas")),
            ("grupos", GruposOrdenados(grupos, columna=0)),
        ]
    )


def nombre_corrida(model: str, params: dict, cfg: P3Config = CONFIG_P3) -> str:
    """Nombre descriptivo del run, por ejemplo ``p3-kmeans-uso-k5`` o ``p3-gmm-uso-n5-full``."""
    p = hiperparametros_efectivos(model, params, cfg)
    if model == "kmeans_uso":
        return f"p3-kmeans-uso-k{p['k']}"
    return f"p3-gmm-uso-n{p['n_components']}-{p['covariance_type']}"
