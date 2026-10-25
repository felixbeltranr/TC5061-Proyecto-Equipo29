"""Selección del campeón de la pregunta 3 (P3) y su registro en el Model Registry de MLflow.

Candidatas (entre las configuraciones de las búsquedas):

- **C1**: KMeans de uso con ``k = CONFIG_P3.k_baseline`` (la referencia de la entrega 1).
- **C2**: el mejor KMeans de uso de la búsqueda (el segundo si el mejor es C1).
- **C3**: la mejor GaussianMixture de la búsqueda.

Regla (todo con datos anteriores a mayo de 2026):

1. Ranking cuantitativo de ``busqueda_p3.ordenar``: silueta de validación, empate a menos de
   ``empate_silueta`` -> estabilidad (ARI), luego Davies-Bouldin, luego menor complejidad.
2. Criterios de interpretabilidad y utilidad (pasa / no pasa), con el modelo reentrenado en todo
   el periodo de entrenamiento:

   - **I1 tamaño**: cada grupo tiene al menos ``tamano_minimo_grupo`` de las estaciones.
   - **I2 distinguibles**: cada grupo se aleja al menos ``umbral_z`` desviaciones estándar del
     promedio en alguna variable, y cada par de grupos difiere al menos ``umbral_z`` en alguna.
   - **I3 nombrables**: cada grupo recibe un nombre por sus dos variables dominantes (con
     ``|z| >= umbral_z``) y no hay dos grupos con el mismo nombre.
   - **U1 rebalanceo**: hay un grupo que se vacía en la mañana (``net_am`` medio < 0) y otro que
     se llena (``net_am`` medio > 0).

3. Campeón = la primera del ranking que cumple I1, I2, I3 y U1; retador (``challenger``) = la
   siguiente que cumple. Si el campeón no es la primera del ranking, se documenta el motivo.
"""

from itertools import combinations

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd

from mibici.config import CONFIG_P3, P3Config
from mibici.modeling.artefactos_p3 import perfiles_z
from mibici.modeling.busqueda_p3 import ordenar

# Nombre de cada variable según su signo (debajo, encima del promedio) para el criterio I3
DESCRIPTORES = {
    "log_trips_day": ("poco uso", "mucho uso"),
    "weekend_share": ("de día hábil", "de fin de semana"),
    "net_am": ("se vacía en la mañana", "se llena en la mañana"),
    "net_pm": ("se vacía en la tarde", "se llena en la tarde"),
    "dep_am": ("pocas salidas de mañana", "salidas de mañana"),
    "dep_mid": ("pocas salidas de mediodía", "salidas de mediodía"),
    "dep_pm": ("pocas salidas de tarde", "salidas de tarde"),
    "dep_night": ("pocas salidas de noche", "salidas de noche"),
    "arr_am": ("pocas llegadas de mañana", "llegadas de mañana"),
    "arr_mid": ("pocas llegadas de mediodía", "llegadas de mediodía"),
    "arr_pm": ("pocas llegadas de tarde", "llegadas de tarde"),
    "arr_night": ("pocas llegadas de noche", "llegadas de noche"),
}
CRITERIO = (
    "Ranking: mayor silueta de validación por estaciones (5 particiones); a menos de 0.01 de la "
    "mejor decide la estabilidad (ARI), luego Davies-Bouldin y luego la menor complejidad. "
    "Campeón: la primera del ranking que cumple I1 tamaño, I2 distinguibles, I3 nombrables y "
    "U1 rebalanceo. Solo datos anteriores a mayo de 2026."
)


# Variables cuyo nombre describe un sentido de flujo: se nombran con el signo del valor original
FLUJOS = ("net_am", "net_pm")
EQUILIBRADO = {"net_am": "equilibrada en la mañana", "net_pm": "equilibrada en la tarde"}


def nombre_grupo(fila_z: pd.Series, umbral: float, fila_original: pd.Series | None = None) -> str:
    """Nombre de un grupo por sus dos variables dominantes; vacío si ninguna llega al umbral.

    La dominancia se mide con el valor estandarizado (``fila_z``: qué tanto se aparta del
    promedio). En ``net_am`` y ``net_pm`` el sentido ("se vacía" o "se llena") sale del signo
    del promedio original del grupo (``fila_original``), no del estandarizado: un grupo que se
    llena menos que el promedio sigue llenándose. Un flujo original exactamente 0 se nombra
    "equilibrada".
    """
    dominantes = fila_z[fila_z.abs() >= umbral].abs().sort_values(ascending=False).index[:2]

    def descriptor(v):
        if v in FLUJOS and fila_original is not None:
            if fila_original[v] == 0:
                return EQUILIBRADO[v]
            return DESCRIPTORES[v][int(fila_original[v] > 0)]
        return DESCRIPTORES[v][int(fila_z[v] > 0)]

    return " · ".join(descriptor(v) for v in dominantes)


def criterios(
    perfiles: pd.DataFrame, etiquetas: np.ndarray, variables: list[str], cfg: P3Config = CONFIG_P3
) -> tuple[dict, pd.DataFrame]:
    """Evalúa I1, I2, I3 y U1; regresa el resumen y una tabla por grupo.

    ``perfiles`` son las estaciones del periodo de entrenamiento y ``etiquetas`` su grupo.
    """
    z = perfiles_z(perfiles, etiquetas, variables)
    proporcion = pd.Series(etiquetas).value_counts(normalize=True).sort_index()
    net_am = perfiles.assign(grupo=etiquetas).groupby("grupo")["net_am"].mean()
    medias = perfiles[variables].assign(grupo=etiquetas).groupby("grupo").mean()
    nombres = pd.Series({g: nombre_grupo(z.loc[g], cfg.umbral_z, medias.loc[g]) for g in z.index})
    z_grupo = z.abs().max(axis=1)
    z_par = [float((z.loc[a] - z.loc[b]).abs().max()) for a, b in combinations(z.index, 2)]
    resumen = {
        "I1_tamano": bool(proporcion.min() >= cfg.tamano_minimo_grupo),
        "I2_distinguibles": bool(
            (z_grupo >= cfg.umbral_z).all() and min(z_par, default=0) >= cfg.umbral_z
        ),
        "I3_nombrables": bool((nombres != "").all() and nombres.is_unique),
        "U1_rebalanceo": bool((net_am < 0).any() and (net_am > 0).any()),
        "proporcion_minima": float(proporcion.min()),
        "z_minimo_grupo": float(z_grupo.min()),
        "z_minimo_par": min(z_par, default=0.0),
    }
    resumen["cumple"] = all(
        resumen[c] for c in ("I1_tamano", "I2_distinguibles", "I3_nombrables", "U1_rebalanceo")
    )
    tabla = pd.DataFrame(
        {
            "n_estaciones": pd.Series(etiquetas).value_counts().sort_index(),
            "proporcion": proporcion,
            "net_am_medio": net_am,
            "z_maximo": z_grupo,
            "nombre": nombres,
        }
    ).rename_axis("grupo")
    return resumen, tabla


def elegir_candidatas(ranking_kmeans: list[dict], ranking_gmm: list[dict], cfg=CONFIG_P3):
    """C1, C2 y C3 a partir de los rankings de las búsquedas (ver docstring del módulo)."""
    c1 = next(r for r in ranking_kmeans if r["params"]["k"] == cfg.k_baseline)
    c2 = next(r for r in ranking_kmeans if r["params"]["k"] != cfg.k_baseline)
    return {"C1": c1, "C2": c2, "C3": ranking_gmm[0]}


def elegir_campeon(candidatas: dict[str, dict], cfg: P3Config = CONFIG_P3) -> list[dict]:
    """Ranking de candidatas con su papel (champion, challenger o vacío) y el motivo.

    Cada candidata trae sus métricas de validación y ``criterios`` (resumen de ``criterios``).
    Solo pueden ser champion o challenger las candidatas con validación completa (silueta de
    validación definida en todas las particiones) que además cumplen los criterios.
    """
    filas = [{**c, "candidata": clave} for clave, c in candidatas.items()]
    ranking = ordenar(filas, cfg)
    completa = {id(r): not np.isnan(r["val_silhouette_uso"]) for r in ranking}
    aptas = [r for r in ranking if completa[id(r)] and r["criterios"]["cumple"]]
    if not aptas:
        raise ValueError(
            "Ninguna candidata tiene validación completa y cumple los criterios de "
            "interpretabilidad"
        )
    campeon = aptas[0]
    retador = aptas[1] if len(aptas) > 1 else None
    for r in ranking:
        r["papel"] = (
            cfg.alias_campeon if r is campeon else cfg.alias_retador if r is retador else ""
        )
        fallas = [
            c
            for c in ("I1_tamano", "I2_distinguibles", "I3_nombrables", "U1_rebalanceo")
            if not r["criterios"][c]
        ]
        r["motivo"] = (
            "primera del ranking que cumple los criterios"
            if r is campeon and r["posicion"] == 1
            else "las anteriores del ranking no cumplen los criterios"
            if r is campeon
            else "validación incompleta (alguna partición inválida)"
            if not completa[id(r)]
            else "no cumple " + ", ".join(fallas)
            if fallas
            else "cumple, pero queda detrás en el ranking"
        )
    return ranking


def tabla_seleccion(ranking: list[dict]) -> pd.DataFrame:
    """Tabla de candidatas para ``data/criterios_interpretabilidad.csv``."""
    return pd.DataFrame(
        [
            {
                "candidata": r["candidata"],
                "posicion": r["posicion"],
                "papel": r["papel"],
                "modelo": r["model"],
                **r["params"],
                "val_silhouette_uso": r["val_silhouette_uso"],
                "val_ari_estabilidad": r["val_ari_estabilidad"],
                "val_davies_bouldin_uso": r["val_davies_bouldin_uso"],
                **r["criterios"],
                "motivo": r["motivo"],
            }
            for r in ranking
        ]
    )


def comprobar_version(
    nombre: str, version: str, ejemplo: pd.DataFrame, cfg: P3Config = CONFIG_P3
) -> np.ndarray:
    """Carga una versión por ``models:/<nombre>/<versión>`` y comprueba su predicción.

    La predicción debe tener un grupo por fila, sin faltantes, y grupos enteros no negativos.
    Lanza ValueError si no se cumple (o el error de MLflow si no carga).
    """
    modelo = mlflow.pyfunc.load_model(f"models:/{nombre}/{version}")
    entrada = ejemplo[list(cfg.prioridad_variables)].astype("float64")
    pred = np.asarray(modelo.predict(entrada), dtype=float).ravel()
    if len(pred) != len(entrada):
        raise ValueError(
            f"La versión {version} predijo {len(pred)} grupos para {len(entrada)} filas"
        )
    if not np.isfinite(pred).all() or (pred < 0).any() or (pred != np.round(pred)).any():
        raise ValueError(f"La versión {version} predijo grupos inválidos: {pred.tolist()}")
    return pred.astype(int)


def registrar_en_registry(
    ranking: list[dict],
    run_ids: dict[str, str],
    experimento: str,
    ejemplo: pd.DataFrame,
    cfg: P3Config = CONFIG_P3,
) -> dict[str, str]:
    """Registra campeón y retador como versiones nuevas y solo después mueve sus aliases.

    1. Registra cada versión con su descripción y su linaje: run de origen, commit, versión de
       datos, experimento, candidata, posición en el ranking y motivo.
    2. Carga cada versión nueva por ``models:/<nombre>/<versión>`` y comprueba su predicción
       con ``ejemplo`` (``comprobar_version``).
    3. Solo si todas pasan, mueve los aliases. Si alguna falla, marca las versiones nuevas con
       ``comprobacion = fallida``, no toca los aliases (siguen en las versiones anteriores) y
       lanza el error.

    Regresa {alias: versión}.
    """
    cliente = MlflowClient()
    nombre = cfg.modelo_registrado
    try:
        cliente.get_registered_model(nombre)
    except MlflowException:
        cliente.create_registered_model(
            nombre,
            tags={"question": "P3", "team": cfg.team},
            description=(
                "P3: grupos de estaciones MiBici con perfiles de uso similares. Entrada: las 12 "
                "variables de uso por estación (ver firma); salida: grupo (0 = menor uso). "
                f"{CRITERIO}"
            ),
        )
    nuevas = {}  # alias -> versión registrada, todavía sin alias
    for r in ranking:
        if not r["papel"]:
            continue
        run_id = run_ids[r["candidata"]]
        run = cliente.get_run(run_id)
        version = mlflow.register_model(
            run.data.tags.get("model_uri", f"runs:/{run_id}/model"),
            nombre,
            tags={
                "run_id": run_id,
                "git_commit": run.data.tags.get("git_commit", ""),
                "data_version": run.data.tags.get("data_version", ""),
                "experimento": experimento,
                "candidata": r["candidata"],
                "posicion_ranking": str(r["posicion"]),
                "motivo": r["motivo"],
            },
        )
        cliente.update_model_version(
            nombre,
            version.version,
            description=(
                f"{r['candidata']} ({r['model']} {r['params']}), {r['papel']}. Validación por "
                f"estaciones: silueta {r['val_silhouette_uso']:.3f}, ARI "
                f"{r['val_ari_estabilidad']:.3f}, Davies-Bouldin "
                f"{r['val_davies_bouldin_uso']:.3f}. Posición {r['posicion']} del ranking; "
                f"{r['motivo']}."
            ),
        )
        nuevas[r["papel"]] = version.version
    try:
        for version in nuevas.values():
            comprobar_version(nombre, version, ejemplo, cfg)
    except Exception:
        for version in nuevas.values():
            cliente.set_model_version_tag(nombre, version, "comprobacion", "fallida")
        raise
    for alias, version in nuevas.items():
        cliente.set_model_version_tag(nombre, version, "comprobacion", "ok")
        cliente.set_registered_model_alias(nombre, alias, version)
    return nuevas


def verificar_alias(ejemplo: pd.DataFrame, alias: str, cfg: P3Config = CONFIG_P3) -> np.ndarray:
    """Carga el modelo por nombre y alias desde el Registry y asigna grupo al ejemplo."""
    modelo = mlflow.pyfunc.load_model(f"models:/{cfg.modelo_registrado}@{alias}")
    return np.asarray(modelo.predict(ejemplo[list(cfg.prioridad_variables)].astype("float64")))
