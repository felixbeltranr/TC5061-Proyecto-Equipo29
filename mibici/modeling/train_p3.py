"""Entrenamiento y registro en MLflow de la pregunta 3 (P3): baseline y candidatas de uso.

- ``kmeans_geo`` (baseline): KMeans sobre la ubicación de las estaciones (lat/lon en km).
- ``kmeans_uso`` y ``gmm_uso`` (candidatas): pipelines de ``modelos_p3.py`` sobre las variables
  de uso (selección por VIF, escalado y KMeans o GaussianMixture).

Todos se entrenan solo con el periodo de entrenamiento y se califican en el mismo espacio de uso:
silueta, Davies-Bouldin y Calinski-Harabasz sobre las variables de uso estandarizadas, en
entrenamiento (``train_``) y en evaluación (``test_``).

Lee los perfiles que genera ``python -m mibici.features_p3`` en ``data/interim``. Los
parámetros (fechas, semilla, k) están en ``CONFIG_P3`` (``mibici/config.py``).

Uso desde la raíz del repositorio::

    python -m mibici.dataset                           # 1. ingesta del equipo
    python -m mibici.features_p3                       # 2. perfiles de uso de P3
    python -m mibici.modeling.train_p3                 # 3. baseline con k = 4, 5 y 6
    python -m mibici.modeling.train_p3 --k 5           #    solo k = 5
    python -m mibici.modeling.train_p3 --no-registrar  #    métricas sin escribir en MLflow
    python -m mibici.modeling.train_p3 --model kmeans_uso --k 5
    python -m mibici.modeling.train_p3 --model gmm_uso --n-components 5 --covariance-type diag
    python -m mibici.modeling.train_p3 --model kmeans_uso --buscar   # búsqueda con runs anidados
    python -m mibici.modeling.train_p3 --model seleccion   # flujo completo: baseline, búsquedas,
                                                           # C1-C3, campeón y Model Registry

Además de las métricas de entrenamiento y evaluación, cada run lleva las de validación interna
por estaciones (``val_``) y la estabilidad (``val_ari_estabilidad``); ver ``busqueda_p3.py``.

La interfaz de MLflow se abre con ``mlflow ui --backend-store-uri sqlite:///mlflow.db``.
"""

from dataclasses import replace
from pathlib import Path
import subprocess
import tempfile
from typing import Annotated

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
import typer

from mibici.config import CONFIG, CONFIG_P3, PROJ_ROOT, P3Config
from mibici.features_p3 import DatasetP3, a_km, cargar_dataset
from mibici.modeling import artefactos_p3 as art
from mibici.modeling import seleccion_p3 as sel
from mibici.modeling.busqueda_p3 import evaluar_validacion, ordenar, rejilla
from mibici.modeling.modelos_p3 import (
    construir_modelo,
    hiperparametros_efectivos,
    nombre_corrida,
    validar_hiperparametros,
)

app = typer.Typer()


class Renumerado:
    """Envuelve un KMeans y numera sus grupos según una columna de los centroides.

    Así "zona 0" significa lo mismo en cualquier computadora (para el baseline, la zona más al
    poniente).
    """

    def __init__(self, km: KMeans, columna: int):
        self.km = km
        self.mapa = np.argsort(np.argsort(km.cluster_centers_[:, columna]))
        self.labels_ = self.mapa[km.labels_]

    def predict(self, X):
        return self.mapa[self.km.predict(X)]


def metricas(prefijo: str, X: np.ndarray, etiquetas) -> dict[str, float]:
    """Métricas internas de clustering calculadas en el espacio de uso ``X``."""
    return {
        f"{prefijo}_silhouette_uso": float(silhouette_score(X, etiquetas)),
        f"{prefijo}_davies_bouldin_uso": float(davies_bouldin_score(X, etiquetas)),
        f"{prefijo}_calinski_harabasz_uso": float(calinski_harabasz_score(X, etiquetas)),
    }


def espacio_uso(ds: DatasetP3) -> tuple[np.ndarray, np.ndarray]:
    """Variables de uso estandarizadas de entrenamiento y evaluación, donde se califica todo modelo.

    El escalado se aprende solo con el periodo de entrenamiento.
    """
    en, ev, vars_ = ds.entrenamiento, ds.evaluacion, ds.variables
    escalador = StandardScaler().fit(en[vars_])
    return escalador.transform(en[vars_]), escalador.transform(ev[vars_])


def entrenar_baseline(
    ds: DatasetP3, k: int, cfg: P3Config = CONFIG_P3
) -> tuple[Renumerado, dict[str, float]]:
    """Ajusta el baseline geográfico con k zonas y regresa el modelo y sus métricas."""
    en, ev = ds.entrenamiento, ds.evaluacion
    X_en, X_ev = espacio_uso(ds)
    G_en, G_ev = a_km(ds.nom, en.index, cfg.lat_ref), a_km(ds.nom, ev.index, cfg.lat_ref)
    km = Renumerado(KMeans(n_clusters=k, n_init=cfg.n_init, random_state=cfg.seed).fit(G_en), 0)
    m = {
        **metricas("train", X_en, km.labels_),
        **metricas("test", X_ev, km.predict(G_ev)),
        "train_silhouette_geo": float(silhouette_score(G_en, km.labels_)),
    }
    return km, m


def entrenar_candidata(
    ds: DatasetP3, model: str, params: dict, cfg: P3Config = CONFIG_P3
) -> tuple[object, dict[str, float]]:
    """Entrena una candidata de uso con el periodo de entrenamiento y regresa el modelo y métricas.

    El pipeline recibe las 12 variables candidatas y elige las suyas por VIF; las métricas se
    calculan en el mismo espacio de uso que el baseline.
    """
    entrada = list(cfg.prioridad_variables)
    modelo = construir_modelo(model, params, cfg).fit(ds.entrenamiento[entrada])
    X_en, X_ev = espacio_uso(ds)
    m = {
        **metricas("train", X_en, modelo.predict(ds.entrenamiento[entrada])),
        **metricas("test", X_ev, modelo.predict(ds.evaluacion[entrada])),
    }
    return modelo, m


def _git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=PROJ_ROOT, capture_output=True, text=True, check=False)
    return r.stdout.strip()


def etiquetas(
    ds: DatasetP3,
    cfg: P3Config = CONFIG_P3,
    familia: str = "kmeans_geo",
    etapa: str = "baseline",
) -> dict[str, str]:
    """Etiquetas del run con el mismo esquema que P2 (question, team, author, ...).

    ``familia`` va en ``model_family`` y ``etapa`` en ``stage_of_work`` (baseline, search o
    candidate).
    """
    sha = _git("rev-parse", "HEAD") or "sin-git"
    return {
        "question": "P3",
        "team": cfg.team,
        "author": cfg.author,
        "data_range": f"{cfg.meses_inicio}..{cfg.meses_fin}",
        "seed": str(cfg.seed),
        "data_version": ds.data_version,
        "mlflow.source.git.commit": sha,
        "git_commit": sha,
        "git_dirty": str(bool(_git("status", "--porcelain", "--untracked-files=no"))),
        "model_family": familia,
        "stage_of_work": etapa,
        "origen": "mibici/modeling/train_p3.py",
    }


def registrar_baseline(
    ds: DatasetP3, k: int, km: Renumerado, m: dict, cfg: P3Config = CONFIG_P3
) -> str:
    """Registra un run en MLflow con parámetros, métricas, etiquetas y artefactos."""
    nom, en, ev = ds.nom, ds.entrenamiento, ds.evaluacion
    with mlflow.start_run(run_name=f"p3-baseline-kmeans-geo-k{k}") as run:
        mlflow.set_tags(etiquetas(ds, cfg))
        mlflow.log_params(
            {
                "fecha_inicio_entrenamiento": cfg.fecha_inicio_entrenamiento,
                "fecha_fin_entrenamiento": cfg.fecha_fin_entrenamiento,
                "fecha_inicio_evaluacion": cfg.fecha_inicio_evaluacion,
                "fecha_fin_evaluacion": cfg.fecha_fin_evaluacion,
                "data_version": ds.data_version,
                "seed": cfg.seed,
                "modelo": "KMeans",
                "k": k,
                "n_init": cfg.n_init,
                "entrada_modelo": "x_km,y_km",
                "lat_ref": cfg.lat_ref,
                "variables_evaluacion": ",".join(ds.variables),
                "min_dias_activos": cfg.min_dias_activos,
                "duracion_min_seg": cfg.duracion_min_seg,
                "n_estaciones_entrenamiento": len(en),
                "n_estaciones_evaluacion": len(ev),
            }
        )
        mlflow.log_metrics(m)
        et_nv = km.predict(a_km(nom, ds.nuevas, cfg.lat_ref))
        _registrar_archivos(
            ds,
            cfg,
            et_en=km.labels_,
            et_ev=km.predict(a_km(nom, ev.index, cfg.lat_ref)),
            idx_nv=ds.nuevas,
            et_nv=et_nv,
            etiqueta="zona",
            titulo=f"P3 baseline: zonas geográficas (k={k})",
            config=art.config_corrida(
                ds, "kmeans_geo", {"k": k, "n_init": cfg.n_init}, ds.variables, cfg
            )
            | {"variables_entrada": ["x_km", "y_km"]},  # el baseline agrupa por ubicación
        )
        return run.info.run_id


def _registrar_archivos(
    ds: DatasetP3,
    cfg: P3Config,
    et_en: np.ndarray,
    et_ev: np.ndarray,
    idx_nv: pd.Index,
    et_nv: np.ndarray,
    etiqueta: str,
    titulo: str,
    config: dict,
) -> None:
    """Artefactos comunes del run activo en data/, plots/ y config/ (ver artefactos_p3.py)."""
    en = ds.entrenamiento
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for sub in ("data", "plots", "config"):
            (tmp / sub).mkdir()
        if cfg.manifiesto_path.exists():
            (tmp / "data" / cfg.manifiesto_path.name).write_bytes(cfg.manifiesto_path.read_bytes())
        art.asignacion_estaciones(ds, et_en, et_ev, etiqueta).to_csv(
            tmp / "data" / "asignacion_estaciones.csv"
        )
        perfil = art.perfil_por_grupo(en, et_en, list(cfg.prioridad_variables))
        perfil.rename_axis(etiqueta).round(4).to_csv(
            tmp / "data" / f"perfil_uso_por_{etiqueta}.csv"
        )
        ds.bitacora.to_csv(tmp / "data" / "bitacora_limpieza.csv", index=False)
        art.mapa_grupos(
            ds.nom,
            en.index,
            et_en,
            idx_nv,
            et_nv,
            titulo,
            tmp / "plots" / f"mapa_{etiqueta}s.png",
            etiqueta,
            cfg,
        )
        z = art.perfiles_z(en, et_en, ds.variables)
        art.figura_perfiles(
            z.rename_axis(etiqueta),
            perfil["n_estaciones"],
            f"{titulo}: perfil de uso (desv. estándar vs. promedio)",
            tmp / "plots" / f"perfiles_{etiqueta}s.png",
        )
        art.guardar_json(config, tmp / "config" / "config_corrida.json")
        mlflow.log_artifacts(str(tmp))


def registrar_candidata(
    ds: DatasetP3,
    model: str,
    params: dict,
    modelo,
    m: dict,
    cfg: P3Config = CONFIG_P3,
    extra_tags: dict | None = None,
    extra_data: dict[str, pd.DataFrame] | None = None,
) -> str:
    """Registra un run de una candidata de uso con parámetros, métricas, etiquetas y artefactos.

    Artefactos: el modelo con firma en ``model/``, los perfiles como entradas del run
    (``log_input``) y los archivos de ``data/``, ``plots/`` y ``config/``. ``extra_tags`` y
    ``extra_data`` (tablas que van a ``data/``) los agrega la selección del campeón.
    """
    nombre = nombre_corrida(model, params, cfg)
    with mlflow.start_run(run_name=nombre) as run:
        mlflow.set_tags(etiquetas(ds, cfg, familia=model, etapa="candidate") | (extra_tags or {}))
        for archivo, tabla in (extra_data or {}).items():
            with tempfile.TemporaryDirectory() as tmp:
                ruta = Path(tmp) / archivo
                tabla.round(4).to_csv(ruta, index=tabla.index.name is not None)
                mlflow.log_artifact(str(ruta), "data")
        mlflow.log_params(
            {
                "fecha_inicio_entrenamiento": cfg.fecha_inicio_entrenamiento,
                "fecha_fin_entrenamiento": cfg.fecha_fin_entrenamiento,
                "fecha_inicio_evaluacion": cfg.fecha_inicio_evaluacion,
                "fecha_fin_evaluacion": cfg.fecha_fin_evaluacion,
                "data_version": ds.data_version,
                "seed": cfg.seed,
                "modelo": model,
                **hiperparametros_efectivos(model, params, cfg),
                "entrada_modelo": ",".join(cfg.prioridad_variables),
                "umbral_vif": cfg.umbral_vif,
                "variables_evaluacion": ",".join(ds.variables),
                "min_dias_activos": cfg.min_dias_activos,
                "duracion_min_seg": cfg.duracion_min_seg,
                "n_estaciones_entrenamiento": len(ds.entrenamiento),
                "n_estaciones_evaluacion": len(ds.evaluacion),
            }
        )
        mlflow.log_metrics(m)
        entrada = list(cfg.prioridad_variables)
        en, ev = ds.entrenamiento, ds.evaluacion
        et_en, et_ev = modelo.predict(en[entrada]), modelo.predict(ev[entrada])
        nuevas = ev.index.isin(ds.nuevas)
        variables_vif = list(modelo.named_steps["vif"].variables_)
        mlflow.log_param("variables_vif", ",".join(variables_vif))
        _registrar_archivos(
            ds,
            cfg,
            et_en=et_en,
            et_ev=et_ev,
            idx_nv=ev.index[nuevas],
            et_nv=et_ev[nuevas],
            etiqueta="grupo",
            titulo=nombre,
            config=art.config_corrida(
                ds, model, hiperparametros_efectivos(model, params, cfg), variables_vif, cfg
            ),
        )
        art.registrar_datasets(ds, cfg)
        info = art.registrar_modelo(modelo, en.head(5), cfg)
        mlflow.set_tag("model_uri", info.model_uri)  # lo usa el registro en el Model Registry
        return run.info.run_id


def _nombre_busqueda(model: str, params: dict | None = None) -> str:
    """``p3-busqueda-kmeans-uso`` (padre) o ``p3-busqueda-kmeans-uso-k3`` (hija)."""
    if params is None:
        return f"p3-busqueda-{model.replace('_', '-')}"
    return nombre_corrida(model, params).replace("p3-", "p3-busqueda-", 1)


def buscar(
    ds: DatasetP3, model: str, cfg: P3Config = CONFIG_P3, registrar: bool = True
) -> list[dict]:
    """Búsqueda de ``model`` sobre su rejilla con validación por estaciones.

    Con ``registrar`` deja un run padre por búsqueda y un run hijo (anidado) por combinación.
    El padre guarda el ranking completo en ``config/ranking_busqueda.csv`` y la mejor
    combinación como parámetros ``mejor_*``. Regresa el ranking (ver ``busqueda_p3.ordenar``).
    """
    resultados = []
    contexto = mlflow.start_run(run_name=_nombre_busqueda(model)) if registrar else None
    try:
        if registrar:
            mlflow.set_tags(etiquetas(ds, cfg, familia=model, etapa="search"))
            mlflow.log_params(
                {
                    "modelo": model,
                    "n_particiones": cfg.n_particiones,
                    "empate_silueta": cfg.empate_silueta,
                    "data_version": ds.data_version,
                    "n_combinaciones": len(rejilla(model, cfg)),
                }
            )
        for params in rejilla(model, cfg):
            m = evaluar_validacion(ds, model, params, cfg)
            resultados.append({"model": model, "params": params, **m})
            logger.info(
                f"{model} {params}: silueta val {m['val_silhouette_uso']:.3f} | "
                f"ARI {m['val_ari_estabilidad']:.3f}"
            )
            if registrar:
                with mlflow.start_run(run_name=_nombre_busqueda(model, params), nested=True):
                    mlflow.set_tags(etiquetas(ds, cfg, familia=model, etapa="search"))
                    mlflow.log_params(
                        {"modelo": model, **hiperparametros_efectivos(model, params, cfg)}
                    )
                    mlflow.log_metrics(m)
        ranking = ordenar(resultados, cfg)
        if registrar:
            mejor = ranking[0]
            mlflow.log_params({f"mejor_{k}": v for k, v in mejor["params"].items()})
            mlflow.log_metrics({k: v for k, v in mejor.items() if k.startswith("val_")})
            tabla = pd.DataFrame(
                [
                    {
                        "posicion": r["posicion"],
                        **r["params"],
                        "empate_silueta": r["empate_silueta"],
                        **{k: v for k, v in r.items() if k.startswith("val_")},
                    }
                    for r in ranking
                ]
            )
            with tempfile.TemporaryDirectory() as tmp:
                ruta = Path(tmp) / "ranking_busqueda.csv"
                tabla.round(4).to_csv(ruta, index=False)
                mlflow.log_artifact(str(ruta), "config")
    finally:
        if contexto is not None:
            mlflow.end_run()
    return ranking


def seleccionar(ds: DatasetP3, cfg: P3Config = CONFIG_P3, registrar: bool = True) -> list[dict]:
    """Flujo completo de P3 con la regla de ``seleccion_p3``.

    1. Baseline geográfico con ``k = cfg.k_baseline`` (con métricas de validación).
    2. Búsquedas de KMeans de uso y GaussianMixture (runs anidados).
    3. Candidatas C1, C2 y C3 reentrenadas con todo el periodo de entrenamiento, con sus criterios
       de interpretabilidad.
    4. Ranking, campeón y retador; un run por candidata con ``data/criterios_interpretabilidad.csv``.
    5. Registro del campeón y del retador en el Model Registry: se registran las versiones, se
       carga y se prueba cada una, y solo entonces se mueven los aliases ``champion`` y
       ``challenger``; al final, prueba de carga por nombre y alias.
    """
    k = cfg.k_baseline
    km, m = entrenar_baseline(ds, k, cfg)
    m |= evaluar_validacion(ds, "kmeans_geo", {"k": k}, cfg)
    if registrar:
        registrar_baseline(ds, k, km, m, cfg)
    candidatas = sel.elegir_candidatas(
        buscar(ds, "kmeans_uso", cfg, registrar), buscar(ds, "gmm_uso", cfg, registrar), cfg
    )
    entrada = list(cfg.prioridad_variables)
    modelos, metricas_ = {}, {}
    for clave, c in candidatas.items():
        modelo, mm = entrenar_candidata(ds, c["model"], c["params"], cfg)
        modelos[clave] = modelo
        metricas_[clave] = mm | {n: v for n, v in c.items() if n.startswith("val_")}
        c["criterios"], c["grupos"] = sel.criterios(
            ds.entrenamiento, modelo.predict(ds.entrenamiento[entrada]), ds.variables, cfg
        )
    ranking = sel.elegir_campeon(candidatas, cfg)
    tabla = sel.tabla_seleccion(ranking)
    logger.info("Selección de P3:\n" + tabla.to_string(index=False))
    if not registrar:
        return ranking
    run_ids = {}
    for r in ranking:
        clave = r["candidata"]
        run_ids[clave] = registrar_candidata(
            ds,
            r["model"],
            r["params"],
            modelos[clave],
            metricas_[clave],
            cfg,
            extra_tags={
                "candidata": clave,
                "posicion_ranking": str(r["posicion"]),
                "seleccion": r["papel"] or "ninguno",
                "cumple_criterios": str(r["criterios"]["cumple"]),
            },
            extra_data={
                "criterios_interpretabilidad.csv": tabla,
                "grupos_interpretabilidad.csv": r["grupos"],
            },
        )
    ejemplo = ds.entrenamiento.head(5)
    versiones = sel.registrar_en_registry(ranking, run_ids, cfg.experimento, ejemplo, cfg)
    grupos = sel.verificar_alias(ejemplo, cfg.alias_campeon, cfg)  # verificación final por alias
    logger.success(
        f"{cfg.modelo_registrado}: versiones {versiones}; carga por "
        f"models:/{cfg.modelo_registrado}@{cfg.alias_campeon} OK (grupos {grupos.tolist()})"
    )
    return ranking


MODELOS_CLI = ("kmeans_geo", "kmeans_uso", "gmm_uso", "seleccion")


def validar_opciones(
    model: str,
    k: list[int] | None,
    n_components: list[int] | None,
    covariance_type: str | None,
    n_init: int | None,
    buscar_: bool = False,
) -> None:
    """Rechaza opciones de la CLI que no aplican al modelo, en lugar de ignorarlas.

    - ``kmeans_geo`` y ``kmeans_uso``: no aceptan ``--n-components`` ni ``--covariance-type``.
    - ``gmm_uso``: no acepta ``--k``.
    - ``seleccion``: no acepta hiperparámetros ni ``--buscar`` (todo sale de ``CONFIG_P3``).
    - ``--buscar``: solo con ``kmeans_uso`` o ``gmm_uso`` y sin hiperparámetros (usa la rejilla).
    """
    if model not in MODELOS_CLI:
        raise ValueError(f"Modelo de P3 no soportado: {model}. Opciones: {list(MODELOS_CLI)}")
    dadas = {
        "--k": k,
        "--n-components": n_components,
        "--covariance-type": covariance_type,
        "--n-init": n_init,
    }
    dadas = [nombre for nombre, valor in dadas.items() if valor not in (None, [])]
    if model == "seleccion" or buscar_:
        if buscar_ and model not in ("kmeans_uso", "gmm_uso"):
            raise ValueError("--buscar solo aplica a kmeans_uso o gmm_uso")
        no_aplican = dadas
    elif model == "gmm_uso":
        no_aplican = [o for o in dadas if o == "--k"]
    else:
        no_aplican = [o for o in dadas if o in ("--n-components", "--covariance-type")]
    if no_aplican:
        contexto = "--buscar" if buscar_ else model
        raise ValueError(f"Opciones que no aplican a {contexto}: {no_aplican}")


def combinaciones(
    model: str,
    k: list[int] | None,
    n_components: list[int] | None,
    covariance_type: str | None,
    n_init: int | None,
    cfg: P3Config = CONFIG_P3,
) -> list[dict]:
    """Hiperparámetros de cada run que pide la CLI (uno por valor de k o de n_components)."""
    extra = {"n_init": n_init} if n_init is not None else {}
    if model == "kmeans_geo":
        return [{"k": kk, **extra} for kk in k or cfg.k_comparacion]
    if model == "kmeans_uso":
        return [{"k": kk, **extra} for kk in k or [cfg.k_baseline]]
    cov = {"covariance_type": covariance_type or cfg.covarianza_gmm}
    return [{"n_components": n, **cov, **extra} for n in n_components or [cfg.k_baseline]]


@app.command()
def main(
    k: Annotated[list[int] | None, typer.Option(help="Número de zonas; se puede repetir.")] = None,
    perfiles_dir: Path = CONFIG.interim_data_dir,
    tracking_uri: str = CONFIG_P3.tracking_uri,
    registrar: Annotated[bool, typer.Option(help="Escribir los runs en MLflow.")] = True,
    model: Annotated[
        str,
        typer.Option(
            help="kmeans_geo (baseline), kmeans_uso, gmm_uso o seleccion (flujo completo)."
        ),
    ] = CONFIG_P3.model,
    n_init: int | None = None,
    n_components: Annotated[
        list[int] | None, typer.Option(help="Componentes de gmm_uso; se puede repetir.")
    ] = None,
    covariance_type: Annotated[
        str | None, typer.Option(help="Covarianza de gmm_uso: full, tied, diag o spherical.")
    ] = None,
    buscar_: Annotated[
        bool,
        typer.Option(
            "--buscar",
            help="Búsqueda de kmeans_uso o gmm_uso con runs anidados (rejilla de config).",
        ),
    ] = False,
):
    """Entrena el baseline o una candidata de P3 y registra un run de MLflow por combinación.

    Con ``--model seleccion`` corre el flujo completo (ver ``seleccionar``).
    """
    validar_opciones(model, k, n_components, covariance_type, n_init, buscar_)
    corridas = []
    if model != "seleccion":
        corridas = combinaciones(model, k, n_components, covariance_type, n_init)
        for params in corridas:
            validar_hiperparametros(model, params)
    cfg = replace(CONFIG_P3, n_init=n_init) if n_init is not None else CONFIG_P3
    ds = cargar_dataset(perfiles_dir)
    logger.info(
        f"data_version {ds.data_version} | entrenamiento {len(ds.entrenamiento)} | "
        f"evaluación {len(ds.evaluacion)} estaciones | variables {ds.variables}"
    )
    if registrar:
        mlflow.set_tracking_uri(tracking_uri)
        if mlflow.get_experiment_by_name(cfg.experimento) is None:
            mlflow.create_experiment(cfg.experimento, artifact_location=cfg.artifact_location)
        mlflow.set_experiment(cfg.experimento)
    if model == "seleccion":
        seleccionar(ds, cfg, registrar)
        return
    if buscar_:
        ranking = buscar(ds, model, cfg, registrar)
        mejor = ranking[0]
        logger.success(
            f"Mejor {model}: {mejor['params']} | silueta val {mejor['val_silhouette_uso']:.3f} "
            f"| ARI {mejor['val_ari_estabilidad']:.3f}"
        )
        return
    for params in corridas:
        if model == "kmeans_geo":
            km, m = entrenar_baseline(ds, params["k"], cfg)
        else:
            modelo, m = entrenar_candidata(ds, model, params, cfg)
        m |= evaluar_validacion(ds, model, params, cfg)
        resumen = " | ".join(f"{n} {v:.3f}" for n, v in m.items())
        logger.info(f"{model} {params}: {resumen}")
        if not registrar:
            continue
        if model == "kmeans_geo":
            run_id = registrar_baseline(ds, params["k"], km, m, cfg)
        else:
            run_id = registrar_candidata(ds, model, params, modelo, m, cfg)
        logger.success(f"Run registrado: {model} {params} ({run_id[:8]})")


if __name__ == "__main__":
    app()
