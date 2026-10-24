"""Entrenamiento y registro en MLflow del baseline de la pregunta 3 (P3).

Baseline: KMeans sobre la ubicación de las estaciones (lat/lon en km). Se entrena solo con el
periodo de entrenamiento y se califica en el espacio de uso: silueta, Davies-Bouldin y
Calinski-Harabasz sobre las variables de uso estandarizadas, en entrenamiento (``train_``) y en
evaluación (``test_``).

Lee los perfiles que genera ``python -m mibici.features_p3`` en ``data/interim``. Los
parámetros (fechas, semilla, k) están en ``CONFIG_P3`` (``mibici/config.py``).

Uso desde la raíz del repositorio::

    python -m mibici.dataset                           # 1. ingesta del equipo
    python -m mibici.features_p3                       # 2. perfiles de uso de P3
    python -m mibici.modeling.train_p3                 # 3. baseline con k = 4, 5 y 6
    python -m mibici.modeling.train_p3 --k 5           #    solo k = 5
    python -m mibici.modeling.train_p3 --no-registrar  #    métricas sin escribir en MLflow

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

app = typer.Typer()
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARCADORES = ["o", "s", "^", "D", "v", "P", "X", "*"]


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


def entrenar_baseline(
    ds: DatasetP3, k: int, cfg: P3Config = CONFIG_P3
) -> tuple[Renumerado, dict[str, float]]:
    """Ajusta el baseline geográfico con k zonas y regresa el modelo y sus métricas."""
    en, ev, vars_ = ds.entrenamiento, ds.evaluacion, ds.variables
    escalador = StandardScaler().fit(en[vars_])  # solo con el periodo de entrenamiento
    X_en, X_ev = escalador.transform(en[vars_]), escalador.transform(ev[vars_])
    G_en, G_ev = a_km(ds.nom, en.index, cfg.lat_ref), a_km(ds.nom, ev.index, cfg.lat_ref)
    km = Renumerado(KMeans(n_clusters=k, n_init=cfg.n_init, random_state=cfg.seed).fit(G_en), 0)
    m = {
        **metricas("train", X_en, km.labels_),
        **metricas("test", X_ev, km.predict(G_ev)),
        "train_silhouette_geo": float(silhouette_score(G_en, km.labels_)),
    }
    return km, m


def _git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=PROJ_ROOT, capture_output=True, text=True, check=False)
    return r.stdout.strip()


def etiquetas(ds: DatasetP3, cfg: P3Config = CONFIG_P3) -> dict[str, str]:
    """Etiquetas del run con el mismo esquema que P2 (question, team, author, ...)."""
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
        "tipo": "baseline",
        "origen": "mibici/modeling/train_p3.py",
    }


def _mapa(ds: DatasetP3, km: Renumerado, k: int, ruta: Path, cfg: P3Config = CONFIG_P3) -> None:
    """Mapa de zonas: relleno = entrenamiento; hueco = estaciones nuevas asignadas con predict."""
    import matplotlib

    matplotlib.use("Agg")  # sin ventana: la figura solo se guarda como artefacto
    import matplotlib.pyplot as plt

    nom, en = ds.nom, ds.entrenamiento
    et_nuevas = km.predict(a_km(nom, ds.nuevas, cfg.lat_ref))
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for g in range(k):
        est = en.index[km.labels_ == g]
        ax.scatter(
            nom.loc[est, "longitude"],
            nom.loc[est, "latitude"],
            s=16,
            color=COLORES[g],
            marker=MARCADORES[g],
            label=f"zona {g}",
        )
        nv = ds.nuevas[et_nuevas == g]
        ax.scatter(
            nom.loc[nv, "longitude"],
            nom.loc[nv, "latitude"],
            s=22,
            facecolor="none",
            edgecolor=COLORES[g],
            marker=MARCADORES[g],
        )
    ax.set_aspect(1 / np.cos(np.radians(cfg.lat_ref)))
    ax.set_title(f"P3 baseline: zonas geográficas (k={k})", loc="left")
    ax.set_xlabel("longitud")
    ax.set_ylabel("latitud")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=k, fontsize=8, frameon=False)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _mapa(ds, km, k, tmp / f"p3_mapa_baseline_k{k}.png", cfg)
            cols = ["name", "municipio", "latitude", "longitude"]
            asignacion = pd.concat(
                [
                    nom.loc[en.index, cols].assign(zona=km.labels_, periodo="entrenamiento"),
                    nom.loc[ds.nuevas, cols].assign(
                        zona=km.predict(a_km(nom, ds.nuevas, cfg.lat_ref)),
                        periodo="nueva (predict)",
                    ),
                ]
            )
            asignacion.rename_axis("estacion").to_csv(tmp / "asignacion_estaciones.csv")
            (
                en.assign(zona=km.labels_)
                .groupby("zona")[["trips_day", *ds.variables]]
                .mean()
                .round(4)
                .to_csv(tmp / "perfil_uso_por_zona.csv")
            )
            ds.bitacora.to_csv(tmp / "bitacora_limpieza.csv", index=False)
            mlflow.log_artifacts(str(tmp))
        return run.info.run_id


@app.command()
def main(
    k: Annotated[list[int] | None, typer.Option(help="Número de zonas; se puede repetir.")] = None,
    perfiles_dir: Path = CONFIG.interim_data_dir,
    tracking_uri: str = CONFIG_P3.tracking_uri,
    registrar: Annotated[bool, typer.Option(help="Escribir los runs en MLflow.")] = True,
    model: str = CONFIG_P3.model,
    n_init: int | None = None,
):
    """Entrena el baseline geográfico de P3 y registra un run de MLflow por cada k."""
    if model != CONFIG_P3.model:
        raise ValueError(f"Unsupported P3 model: {model}")
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
    for kk in k or cfg.k_comparacion:
        km, m = entrenar_baseline(ds, kk, cfg)
        resumen = " | ".join(f"{n} {v:.3f}" for n, v in m.items())
        logger.info(f"k={kk}: {resumen}")
        if registrar:
            run_id = registrar_baseline(ds, kk, km, m, cfg)
            logger.success(f"Run registrado: p3-baseline-kmeans-geo-k{kk} ({run_id[:8]})")


if __name__ == "__main__":
    app()
