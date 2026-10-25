"""Artefactos de los runs de la pregunta 3 (P3), organizados por carpeta.

- ``model/``: el pipeline con firma (12 variables de uso -> grupo) y ejemplo de entrada.
- ``data/``: manifiesto de datos, asignación de estaciones, perfil de uso por grupo y bitácora
  de limpieza. Solo datos por estación (id, nombre, municipio, coordenadas y promedios de uso);
  ningún dato de las personas usuarias.
- ``plots/``: mapa de grupos y perfiles de uso por grupo.
- ``config/``: configuración de la corrida.
"""

from importlib.metadata import version
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: las figuras solo se guardan como artefactos
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
import skops.io as sio

from mibici.config import CONFIG_P3, P3Config
from mibici.features_p3 import DatasetP3

COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARCADORES = ["o", "s", "^", "D", "v", "P", "X", "*"]
# divergente: azul (bajo el promedio) - gris neutro (promedio) - rojo (sobre el promedio)
DIVERGENTE = LinearSegmentedColormap.from_list("p3_div", ["#2a78d6", "#f0efec", "#e34948"])
TINTA = "#0b0b0b"
# tipos que skops puede cargar: scikit-learn, NumPy y las clases propias de modelos_p3
PREFIJOS_CONFIABLES = ("sklearn.", "numpy.", "mibici.modeling.modelos_p3.")
COLUMNAS_ESTACION = ["name", "municipio", "latitude", "longitude"]


def mapa_grupos(
    nom: pd.DataFrame,
    idx_en: pd.Index,
    et_en: np.ndarray,
    idx_nv: pd.Index,
    et_nv: np.ndarray,
    titulo: str,
    ruta: Path,
    etiqueta: str = "grupo",
    cfg: P3Config = CONFIG_P3,
) -> Path:
    """Mapa de estaciones: relleno = periodo de entrenamiento; hueco = estaciones nuevas."""
    n = int(max(et_en.max(), et_nv.max() if len(et_nv) else 0)) + 1
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for g in range(n):
        est = idx_en[et_en == g]
        ax.scatter(
            nom.loc[est, "longitude"],
            nom.loc[est, "latitude"],
            s=16,
            color=COLORES[g],
            marker=MARCADORES[g],
            label=f"{etiqueta} {g}",
        )
        nv = idx_nv[et_nv == g]
        ax.scatter(
            nom.loc[nv, "longitude"],
            nom.loc[nv, "latitude"],
            s=22,
            facecolor="none",
            edgecolor=COLORES[g],
            marker=MARCADORES[g],
        )
    ax.scatter([], [], s=22, facecolor="none", edgecolor="#52514e", label="hueco: estación nueva")
    ax.set_aspect(1 / np.cos(np.radians(cfg.lat_ref)))
    ax.set_title(titulo, loc="left")
    ax.set_xlabel("longitud")
    ax.set_ylabel("latitud")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=min(n + 1, 5),
        fontsize=8,
        frameon=False,
    )
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ruta


def perfil_por_grupo(perfiles: pd.DataFrame, etiquetas: np.ndarray, variables: list[str]):
    """Número de estaciones, viajes por día y promedio de cada variable de uso por grupo."""
    tabla = perfiles.assign(grupo=etiquetas).groupby("grupo")
    return pd.concat(
        [tabla.size().rename("n_estaciones"), tabla[["trips_day", *variables]].mean()], axis=1
    )


def perfiles_z(perfiles: pd.DataFrame, etiquetas: np.ndarray, variables: list[str]):
    """Promedio de cada variable por grupo en desviaciones estándar del periodo de entrenamiento."""
    z = (perfiles[variables] - perfiles[variables].mean()) / perfiles[variables].std(ddof=0)
    return z.assign(grupo=etiquetas).groupby("grupo").mean()


def figura_perfiles(z: pd.DataFrame, n_estaciones: pd.Series, titulo: str, ruta: Path) -> Path:
    """Mapa de calor grupo x variable (desviaciones estándar respecto del promedio)."""
    lim = max(1.0, float(np.nanmax(np.abs(z.to_numpy()))))
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * z.shape[1], 0.8 + 0.55 * z.shape[0]))
    ax.imshow(z.to_numpy(), cmap=DIVERGENTE, vmin=-lim, vmax=lim, aspect="auto")
    for (i, j), v in np.ndenumerate(z.to_numpy()):
        ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8, color=TINTA)
    ax.set_xticks(range(z.shape[1]), z.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(
        range(z.shape[0]), [f"grupo {g} ({n_estaciones[g]})" for g in z.index], fontsize=8
    )
    ax.set_title(titulo, loc="left", fontsize=10)
    ax.tick_params(length=0)
    for lado in ax.spines.values():
        lado.set_visible(False)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ruta


def asignacion_estaciones(
    ds: DatasetP3, et_en: np.ndarray, et_ev: np.ndarray, etiqueta: str = "grupo"
) -> pd.DataFrame:
    """Grupo de cada estación en entrenamiento y en evaluación (solo datos de la estación)."""
    nom = ds.nom
    en = nom.loc[ds.entrenamiento.index, COLUMNAS_ESTACION].assign(
        **{etiqueta: et_en}, periodo="entrenamiento"
    )
    ev = nom.loc[ds.evaluacion.index, COLUMNAS_ESTACION].assign(
        **{etiqueta: et_ev}, periodo="evaluacion"
    )
    tabla = pd.concat([en, ev]).rename_axis("estacion")
    return tabla.assign(nueva=tabla.index.isin(ds.nuevas))


def tipos_confiables(modelo) -> list[str]:
    """Tipos que skops pide confirmar; se rechaza cualquiera fuera de PREFIJOS_CONFIABLES."""
    tipos = sio.get_untrusted_types(data=sio.dumps(modelo))
    ajenos = [t for t in tipos if not t.startswith(PREFIJOS_CONFIABLES)]
    if ajenos:
        raise ValueError(f"Tipos no confiables en el modelo: {ajenos}")
    return sorted(tipos)


def requisitos_modelo() -> list[str]:
    """Versiones instaladas de lo que necesita cargar el modelo."""
    return [f"{p}=={version(p)}" for p in ("scikit-learn", "pandas", "numpy", "skops")]


def registrar_modelo(modelo, ejemplo: pd.DataFrame, cfg: P3Config = CONFIG_P3):
    """Guarda el pipeline con firma (12 variables -> grupo) y ejemplo de entrada en ``model/``.

    Formato skops; las clases de ``mibici.modeling.modelos_p3`` se declaran confiables, así que
    quien lo cargue necesita el paquete ``mibici`` instalado (``pip install -e .``).
    """
    entrada = ejemplo[list(cfg.prioridad_variables)].astype("float64")
    salida = pd.Series(modelo.predict(entrada), name="grupo")
    return mlflow.sklearn.log_model(
        modelo,
        name="model",
        signature=infer_signature(entrada, salida),
        input_example=entrada,
        serialization_format="skops",
        skops_trusted_types=tipos_confiables(modelo),
        pip_requirements=requisitos_modelo(),
    )


def registrar_datasets(ds: DatasetP3, cfg: P3Config = CONFIG_P3) -> None:
    """Perfiles de entrenamiento y evaluación como entradas del run (esquema, perfil y digest)."""
    for contexto, df in (("training", ds.entrenamiento), ("evaluation", ds.evaluacion)):
        dataset = mlflow.data.from_pandas(
            df[list(cfg.prioridad_variables)].reset_index(), name=f"p3_perfiles_{contexto}"
        )
        mlflow.log_input(
            dataset,
            context=contexto,
            tags={"data_version": ds.data_version, "n_estaciones": str(len(df))},
        )


def guardar_json(datos: dict, ruta: Path) -> Path:
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False, default=str) + "\n")
    return ruta


def config_corrida(
    ds: DatasetP3, model: str, hiperparametros: dict, variables_vif: list[str], cfg: P3Config
) -> dict:
    """Configuración con que se ejecutó la corrida (para ``config/config_corrida.json``)."""
    return {
        "question": "P3",
        "modelo": model,
        "hiperparametros": hiperparametros,
        "variables_entrada": list(cfg.prioridad_variables),
        "variables_vif": variables_vif,
        "umbral_vif": cfg.umbral_vif,
        "periodo_entrenamiento": [cfg.fecha_inicio_entrenamiento, cfg.fecha_fin_entrenamiento],
        "periodo_evaluacion": [cfg.fecha_inicio_evaluacion, cfg.fecha_fin_evaluacion],
        "validacion": {
            "tipo": "KFold por estaciones",
            "n_particiones": cfg.n_particiones,
            "empate_silueta": cfg.empate_silueta,
        },
        "franjas": cfg.franjas,
        "min_dias_activos": cfg.min_dias_activos,
        "duracion_min_seg": cfg.duracion_min_seg,
        "semilla": cfg.seed,
        "data_version": ds.data_version,
        "nomenclatura": ds.nomenclatura,
    }
