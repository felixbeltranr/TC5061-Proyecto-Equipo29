"""A/B testing de P3: versión vigente (A) contra versión reentrenada (B) en el mes actual.

- **A**: el campeón. Por omisión se reconstruye con la misma configuración y datos con que se
  entrenó (``kmeans_uso`` con k = 3, enero de 2025 a abril de 2026); con ``--modelo-a`` se carga
  del Model Registry (por ejemplo ``models:/mibici-equipo29-p3-grupos@champion``).
- **B**: el mismo pipeline reentrenado con más datos (enero de 2025 a ``--fin-entrenamiento-b``,
  que incluye las estaciones nuevas), o una versión del Registry con ``--modelo-b``.

Las dos versiones asignan grupo a las mismas estaciones: los perfiles del mes actual (por
ejemplo, octubre de 2026), que ninguna vio al entrenar. Es un A/B fuera de línea con diseño
pareado: no se reparte tráfico, cada estación recibe el grupo de A y el de B.

Como un modelo de grupos no tiene etiqueta verdadera, se comparan:

1. **Separación de los grupos** (métrica principal): silueta, Davies-Bouldin y
   Calinski-Harabasz. Cada versión elige sus variables por VIF, así que calificar en el espacio
   de una favorece a esa versión; la decisión se toma en el espacio neutral de las 12 variables
   de entrada estandarizadas con el mes actual, y los otros dos espacios se reportan aparte.
2. **Estabilidad**: con los perfiles de la primera y la segunda quincena del mes, qué tanto se
   mantiene el grupo de cada estación (ARI entre quincenas).
3. **Tamaño mínimo de grupo** (criterio I1 de la selección del campeón).

Las diferencias B - A llevan un intervalo de confianza del 95 % con bootstrap pareado por
estaciones. **Regla de decisión** (fijada antes de ver los resultados): B reemplaza a A solo si
su silueta en el espacio neutral es mayor con un intervalo que no incluye el cero, su
estabilidad no es significativamente menor y todos sus grupos tienen al menos
``CONFIG_P3.tamano_minimo_grupo`` de las estaciones. Si no, se mantiene A.

Con ``--registrar`` la versión B se registra en MLflow como nueva versión de
``CONFIG_P3.modelo_registrado`` con el alias ``challenger``; el alias ``champion`` nunca se mueve
aquí: promover es una decisión que se toma con este reporte.

Uso desde la raíz del repositorio (después del reporte de drift de P3)::

    python -m mibici.monitoring.p3_ab
    python -m mibici.monitoring.p3_ab --registrar --tracking-uri http://127.0.0.1:5000
    python -m mibici.monitoring.p3_ab --tracking-uri http://127.0.0.1:5000 \\
        --modelo-a models:/mibici-equipo29-p3-grupos@champion \\
        --modelo-b models:/mibici-equipo29-p3-grupos@challenger
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Annotated

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
import typer

from mibici.config import CONFIG, CONFIG_P3, P3Config
from mibici.features_p3 import (
    cargar_flujos,
    elegibles,
    filtrar_periodo,
    leer_nomenclatura,
    nomenclatura_mas_reciente,
    perfil_estaciones,
    version_manifiesto,
)
from mibici.modeling import artefactos_p3 as art
from mibici.modeling.modelos_p3 import construir_modelo, nombre_corrida
from mibici.monitoring.p1_drift import file_sha256, month_bounds
from mibici.monitoring.p3_datasets import (
    MIN_DIAS_ACTIVOS_MES,
    P3_FEATURES,
    flujos_del_mes_actual,
)

app = typer.Typer()

MODELO = "kmeans_uso"  # familia del campeón vigente
MIN_DIAS_QUINCENA = 7
N_BOOTSTRAP = 1000


def perfiles_periodo(
    flujos: pd.DataFrame, inicio: str, fin: str, ids_con_ubicacion, min_dias: int, cfg: P3Config
) -> pd.DataFrame:
    """Perfil de uso (12 variables de entrada) de cada estación elegible entre dos fechas."""
    perfil = perfil_estaciones(filtrar_periodo(flujos, inicio, fin), cfg.franjas)
    return elegibles(perfil, ids_con_ubicacion, min_dias)[list(P3_FEATURES)]


def entrenar(perfiles: pd.DataFrame, params: dict, cfg: P3Config = CONFIG_P3):
    """Pipeline de P3 (VIF, escalado y grupos) entrenado con los perfiles dados."""
    return construir_modelo(MODELO, params, cfg).fit(perfiles[list(P3_FEATURES)])


def variables_vif(modelo) -> list[str]:
    """Variables que conservó el paso de VIF del pipeline."""
    return list(modelo.named_steps["vif"].variables_)


def predecir(modelo, perfiles: pd.DataFrame) -> np.ndarray:
    return np.asarray(modelo.predict(perfiles[list(P3_FEATURES)].astype("float64")), dtype=int)


def calidad(X: np.ndarray, etiquetas: np.ndarray) -> dict[str, float]:
    """Silueta, Davies-Bouldin y Calinski-Harabasz de una asignación en el espacio ``X``."""
    return {
        "silueta": float(silhouette_score(X, etiquetas)),
        "davies_bouldin": float(davies_bouldin_score(X, etiquetas)),
        "calinski_harabasz": float(calinski_harabasz_score(X, etiquetas)),
    }


def intervalo_pareado(
    funcion, n: int, n_bootstrap: int = N_BOOTSTRAP, seed: int = CONFIG_P3.seed
) -> tuple[float, float]:
    """IC del 95 % de ``funcion(indices)`` remuestreando estaciones con reemplazo.

    ``funcion`` regresa la diferencia B - A calculada con las mismas estaciones para las dos
    versiones; las muestras donde no se puede calcular (un solo grupo) se descartan.
    """
    rng = np.random.default_rng(seed)
    valores = []
    for _ in range(n_bootstrap):
        try:
            valores.append(funcion(rng.integers(0, n, n)))
        except ValueError:
            continue
    bajo, alto = np.percentile(valores, [2.5, 97.5])
    return float(bajo), float(alto)


def comparar(
    modelo_a,
    modelo_b,
    actual: pd.DataFrame,
    quincena_1: pd.DataFrame,
    quincena_2: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = CONFIG_P3.seed,
) -> pd.DataFrame:
    """Tabla A/B: una fila por métrica y espacio, con A, B, B - A y su IC del 95 %."""
    g_a, g_b = predecir(modelo_a, actual), predecir(modelo_b, actual)
    espacios = {
        "neutral_12_variables": list(P3_FEATURES),
        "variables_vif_a": variables_vif(modelo_a),
        "variables_vif_b": variables_vif(modelo_b),
    }
    filas = []
    for espacio, variables in espacios.items():
        X = StandardScaler().fit_transform(actual[variables])
        q_a, q_b = calidad(X, g_a), calidad(X, g_b)
        for metrica, mejor in (
            ("silueta", "mayor"),
            ("davies_bouldin", "menor"),
            ("calinski_harabasz", "mayor"),
        ):
            fn = {
                "silueta": silhouette_score,
                "davies_bouldin": davies_bouldin_score,
                "calinski_harabasz": calinski_harabasz_score,
            }[metrica]
            bajo, alto = intervalo_pareado(
                lambda i, X=X, fn=fn: fn(X[i], g_b[i]) - fn(X[i], g_a[i]),
                len(X),
                n_bootstrap,
                seed,
            )
            filas.append(
                {
                    "metrica": metrica,
                    "espacio": espacio,
                    "mejor": mejor,
                    "a": q_a[metrica],
                    "b": q_b[metrica],
                    "b_menos_a": q_b[metrica] - q_a[metrica],
                    "ic95_bajo": bajo,
                    "ic95_alto": alto,
                }
            )

    comunes = quincena_1.index.intersection(quincena_2.index)
    pares = {
        v: (predecir(m, quincena_1.loc[comunes]), predecir(m, quincena_2.loc[comunes]))
        for v, m in (("a", modelo_a), ("b", modelo_b))
    }
    ari = {v: adjusted_rand_score(*p) for v, p in pares.items()}
    bajo, alto = intervalo_pareado(
        lambda i: (
            adjusted_rand_score(pares["b"][0][i], pares["b"][1][i])
            - adjusted_rand_score(pares["a"][0][i], pares["a"][1][i])
        ),
        len(comunes),
        n_bootstrap,
        seed,
    )
    filas.append(
        {
            "metrica": "ari_entre_quincenas",
            "espacio": "asignaciones",
            "mejor": "mayor",
            "a": ari["a"],
            "b": ari["b"],
            "b_menos_a": ari["b"] - ari["a"],
            "ic95_bajo": bajo,
            "ic95_alto": alto,
        }
    )
    minimo = {v: np.bincount(g).min() / len(g) for v, g in (("a", g_a), ("b", g_b))}
    filas.append(
        {
            "metrica": "proporcion_grupo_mas_chico",
            "espacio": "asignaciones",
            "mejor": "mayor",
            "a": minimo["a"],
            "b": minimo["b"],
            "b_menos_a": minimo["b"] - minimo["a"],
            "ic95_bajo": np.nan,
            "ic95_alto": np.nan,
        }
    )
    return pd.DataFrame(filas)


def decidir(tabla: pd.DataFrame, cfg: P3Config = CONFIG_P3) -> dict:
    """Aplica la regla de decisión del módulo a la tabla de ``comparar``."""
    fila = tabla.set_index(["metrica", "espacio"])
    silueta = fila.loc[("silueta", "neutral_12_variables")]
    estabilidad = fila.loc[("ari_entre_quincenas", "asignaciones")]
    tamano = fila.loc[("proporcion_grupo_mas_chico", "asignaciones")]
    condiciones = {
        "silueta_neutral_mejor": bool(silueta.ic95_bajo > 0),
        "estabilidad_no_peor": bool(estabilidad.ic95_alto >= 0),
        "grupos_suficientes": bool(tamano.b >= cfg.tamano_minimo_grupo),
    }
    promover = all(condiciones.values())
    return {
        "decision": "promover B" if promover else "mantener A",
        "condiciones": condiciones,
        "silueta_neutral": {
            "a": silueta.a,
            "b": silueta.b,
            "ic95": [silueta.ic95_bajo, silueta.ic95_alto],
        },
        "ari_entre_quincenas": {
            "a": estabilidad.a,
            "b": estabilidad.b,
            "ic95": [estabilidad.ic95_bajo, estabilidad.ic95_alto],
        },
    }


def asignaciones(
    modelo_a, modelo_b, actual: pd.DataFrame, entrenamiento_a: pd.DataFrame
) -> pd.DataFrame:
    """Grupo de A y de B por estación del mes actual, y si es nueva para A."""
    return pd.DataFrame(
        {
            "grupo_a": predecir(modelo_a, actual),
            "grupo_b": predecir(modelo_b, actual),
            "nueva_para_a": ~actual.index.isin(entrenamiento_a.index),
        },
        index=actual.index,
    )


def registrar_version_b(
    modelo_b,
    perfiles_b: pd.DataFrame,
    params: dict,
    tabla: pd.DataFrame,
    resumen: dict,
    archivos: list[Path],
    cfg: P3Config,
    tracking_uri: str,
) -> str:
    """Registra B en un run y como nueva versión del modelo de P3 con el alias ``challenger``."""
    from mlflow import MlflowClient

    from mibici.modeling.seleccion_p3 import comprobar_version
    from mibici.tracking import configurar_experimento, etiquetas_base

    configurar_experimento(tracking_uri, cfg.experimento, cfg.artifact_location)
    data_version = version_manifiesto(cfg.manifiesto_path)
    fin = cfg.fecha_fin_entrenamiento
    nombre = f"{nombre_corrida(MODELO, params, cfg)}-reentrenado-{fin[:7]}"
    metricas = {
        f"ab_{r.metrica}_{r.espacio}_{v}": float(getattr(r, v))
        for r in tabla.itertuples()
        for v in ("a", "b", "b_menos_a")
    }
    with mlflow.start_run(run_name=nombre) as run:
        mlflow.set_tags(
            etiquetas_base(
                "P3",
                cfg.author,
                cfg.seed,
                f"{cfg.fecha_inicio_entrenamiento}..{fin}",
                data_version,
            )
            | {
                "model_family": MODELO,
                "stage_of_work": "ab_test",
                "origen": "mibici/monitoring/p3_ab.py",
                "ab_decision": resumen["decision"],
                "ab_periodo": resumen["periodo_actual"],
            }
        )
        mlflow.log_params(
            {
                "fecha_inicio_entrenamiento": cfg.fecha_inicio_entrenamiento,
                "fecha_fin_entrenamiento": fin,
                "modelo": MODELO,
                **params,
                "seed": cfg.seed,
                "entrada_modelo": ",".join(P3_FEATURES),
                "variables_vif": ",".join(variables_vif(modelo_b)),
                "min_dias_activos": cfg.min_dias_activos,
                "n_estaciones_entrenamiento": len(perfiles_b),
                "ab_min_dias_mes": resumen["min_dias_mes"],
                "ab_n_estaciones": resumen["n_estaciones_actual"],
            }
        )
        mlflow.log_metrics({k: v for k, v in metricas.items() if np.isfinite(v)})
        for archivo in archivos:
            mlflow.log_artifact(str(archivo), "ab_test")
        info = art.registrar_modelo(modelo_b, perfiles_b.head(5), cfg)
        run_id = run.info.run_id

    cliente = MlflowClient()
    version = mlflow.register_model(
        info.model_uri,
        cfg.modelo_registrado,
        tags={
            "run_id": run_id,
            "data_version": data_version,
            "origen": "mibici/monitoring/p3_ab.py",
            "ab_decision": resumen["decision"],
        },
    )
    cliente.update_model_version(
        cfg.modelo_registrado,
        version.version,
        description=(
            f"{MODELO} {params} reentrenado con {cfg.fecha_inicio_entrenamiento} a {fin}. "
            f"A/B en {resumen['periodo_actual']}: {resumen['decision']}."
        ),
    )
    comprobar_version(cfg.modelo_registrado, version.version, perfiles_b.head(5), cfg)
    cliente.set_model_version_tag(cfg.modelo_registrado, version.version, "comprobacion", "ok")
    cliente.set_registered_model_alias(cfg.modelo_registrado, cfg.alias_retador, version.version)
    return version.version


def cargar_modelo(uri: str, tracking_uri: str):
    """Carga un pipeline de P3 del Registry (formato sklearn, para ver su paso de VIF)."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    return mlflow.sklearn.load_model(uri)


@app.command()
def main(
    historical_trips: Path = CONFIG.processed_data_dir / CONFIG.output_filename,
    current_trips: Annotated[
        Path | None, typer.Option(help="CSV del mes actual (por omisión el de data/raw).")
    ] = None,
    current_start: str = "2026-10-01",
    current_end_exclusive: str = "2026-11-01",
    nomenclature: Path | None = None,
    fin_entrenamiento_b: str = "2026-09-30",
    k: int = 3,
    modelo_a: Annotated[str | None, typer.Option(help="URI del Registry para A.")] = None,
    modelo_b: Annotated[str | None, typer.Option(help="URI del Registry para B.")] = None,
    registrar: Annotated[bool, typer.Option(help="Registrar B en MLflow (challenger).")] = False,
    tracking_uri: str = CONFIG_P3.tracking_uri,
    output: Path | None = None,
    n_bootstrap: int = N_BOOTSTRAP,
):
    """Compara la versión vigente de P3 (A) con la reentrenada (B) en el mes actual."""
    if registrar and modelo_b:
        raise typer.BadParameter("--registrar entrena B; no se combina con --modelo-b.")
    year, month = month_bounds(current_start, current_end_exclusive)
    output = output or CONFIG.reports_dir / "monitoring" / f"{year}_{month:02d}" / "p3"
    current_trips = current_trips or CONFIG.raw_data_dir / f"datos_abiertos_{year}_{month:02d}.csv"
    cfg_a = CONFIG_P3
    cfg_b = replace(CONFIG_P3, fecha_fin_entrenamiento=fin_entrenamiento_b)
    if pd.Timestamp(fin_entrenamiento_b) >= pd.Timestamp(current_start):
        raise typer.BadParameter("B debe entrenarse con datos anteriores al mes actual.")
    params = {"k": k}

    flujos, _, _ = cargar_flujos(historical_trips)
    nom = leer_nomenclatura(nomenclature or nomenclatura_mas_reciente())
    inicio = pd.Timestamp(current_start)
    fin_excl = pd.Timestamp(current_end_exclusive)
    flujos_act, _, _ = flujos_del_mes_actual(current_trips, inicio, fin_excl)
    ultimo = str((fin_excl - pd.Timedelta(days=1)).date())
    mitad = str((inicio + pd.Timedelta(days=14)).date())
    actual = perfiles_periodo(
        flujos_act, current_start, ultimo, nom.index, MIN_DIAS_ACTIVOS_MES, cfg_a
    )
    quincena_1 = perfiles_periodo(
        flujos_act, current_start, mitad, nom.index, MIN_DIAS_QUINCENA, cfg_a
    )
    quincena_2 = perfiles_periodo(
        flujos_act,
        str((inicio + pd.Timedelta(days=15)).date()),
        ultimo,
        nom.index,
        MIN_DIAS_QUINCENA,
        cfg_a,
    )

    perfiles_a = perfiles_periodo(
        flujos,
        cfg_a.fecha_inicio_entrenamiento,
        cfg_a.fecha_fin_entrenamiento,
        nom.index,
        cfg_a.min_dias_activos,
        cfg_a,
    )
    perfiles_b = perfiles_periodo(
        flujos,
        cfg_b.fecha_inicio_entrenamiento,
        cfg_b.fecha_fin_entrenamiento,
        nom.index,
        cfg_b.min_dias_activos,
        cfg_b,
    )
    a = cargar_modelo(modelo_a, tracking_uri) if modelo_a else entrenar(perfiles_a, params, cfg_a)
    b = cargar_modelo(modelo_b, tracking_uri) if modelo_b else entrenar(perfiles_b, params, cfg_b)
    logger.info(
        f"A: {modelo_a or 'réplica del campeón'} ({len(perfiles_a)} estaciones) | "
        f"B: {modelo_b or f'reentrenado hasta {fin_entrenamiento_b}'} ({len(perfiles_b)}) | "
        f"mes actual: {len(actual)} estaciones"
    )

    tabla = comparar(a, b, actual, quincena_1, quincena_2, n_bootstrap, cfg_a.seed)
    resumen = decidir(tabla, cfg_a) | {
        "periodo_actual": f"{current_start}..{ultimo}",
        "modelo_a": modelo_a or f"réplica {MODELO} {params} hasta {cfg_a.fecha_fin_entrenamiento}",
        "modelo_b": modelo_b or f"{MODELO} {params} reentrenado hasta {fin_entrenamiento_b}",
        "variables_vif_a": variables_vif(a),
        "variables_vif_b": variables_vif(b),
        "n_estaciones_entrenamiento_a": len(perfiles_a),
        "n_estaciones_entrenamiento_b": len(perfiles_b),
        "n_estaciones_actual": len(actual),
        "n_estaciones_quincenas": len(quincena_1.index.intersection(quincena_2.index)),
        "min_dias_mes": MIN_DIAS_ACTIVOS_MES,
        "min_dias_quincena": MIN_DIAS_QUINCENA,
        "n_bootstrap": n_bootstrap,
        "csv_actual_sha256": file_sha256(current_trips),
    }
    asig = asignaciones(a, b, actual, perfiles_a)
    resumen["acuerdo_a_b_ari"] = float(adjusted_rand_score(asig.grupo_a, asig.grupo_b))
    resumen["tamanos_grupo"] = {
        "a": np.bincount(asig.grupo_a).tolist(),
        "b": np.bincount(asig.grupo_b).tolist(),
        "a_estaciones_nuevas": np.bincount(asig.grupo_a[asig.nueva_para_a], minlength=k).tolist(),
        "b_estaciones_nuevas": np.bincount(asig.grupo_b[asig.nueva_para_a], minlength=k).tolist(),
    }

    output.mkdir(parents=True, exist_ok=True)
    archivos = [output / "p3_ab_resultados.csv", output / "p3_ab_resumen.json"]
    archivos.append(output / "p3_ab_asignaciones.csv")
    tabla.to_csv(archivos[0], index=False)
    archivos[1].write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    asig.to_csv(archivos[2])
    logger.info("A/B de P3:\n" + tabla.round(4).to_string(index=False))
    logger.success(f"Decisión: {resumen['decision']} {resumen['condiciones']} | {output}")

    if registrar:
        version = registrar_version_b(
            b, perfiles_b, params, tabla, resumen, archivos, cfg_b, tracking_uri
        )
        logger.success(
            f"B registrado como {cfg_b.modelo_registrado} v{version} (@{cfg_b.alias_retador}); "
            f"@{cfg_b.alias_campeon} no se movió."
        )


if __name__ == "__main__":
    app()
