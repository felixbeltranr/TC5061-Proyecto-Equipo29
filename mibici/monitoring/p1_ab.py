"""A/B testing de P1: campeón (A) contra el mismo modelo reentrenado con datos recientes (B).

- **A**: la versión vigente del Model Registry (por omisión
  ``models:/mibici-equipo29-p1-demanda@champion``) o una carpeta local de MLflow con el
  ``model.skops`` del campeón (``--modelo-a``).
- **B**: HistGradientBoosting con pérdida de Poisson y los mismos hiperparámetros del campeón,
  entrenado con las filas elegibles del 2025-01-01 al último día conocido antes del mes
  evaluado. Es el embargo de P1: el pronóstico de D se emite a las 23:00 de D-1 y usa salidas
  hasta D-2, así que B termina en el inicio del mes menos ``LAG_MINIMO`` días. Con
  ``--modelo-b`` se carga una versión del Registry en vez de entrenarla; su etiqueta
  ``entrenamiento_fin`` (de la versión o de su run) debe existir y cumplir ese embargo. Una
  carpeta local no declara su ventana, así que no se acepta como B.

Las dos versiones pronostican las mismas filas estación-día del mes evaluado (por ejemplo,
octubre de 2026), que ninguna vio al entrenar: es un A/B fuera de línea con diseño pareado; no
se reparte tráfico. El panel se arma en memoria con el histórico y el CSV del mes, como en
:mod:`mibici.monitoring.p1_datasets`; el histórico debe llegar al día anterior al mes y el CSV
debe traer viajes en cada día del mes, porque el panel rellena con cero los días sin viajes.
Antes de entrenar B se reentrena A' con la ventana del campeón y se exige que reproduzca las
predicciones de A: así B solo difiere de A en los datos.

Métricas por grupo (todas, estaciones existentes en el entrenamiento de A y estaciones
nuevas): MAE, RMSE, WAPE y sesgo (pronóstico menos real). La diferencia de MAE (B - A) lleva
un IC del 95 % con bootstrap pareado por estación (las filas de una estación se remuestrean
juntas), 1,000 réplicas y semilla 29. **Regla de decisión** (fijada antes de ver resultados):
B cumple la regla solo si

1. el IC de la diferencia de MAE (B - A) en todas las estaciones queda completo bajo 0,
2. en ningún grupo (existentes, nuevas) el IC queda completo arriba de 0, y
3. las predicciones de B son finitas y no negativas (política de dominio de P1).

Cumplir la regla habilita proponer B; no lo promueve ni lo despliega. La sensibilidad
(bootstrap por día y concentración de la mejora por estación) se reporta aparte y no cambia
la regla.

Con ``--registrar`` el B entrenado se registra en MLflow como nueva versión de
``CONFIG_P1.modelo_registrado``. El alias ``CONFIG_P1.alias_retador`` se asigna solo después
de comprobar que la versión registrada reproduce las predicciones evaluadas; el alias
``champion`` nunca se mueve aquí: promover es una decisión que se toma con este reporte. Si ya
hay una versión de este módulo para el mismo mes y la misma ventana con ``comprobacion=ok``, no
se registra otra (ni se mueve el alias del retador) salvo con ``--forzar``. La versión nace con
``comprobacion=pendiente`` y el run con ``registro=pendiente``; una falla deja ``fallida`` o
``fallido`` en lugar de una versión o un run sin marca.

Uso desde la raíz del repositorio (después del reporte de drift de P1). En Windows, ``-X utf8``
evita que la consola falle al imprimir el enlace del run (el CLI además reemplaza los
caracteres que la consola no puede escribir)::

    python -X utf8 -m mibici.monitoring.p1_ab
    python -X utf8 -m mibici.monitoring.p1_ab --modelo-a <carpeta de MLflow con model.skops>
    python -X utf8 -m mibici.monitoring.p1_ab --registrar --tracking-uri http://127.0.0.1:5000
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from typing import Annotated

from loguru import logger
import mlflow
from mlflow.models import Model
import numpy as np
import pandas as pd
import typer

from mibici import features as fe
from mibici.config import CONFIG, CONFIG_P1, P1Config
from mibici.modeling import train_p1
from mibici.monitoring.p1_drift import file_sha256, month_bounds

app = typer.Typer()

MODELO = "hgb_poisson"  # familia del campeón vigente
# Hiperparámetros del campeón, elegidos en validación (tercera configuración de
# CONFIG_P1.rejilla_hgb); la comprobación de reproducción confirma que son los de A.
PARAMS_CAMPEON = {
    "learning_rate": 0.05,
    "max_leaf_nodes": 63,
    "max_iter": 400,
    "l2_regularization": 1.0,
}
INICIO_PANEL = CONFIG_P1.train_inicio  # primer día del panel y de las dos ventanas
INICIO_USO_A = CONFIG_P1.test_inicio  # el campeón se entrenó para usarse desde este día
N_BOOTSTRAP = 1000
TOLERANCIA = 1e-9  # diferencia absoluta máxima de predicción al reproducir o al recargar
GRUPOS = ("todas", "existentes", "nuevas")  # orden en que el bootstrap consume la semilla
DECISION_CUMPLE = "B cumple la regla"
DECISION_MANTENER = "mantener A"
LECTURA_DECISION = (
    "Cumplir la regla habilita proponer B como reemplazo de A; no lo promueve ni lo despliega. "
    "Promover es una decisión aparte que se toma con este reporte."
)
ORIGEN = "mibici/monitoring/p1_ab.py"
MODELO_A_POR_OMISION = f"models:/{CONFIG_P1.modelo_registrado}@{CONFIG_P1.alias_campeon}"


def metricas(y: np.ndarray, p: np.ndarray) -> dict:
    """MAE, RMSE, WAPE (razón, no %), sesgo (pronóstico menos real) y número de filas."""
    err = p - y
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "wape": float(np.abs(err).sum() / y.sum()),
        "sesgo": float(err.mean()),
        "n": len(y),
    }


def agregar(filas: pd.DataFrame, unidad: str) -> pd.DataFrame:
    """Suma del error absoluto de A y de B y número de filas por estación o por día."""
    return filas.groupby(unidad).agg(
        abs_a=("abs_a", "sum"), abs_b=("abs_b", "sum"), n=("abs_a", "size")
    )


def bootstrap_pareado(
    filas: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int = N_BOOTSTRAP,
    unidad: str = "estacion",
) -> tuple[float, float, float]:
    """Diferencia de MAE (B - A) y su IC del 95 % remuestreando ``unidad`` con reemplazo.

    Es pareado: cada réplica toma las mismas estaciones (o días), con todas sus filas, para A y
    para B, así que la diferencia no mezcla el ruido de dos muestras distintas.
    """
    bloques = agregar(filas, unidad)
    a, b, n = (bloques[c].to_numpy() for c in ("abs_a", "abs_b", "n"))
    diff = b.sum() / n.sum() - a.sum() / n.sum()
    idx = rng.integers(0, len(bloques), size=(n_bootstrap, len(bloques)))
    sa, sb, sn = a[idx].sum(1), b[idx].sum(1), n[idx].sum(1)
    reps = sb / sn - sa / sn
    bajo, alto = np.percentile(reps, [2.5, 97.5])
    return float(diff), float(bajo), float(alto)


def ventanas(current_start: str | pd.Timestamp) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Ventanas de entrenamiento (inclusivas) de A y de B.

    A es la del campeón: objetivos hasta el último día conocido al emitir el primer pronóstico
    de ``INICIO_USO_A`` (2026-04-29). B llega al último día conocido al emitir el primer
    pronóstico del mes evaluado: ``current_start`` menos ``LAG_MINIMO`` días.
    """
    inicio = pd.Timestamp(INICIO_PANEL)
    return {
        "A": (inicio, fe.ultimo_dia_conocido(pd.Timestamp(INICIO_USO_A))),
        "B": (inicio, fe.ultimo_dia_conocido(pd.Timestamp(current_start))),
    }


def exigir_mes_posterior_a_a(inicio: pd.Timestamp) -> None:
    """Error si el mes evaluado empieza dentro de la ventana de entrenamiento de A.

    A ya vio esas etiquetas al entrenar, así que la comparación no sería fuera de muestra.
    """
    fin_a = ventanas(inicio)["A"][1]
    if pd.Timestamp(inicio) <= fin_a:
        raise ValueError(
            f"El mes evaluado empieza el {pd.Timestamp(inicio).date()}, dentro de la ventana de "
            f"entrenamiento de A (hasta el {fin_a.date()}): A ya vio esas etiquetas y no se "
            "compara."
        )


def construir_panel(
    historical_trips: Path,
    current_trips: Path,
    inicio: pd.Timestamp,
    fin_excl: pd.Timestamp,
) -> tuple[pd.DataFrame, dict]:
    """Panel estación-día del histórico más el mes evaluado y procedencia de las entradas.

    El histórico debe terminar antes del mes y los viajes actuales deben caer dentro de él,
    para no duplicar observaciones (la misma condición que ``p1_datasets``). Además el
    histórico debe llegar al día anterior al mes y el CSV del mes debe traer viajes en cada
    día: ``construir_panel_diario`` rellena con cero los días sin viajes, y un hueco se
    convertiría en ceros falsos en la ventana de B, en los rezagos y en el objetivo evaluado.
    """
    hist = fe.cargar_salidas(historical_trips)
    act = fe.cargar_salidas(current_trips)
    fechas_h = hist["Inicio_del_viaje"].dropna().dt.normalize()
    fechas_a = act["Inicio_del_viaje"].dropna().dt.normalize()
    if fechas_h.empty or fechas_a.empty:
        raise ValueError("El histórico y el mes evaluado deben tener fechas de viaje legibles.")
    if (fechas_h >= inicio).any():
        raise ValueError("El histórico debe terminar antes del mes evaluado.")
    if (fechas_a < inicio).any() or (fechas_a >= fin_excl).any():
        raise ValueError("Los viajes actuales deben caer dentro del mes evaluado.")
    vispera = inicio - pd.Timedelta(days=1)
    if fechas_h.max() != vispera:
        raise ValueError(
            f"El histórico termina el {fechas_h.max().date()} y debe llegar al {vispera.date()}, "
            "el día anterior al mes evaluado: los días que faltan se rellenarían con ceros."
        )
    dias_mes = pd.date_range(inicio, fin_excl - pd.Timedelta(days=1), freq="D")
    faltan = dias_mes[~dias_mes.isin(fechas_a.unique())]
    if len(faltan):
        raise ValueError(
            f"El mes evaluado no está completo en los viajes actuales: {len(faltan)} días sin "
            f"viajes (el primero, {faltan[0].date()}) se rellenarían con ceros."
        )
    panel, _ = fe.construir_panel_diario(
        pd.concat([hist, act], ignore_index=True),
        inicio=pd.Timestamp(INICIO_PANEL),
        fin=fin_excl - pd.Timedelta(days=1),
    )
    entradas = {
        "viajes_historicos": {
            "ruta": str(historical_trips),
            "sha256": file_sha256(historical_trips),
        },
        "viajes_actuales": {
            "ruta": str(current_trips),
            "sha256": file_sha256(current_trips),
            "fecha_min": str(fechas_a.min().date()),
            "fecha_max": str(fechas_a.max().date()),
        },
    }
    return panel, entradas


def particionar(
    feats: pd.DataFrame, inicio: pd.Timestamp, fin_excl: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filas elegibles para entrenar A, para entrenar B y del mes evaluado (completo o error)."""
    exigir_mes_posterior_a_a(inicio)
    eleg = feats.loc[feats["elegible"]].copy()
    v = ventanas(inicio)
    train_a = eleg[eleg["fecha"].between(*v["A"])]
    train_b = eleg[eleg["fecha"].between(*v["B"])]
    evalua = eleg[(eleg["fecha"] >= inicio) & (eleg["fecha"] < fin_excl)].copy()
    if evalua.empty:
        raise ValueError("No hay filas elegibles en el mes evaluado.")
    dias_mes = pd.date_range(inicio, fin_excl - pd.Timedelta(days=1), freq="D")
    if set(evalua["fecha"].dt.normalize().unique()) != set(dias_mes):
        raise ValueError("El mes evaluado no está completo en los datos; no se compara.")
    return train_a, train_b, evalua


def diferencia_pred(a, b) -> float:
    """Diferencia absoluta máxima entre dos vectores de predicción, o ``inf`` si no se comparan.

    Devuelve ``inf`` si las longitudes difieren, si no hay filas o si alguno trae valores no
    finitos: así toda comprobación ``dif <= TOLERANCIA`` falla cerrada. Con ``max`` a secas un
    NaN daba NaN, y ``NaN > TOLERANCIA`` es falso, de modo que la comprobación pasaba.
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        return float("inf")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("inf")
    return float(np.abs(a - b).max())


def comprobar_reproduccion(
    train_a: pd.DataFrame,
    evalua: pd.DataFrame,
    pred_a: np.ndarray,
    params: dict = PARAMS_CAMPEON,
    semilla: int = CONFIG_P1.semilla,
) -> float:
    """Reentrena A' con la ventana del campeón y exige que reproduzca las predicciones de A."""
    a_rep = train_p1.ajustar_candidato(MODELO, train_a, params, semilla)
    pred_rep = np.asarray(a_rep.predict(evalua[fe.FEATURES]), dtype=float)
    dif = diferencia_pred(pred_rep, pred_a)
    if not dif <= TOLERANCIA:
        raise ValueError(
            f"El camino de entrenamiento no reproduce a A (dif. máx. {dif}; inf = longitudes "
            "distintas o valores no finitos)."
        )
    return dif


def comparar(
    evalua: pd.DataFrame,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    estaciones_a,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = CONFIG_P1.semilla,
) -> tuple[dict, pd.DataFrame]:
    """Tabla A/B por grupo y filas evaluadas con su error absoluto y su grupo.

    Un solo generador se consume en el orden de ``GRUPOS`` (todas, existentes, nuevas); un
    grupo sin filas se omite sin consumirlo.
    """
    filas = evalua.copy()
    y = filas["objetivo"].to_numpy(dtype=float)
    filas["abs_a"] = np.abs(pred_a - y)
    filas["abs_b"] = np.abs(pred_b - y)
    filas["grupo"] = np.where(filas["estacion"].isin(estaciones_a), "existentes", "nuevas")
    rng = np.random.default_rng(seed)
    tabla = {}
    for nombre in GRUPOS:
        if nombre == "todas":
            mask = np.ones(len(filas), dtype=bool)
        else:
            mask = (filas["grupo"] == nombre).to_numpy()
        if not mask.any():
            continue
        sub = filas[mask]
        diff, bajo, alto = bootstrap_pareado(sub, rng, n_bootstrap)
        tabla[nombre] = {
            "estaciones": int(sub["estacion"].nunique()),
            "A": metricas(y[mask], pred_a[mask]),
            "B": metricas(y[mask], pred_b[mask]),
            "dif_mae_B_menos_A": diff,
            "ic95": [bajo, alto],
        }
    return tabla, filas


def sensibilidad(
    filas: pd.DataFrame, n_bootstrap: int = N_BOOTSTRAP, seed: int = CONFIG_P1.semilla
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Bootstrap por día y concentración de la mejora por estación (no cambia la regla).

    El bootstrap por día usa un generador nuevo con la misma semilla.
    """
    por_dia = agregar(filas, "fecha")
    _, bajo, alto = bootstrap_pareado(filas, np.random.default_rng(seed), n_bootstrap, "fecha")
    por_est = filas.groupby("estacion").agg(abs_a=("abs_a", "sum"), abs_b=("abs_b", "sum"))
    por_est["mejora"] = por_est["abs_a"] - por_est["abs_b"]
    mejora_total = float(por_est["mejora"].sum())
    top5 = por_est["mejora"].sort_values(ascending=False).head(5)
    resumen = {
        "ic95_bootstrap_por_dia_todas": [bajo, alto],
        "dias_con_B_mejor": int((por_dia["abs_b"] < por_dia["abs_a"]).sum()),
        "dias": len(por_dia),
        "estaciones_con_B_mejor": int((por_est["mejora"] > 0).sum()),
        "estaciones": len(por_est),
        "mejora_total_error_abs": mejora_total,
        "top5_estaciones": {str(k): float(v) for k, v in top5.items()},
        "fraccion_top5": float(top5.sum() / mejora_total) if mejora_total else None,
    }
    return resumen, por_est, por_dia


def dominio_valido(pred: np.ndarray) -> bool:
    """Política de dominio de P1: predicciones finitas y no negativas."""
    pred = np.asarray(pred, dtype=float)
    return bool(np.isfinite(pred).all() and (pred >= 0).all())


def decidir(tabla: dict, pred_b: np.ndarray) -> dict:
    """Aplica la regla fijada a la tabla de ``comparar`` y a las predicciones de B."""
    regla = {
        "c1_mejora_mae_ic_bajo_cero": bool(tabla["todas"]["ic95"][1] < 0),
        "c2_no_peor_en_grupos": bool(
            all(tabla[g]["ic95"][0] <= 0 for g in ("existentes", "nuevas") if g in tabla)
        ),
        "c3_dominio_b": dominio_valido(pred_b),
    }
    decision = DECISION_CUMPLE if all(regla.values()) else DECISION_MANTENER
    return {"regla": regla, "decision": decision, "lectura_decision": LECTURA_DECISION}


def tabla_resultados(tabla: dict) -> pd.DataFrame:
    """Tabla A/B por grupo en formato largo para el CSV y el reporte."""
    filas = []
    for grupo, t in tabla.items():
        fila = {"grupo": grupo, "estaciones": t["estaciones"], "n": t["A"]["n"]}
        for metrica in ("mae", "rmse", "wape", "sesgo"):
            fila[f"{metrica}_a"] = t["A"][metrica]
            fila[f"{metrica}_b"] = t["B"][metrica]
        fila["dif_mae_b_menos_a"] = t["dif_mae_B_menos_A"]
        fila["ic95_bajo"], fila["ic95_alto"] = t["ic95"]
        filas.append(fila)
    return pd.DataFrame(filas)


def es_local(referencia: str) -> bool:
    """La referencia apunta a una carpeta local de MLflow (y no a una URI del Registry)."""
    return Path(referencia).exists()


def cargar_modelo(referencia: str, tracking_uri: str):
    """Carga un modelo de P1 (formato sklearn) de una carpeta local o de una URI de MLflow."""
    if not es_local(referencia):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
    return mlflow.sklearn.load_model(referencia)


def version_registrada(referencia: str, tracking_uri: str):
    """Versión del Registry que nombra ``models:/<nombre>/<n>`` o ``models:/<nombre>@<alias>``.

    Devuelve ``None`` si la referencia es una carpeta local o una URI que no nombra una versión.
    """
    if es_local(referencia) or not referencia.startswith("models:/"):
        return None
    from mlflow import MlflowClient

    resto = referencia.removeprefix("models:/")
    cliente = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    if "@" in resto:
        nombre, alias = resto.split("@", 1)
        return cliente.get_model_version_by_alias(nombre, alias)
    if "/" in resto:
        nombre, numero = resto.rsplit("/", 1)
        return cliente.get_model_version(nombre, numero)
    return None


def describir_modelo(referencia: str, tracking_uri: str) -> dict:
    """Identidad del modelo usado: huella de ``model.skops`` local o versión del Registry."""
    info: dict = {"referencia": referencia}
    if es_local(referencia):
        carpeta = Path(referencia)
        if (carpeta / "model.skops").is_file():
            info["model_skops_sha256"] = file_sha256(carpeta / "model.skops")
        if (carpeta / "MLmodel").is_file():
            info["run_id"] = Model.load(str(carpeta / "MLmodel")).run_id
        return info
    version = version_registrada(referencia, tracking_uri)
    if version is not None:
        info |= {"nombre": version.name, "version": int(version.version), "run_id": version.run_id}
    return info


def ventana_registrada(referencia: str, tracking_uri: str, limite: pd.Timestamp) -> dict:
    """Identidad y ventana de entrenamiento de un B del Registry, con el embargo exigido.

    La ventana sale de las etiquetas ``entrenamiento_inicio`` y ``entrenamiento_fin`` que
    escribe ``--registrar``; las de la versión mandan sobre las de su run. Sin
    ``entrenamiento_fin``, o con un fin posterior a ``limite`` (el último día conocido al
    pronosticar el primer día del mes), no se puede descartar que B haya visto etiquetas del
    mes evaluado: es error. Una carpeta local no declara su ventana, así que no se acepta.
    """
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    version = version_registrada(referencia, tracking_uri)
    if version is None:
        raise ValueError(
            "--modelo-b debe ser una versión del Registry (models:/<nombre>/<n> o "
            f"models:/<nombre>@<alias>), no {referencia}: una carpeta local o una URI sin "
            "versión no declaran su ventana de entrenamiento y no se puede comprobar el "
            "embargo de B."
        )
    etiquetas_version = dict(version.tags or {})
    etiquetas_run: dict = {}
    if version.run_id:
        cliente = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        try:
            etiquetas_run = dict(cliente.get_run(version.run_id).data.tags)
        except MlflowException as exc:
            logger.warning(f"No se pudo leer el run {version.run_id} de {referencia}: {exc}")
    etiquetas = etiquetas_run | etiquetas_version
    nombre_version = f"{version.name} v{version.version}"
    texto = etiquetas.get("entrenamiento_fin")
    if texto is None:
        raise ValueError(
            f"{nombre_version} no tiene la etiqueta entrenamiento_fin (ni en la versión ni en su "
            "run): no se puede comprobar que B no vio etiquetas del mes evaluado."
        )
    try:
        fin = pd.Timestamp(texto)
    except ValueError:  # las etiquetas son texto; uno que no es fecha da ValueError
        fin = pd.NaT
    if pd.isna(fin):  # pd.Timestamp("") da NaT, y NaT > limite es falso: se rechaza aquí
        raise ValueError(f"entrenamiento_fin={texto!r} de {nombre_version} no es una fecha.")
    fin = fin.normalize()
    if fin > limite:
        raise ValueError(
            f"{nombre_version} se entrenó hasta el {fin.date()}, después del {limite.date()} "
            f"(último día conocido al pronosticar el primer día del mes, embargo de "
            f"{fe.LAG_MINIMO} días): pudo ver etiquetas del mes evaluado y no se compara."
        )
    fuente = "versión" if "entrenamiento_fin" in etiquetas_version else "run"
    return {
        "referencia": referencia,
        "nombre": version.name,
        "version": int(version.version),
        "run_id": version.run_id,
        "entrenamiento_inicio": etiquetas.get("entrenamiento_inicio"),
        "entrenamiento_fin": str(fin.date()),
        "fuente_entrenamiento_fin": fuente,
        "comprobacion": etiquetas_version.get("comprobacion"),
    }


def metricas_registro(resumen: dict) -> dict[str, float]:
    """Métricas del A/B con nombres planos para el run de MLflow (solo valores finitos)."""
    m: dict[str, float] = {}
    for grupo, t in resumen["tabla"].items():
        for version in ("A", "B"):
            for metrica in ("mae", "rmse", "wape", "sesgo"):
                m[f"ab_{grupo}_{metrica}_{version.lower()}"] = t[version][metrica]
        m[f"ab_{grupo}_n"] = t["A"]["n"]
        m[f"ab_{grupo}_estaciones"] = t["estaciones"]
        m[f"ab_{grupo}_dif_mae_b_menos_a"] = t["dif_mae_B_menos_A"]
        m[f"ab_{grupo}_ic95_bajo"], m[f"ab_{grupo}_ic95_alto"] = t["ic95"]
    s = resumen["sensibilidad"]
    m["ab_dia_ic95_bajo"], m["ab_dia_ic95_alto"] = s["ic95_bootstrap_por_dia_todas"]
    m["ab_dias_con_b_mejor"] = s["dias_con_B_mejor"]
    m["ab_estaciones_con_b_mejor"] = s["estaciones_con_B_mejor"]
    if s["fraccion_top5"] is not None:
        m["ab_fraccion_top5"] = s["fraccion_top5"]
    if resumen["reproduccion_A"].get("max_abs_dif_pred") is not None:
        m["ab_reproduccion_a_max_abs_dif"] = resumen["reproduccion_A"]["max_abs_dif_pred"]
    return {k: float(v) for k, v in m.items() if np.isfinite(v)}


def sha256_model_skops(model_uri: str) -> str | None:
    """Huella del ``model.skops`` de una versión registrada, descargado a una carpeta temporal."""
    from mlflow.artifacts import download_artifacts
    from mlflow.exceptions import MlflowException

    try:
        with tempfile.TemporaryDirectory() as tmp:
            carpeta = Path(download_artifacts(model_uri, dst_path=tmp))
            skops = next(carpeta.rglob("model.skops"), None)
            return file_sha256(skops) if skops else None
    except (MlflowException, OSError) as exc:  # la huella es evidencia extra; no detiene
        logger.warning(f"No se pudo descargar {model_uri} para calcular su huella: {exc}")
        return None


def proteger_consola(*flujos) -> None:
    """Evita que un carácter que la consola no puede codificar detenga el registro.

    Con tracking HTTP, MLflow escribe en stdout el enlace del run con un emoji al cerrarlo. En
    una consola de Windows en cp1252 eso lanzaba ``UnicodeEncodeError`` con el run y el modelo
    ya guardados y antes de registrar la versión. Con ``errors="backslashreplace"`` el carácter
    se escribe como ``\\U0001f3c3`` y el registro sigue. Sin argumentos protege stdout y stderr.
    """
    for flujo in flujos or (sys.stdout, sys.stderr):
        reconfigurar = getattr(flujo, "reconfigure", None)
        if reconfigurar is not None:
            reconfigurar(errors="backslashreplace")


def version_equivalente(
    cliente, nombre: str, ab_periodo: str, entrenamiento_fin: str
) -> int | None:
    """Versión más reciente de este módulo para el mismo mes y la misma ventana, ya comprobada.

    Cuenta solo las versiones con ``origen`` de este módulo, el mismo ``ab_periodo``, el mismo
    ``entrenamiento_fin`` y ``comprobacion=ok``. No exige el mismo ``git_commit``: con cualquier
    commit nuevo la corrida repetida volvería a registrar y a mover el alias del retador.
    """
    criterios = {
        "origen": ORIGEN,
        "ab_periodo": ab_periodo,
        "entrenamiento_fin": entrenamiento_fin,
        "comprobacion": "ok",
    }
    iguales = [
        int(v.version)
        for v in cliente.search_model_versions(f"name='{nombre}'")
        if all((v.tags or {}).get(clave) == valor for clave, valor in criterios.items())
    ]
    return max(iguales) if iguales else None


def exigir_sin_equivalente(
    cliente, cfg: P1Config, ab_periodo: str, entrenamiento_fin: str, forzar: bool
) -> int | None:
    """Detiene un registro repetido salvo con ``forzar``; devuelve la versión equivalente."""
    previa = version_equivalente(cliente, cfg.modelo_registrado, ab_periodo, entrenamiento_fin)
    if previa is not None and not forzar:
        raise ValueError(
            f"{cfg.modelo_registrado} v{previa} ya es un B de {ORIGEN} para {ab_periodo}, "
            f"entrenado hasta {entrenamiento_fin} y con comprobacion=ok: no se registra otra "
            f"versión igual ni se mueve @{cfg.alias_retador}. Usa --forzar para hacerlo."
        )
    return previa


def marcar_run_fallido(cliente, run_id: str, exc: BaseException) -> None:
    """Deja ``registro=fallido`` y el motivo en el run y lo cierra como FAILED si sigue abierto.

    Un error al marcarlo solo se avisa, para no ocultar la excepción original.
    """
    try:
        cliente.set_tag(run_id, "registro", "fallido")
        cliente.set_tag(run_id, "registro_detalle", f"{type(exc).__name__}: {exc}"[:500])
        if cliente.get_run(run_id).info.status == "RUNNING":
            cliente.set_terminated(run_id, "FAILED")
    except Exception as otro:  # noqa: BLE001 - la excepción original es la que se propaga
        logger.warning(f"No se pudo marcar el run {run_id} como fallido: {otro}")


def deshacer_cierre(
    cliente,
    cfg: P1Config,
    numero: str,
    run_id: str,
    retador_anterior: int | None,
    alias_movido: bool,
    exc: BaseException,
) -> None:
    """Deja una versión cuyo cierre falló como fallida, sin alias y con su run fallido.

    Si el alias del retador ya se había movido, vuelve a la versión anterior (o se borra si no
    había) y se quita ``reemplazada_por`` de esa versión. Así la versión no cuenta como
    equivalente (:func:`version_equivalente` exige ``comprobacion=ok``) y el reintento normal
    registra otra. Cada paso es independiente: un error al deshacer solo se avisa, para no
    ocultar la excepción original. El alias del campeón no se toca.
    """
    nombre = cfg.modelo_registrado
    pasos = [
        lambda: cliente.set_model_version_tag(nombre, numero, "comprobacion", "fallida"),
        lambda: cliente.set_model_version_tag(
            nombre, numero, "comprobacion_detalle", f"{type(exc).__name__}: {exc}"[:500]
        ),
    ]
    if alias_movido and retador_anterior is not None:
        pasos += [
            lambda: cliente.set_registered_model_alias(
                nombre, cfg.alias_retador, str(retador_anterior)
            ),
            lambda: cliente.delete_model_version_tag(
                nombre, str(retador_anterior), "reemplazada_por"
            ),
        ]
    elif alias_movido:
        pasos.append(lambda: cliente.delete_registered_model_alias(nombre, cfg.alias_retador))
    for paso in pasos:
        try:
            paso()
        except Exception as otro:  # noqa: BLE001 - la excepción original es la que se propaga
            logger.warning(f"No se pudo deshacer el cierre de {nombre} v{numero}: {otro}")
    marcar_run_fallido(cliente, run_id, exc)


def registrar_version_b(
    modelo_b,
    train_b: pd.DataFrame,
    filas: pd.DataFrame,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    resumen: dict,
    archivos: list[Path],
    tracking_uri: str,
    cfg: P1Config = CONFIG_P1,
    autor: str | None = None,
    forzar: bool = False,
) -> dict:
    """Registra B como nueva versión del modelo de P1 y le asigna el alias del retador.

    Orden: (1) lee la versión de ``@champion`` y exige que sus predicciones sean las de A en
    las filas evaluadas, antes de escribir nada; (2) sin ``forzar``, se detiene si ya hay una
    versión equivalente (:func:`version_equivalente`); (3) crea un run con etiquetas,
    parámetros, métricas y archivos del A/B (``registro=pendiente``), y guarda B con
    ``train_p1.registrar_modelo``; si algo falla ahí, el run queda con ``registro=fallido``;
    (4) registra la versión con etiquetas de linaje y ``comprobacion=pendiente``; (5) carga esa
    versión con pyfunc, predice todas las filas evaluadas y exige la misma predicción que B en
    memoria y dominio válido. Si la descripción o la comprobación fallan, la versión queda con
    ``comprobacion=fallida`` y sin alias. (6) Solo si pasa se asigna ``cfg.alias_retador``; si
    apuntaba a otra versión, esa queda con ``reemplazada_por`` y la nueva con
    ``retador_anterior``. Se confirma que ``@champion`` sigue en la misma versión y, como
    última escritura, la versión pasa a ``comprobacion=ok``. Si este cierre falla en cualquier
    punto, :func:`deshacer_cierre` deja la versión fallida, devuelve el alias del retador a
    donde estaba y marca el run como fallido, así que el reintento normal no queda bloqueado.
    El alias del campeón nunca se mueve, borra ni reasigna aquí.
    """
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    from mibici import registro_p1, tracking

    if cfg.alias_retador == cfg.alias_campeon:
        raise ValueError("El alias del retador no puede ser el del campeón.")
    nombre = cfg.modelo_registrado
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    cliente = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    X = filas[fe.FEATURES].astype("float64")

    try:
        campeon_antes = int(cliente.get_model_version_by_alias(nombre, cfg.alias_campeon).version)
    except MlflowException as exc:
        raise ValueError(
            f"No existe {nombre}@{cfg.alias_campeon} en {tracking_uri}; B no se registra."
        ) from exc
    modelo_campeon = mlflow.pyfunc.load_model(f"models:/{nombre}/{campeon_antes}")
    pred_campeon = np.asarray(modelo_campeon.predict(X), dtype=float).reshape(-1)
    dif_campeon = diferencia_pred(pred_campeon, pred_a)
    if not dif_campeon <= TOLERANCIA:
        raise ValueError(
            f"A no es {nombre} v{campeon_antes} (@{cfg.alias_campeon}): sus predicciones "
            f"difieren hasta {dif_campeon} (inf = longitudes distintas o valores no finitos). "
            "El A/B no se hizo contra el campeón; B no se registra."
        )

    inicio_b, fin_b = resumen["B"]["entrenamiento"]
    periodo = resumen["ab_periodo"]
    equivalente_previa = exigir_sin_equivalente(cliente, cfg, periodo, fin_b, forzar)

    tracking.configurar_experimento(
        tracking_uri, cfg.experimento, cfg.artefactos_dir.resolve().as_uri()
    )
    data_version = tracking.version_datos(cfg.manifiesto_path)
    etiqueta = train_p1.NOMBRES_RUN[MODELO][1]
    etiquetas = tracking.etiquetas_base(
        "P1", autor or tracking.autor_git(), cfg.semilla, f"{inicio_b}..{fin_b}", data_version
    )
    run_id = None
    try:
        with mlflow.start_run(run_name=f"p1_ab_{MODELO}_hasta_{fin_b}") as run:
            run_id = run.info.run_id
            mlflow.set_tags(
                etiquetas
                | train_p1.etiquetas_modelo(etiqueta, train_p1.FAMILIAS[MODELO], "ab_test")
                | {
                    "origen": ORIGEN,
                    "registro": "pendiente",
                    "ab_decision": resumen["decision"],
                    "ab_periodo": periodo,
                    "entrenamiento_inicio": inicio_b,
                    "entrenamiento_fin": fin_b,
                    "modelo_a": resumen["entradas"]["modelo_a"]["referencia"],
                    "campeon_version": str(campeon_antes),
                }
                | {
                    f"sha256_{clave}": entrada["sha256"]
                    for clave, entrada in resumen["entradas"].items()
                    if "sha256" in entrada
                }
            )
            mlflow.log_params(
                {
                    "modelo": etiqueta,
                    **resumen["B"]["params"],
                    "semilla": cfg.semilla,
                    "features": ",".join(fe.FEATURES),
                    "entrenamiento_inicio": inicio_b,
                    "entrenamiento_fin": fin_b,
                    "lag_minimo_dias": fe.LAG_MINIMO,
                    "n_train": len(train_b),
                    "estaciones_train": int(train_b["estacion"].nunique()),
                    "ab_periodo": periodo,
                    "ab_n_filas": len(filas),
                    "ab_n_bootstrap": resumen["bootstrap"]["replicas"],
                    "ab_unidad_bootstrap": resumen["bootstrap"]["unidad"],
                }
            )
            mlflow.log_metrics(metricas_registro(resumen))
            for archivo in archivos:
                mlflow.log_artifact(str(archivo), "ab_test")
            info = train_p1.registrar_modelo(modelo_b, train_b.sample(5, random_state=cfg.semilla))
        version = mlflow.register_model(
            info.model_uri,
            nombre,
            tags={
                "run_id": run_id,
                "git_commit": etiquetas["git_commit"],
                "git_dirty": etiquetas["git_dirty"],
                "origen": ORIGEN,
                "ab_decision": resumen["decision"],
                "ab_periodo": periodo,
                "entrenamiento_fin": fin_b,
                "comprobacion": "pendiente",
            },
        )
    except Exception as exc:
        if run_id is not None:
            marcar_run_fallido(cliente, run_id, exc)
        raise

    numero = str(version.version)
    try:
        cliente.update_model_version(
            nombre,
            numero,
            description=(
                f"HistGradientBoosting (Poisson) con los hiperparámetros del campeón, "
                f"reentrenado con objetivos del {inicio_b} al {fin_b} (embargo de "
                f"{fe.LAG_MINIMO} días antes del mes evaluado). A/B contra v{campeon_antes} en "
                f"{periodo}: {resumen['decision']}. {LECTURA_DECISION}"
            ),
        )
        pred_registrada = np.asarray(
            registro_p1.verificar_version(nombre, numero, X), dtype=float
        ).reshape(-1)
        dif = diferencia_pred(pred_registrada, pred_b)
        if not dif <= TOLERANCIA:
            raise ValueError(
                f"la versión registrada no reproduce las predicciones de B (dif. máx. {dif})"
            )
    except Exception as exc:
        try:
            cliente.set_model_version_tag(nombre, numero, "comprobacion", "fallida")
            cliente.set_model_version_tag(nombre, numero, "comprobacion_detalle", str(exc)[:500])
        except Exception as otro:  # noqa: BLE001 - queda comprobacion=pendiente, tampoco válida
            logger.warning(f"No se pudo marcar {nombre} v{numero} como fallida: {otro}")
        marcar_run_fallido(cliente, run_id, exc)
        raise ValueError(f"{nombre} v{numero} queda sin alias: {exc}") from exc

    # Cierre del registro. ``comprobacion=ok`` es la última escritura: una versión con esa
    # etiqueta ya tiene su alias y su run cerrado. Si algo falla antes, todo se deshace.
    retador_anterior = None
    alias_movido = False
    try:
        cliente.set_model_version_tag(nombre, numero, "comprobacion_max_abs_dif", repr(dif))
        cliente.set_model_version_tag(nombre, numero, "comprobacion_filas", str(len(X)))
        try:  # el alias del retador sí se reasigna; se deja constancia en las dos versiones
            retador_anterior = int(
                cliente.get_model_version_by_alias(nombre, cfg.alias_retador).version
            )
        except MlflowException:
            retador_anterior = None
        if retador_anterior is not None:
            cliente.set_model_version_tag(
                nombre, numero, "retador_anterior", str(retador_anterior)
            )
        cliente.set_registered_model_alias(nombre, cfg.alias_retador, numero)
        alias_movido = True
        if retador_anterior is not None:
            cliente.set_model_version_tag(nombre, str(retador_anterior), "reemplazada_por", numero)
        campeon_despues = int(
            cliente.get_model_version_by_alias(nombre, cfg.alias_campeon).version
        )
        if campeon_despues != campeon_antes:
            raise RuntimeError(
                f"@{cfg.alias_campeon} cambió de v{campeon_antes} a v{campeon_despues} mientras "
                "se registraba B. Este módulo no lo mueve: revisa el Registry antes de seguir."
            )
        cliente.set_tag(run_id, "registro_version", numero)
        cliente.set_tag(run_id, "registro", "ok")
        cliente.set_model_version_tag(nombre, numero, "comprobacion", "ok")
    except Exception as exc:
        deshacer_cierre(cliente, cfg, numero, run_id, retador_anterior, alias_movido, exc)
        raise ValueError(f"{nombre} v{numero} queda sin alias: {exc}") from exc
    model_uri = f"models:/{nombre}/{numero}"
    return {
        "estado": "registrado",
        "tracking_uri": tracking_uri,
        "nombre": nombre,
        "version": int(numero),
        "run_id": run_id,
        "model_uri": model_uri,
        "logged_model_uri": info.model_uri,
        "alias": cfg.alias_retador,
        "retador_version_anterior": retador_anterior,
        "version_equivalente_previa": equivalente_previa,
        "campeon_alias": cfg.alias_campeon,
        "campeon_version_antes": campeon_antes,
        "campeon_version_despues": campeon_despues,
        "campeon_max_abs_dif_pred_a": dif_campeon,
        "comprobacion": "ok",
        "max_abs_dif_pred": dif,
        "filas_comprobadas": len(X),
        "model_skops_sha256": sha256_model_skops(model_uri),
    }


def texto_rango(ventana: tuple[pd.Timestamp, pd.Timestamp]) -> list[str]:
    """Ventana inclusiva como par de fechas ISO para el resumen."""
    return [str(ventana[0].date()), str(ventana[1].date())]


def escribir_json(ruta: Path, datos: dict) -> None:
    """Escribe el resumen en UTF-8 con acentos legibles."""
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")


def ejecutar(
    historical_trips: Path,
    current_trips: Path,
    current_start: str,
    current_end_exclusive: str,
    output: Path,
    modelo_a: str = MODELO_A_POR_OMISION,
    modelo_b: str | None = None,
    registrar: bool = False,
    tracking_uri: str = CONFIG_P1.tracking_uri,
    n_bootstrap: int = N_BOOTSTRAP,
    forzar: bool = False,
) -> dict:
    """A/B completo: panel, ventanas, predicciones, tabla, regla, salidas y registro opcional.

    Con ``modelo_b`` se exige, antes de leer datos, que la versión declare una ventana que
    cumpla el embargo (:func:`ventana_registrada`). Con ``registrar`` se revisa, antes de
    entrenar, que no exista ya una versión equivalente (salvo con ``forzar``).
    """
    if registrar and modelo_b:
        raise ValueError("--registrar entrena B; no se combina con --modelo-b.")
    if forzar and not registrar:
        raise ValueError("--forzar solo aplica con --registrar.")
    month_bounds(current_start, current_end_exclusive)
    inicio, fin_excl = pd.Timestamp(current_start), pd.Timestamp(current_end_exclusive)
    exigir_mes_posterior_a_a(inicio)
    v = ventanas(inicio)
    ultimo = str((fin_excl - pd.Timedelta(days=1)).date())
    periodo = f"{inicio.date()}..{ultimo}"
    info_b = ventana_registrada(modelo_b, tracking_uri, v["B"][1]) if modelo_b else None
    if registrar:
        from mlflow import MlflowClient

        cliente = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        exigir_sin_equivalente(cliente, CONFIG_P1, periodo, texto_rango(v["B"])[1], forzar)
    panel, entradas = construir_panel(historical_trips, current_trips, inicio, fin_excl)
    feats = fe.construir_features(panel)
    train_a, train_b, evalua = particionar(feats, inicio, fin_excl)

    entradas["modelo_a"] = describir_modelo(modelo_a, tracking_uri)
    a = cargar_modelo(modelo_a, tracking_uri)
    pred_a = np.asarray(a.predict(evalua[fe.FEATURES].astype("float64")), dtype=float)
    if info_b:
        entradas["modelo_b"] = info_b
        # se carga la versión comprobada, no el alias, por si el alias se mueve entretanto
        b = cargar_modelo(f"models:/{info_b['nombre']}/{info_b['version']}", tracking_uri)
        pred_b = np.asarray(b.predict(evalua[fe.FEATURES].astype("float64")), dtype=float)
        reproduccion = {
            "max_abs_dif_pred": None,
            "nota": (
                f"B es {info_b['nombre']} v{info_b['version']} del Registry ({modelo_b}), "
                "cargada sin entrenar, así que no se reproduce A. Su ventana "
                f"({info_b['entrenamiento_inicio']}..{info_b['entrenamiento_fin']}) sale de la "
                f"etiqueta entrenamiento_fin ({info_b['fuente_entrenamiento_fin']}) y cumple el "
                f"embargo (fin <= {v['B'][1].date()})."
            ),
        }
    else:
        reproduccion = {"max_abs_dif_pred": comprobar_reproduccion(train_a, evalua, pred_a)}
        b = train_p1.ajustar_candidato(MODELO, train_b, PARAMS_CAMPEON, CONFIG_P1.semilla)
        pred_b = np.asarray(b.predict(evalua[fe.FEATURES]), dtype=float)

    estaciones_a = set(train_a["estacion"].unique())
    tabla, filas = comparar(evalua, pred_a, pred_b, estaciones_a, n_bootstrap, CONFIG_P1.semilla)
    sens, por_est, por_dia = sensibilidad(filas, n_bootstrap, CONFIG_P1.semilla)
    decision = decidir(tabla, pred_b)
    resumen = {
        "origen": ORIGEN,
        "mes_evaluado": [str(inicio.date()), str(fin_excl.date())],
        "ab_periodo": periodo,
        "entradas": entradas,
        "panel_rango": [str(panel["fecha"].min().date()), str(panel["fecha"].max().date())],
        "A": {
            "modelo": modelo_a,
            "entrenamiento": texto_rango(v["A"]),
            "filas": len(train_a),
            "estaciones": int(train_a["estacion"].nunique()),
        },
        "B": {
            "modelo": modelo_b or f"{MODELO} reentrenado",
            "params": None if modelo_b else PARAMS_CAMPEON,
            "entrenamiento": (
                [info_b["entrenamiento_inicio"], info_b["entrenamiento_fin"]]
                if info_b
                else texto_rango(v["B"])
            ),
            "filas": None if modelo_b else len(train_b),
            "estaciones": None if modelo_b else int(train_b["estacion"].nunique()),
            "dominio_ok": decision["regla"]["c3_dominio_b"],
        },
        "reproduccion_A": reproduccion,
        "tabla": tabla,
        **decision,
        "sensibilidad": sens,
        "bootstrap": {"replicas": n_bootstrap, "semilla": CONFIG_P1.semilla, "unidad": "estacion"},
    }

    output.mkdir(parents=True, exist_ok=True)
    rutas = {
        "resumen": output / "p1_ab_resumen.json",
        "resultados": output / "p1_ab_resultados.csv",
        "por_estacion": output / "p1_ab_por_estacion.csv",
        "por_dia": output / "p1_ab_por_dia.csv",
        "predicciones": output / "p1_ab_predicciones.parquet",
    }
    tabla_resultados(tabla).to_csv(rutas["resultados"], index=False)
    por_est.to_csv(rutas["por_estacion"])
    por_dia.to_csv(rutas["por_dia"])
    predicciones = filas[["estacion", "fecha", "objetivo"]].reset_index(drop=True)
    predicciones["pred_a"] = pred_a
    predicciones["pred_b"] = pred_b
    predicciones["grupo"] = filas["grupo"].to_numpy()
    predicciones.to_parquet(rutas["predicciones"], index=False)
    resumen["salidas"] = {clave: ruta.name for clave, ruta in rutas.items()}
    escribir_json(rutas["resumen"], resumen)
    logger.info("A/B de P1:\n" + tabla_resultados(tabla).round(4).to_string(index=False))
    logger.success(f"Decisión: {resumen['decision']} {resumen['regla']} | {output}")

    if registrar:
        resumen["registro"] = registrar_version_b(
            b,
            train_b,
            filas,
            pred_a,
            pred_b,
            resumen,
            list(rutas.values()),
            tracking_uri,
            forzar=forzar,
        )
        escribir_json(rutas["resumen"], resumen)
        r = resumen["registro"]
        logger.success(
            f"B registrado como {r['nombre']} v{r['version']} (@{r['alias']}); "
            f"@{r['campeon_alias']} sigue en v{r['campeon_version_despues']}."
        )
    return resumen


@app.command()
def main(
    historical_trips: Path = CONFIG.processed_data_dir / CONFIG.output_filename,
    current_trips: Annotated[
        Path | None, typer.Option(help="CSV del mes actual (por omisión el de data/raw).")
    ] = None,
    current_start: str = "2026-10-01",
    current_end_exclusive: str = "2026-11-01",
    modelo_a: Annotated[
        str, typer.Option(help="URI del Registry o carpeta local de MLflow para A.")
    ] = MODELO_A_POR_OMISION,
    modelo_b: Annotated[
        str | None,
        typer.Option(
            help="Versión del Registry para B (models:/<nombre>/<n> o @alias); se carga, no se "
            "entrena, y su etiqueta entrenamiento_fin debe cumplir el embargo."
        ),
    ] = None,
    registrar: Annotated[
        bool, typer.Option(help="Registrar B en MLflow con el alias del retador.")
    ] = False,
    forzar: Annotated[
        bool,
        typer.Option(help="Con --registrar, registrar aunque ya exista una versión equivalente."),
    ] = False,
    tracking_uri: str = CONFIG_P1.tracking_uri,
    output: Path | None = None,
    n_bootstrap: int = N_BOOTSTRAP,
):
    """Compara el campeón de P1 (A) con el modelo reentrenado (B) en el mes actual."""
    if registrar and modelo_b:
        raise typer.BadParameter("--registrar entrena B; no se combina con --modelo-b.")
    if forzar and not registrar:
        raise typer.BadParameter("--forzar solo aplica con --registrar.")
    proteger_consola()
    year, month = month_bounds(current_start, current_end_exclusive)
    output = output or CONFIG.reports_dir / "monitoring" / f"{year}_{month:02d}" / "p1"
    current_trips = current_trips or CONFIG.raw_data_dir / f"datos_abiertos_{year}_{month:02d}.csv"
    ejecutar(
        historical_trips,
        current_trips,
        current_start,
        current_end_exclusive,
        output,
        modelo_a,
        modelo_b,
        registrar,
        tracking_uri,
        n_bootstrap,
        forzar,
    )


if __name__ == "__main__":
    app()
