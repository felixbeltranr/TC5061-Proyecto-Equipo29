"""Drift calibrado de P1: umbrales por variable, estaciones existentes y nuevas, y control anual.

Uso desde la raíz del repositorio (después del reporte de Evidently de P1)::

    python -m mibici.monitoring --question p1
    python -m mibici.monitoring.p1_drift_calibrado \\
        --historical-trips data/processed/combined_dataset.parquet \\
        --current-trips data/raw/datos_abiertos_2026_10.csv \\
        --panel-campeon data/interim/p1_panel_diario.parquet

Arma en memoria el panel estación-día del histórico más el mes actual con el mismo constructor de
``mibici/monitoring/p1_datasets.py`` (``features.construir_panel_diario`` desde
``CONFIG_P1.train_inicio`` y ``features.construir_features``). El histórico debe terminar antes del
mes actual y los viajes del mes actual deben caer dentro de él. La referencia son las filas
elegibles de ``CONFIG_P1.train_inicio`` a ``features.ultimo_dia_conocido(CONFIG_P1.test_inicio)``
(2025-01-01 a 2026-04-29), la ventana con que se entrenó el campeón.

**Distancias.** Replican el cálculo de Evidently 0.7.23 para cada columna
(``evidently/legacy/calculations/stattests``):

- Numéricas (rezagos y medias): distancia de Wasserstein entre referencia y actual dividida entre
  la desviación estándar de la referencia con un piso de 0.001, ``max(std, 0.001)``, igual que
  Evidently ("Wasserstein distance (normed)"). Con una referencia constante la distancia no se
  anula y un cambio en el mes actual se marca. ``p3_drift.distancia_normada`` no usa ese piso
  (con desviación 0 da infinito); con los datos de P1 la desviación de la referencia es mucho
  mayor que 0.001 y el piso no se activa.
- Categóricas (``dia_semana``, ``festivo``, ``festivo_lag7``): distancia de Jensen-Shannon entre
  las proporciones de cada categoría ("Jensen-Shannon distance"), sin rellenar ceros y con
  logaritmo natural, como Evidently.

Evidently marca drift cuando la distancia es mayor o igual que 0.1; la columna ``drift_evidently``
usa ``> 0.1``, igual que las tablas de drift de P3. Solo difieren si una distancia vale
exactamente 0.1.

**Verificaciones.** Con ``--panel-campeon`` se comprueba que agregar el mes actual no cambia
ninguna fila del panel con que se entrenó el campeón. Con ``--evidently-json`` se comparan las
distancias del total contra los scores del reporte de Evidently del mismo mes: cada variable
debe tener score con el mismo método, la diferencia máxima no debe pasar de
``TOLERANCIA_EVIDENTLY`` y el drift con 0.1 debe coincidir con el de Evidently (si una distancia
vale exactamente 0.1 no coincide y hay que revisarla). Las fallas quedan en ``verificaciones`` del
resumen; el CLI escribe las salidas para revisarlas, registra cada falla y termina con código 1.

**Umbral calibrado.** Con más de un año de referencia contra un solo mes, el umbral fijo de 0.1
salta por la variación normal entre meses (vacaciones, festivos, calendario). Cada mes de la
referencia se compara contra el resto de la referencia y el umbral de cada variable es la mayor
de esas distancias: el mes actual solo se marca si se aleja más que cualquier mes del
entrenamiento (mismo criterio que ``p3_drift.py``). Solo calibran los meses con al menos
``MIN_DIAS_CALIBRACION`` días elegibles; enero de 2025 queda fuera porque los rezagos necesitan
28 días de historia y solo tiene filas elegibles el 30 y el 31. El resto contra el que se compara
cada mes sí incluye esos días.

**Grupos.** La distancia del mes actual se calcula con todas las estaciones, con las existentes
(estaban en la referencia) y con las nuevas, para separar un cambio de comportamiento de un
cambio en la composición de la red. Como control, el mismo mes del año anterior (octubre de 2025
para octubre de 2026) se compara contra la referencia, y las existentes del mes actual contra ese
mes. Las filas sin historia suficiente o inactivas se excluyen y se cuentan en el resumen.

**Meses intermedios.** ``p1_drift_mensual.csv`` repite la medida para cada mes entre el fin de la
referencia y el mes actual (mayo a septiembre de 2026). Es una referencia histórica y no una
prueba intacta: el README documenta que la política de dominio de P1 se agregó después de ver los
resultados de prueba de ese periodo. La comparación posterior es el mes actual.

El drift es una señal sobre la distribución de las variables, no una prueba de pérdida de
desempeño ni una decisión automática de reentrenar; se confirma con la evaluación A/B.

Salidas en ``--output`` (por omisión ``reports/monitoring/AAAA_MM/p1``): ``p1_drift_calibrado.csv``
(tabla del mes actual), ``p1_umbrales_calibrados.csv``, ``p1_distancias_mes_vs_resto.csv``,
``p1_drift_mensual.csv`` y ``p1_drift_calibrado_resumen.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
import platform
from typing import Annotated

from loguru import logger
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import typer

from mibici import features
from mibici.config import CONFIG, CONFIG_P1
from mibici.features import FEATURES
from mibici.monitoring.p1_datasets import P1_CATEGORICAL_FEATURES
from mibici.monitoring.p1_drift import file_sha256, month_bounds

app = typer.Typer()

UMBRAL_EVIDENTLY = 0.1  # umbral por omisión de Evidently para las dos distancias
PISO_DESVIACION = 0.001  # Evidently divide la distancia de Wasserstein entre max(std, 0.001)
TOLERANCIA_EVIDENTLY = 1e-9  # diferencia máxima aceptada contra los scores del reporte
MIN_DIAS_CALIBRACION = 20  # días elegibles mínimos para que un mes de la referencia calibre
METODOS_EVIDENTLY = {
    "wasserstein_normada": "Wasserstein distance (normed)",
    "jensen_shannon": "Jensen-Shannon distance",
}
GRUPOS = {"todas": "", "existentes": "_existentes", "nuevas": "_nuevas"}
ARCHIVOS = {
    "tabla_mes_actual": "p1_drift_calibrado.csv",
    "umbrales": "p1_umbrales_calibrados.csv",
    "distancias_mes_vs_resto": "p1_distancias_mes_vs_resto.csv",
    "drift_mensual": "p1_drift_mensual.csv",
    "resumen": "p1_drift_calibrado_resumen.json",
}


def wasserstein_normada(referencia: pd.Series, actual: pd.Series) -> float:
    """Distancia de Wasserstein dividida entre la desviación estándar de la referencia.

    Es la medida "Wasserstein distance (normed)" de Evidently: divide entre
    ``max(desviación, PISO_DESVIACION)``, así que una referencia constante no anula la distancia.
    """
    desviacion = max(float(np.std(referencia)), PISO_DESVIACION)
    return float(wasserstein_distance(referencia, actual) / desviacion)


def jensen_shannon(referencia: pd.Series, actual: pd.Series) -> float:
    """Distancia de Jensen-Shannon entre las proporciones de cada categoría (logaritmo natural).

    Las categorías que solo aparecen en una de las dos muestras cuentan con proporción 0 en la
    otra, sin rellenar ceros, como Evidently.
    """
    categorias = sorted(set(referencia.unique()) | set(actual.unique()))
    p = referencia.value_counts(normalize=True).reindex(categorias, fill_value=0).to_numpy()
    q = actual.value_counts(normalize=True).reindex(categorias, fill_value=0).to_numpy()
    return float(jensenshannon(p, q))


def metodo(variable: str) -> str:
    """Nombre del método de distancia de una variable de P1."""
    return "jensen_shannon" if variable in P1_CATEGORICAL_FEATURES else "wasserstein_normada"


def distancia(variable: str, referencia: pd.Series, actual: pd.Series) -> float:
    """Distancia de la variable con su método; NaN si alguna de las dos muestras está vacía."""
    if referencia.empty or actual.empty:
        return float("nan")
    if variable in P1_CATEGORICAL_FEATURES:
        return jensen_shannon(referencia, actual)
    return wasserstein_normada(referencia, actual)


def _media(serie: pd.Series) -> float:
    return float(serie.mean()) if len(serie) else float("nan")


def construir_panel(
    historical_trips_path: Path,
    current_trips_path: Path,
    current_start: str,
    current_end_exclusive: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Panel estación-día del histórico más el mes actual y sus features de P1.

    Igual que ``p1_datasets.build_p1_populations``: el panel empieza en ``CONFIG_P1.train_inicio``
    y termina el último día del mes actual. Falla si el histórico tiene viajes desde el inicio del
    mes actual (se duplicarían) o si los viajes del mes actual caen fuera de él.
    """
    month_bounds(current_start, current_end_exclusive)
    inicio, fin_excl = pd.Timestamp(current_start), pd.Timestamp(current_end_exclusive)
    historico = features.cargar_salidas(historical_trips_path)
    actual = features.cargar_salidas(current_trips_path)

    fechas_hist = historico["Inicio_del_viaje"].dropna().dt.normalize()
    if fechas_hist.empty:
        raise ValueError("El histórico no tiene fechas de inicio legibles.")
    if (fechas_hist >= inicio).any():
        raise ValueError(
            f"El histórico debe terminar antes del mes actual ({inicio.date()}); "
            f"llega a {fechas_hist.max().date()} y duplicaría viajes del mes actual."
        )
    fechas_act = actual["Inicio_del_viaje"].dropna().dt.normalize()
    if fechas_act.empty:
        raise ValueError("El archivo del mes actual no tiene fechas de inicio legibles.")
    if (fechas_act < inicio).any() or (fechas_act >= fin_excl).any():
        raise ValueError(
            f"Los viajes del mes actual deben caer en [{inicio.date()}, {fin_excl.date()}); "
            f"van de {fechas_act.min().date()} a {fechas_act.max().date()}."
        )

    panel, resumen_panel = features.construir_panel_diario(
        pd.concat([historico, actual], ignore_index=True),
        inicio=pd.Timestamp(CONFIG_P1.train_inicio),
        fin=fin_excl - pd.Timedelta(days=1),
    )
    info = {
        "viajes_historicos": len(historico),
        "viajes_mes_actual": len(actual),
        "fechas_ilegibles_historico": int(historico["Inicio_del_viaje"].isna().sum()),
        "fechas_ilegibles_mes_actual": int(actual["Inicio_del_viaje"].isna().sum()),
        "historico_inicio": str(fechas_hist.min().date()),
        "historico_fin": str(fechas_hist.max().date()),
        "mes_actual_inicio": str(fechas_act.min().date()),
        "mes_actual_fin": str(fechas_act.max().date()),
        "panel": resumen_panel,
    }
    return panel, features.construir_features(panel), info


@dataclass(frozen=True)
class Poblaciones:
    """Filas de P1 que se comparan, todas con estacion, fecha, mes, objetivo y FEATURES."""

    referencia: pd.DataFrame  # elegibles de la ventana del campeón
    actual_todas: pd.DataFrame  # mes actual antes del filtro de elegibilidad
    actual: pd.DataFrame  # elegibles del mes actual
    control: pd.DataFrame  # elegibles de la referencia en el mismo mes del año anterior
    intermedias: pd.DataFrame  # elegibles entre el fin de la referencia y el mes actual
    referencia_inicio: pd.Timestamp
    referencia_fin: pd.Timestamp
    control_inicio: pd.Timestamp
    control_fin_exclusivo: pd.Timestamp


def separar_poblaciones(
    features_p1: pd.DataFrame, current_start: str, current_end_exclusive: str
) -> Poblaciones:
    """Separa referencia, mes actual, control del año anterior y meses intermedios."""
    inicio, fin_excl = pd.Timestamp(current_start), pd.Timestamp(current_end_exclusive)
    ref_inicio = pd.Timestamp(CONFIG_P1.train_inicio)
    ref_fin = features.ultimo_dia_conocido(pd.Timestamp(CONFIG_P1.test_inicio))
    if inicio <= ref_fin:
        raise ValueError(
            f"El mes actual ({inicio.date()}) debe empezar después de la referencia "
            f"({ref_inicio.date()} a {ref_fin.date()})."
        )
    elegibles = features_p1.loc[features_p1["elegible"]].copy()
    elegibles["mes"] = elegibles["fecha"].dt.to_period("M").astype(str)
    referencia = elegibles[elegibles["fecha"].between(ref_inicio, ref_fin)]
    en_mes = (features_p1["fecha"] >= inicio) & (features_p1["fecha"] < fin_excl)
    actual = elegibles[(elegibles["fecha"] >= inicio) & (elegibles["fecha"] < fin_excl)]
    if actual.empty:
        raise ValueError("El mes actual no tiene filas elegibles.")
    if referencia.empty:
        raise ValueError("La referencia no tiene filas elegibles.")
    control_inicio = inicio - pd.DateOffset(years=1)
    control_fin = fin_excl - pd.DateOffset(years=1)
    control = referencia[
        (referencia["fecha"] >= control_inicio) & (referencia["fecha"] < control_fin)
    ]
    intermedias = elegibles[
        (elegibles["fecha"] >= pd.Timestamp(CONFIG_P1.test_inicio)) & (elegibles["fecha"] < inicio)
    ]
    return Poblaciones(
        referencia=referencia,
        actual_todas=features_p1.loc[en_mes],
        actual=actual,
        control=control,
        intermedias=intermedias,
        referencia_inicio=ref_inicio,
        referencia_fin=ref_fin,
        control_inicio=control_inicio,
        control_fin_exclusivo=control_fin,
    )


def calibrar(
    referencia: pd.DataFrame, min_dias: int = MIN_DIAS_CALIBRACION
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Distancia de cada mes de la referencia contra el resto y umbral calibrado por variable.

    ``referencia`` trae las columnas ``mes``, ``fecha`` y FEATURES. Solo calibran los meses con al
    menos ``min_dias`` fechas distintas; el resto de cada comparación es toda la referencia menos
    ese mes. Regresa la tabla mes × variable, los umbrales (``umbral_calibrado`` y
    ``mes_del_maximo``) y los días elegibles por mes.
    """
    dias_por_mes = referencia.groupby("mes")["fecha"].nunique()
    meses = dias_por_mes.index[dias_por_mes >= min_dias]
    if len(meses) < 2:
        raise ValueError(
            f"Se necesitan al menos dos meses con {min_dias} días elegibles para calibrar."
        )
    distancias = pd.DataFrame(
        {
            mes: {
                col: distancia(col, referencia.loc[referencia["mes"] != mes, col], grupo[col])
                for col in FEATURES
            }
            for mes, grupo in referencia[referencia["mes"].isin(meses)].groupby("mes")
        }
    ).T.rename_axis("mes")
    umbrales = pd.DataFrame(
        {"umbral_calibrado": distancias.max(), "mes_del_maximo": distancias.idxmax()}
    ).rename_axis("variable")
    return distancias, umbrales, dias_por_mes


def _distancias_por_grupo(
    referencia: pd.DataFrame, actual: pd.DataFrame, umbrales: pd.DataFrame
) -> list[dict]:
    """Distancias y medias de cada variable con todas las estaciones, existentes y nuevas."""
    ya_estaban = actual["estacion"].isin(referencia["estacion"].unique())
    existentes, nuevas = actual[ya_estaban], actual[~ya_estaban]
    return [
        {
            "variable": col,
            "metodo": metodo(col),
            "distancia": distancia(col, referencia[col], actual[col]),
            "distancia_existentes": distancia(col, referencia[col], existentes[col]),
            "distancia_nuevas": distancia(col, referencia[col], nuevas[col]),
            "umbral_evidently": UMBRAL_EVIDENTLY,
            "umbral_calibrado": float(umbrales.loc[col, "umbral_calibrado"]),
            "media_referencia": _media(referencia[col]),
            "media_actual": _media(actual[col]),
            "media_existentes": _media(existentes[col]),
            "media_nuevas": _media(nuevas[col]),
        }
        for col in FEATURES
    ]


def marcar_drift(tabla: pd.DataFrame) -> pd.DataFrame:
    """Agrega ``drift_evidently`` (> 0.1) y ``drift_calibrado`` (> umbral) para cada grupo.

    Una distancia NaN (grupo sin filas) no se marca como drift.
    """
    tabla = tabla.copy()
    for sufijo in GRUPOS.values():
        valor = tabla[f"distancia{sufijo}"]
        tabla[f"drift_evidently{sufijo}"] = valor > UMBRAL_EVIDENTLY
        tabla[f"drift_calibrado{sufijo}"] = valor > tabla["umbral_calibrado"]
    return tabla


def tabla_mes_actual(poblaciones: Poblaciones, umbrales: pd.DataFrame) -> pd.DataFrame:
    """Drift por variable del mes actual por grupo, con el control del mismo mes del año anterior.

    ``control_vs_referencia`` es la distancia del mismo mes del año anterior contra la referencia
    (cuánto se aleja un mes normal) y ``existentes_vs_control`` la de las estaciones existentes
    del mes actual contra ese mes.
    """
    ref, actual, control = poblaciones.referencia, poblaciones.actual, poblaciones.control
    existentes = actual[actual["estacion"].isin(ref["estacion"].unique())]
    tabla = pd.DataFrame(_distancias_por_grupo(ref, actual, umbrales)).set_index("variable")
    tabla["control_vs_referencia"] = [distancia(c, ref[c], control[c]) for c in FEATURES]
    tabla["existentes_vs_control"] = [distancia(c, control[c], existentes[c]) for c in FEATURES]
    tabla["media_control"] = [_media(control[c]) for c in FEATURES]
    return marcar_drift(tabla)


def drift_mensual(poblaciones: Poblaciones, umbrales: pd.DataFrame) -> pd.DataFrame:
    """Drift por variable y grupo de cada mes entre el fin de la referencia y el mes actual."""
    filas = [
        {"mes": mes} | fila
        for mes, grupo in poblaciones.intermedias.groupby("mes")
        for fila in _distancias_por_grupo(poblaciones.referencia, grupo, umbrales)
    ]
    columnas = ["mes", "variable", "metodo", "umbral_calibrado"]
    columnas += [f"distancia{sufijo}" for sufijo in GRUPOS.values()]
    return marcar_drift(pd.DataFrame(filas) if filas else pd.DataFrame(columns=columnas))


def _variables_con_drift(tabla: pd.DataFrame, criterio: str) -> dict[str, list[str]]:
    variables = tabla["variable"] if "variable" in tabla else tabla.index.to_series()
    return {
        grupo: variables[tabla[f"drift_{criterio}{sufijo}"].to_numpy()].tolist()
        for grupo, sufijo in GRUPOS.items()
    }


def leer_scores_evidently(ruta: Path) -> pd.DataFrame:
    """Scores ``ValueDrift`` por columna de un reporte JSON de Evidently (método y umbral)."""
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    filas = [
        {
            "variable": metrica["config"]["column"],
            "metodo_evidently": metrica["config"]["method"],
            "umbral": float(metrica["config"]["threshold"]),
            "score": float(metrica["value"]),
        }
        for metrica in datos.get("metrics", [])
        if metrica.get("config", {}).get("type") == "evidently:metric_v2:ValueDrift"
    ]
    if not filas:
        raise ValueError(f"{ruta} no tiene métricas ValueDrift de Evidently.")
    return pd.DataFrame(filas).set_index("variable")


def comparar_con_evidently(tabla: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Distancia del total contra el score de Evidently de cada variable.

    ``drift_segun_evidently`` aplica la regla de Evidently (score mayor o igual que su umbral).
    """
    comparacion = tabla[["metodo", "distancia", "drift_evidently"]].join(scores, how="outer")
    comparacion["metodo_coincide"] = (
        comparacion["metodo"].map(METODOS_EVIDENTLY) == comparacion["metodo_evidently"]
    )
    comparacion["dif_abs"] = (comparacion["distancia"] - comparacion["score"]).abs()
    comparacion["drift_segun_evidently"] = comparacion["score"] >= comparacion["umbral"]
    return comparacion


def comparar_con_panel_campeon(panel: pd.DataFrame, original: pd.DataFrame) -> dict:
    """Comprueba que las filas del panel hasta el último día de ``original`` son idénticas.

    Sirve para confirmar que agregar el mes actual no cambia ninguna fila del panel con que se
    entrenó el campeón (mismas columnas, tipos y valores).
    """
    hasta = original["fecha"].max()
    columnas = list(original.columns)
    nuevo = panel.loc[panel["fecha"] <= hasta, columnas]
    orden = ["estacion", "fecha"]
    igual = (
        nuevo.sort_values(orden)
        .reset_index(drop=True)
        .equals(original.sort_values(orden).reset_index(drop=True))
    )
    return {
        "igual": bool(igual),
        "filas_campeon": len(original),
        "filas_panel_hasta_esa_fecha": len(nuevo),
        "fecha_min": str(original["fecha"].min().date()),
        "fecha_max": str(hasta.date()),
    }


def _sin_nan(valor: float) -> float | None:
    return None if pd.isna(valor) else float(valor)


def fallas_de_verificacion(resumen: dict, tolerancia: float = TOLERANCIA_EVIDENTLY) -> list[str]:
    """Fallas de las comprobaciones contra el panel del campeón y el reporte de Evidently.

    Revisa las secciones ``panel_campeon`` y ``equivalencia_evidently`` del resumen que existan.
    Regresa una lista vacía si todo coincide; una diferencia máxima indefinida (alguna variable
    sin score o sin distancia) cuenta como falla.
    """
    fallas = []
    panel = resumen.get("panel_campeon")
    if panel is not None and not panel["igual"]:
        fallas.append(
            f"El panel hasta {panel['fecha_max']} no es idéntico al panel del campeón "
            f"({panel['filas_panel_hasta_esa_fecha']} filas contra {panel['filas_campeon']})."
        )
    equivalencia = resumen.get("equivalencia_evidently")
    if equivalencia is None:
        return fallas
    sin_score = equivalencia["variables_sin_score"]
    solo_evidently = equivalencia["variables_solo_en_evidently"]
    if sin_score or solo_evidently:
        fallas.append(
            f"Las variables no coinciden con el reporte de Evidently: sin score {sin_score}, "
            f"solo en Evidently {solo_evidently}."
        )
    if not equivalencia["metodos_coinciden"]:
        fallas.append("El método de alguna variable no coincide con el del reporte de Evidently.")
    maxima = equivalencia["max_dif_abs"]
    if maxima is None or maxima > tolerancia:
        fallas.append(
            f"Diferencia máxima contra Evidently {maxima} indefinida o mayor que {tolerancia:g}."
        )
    if not equivalencia["drift_evidently_coincide"]:
        fallas.append("El drift con umbral 0.1 no coincide con el que marca Evidently.")
    return fallas


def resumir(
    poblaciones: Poblaciones,
    tabla: pd.DataFrame,
    mensual: pd.DataFrame,
    umbrales: pd.DataFrame,
    dias_por_mes: pd.Series,
    min_dias: int = MIN_DIAS_CALIBRACION,
) -> dict:
    """Conteos de filas y estaciones, exclusiones, calibración y variables con drift."""
    ref, actual, todas = poblaciones.referencia, poblaciones.actual, poblaciones.actual_todas
    estaciones_ref = ref["estacion"].unique()
    ya_estaban = actual["estacion"].isin(estaciones_ref)
    meses_calibracion = dias_por_mes.index[dias_por_mes >= min_dias]
    excluidos = dias_por_mes[dias_por_mes < min_dias]
    intermedios = [
        {
            "mes": mes,
            "filas": len(grupo),
            "estaciones": int(grupo["estacion"].nunique()),
            "estaciones_nuevas": int(
                grupo.loc[~grupo["estacion"].isin(estaciones_ref), "estacion"].nunique()
            ),
        }
        for mes, grupo in poblaciones.intermedias.groupby("mes")
    ]
    return {
        "problema": "P1",
        "unidad": "estación-día",
        "umbral_evidently": UMBRAL_EVIDENTLY,
        "referencia": {
            "inicio": str(poblaciones.referencia_inicio.date()),
            "fin": str(poblaciones.referencia_fin.date()),
            "filas": len(ref),
            "estaciones": len(estaciones_ref),
            "meses": int(ref["mes"].nunique()),
            "min_dias_calibracion": min_dias,
            "meses_calibracion": list(meses_calibracion),
            "meses_excluidos_de_calibracion": {m: int(d) for m, d in excluidos.items()},
        },
        "mes_actual": {
            "filas_antes_de_elegibilidad": len(todas),
            "estaciones_antes_de_elegibilidad": int(todas["estacion"].nunique()),
            "filas_sin_historia": int(todas["sin_historia"].sum()),
            "filas_inactivas": int(todas["inactiva"].sum()),
            "estaciones_sin_historia": sorted(
                map(str, todas.loc[todas["sin_historia"], "estacion"].unique())
            ),
            "filas_elegibles": len(actual),
            "estaciones_elegibles": int(actual["estacion"].nunique()),
            "filas_existentes": int(ya_estaban.sum()),
            "filas_nuevas": int((~ya_estaban).sum()),
            "estaciones_existentes": int(actual.loc[ya_estaban, "estacion"].nunique()),
            "estaciones_nuevas": int(actual.loc[~ya_estaban, "estacion"].nunique()),
        },
        "control": {
            "inicio": str(poblaciones.control_inicio.date()),
            "fin_exclusivo": str(poblaciones.control_fin_exclusivo.date()),
            "filas": len(poblaciones.control),
            "estaciones": int(poblaciones.control["estacion"].nunique()),
        },
        "meses_intermedios": intermedios,
        "objetivo_medio": {
            "referencia": _sin_nan(ref["objetivo"].mean()),
            "control": _sin_nan(poblaciones.control["objetivo"].mean()),
            "existentes": _sin_nan(actual.loc[ya_estaban, "objetivo"].mean()),
            "nuevas": _sin_nan(actual.loc[~ya_estaban, "objetivo"].mean()),
        },
        "umbrales": {
            var: {"umbral_calibrado": float(fila.umbral_calibrado), "mes": fila.mes_del_maximo}
            for var, fila in umbrales.iterrows()
        },
        "drift_mes_actual": {
            criterio: _variables_con_drift(tabla, criterio)
            for criterio in ("evidently", "calibrado")
        },
        "drift_mensual": {
            criterio: {
                mes: _variables_con_drift(grupo, criterio) for mes, grupo in mensual.groupby("mes")
            }
            for criterio in ("evidently", "calibrado")
        },
    }


def run(
    historical_trips_path: Path,
    current_trips_path: Path,
    output_dir: Path,
    current_start: str = "2026-10-01",
    current_end_exclusive: str = "2026-11-01",
    panel_campeon_path: Path | None = None,
    evidently_json_path: Path | None = None,
    min_dias: int = MIN_DIAS_CALIBRACION,
) -> dict:
    """Calcula el drift calibrado de P1, escribe las tablas en ``output_dir`` y regresa el resumen.

    Las salidas se escriben aunque falle alguna verificación; las fallas quedan en
    ``resumen["verificaciones"]["fallas"]`` y quien llama decide qué hacer con ellas (el CLI
    termina con código 1).
    """
    for ruta in (historical_trips_path, current_trips_path):
        if not Path(ruta).is_file():
            raise FileNotFoundError(f"No existe {ruta}.")
    panel, feats, info = construir_panel(
        historical_trips_path, current_trips_path, current_start, current_end_exclusive
    )
    poblaciones = separar_poblaciones(feats, current_start, current_end_exclusive)
    distancias, umbrales, dias_por_mes = calibrar(poblaciones.referencia, min_dias)
    tabla = tabla_mes_actual(poblaciones, umbrales)
    mensual = drift_mensual(poblaciones, umbrales)
    resumen = resumir(poblaciones, tabla, mensual, umbrales, dias_por_mes, min_dias)
    resumen["periodo_actual"] = {
        "inicio": current_start,
        "fin_exclusivo": current_end_exclusive,
    }
    resumen["viajes"] = info

    entradas = {
        "historical_trips": historical_trips_path,
        "current_trips": current_trips_path,
    }
    if panel_campeon_path is not None:
        resumen["panel_campeon"] = comparar_con_panel_campeon(
            panel, pd.read_parquet(panel_campeon_path)
        )
        entradas["panel_campeon"] = panel_campeon_path
    if evidently_json_path is not None:
        scores = leer_scores_evidently(evidently_json_path)
        comparacion = comparar_con_evidently(tabla, scores)
        resumen["equivalencia_evidently"] = {
            "variables": len(comparacion),
            "variables_sin_score": [var for var in tabla.index if var not in scores.index],
            "variables_solo_en_evidently": [var for var in scores.index if var not in tabla.index],
            "metodos_coinciden": bool(comparacion["metodo_coincide"].all()),
            "max_dif_abs": _sin_nan(comparacion["dif_abs"].max(skipna=False)),
            "drift_evidently_coincide": bool(
                (comparacion["drift_evidently"] == comparacion["drift_segun_evidently"]).all()
            ),
            "por_variable": {
                var: {
                    "distancia": _sin_nan(fila.distancia),
                    "score_evidently": _sin_nan(fila.score),
                    "dif_abs": _sin_nan(fila.dif_abs),
                }
                for var, fila in comparacion.iterrows()
            },
        }
        entradas["evidently_json"] = evidently_json_path
    resumen["verificaciones"] = {
        "realizadas": [
            nombre for nombre in ("panel_campeon", "evidently_json") if nombre in entradas
        ],
        "tolerancia_evidently": TOLERANCIA_EVIDENTLY,
        "fallas": fallas_de_verificacion(resumen),
    }
    resumen["entradas"] = {
        nombre: {"path": str(ruta), "sha256": file_sha256(Path(ruta))}
        for nombre, ruta in entradas.items()
    }
    resumen["runtime"] = {
        "python": platform.python_version(),
        **{paquete: version(paquete) for paquete in ("numpy", "pandas", "scipy")},
    }
    resumen["salidas"] = ARCHIVOS
    resumen["interpretacion"] = (
        "El drift es una señal sobre la distribución de las variables, no una prueba de pérdida "
        "de desempeño ni una decisión automática de reentrenar; se confirma con la evaluación A/B."
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(output_dir / ARCHIVOS["tabla_mes_actual"])
    umbrales.to_csv(output_dir / ARCHIVOS["umbrales"])
    distancias.to_csv(output_dir / ARCHIVOS["distancias_mes_vs_resto"])
    mensual.to_csv(output_dir / ARCHIVOS["drift_mensual"], index=False)
    (output_dir / ARCHIVOS["resumen"]).write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return resumen


@app.command()
def main(
    historical_trips: Annotated[
        Path, typer.Option(help="Viajes históricos que terminan antes del mes actual.")
    ] = CONFIG.processed_data_dir / CONFIG.output_filename,
    current_trips: Annotated[
        Path | None, typer.Option(help="CSV del mes actual (por omisión el de data/raw).")
    ] = None,
    current_start: str = "2026-10-01",
    current_end_exclusive: str = "2026-11-01",
    output: Annotated[
        Path | None,
        typer.Option(help="Carpeta de salida (por omisión reports/monitoring/AAAA_MM/p1)."),
    ] = None,
    panel_campeon: Annotated[
        Path | None,
        typer.Option(help="Panel de entrenamiento del campeón para comprobar que no cambia."),
    ] = None,
    evidently_json: Annotated[
        Path | None,
        typer.Option(help="Reporte JSON de Evidently del mismo mes (por omisión el de --output)."),
    ] = None,
    min_dias_calibracion: int = MIN_DIAS_CALIBRACION,
):
    """Drift calibrado de P1 del mes actual contra la ventana de entrenamiento del campeón."""
    year, month = month_bounds(current_start, current_end_exclusive)
    output = output or CONFIG.reports_dir / "monitoring" / f"{year}_{month:02d}" / "p1"
    current_trips = current_trips or CONFIG.raw_data_dir / f"datos_abiertos_{year}_{month:02d}.csv"
    if evidently_json is None and (output / "p1_data_drift.json").is_file():
        evidently_json = output / "p1_data_drift.json"
    resumen = run(
        historical_trips,
        current_trips,
        output,
        current_start,
        current_end_exclusive,
        panel_campeon,
        evidently_json,
        min_dias_calibracion,
    )
    actual = resumen["mes_actual"]
    logger.info(
        f"Referencia: {resumen['referencia']['filas']} filas | mes actual: "
        f"{actual['filas_elegibles']} filas elegibles ({actual['estaciones_existentes']} "
        f"estaciones existentes, {actual['estaciones_nuevas']} nuevas)"
    )
    for criterio, grupos in resumen["drift_mes_actual"].items():
        logger.info(f"Drift con umbral {criterio}: {grupos}")
    if "panel_campeon" in resumen:
        panel = resumen["panel_campeon"]
        logger.info(
            f"Panel hasta {panel['fecha_max']} idéntico al del campeón: {panel['igual']} "
            f"({panel['filas_panel_hasta_esa_fecha']} filas contra {panel['filas_campeon']})"
        )
    if "equivalencia_evidently" in resumen:
        equivalencia = resumen["equivalencia_evidently"]
        logger.info(
            f"Contra Evidently: diferencia máxima {equivalencia['max_dif_abs']}, métodos "
            f"coinciden {equivalencia['metodos_coinciden']}, drift coincide "
            f"{equivalencia['drift_evidently_coincide']}"
        )
    fallas = resumen["verificaciones"]["fallas"]
    for falla in fallas:
        logger.error(falla)
    if fallas:
        logger.error(
            f"Las salidas en {output} no pasan las verificaciones; ver 'verificaciones' en "
            f"{ARCHIVOS['resumen']}."
        )
        raise typer.Exit(1)
    logger.success(f"Drift calibrado de P1 en {output}")


if __name__ == "__main__":
    app()
