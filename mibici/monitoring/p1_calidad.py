"""Reporte de calidad de datos de P1 (estación-día) para el mes actual, separado del de drift.

El reporte de drift (``mibici.monitoring.p1_drift``) compara distribuciones de las 11 variables
del modelo; este módulo revisa que los datos del mes actual sean **válidos** antes de usarlos en
el monitoreo o en el A/B:

- **Población**: la referencia son las filas elegibles con las que se entrenó el campeón
  (``CONFIG_P1.train_inicio`` a ``ultimo_dia_conocido(CONFIG_P1.test_inicio)``) y la actual son
  las filas elegibles del mes (por ejemplo, octubre de 2026), construidas con el mismo
  ``construir_panel_diario`` y ``construir_features`` de ``mibici.features``. A diferencia del
  reporte de drift, aquí entran también ``estacion``, ``fecha`` y ``objetivo``.
- **Evidently** (``p1_data_quality.html`` y ``.json``): estadísticas del conjunto y por columna
  (``DatasetStats`` y ``ValueStats``, los componentes de ``DataSummaryPreset``) con pruebas
  explícitas: filas, columnas, filas y columnas duplicadas, columnas vacías o constantes,
  faltantes por columna, dominio (salidas, rezagos y medias ≥ 0; ``dia_semana`` en 0..6;
  ``festivo`` y ``festivo_lag7`` en {0, 1}), un tope para ``objetivo`` y los rezagos
  (``FACTOR_MAXIMO`` veces el máximo de la referencia) y valores fuera del rango observado en
  la referencia, tanto por conteo (un solo valor ya advierte) como por proporción (más de
  ``UMBRAL_FUERA_DE_RANGO`` falla). No se usa ``DataSummaryPreset`` directamente porque en
  evidently 0.7.23 agrega siempre ``RowCount()`` con la prueba por omisión «igual a la
  referencia ± 10 %», que no aplica aquí: la referencia abarca 16 meses y el periodo actual uno.
- **Integridad del archivo de viajes** (pandas, antes de agregar): del archivo del mes solo se
  leen ``Viaje_Id``, ``Inicio_del_viaje``, ``Fin_del_viaje``, ``Origen_Id`` y ``Destino_Id``,
  sin datos de usuario. ``Viaje_Id`` único; viajes que repiten inicio, fin, origen y destino
  con otro ``Viaje_Id``, que es como se ve una recarga o una duplicación del ETL (advertencia
  con su conteo; falla por encima de ``UMBRAL_VIAJES_REPETIDOS`` de los viajes); viajes sin
  fecha o estación legible, que no entran al panel; conciliación de las salidas del panel en el
  mes con los viajes válidos del archivo; y volumen diario frente a la mediana del mismo día de
  la semana en el mes.
- **Integridad de la tabla estación-día** (pandas): cobertura de todos los días del mes y filas
  excluidas por ``sin_historia`` o ``inactiva``; ``dia_semana`` y ``festivo`` coherentes con la
  fecha y el calendario de ``mibici.features.festivos``; conteos enteros no negativos; y
  consistencia temporal: cada rezago y cada media se recalculan desde el panel con salidas de
  D−29 a D−2 (nunca D−1 ni D), lo que documenta que no hay fuga temporal en las filas del mes.
  La llave (estacion, fecha) única y ``DuplicatedRowCount`` de Evidently son **controles
  estructurales**: ``construir_panel_diario`` cuenta los viajes por estación y día, así que la
  tabla es única por construcción y esas pruebas no pueden ver viajes repetidos; eso lo revisa
  la integridad del archivo de viajes.

Las pruebas que reflejan algo esperado y explicable (por ejemplo, ``festivo`` constante en un
mes sin festivos) no son críticas: quedan como advertencia y el resumen dice por qué. Una prueba
crítica que falla se reporta como falla; con ``--estricto`` el comando termina con código 1.

Uso desde la raíz del repositorio::

    python -m mibici.monitoring.p1_calidad
    python -m mibici.monitoring.p1_calidad --current-start 2026-10-01 \\
        --current-end-exclusive 2026-11-01 --panel-campeon data/interim/p1_panel_diario.parquet
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from importlib.metadata import version
from itertools import combinations
import json
from pathlib import Path
import platform
from typing import Annotated

from evidently import DataDefinition, Dataset, Report
from evidently.metrics import MissingValueCount, OutListValueCount, OutRangeValueCount
from evidently.presets import DatasetStats, ValueStats
from evidently.tests import eq, gte, lte
from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici.config import CONFIG, CONFIG_P1
from mibici.features import (
    FEATURES,
    LAG_MINIMO,
    LAGS_SEMANALES,
    MAX_FRACCION_FECHAS_ILEGIBLES,
    VENTANA_CORTA,
    VENTANA_LARGA,
    cargar_salidas,
    construir_features,
    construir_panel_diario,
    festivos,
    ultimo_dia_conocido,
)
from mibici.monitoring.p1_drift import file_sha256, month_bounds

app = typer.Typer()

LLAVE = ("estacion", "fecha")
OBJETIVO = "objetivo"
CATEGORICAS = ("dia_semana", "festivo", "festivo_lag7")
NUMERICAS = tuple(f for f in FEATURES if f not in CATEGORICAS)
COLUMNAS = [*LLAVE, OBJETIVO, *FEATURES]
REZAGOS = (LAG_MINIMO, *LAGS_SEMANALES)
CONTEOS = (OBJETIVO, *(f"lag_{k}" for k in REZAGOS))
BANDERAS_CALENDARIO = ("festivo", "festivo_lag7")
DOMINIOS = {
    "dia_semana": [str(d) for d in range(7)],
    "festivo": ["0", "1"],
    "festivo_lag7": ["0", "1"],
}
# Valores fuera del rango de la referencia: el campeón no los vio al entrenar y extrapola en
# ellos. Uno solo ya es advertencia (puede ser un pico real de demanda); si pasan de esta
# proporción de las filas de una variable, la prueba falla porque el mes ya no se parece a la
# población con que se entrenó el campeón
UMBRAL_FUERA_DE_RANGO = 0.05
# Tope crítico de objetivo y de los rezagos: este múltiplo del máximo de la referencia. Por
# encima, el conteo de una estación-día se trata como imposible (por ejemplo, viajes repetidos)
FACTOR_MAXIMO = 2
# Volumen diario aceptado, como razón frente a la mediana del mismo día de la semana en el mes:
# un día cargado dos veces ronda 2 y uno cortado a la mitad, 0.5
RAZON_VOLUMEN = (0.5, 1.5)
# Columnas de viaje que se leen del archivo del mes para su integridad (sin datos de usuario)
COLUMNAS_VIAJE = ["Viaje_Id", "Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"]
VIAJE_SIN_ID = ["Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"]
DURACION_CORTA_S = 60  # para describir viajes repetidos de segundos con origen = destino
# Viajes repetidos con otro Viaje_Id: dos viajes que coinciden al segundo son raros, y recargar
# un solo día ya suma del orden de 1/30 de los viajes del mes. Hasta esta proporción del archivo
# es advertencia; por encima se trata como recarga y falla
UMBRAL_VIAJES_REPETIDOS = 0.001
UMBRAL_CASI = 0.95  # mismo umbral que AlmostConstant/AlmostDuplicated de Evidently
TOLERANCIA = 1e-9
ESTADOS = {"SUCCESS": "pass", "FAIL": "fail", "WARNING": "warning", "ERROR": "error"}
OPERADORES = {
    "EqualMetricTest": "=",
    "NotEqualMetricTest": "≠",
    "GreaterOrEqualMetricTest": "≥",
    "GreaterThanMetricTest": ">",
    "LessOrEqualMetricTest": "≤",
    "LessThanMetricTest": "<",
}


@dataclass
class Poblaciones:
    """Filas de referencia y del mes actual, más lo necesario para revisar su integridad."""

    referencia: pd.DataFrame
    actual: pd.DataFrame
    mes: pd.DataFrame  # todas las filas del mes, antes del filtro de elegibilidad
    panel: pd.DataFrame
    dias_con_viajes: pd.DatetimeIndex
    metadatos: dict = field(default_factory=dict)


def construir_poblaciones(
    historicos: pd.DataFrame, actuales: pd.DataFrame, inicio: str, fin_exclusivo: str
) -> Poblaciones:
    """Referencia (entrenamiento del campeón) y mes actual con el constructor de features de P1.

    ``historicos`` y ``actuales`` son viajes como los devuelve ``cargar_salidas``. El histórico
    debe terminar antes del mes actual y el archivo actual debe caer completo en el mes, igual
    que en ``build_p1_populations`` del reporte de drift.
    """
    month_bounds(inicio, fin_exclusivo)
    inicio_ts, fin_ts = pd.Timestamp(inicio), pd.Timestamp(fin_exclusivo)
    fechas_hist = historicos["Inicio_del_viaje"].dropna().dt.normalize()
    if fechas_hist.empty:
        raise ValueError("El histórico no tiene fechas de inicio interpretables.")
    if (fechas_hist >= inicio_ts).any():
        raise ValueError("El histórico debe terminar antes del inicio del mes actual.")
    fechas_act = actuales["Inicio_del_viaje"].dropna().dt.normalize()
    if fechas_act.empty:
        raise ValueError("El archivo del mes actual no tiene fechas de inicio interpretables.")
    if (fechas_act < inicio_ts).any() or (fechas_act >= fin_ts).any():
        raise ValueError(
            f"Los viajes actuales deben caer en [{inicio_ts.date()}, {fin_ts.date()})."
        )

    viajes = pd.concat([historicos, actuales], ignore_index=True)
    panel, _ = construir_panel_diario(
        viajes, inicio=pd.Timestamp(CONFIG_P1.train_inicio), fin=fin_ts - pd.Timedelta(days=1)
    )
    tabla = construir_features(panel)
    ref_inicio = pd.Timestamp(CONFIG_P1.train_inicio)
    ref_fin = ultimo_dia_conocido(pd.Timestamp(CONFIG_P1.test_inicio))
    en_ref = tabla["fecha"].between(ref_inicio, ref_fin)
    en_mes = (tabla["fecha"] >= inicio_ts) & (tabla["fecha"] < fin_ts)
    referencia = tabla.loc[tabla["elegible"] & en_ref, COLUMNAS].reset_index(drop=True)
    actual = tabla.loc[tabla["elegible"] & en_mes, COLUMNAS].reset_index(drop=True)
    mes = tabla.loc[en_mes, [*COLUMNAS, "sin_historia", "inactiva", "elegible"]]
    mes = mes.reset_index(drop=True)
    if referencia.empty or actual.empty:
        raise ValueError("La referencia o el mes actual no tienen filas elegibles.")

    existentes = actual["estacion"].isin(referencia["estacion"].unique())
    sin_historia = mes.loc[mes["sin_historia"], "estacion"]
    inactivas = mes.loc[mes["inactiva"], "estacion"]
    metadatos = {
        "referencia": {
            "inicio_configurado": str(ref_inicio.date()),
            "fin_configurado": str(ref_fin.date()),
            "observado_inicio": str(referencia["fecha"].min().date()),
            "observado_fin": str(referencia["fecha"].max().date()),
            "filas": len(referencia),
            "estaciones": int(referencia["estacion"].nunique()),
        },
        "actual": {
            "inicio": str(inicio_ts.date()),
            "fin_exclusivo": str(fin_ts.date()),
            "observado_inicio": str(actual["fecha"].min().date()),
            "observado_fin": str(actual["fecha"].max().date()),
            "filas": len(actual),
            "estaciones": int(actual["estacion"].nunique()),
            "estaciones_existentes": int(actual.loc[existentes, "estacion"].nunique()),
            "estaciones_nuevas": int(actual.loc[~existentes, "estacion"].nunique()),
            "filas_existentes": int(existentes.sum()),
            "filas_nuevas": int((~existentes).sum()),
            "filas_antes_de_elegibilidad": len(mes),
            "estaciones_antes_de_elegibilidad": int(mes["estacion"].nunique()),
            "filas_sin_historia": len(sin_historia),
            "estaciones_sin_historia": sorted(int(e) for e in sin_historia.unique()),
            "filas_inactivas": len(inactivas),
            "estaciones_inactivas": sorted(int(e) for e in inactivas.unique()),
        },
        "viajes": {
            "historicos": len(historicos),
            "historicos_sin_fecha": int(historicos["Inicio_del_viaje"].isna().sum()),
            "actuales": len(actuales),
            "actuales_sin_fecha": int(actuales["Inicio_del_viaje"].isna().sum()),
        },
    }
    return Poblaciones(
        referencia=referencia,
        actual=actual,
        mes=mes,
        panel=panel,
        dias_con_viajes=pd.DatetimeIndex(np.sort(fechas_act.unique())),
        metadatos=metadatos,
    )


# --- Comprobaciones de integridad con pandas ---


def _comprobacion(
    nombre: str,
    poblacion: str,
    ok: bool,
    observado,
    criterio: str,
    detalle: dict | None = None,
    critica: bool = True,
) -> dict:
    """Una comprobación de integridad; si no es crítica, incumplirla es advertencia, no falla."""
    return {
        "nombre": nombre,
        "poblacion": poblacion,
        "resultado": "pass" if ok else ("fail" if critica else "warning"),
        "critica": critica,
        "valor_observado": observado,
        "criterio": criterio,
        "detalle": detalle or {},
    }


def _ejemplos(frame: pd.DataFrame, mascara, columnas: list[str], n: int = 5) -> list[dict]:
    """Primeras filas que incumplen una comprobación, en tipos que se pueden guardar en JSON."""
    filas = frame.loc[mascara, columnas].head(n)
    return json.loads(filas.to_json(orient="records", date_format="iso"))


def comprobar_llave(frame: pd.DataFrame, poblacion: str) -> dict:
    """La llave (estacion, fecha) no se repite.

    Es un control estructural: con ``construir_panel_diario`` la tabla es única por
    construcción. Detecta una tabla armada de otra forma (por ejemplo, al concatenar dos
    extracciones), no viajes repetidos; esos los revisa ``comprobar_viajes_repetidos``.
    """
    repetidas = frame.duplicated(list(LLAVE), keep=False)
    n = int(frame.duplicated(list(LLAVE)).sum())
    return _comprobacion(
        "llave_estacion_fecha_unica",
        poblacion,
        n == 0,
        n,
        "0 filas con (estacion, fecha) repetida",
        {"ejemplos": _ejemplos(frame, repetidas, list(LLAVE))} if n else None,
    )


def comprobar_calendario(frame: pd.DataFrame, poblacion: str) -> list[dict]:
    """dia_semana, festivo y festivo_lag7 coinciden con la fecha y el calendario de features.py."""
    fecha = frame["fecha"]
    anios = range(int(fecha.dt.year.min()) - 1, int(fecha.dt.year.max()) + 1)
    calendario = festivos(anios)
    esperados = {
        "dia_semana": (
            fecha.dt.dayofweek,
            "dia_semana = día de la semana de fecha (0 = lunes ... 6 = domingo)",
        ),
        "festivo": (
            fecha.isin(calendario).astype(int),
            "festivo = 1 si fecha está en mibici.features.festivos()",
        ),
        "festivo_lag7": (
            (fecha - pd.Timedelta(days=7)).isin(calendario).astype(int),
            "festivo_lag7 = 1 si fecha − 7 días está en mibici.features.festivos()",
        ),
    }
    resultado = []
    for columna, (esperado, criterio) in esperados.items():
        distinto = frame[columna].to_numpy() != esperado.to_numpy()
        n = int(distinto.sum())
        resultado.append(
            _comprobacion(
                f"{columna}_coherente_con_fecha",
                poblacion,
                n == 0,
                n,
                f"0 filas distintas: {criterio}",
                {"ejemplos": _ejemplos(frame, distinto, [*LLAVE, columna])} if n else None,
            )
        )
    return resultado


def comprobar_conteos(frame: pd.DataFrame, poblacion: str) -> dict:
    """objetivo y los rezagos son conteos de salidas: enteros no negativos."""
    malos = {}
    for columna in CONTEOS:
        valores = frame[columna].to_numpy(dtype=float)
        n = int((~np.isfinite(valores) | (valores < 0) | (valores != np.floor(valores))).sum())
        if n:
            malos[columna] = n
    return _comprobacion(
        "conteos_enteros_no_negativos",
        poblacion,
        not malos,
        sum(malos.values()),
        f"0 valores no enteros, negativos o faltantes en {', '.join(CONTEOS)}",
        {"por_columna": malos} if malos else None,
    )


def _matriz_panel(panel: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    """Salidas del panel en una matriz estación × día (NaN donde la estación aún no existía)."""
    estaciones = np.sort(panel["estacion"].unique())
    dia0 = panel["fecha"].min()
    dias = ((panel["fecha"] - dia0) / pd.Timedelta(days=1)).to_numpy().astype(int)
    matriz = np.full((len(estaciones), dias.max() + 1), np.nan)
    matriz[np.searchsorted(estaciones, panel["estacion"].to_numpy()), dias] = panel["salidas"]
    return matriz, estaciones, dia0


def _salidas_previas(
    frame: pd.DataFrame, matriz: np.ndarray, estaciones: np.ndarray, dia0: pd.Timestamp, k: int
) -> np.ndarray:
    """Salidas del panel de la misma estación en D − k para cada fila (NaN si no existen)."""
    estacion = frame["estacion"].to_numpy()
    fila = np.searchsorted(estaciones, estacion).clip(0, len(estaciones) - 1)
    columna = ((frame["fecha"] - dia0) / pd.Timedelta(days=1)).to_numpy().astype(int) - k
    valido = (estaciones[fila] == estacion) & (columna >= 0) & (columna < matriz.shape[1])
    valores = np.full(len(frame), np.nan)
    valores[valido] = matriz[fila[valido], columna[valido]]
    return valores


def comprobar_consistencia_temporal(
    frame: pd.DataFrame, panel: pd.DataFrame, poblacion: str
) -> list[dict]:
    """Recalcula objetivo, rezagos y medias desde el panel y los compara con las features.

    La reconstrucción no reusa ``construir_features``: lee del panel las salidas de la misma
    estación en D − k. objetivo usa k = 0; los rezagos k = 2, 7, 14, 21 y 28; media_7d la media
    de k = 2..8; media_28d la de k = 2..29; y media_misma_dsem_4 la de k = 7, 14, 21 y 28. Las
    features solo usan k ≥ LAG_MINIMO (= 2): si alguna mirara D − 1 o D, no coincidiría.
    """
    matriz, estaciones, dia0 = _matriz_panel(panel)
    previas = {
        k: _salidas_previas(frame, matriz, estaciones, dia0, k)
        for k in range(LAG_MINIMO + VENTANA_LARGA)
    }
    ventana_corta = range(LAG_MINIMO, LAG_MINIMO + VENTANA_CORTA)
    ventana_larga = range(LAG_MINIMO, LAG_MINIMO + VENTANA_LARGA)
    recalculadas = {
        OBJETIVO: (previas[0], "salidas del panel en D"),
        **{f"lag_{k}": (previas[k], f"salidas del panel en D−{k}") for k in REZAGOS},
        "media_7d": (
            np.mean([previas[k] for k in ventana_corta], axis=0),
            f"media de salidas del panel de D−{ventana_corta[-1]} a D−{LAG_MINIMO}",
        ),
        "media_28d": (
            np.mean([previas[k] for k in ventana_larga], axis=0),
            f"media de salidas del panel de D−{ventana_larga[-1]} a D−{LAG_MINIMO}",
        ),
        "media_misma_dsem_4": (
            np.mean([previas[k] for k in LAGS_SEMANALES], axis=0),
            "media de salidas del panel en D−7, D−14, D−21 y D−28",
        ),
    }
    resultado = []
    for columna, (esperado, criterio) in recalculadas.items():
        observado = frame[columna].to_numpy(dtype=float)
        igual = np.isclose(observado, esperado, rtol=0, atol=TOLERANCIA)
        distinto = ~igual
        n = int(distinto.sum())
        detalle = None
        if n:
            ejemplos = frame.loc[distinto, [*LLAVE, columna]].head(5).copy()
            ejemplos["recalculado"] = esperado[distinto][:5]
            detalle = {
                "ejemplos": json.loads(ejemplos.to_json(orient="records", date_format="iso"))
            }
        resultado.append(
            _comprobacion(
                f"{columna}_consistente_con_panel",
                poblacion,
                n == 0,
                n,
                f"0 filas distintas (tolerancia {TOLERANCIA}) de {criterio}",
                detalle,
            )
        )
    return resultado


def comprobar_cobertura(
    actual: pd.DataFrame,
    mes: pd.DataFrame,
    dias_con_viajes: pd.DatetimeIndex,
    inicio: str,
    fin_exclusivo: str,
) -> dict:
    """Todos los días del mes tienen viajes y filas elegibles; reporta las filas excluidas."""
    esperados = pd.date_range(inicio, pd.Timestamp(fin_exclusivo) - pd.Timedelta(days=1))
    esperados = esperados.as_unit("ns")
    con_filas = pd.DatetimeIndex(actual["fecha"].unique()).as_unit("ns")
    sin_filas = esperados.difference(con_filas)
    sin_viajes = esperados.difference(pd.DatetimeIndex(dias_con_viajes).as_unit("ns"))
    por_dia = actual.groupby("fecha").size()
    por_estacion = actual.groupby("estacion").size()
    excluidas = {}
    for bandera in ("sin_historia", "inactiva"):
        filas = mes.loc[mes[bandera]]
        excluidas[bandera] = {
            "filas": len(filas),
            "estaciones": sorted(int(e) for e in filas["estacion"].unique()),
        }
    ok = sin_filas.empty and sin_viajes.empty
    return _comprobacion(
        "cobertura_del_mes",
        "actual",
        ok,
        {
            "dias_esperados": len(esperados),
            "dias_con_filas_elegibles": len(esperados) - len(sin_filas),
            "dias_con_viajes": len(esperados) - len(sin_viajes),
        },
        "todos los días del mes tienen viajes en el archivo actual y filas elegibles",
        {
            "dias_sin_filas_elegibles": [str(d.date()) for d in sin_filas],
            "dias_sin_viajes": [str(d.date()) for d in sin_viajes],
            "filas_por_dia_min": int(por_dia.min()),
            "filas_por_dia_max": int(por_dia.max()),
            "estaciones_con_todos_los_dias": int((por_estacion == len(esperados)).sum()),
            "estaciones_con_dias_parciales": int((por_estacion < len(esperados)).sum()),
            "filas_excluidas": excluidas,
        },
    )


def _festivos_entre(inicio: pd.Timestamp, fin_exclusivo: pd.Timestamp) -> list[str]:
    calendario = festivos(range(inicio.year - 1, fin_exclusivo.year + 1))
    return [str(d.date()) for d in calendario if inicio <= d < fin_exclusivo]


def comprobar_columnas_constantes(frame: pd.DataFrame, inicio: str, fin_exclusivo: str) -> dict:
    """Columnas constantes, casi constantes o duplicadas del mes, y si el calendario las explica.

    Replica las definiciones de Evidently (``nunique() <= 1``; valor más frecuente en al menos
    95 % de las filas; columnas idénticas o iguales en al menos 95 % de las filas con el mismo
    tipo) para decir cuáles son. Solo se consideran esperadas las banderas de calendario:
    ``festivo`` es constante si el mes no tiene festivos y ``festivo_lag7`` si no los hay siete
    días antes; las dos son casi constantes porque hay pocos festivos al año.
    """
    inicio_ts, fin_ts = pd.Timestamp(inicio), pd.Timestamp(fin_exclusivo)
    semana = pd.Timedelta(days=7)
    periodos = {
        "festivo": (inicio_ts, fin_ts),
        "festivo_lag7": (inicio_ts - semana, fin_ts - semana),
    }
    calendario = {bandera: _festivos_entre(*periodo) for bandera, periodo in periodos.items()}
    n = len(frame)
    constantes = [c for c in COLUMNAS if frame[c].nunique() <= 1]
    casi_constantes = [c for c in COLUMNAS if frame[c].value_counts().max() / n >= UMBRAL_CASI]
    duplicadas, casi_duplicadas = [], []
    for a, b in combinations(COLUMNAS, 2):
        if frame[a].dtype.name != frame[b].dtype.name:
            continue
        if frame[a].equals(frame[b]):
            duplicadas.append([a, b])
        if frame[a].eq(frame[b]).sum() / n >= UMBRAL_CASI:
            casi_duplicadas.append([a, b])

    banderas = set(BANDERAS_CALENDARIO)
    no_explicadas = [c for c in constantes if c not in banderas or calendario[c]]
    no_explicadas += [c for c in casi_constantes if c not in banderas]
    no_explicadas += [f"{a}={b}" for a, b in casi_duplicadas if not {a, b} <= banderas]
    explicaciones = {"constantes": "", "casi_constantes": "", "duplicadas": ""}
    for bandera in BANDERAS_CALENDARIO:
        if bandera in constantes and not calendario[bandera]:
            mira = "fecha − 7 días" if bandera == "festivo_lag7" else "la fecha objetivo"
            desde, hasta = periodos[bandera]
            explicaciones["constantes"] += (
                f"{bandera} vale {frame[bandera].iloc[0]} en todas las filas porque el "
                "calendario de mibici.features.festivos() no tiene festivos en lo que mira "
                f"({mira}: {desde.date()} a {(hasta - pd.Timedelta(days=1)).date()}). "
            )
    if any(c in banderas for c in casi_constantes):
        explicaciones["casi_constantes"] = (
            "festivo y festivo_lag7 marcan festivos, que son pocos (nueve por año en "
            "mibici.features.festivos()), así que casi siempre valen 0; una columna constante "
            "cuenta además como casi constante. Es esperado."
        )
    if any({a, b} <= banderas for a, b in casi_duplicadas):
        explicaciones["duplicadas"] = (
            "festivo y festivo_lag7 coinciden en casi todas las filas (en todas si el periodo "
            "no tiene festivos) porque las dos valen 0 fuera de los pocos días festivos."
        )
    explicaciones = {clave: texto.strip() for clave, texto in explicaciones.items()}
    hallazgos = bool(constantes or casi_constantes or casi_duplicadas)
    return {
        "nombre": "columnas_constantes_o_duplicadas_explicables",
        "poblacion": "actual",
        "resultado": "fail" if no_explicadas else ("warning" if hallazgos else "pass"),
        "critica": True,
        "valor_observado": {
            "constantes": constantes,
            "casi_constantes": casi_constantes,
            "duplicadas": duplicadas,
            "casi_duplicadas": casi_duplicadas,
        },
        "criterio": (
            "solo festivo y festivo_lag7 pueden ser constantes, casi constantes o iguales entre "
            "sí, y solo si el calendario lo explica (advertencia); cualquier otra columna falla"
        ),
        "detalle": {
            "festivos_en_periodo": calendario,
            "no_explicadas": no_explicadas,
            "explicaciones": explicaciones,
            "explicacion": " ".join(texto for texto in explicaciones.values() if texto),
        },
    }


def comprobar_referencia(metadatos: dict) -> dict:
    """La referencia no incluye días posteriores al último conocido antes de la evaluación."""
    ref = metadatos["referencia"]
    ok = (
        ref["observado_inicio"] >= ref["inicio_configurado"]
        and ref["observado_fin"] <= ref["fin_configurado"]
    )
    return _comprobacion(
        "referencia_dentro_del_entrenamiento_del_campeon",
        "referencia",
        ok,
        {"observado_inicio": ref["observado_inicio"], "observado_fin": ref["observado_fin"]},
        f"fechas entre {ref['inicio_configurado']} y {ref['fin_configurado']} "
        f"(ultimo_dia_conocido({CONFIG_P1.test_inicio}))",
    )


def comprobar_panel_campeon(panel: pd.DataFrame, panel_campeon: pd.DataFrame) -> dict:
    """El panel reconstruido coincide con el panel del campeón en los días que comparten."""
    fin = panel_campeon["fecha"].max()
    columnas = ["estacion", "fecha", "salidas"]
    propio = panel.loc[panel["fecha"] <= fin, columnas]
    unidos = propio.merge(
        panel_campeon[columnas], on=list(LLAVE), how="outer", suffixes=("", "_campeon")
    )
    solo_propio = int(unidos["salidas_campeon"].isna().sum())
    solo_campeon = int(unidos["salidas"].isna().sum())
    ambos = unidos["salidas"].notna() & unidos["salidas_campeon"].notna()
    distintas = int((ambos & (unidos["salidas"] != unidos["salidas_campeon"])).sum())
    n = solo_propio + solo_campeon + distintas
    return _comprobacion(
        "panel_igual_al_del_campeon",
        "referencia",
        n == 0,
        n,
        f"mismas filas (estacion, fecha) y salidas que el panel del campeón hasta {fin.date()}",
        {
            "filas_comparadas": len(unidos),
            "solo_en_panel_reconstruido": solo_propio,
            "solo_en_panel_del_campeon": solo_campeon,
            "salidas_distintas": distintas,
        },
    )


# --- Integridad del archivo de viajes del mes ---


def cargar_viajes(ruta: Path) -> pd.DataFrame:
    """Lee del archivo de viajes solo ``COLUMNAS_VIAJE``, sin datos de usuario.

    Fechas y estaciones se interpretan como en ``cargar_salidas`` (ISO 8601 fila por fila y
    números); lo que no se puede leer queda como faltante. ``Viaje_Id`` se deja como viene para
    no volver iguales dos identificadores distintos al convertirlos.
    """
    if ruta.suffix == ".parquet":
        viajes = pd.read_parquet(ruta, columns=COLUMNAS_VIAJE)
    else:
        viajes = pd.read_csv(ruta, usecols=COLUMNAS_VIAJE)
    for columna in ("Inicio_del_viaje", "Fin_del_viaje"):
        viajes[columna] = pd.to_datetime(viajes[columna], format="ISO8601", errors="coerce")
    for columna in ("Origen_Id", "Destino_Id"):
        viajes[columna] = pd.to_numeric(viajes[columna], errors="coerce")
    return viajes[COLUMNAS_VIAJE]


def _registros(frame: pd.DataFrame, n: int = 6) -> list[dict]:
    return json.loads(frame.head(n).to_json(orient="records", date_format="iso"))


def comprobar_viaje_id(viajes: pd.DataFrame) -> dict:
    """Cada viaje del archivo tiene ``Viaje_Id`` y ninguno se repite."""
    ids = viajes["Viaje_Id"]
    faltantes = int(ids.isna().sum())
    repetidos = int(ids.dropna().duplicated().sum())
    detalle = {"viajes": len(viajes), "sin_viaje_id": faltantes, "viaje_id_repetidos": repetidos}
    if repetidos:
        mascara = ids.notna() & ids.duplicated(keep=False)
        detalle["ejemplos"] = _registros(viajes.loc[mascara].sort_values("Viaje_Id"))
    return _comprobacion(
        "viaje_id_unico",
        "actual",
        faltantes == 0 and repetidos == 0,
        faltantes + repetidos,
        "0 viajes sin Viaje_Id o con un Viaje_Id repetido en el archivo del mes",
        detalle,
    )


def comprobar_campos_legibles(viajes: pd.DataFrame) -> dict:
    """Viajes sin fecha de inicio o estación de origen legibles, que no entran al panel.

    Hasta ``MAX_FRACCION_FECHAS_ILEGIBLES`` de los viajes es advertencia y más es falla: el
    mismo límite con que ``cargar_salidas`` detiene la carga por fechas ilegibles.
    """
    sin_inicio = viajes["Inicio_del_viaje"].isna()
    sin_origen = viajes["Origen_Id"].isna()
    malos = sin_inicio | sin_origen
    n = int(malos.sum())
    proporcion = n / len(viajes) if len(viajes) else 0.0
    detalle = {
        "sin_inicio_legible": int(sin_inicio.sum()),
        "sin_origen_legible": int(sin_origen.sum()),
        "proporcion": proporcion,
    }
    if n:
        detalle["ejemplos"] = _registros(viajes.loc[malos])
        detalle["explicacion"] = (
            f"{n} viajes ({proporcion:.3%} del archivo) no tienen fecha de inicio u Origen_Id "
            "legible: no entran al panel y no cuentan como salidas."
        )
    return _comprobacion(
        "viajes_con_inicio_y_origen_legibles",
        "actual",
        n == 0,
        n,
        "0 viajes sin fecha de inicio u Origen_Id legible (advertencia hasta "
        f"{MAX_FRACCION_FECHAS_ILEGIBLES:.0%} de los viajes; falla por encima)",
        detalle,
        critica=proporcion > MAX_FRACCION_FECHAS_ILEGIBLES,
    )


def comprobar_viajes_repetidos(viajes: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """Viajes que repiten inicio, fin, origen y destino de otro con distinto ``Viaje_Id``.

    Así se ve una recarga de un día o una duplicación del ETL con identificadores nuevos. Sin
    datos de usuario no se distingue un registro repetido de dos viajes que coinciden al
    segundo, así que unos cuantos son advertencia; por encima de ``UMBRAL_VIAJES_REPETIDOS`` de
    los viajes del archivo la prueba falla. El detalle da el conteo, los días y cuántas salidas
    de más sumaría en el panel la estación-día más afectada si fueran repetidos.
    """
    involucradas = viajes.duplicated(VIAJE_SIN_ID, keep=False)
    de_mas = viajes.duplicated(VIAJE_SIN_ID)
    n = int(de_mas.sum())
    detalle: dict = {"viajes": len(viajes), "filas_involucradas": int(involucradas.sum())}
    if n:
        repetidos = viajes.loc[involucradas]
        duracion = (repetidos["Fin_del_viaje"] - repetidos["Inicio_del_viaje"]).dt.total_seconds()
        cortos = (duracion <= DURACION_CORTA_S) & (
            repetidos["Origen_Id"] == repetidos["Destino_Id"]
        )
        sobrantes = viajes.loc[de_mas].dropna(subset=["Inicio_del_viaje", "Origen_Id"])
        por_estacion_dia = (
            pd.DataFrame(
                {
                    "estacion": sobrantes["Origen_Id"].astype("int64"),
                    "fecha": sobrantes["Inicio_del_viaje"].dt.normalize(),
                }
            )
            .value_counts()
            .rename("salidas_de_mas")
            .reset_index()
            .merge(
                panel[["estacion", "fecha", "salidas"]].astype({"estacion": "int64"}),
                on=list(LLAVE),
                how="left",
            )
        )
        grupos = detalle["filas_involucradas"] - n
        detalle |= {
            "grupos": grupos,
            "filas_de_mas": n,
            "proporcion": n / len(viajes),
            "involucradas_cortas_con_origen_igual_a_destino": int(cortos.sum()),
            "filas_de_mas_por_dia": {
                str(d.date()): int(c)
                for d, c in viajes.loc[de_mas, "Inicio_del_viaje"]
                .dt.normalize()
                .value_counts()
                .sort_index()
                .items()
            },
            "ejemplos": _registros(repetidos.sort_values(VIAJE_SIN_ID)),
        }
        if not por_estacion_dia.empty:
            peor = por_estacion_dia.iloc[0]  # value_counts ordena de mayor a menor
            detalle["estacion_dia_mas_afectada"] = {
                "estacion": int(peor["estacion"]),
                "fecha": str(peor["fecha"].date()),
                "salidas_de_mas": int(peor["salidas_de_mas"]),
                "salidas_en_panel": int(peor["salidas"]) if pd.notna(peor["salidas"]) else None,
            }
            efecto = (
                f" El panel los cuenta como salidas: si fueran repetidos, la estación-día más "
                f"afectada (estación {int(peor['estacion'])}, {peor['fecha'].date()}) tendría "
                f"{int(peor['salidas_de_mas'])} salidas de más de sus "
                f"{detalle['estacion_dia_mas_afectada']['salidas_en_panel']}."
            )
        else:
            efecto = ""
        detalle["explicacion"] = (
            f"{n} viajes ({n / len(viajes):.3%} de {len(viajes)}) repiten inicio, fin, origen y "
            f"destino de otro con distinto Viaje_Id, en {grupos} grupos; "
            f"{int(cortos.sum())} de las {detalle['filas_involucradas']} filas involucradas "
            f"duran {DURACION_CORTA_S} s o menos con origen igual a destino. Sin datos de usuario "
            "no se distingue un registro repetido de dos viajes que coinciden al segundo."
            f"{efecto}"
        )
    return _comprobacion(
        "viajes_repetidos_con_otro_viaje_id",
        "actual",
        n == 0,
        n,
        "0 viajes que repitan inicio, fin, origen y destino de otro con distinto Viaje_Id "
        f"(advertencia hasta {UMBRAL_VIAJES_REPETIDOS:.1%} de los viajes; falla por encima)",
        detalle,
        critica=bool(len(viajes)) and n / len(viajes) > UMBRAL_VIAJES_REPETIDOS,
    )


def comprobar_conciliacion(
    viajes: pd.DataFrame, panel: pd.DataFrame, inicio: str, fin_exclusivo: str
) -> dict:
    """Las salidas del panel en el mes suman exactamente los viajes válidos del archivo.

    Válido es un viaje con fecha de inicio y estación de origen legibles dentro del mes, que es
    lo que cuenta ``construir_panel_diario``. Liga la tabla estación-día con el archivo leído
    aparte: si el panel perdiera o sumara salidas al agregar, no coincidiría.
    """
    inicio_ts, fin_ts = pd.Timestamp(inicio), pd.Timestamp(fin_exclusivo)
    fecha = viajes["Inicio_del_viaje"]
    validos = fecha.notna() & viajes["Origen_Id"].notna()
    en_mes = validos & (fecha >= inicio_ts) & (fecha < fin_ts)
    del_mes = (panel["fecha"] >= inicio_ts) & (panel["fecha"] < fin_ts)
    salidas = int(panel.loc[del_mes, "salidas"].sum())
    n_validos = int(en_mes.sum())
    return _comprobacion(
        "salidas_del_panel_igual_a_viajes_validos",
        "actual",
        salidas == n_validos,
        salidas - n_validos,
        "salidas del panel en el mes − viajes válidos del mes en el archivo = 0",
        {
            "viajes_en_archivo": len(viajes),
            "viajes_validos_del_mes": n_validos,
            "viajes_validos_fuera_del_mes": int((validos & ~en_mes).sum()),
            "salidas_del_panel_en_el_mes": salidas,
        },
    )


def comprobar_volumen_diario(panel: pd.DataFrame, inicio: str, fin_exclusivo: str) -> dict:
    """Viajes válidos de cada día frente a la mediana del mismo día de la semana en el mes.

    Una razón fuera de ``RAZON_VOLUMEN`` advierte de un día cargado de más (cerca de 2) o
    incompleto (cerca de 0.5 o menos). Se compara dentro del mes porque el nivel de demanda
    cambia de un mes a otro: el detalle da los percentiles diarios de la referencia del campeón
    y cuántos días del mes los rebasan, pero eso mide el cambio de nivel que sigue el reporte de
    drift y no distingue una recarga, así que no es el criterio.
    """
    inicio_ts, fin_ts = pd.Timestamp(inicio), pd.Timestamp(fin_exclusivo)
    por_dia = panel.groupby("fecha")["salidas"].sum()
    mes = por_dia.loc[(por_dia.index >= inicio_ts) & (por_dia.index < fin_ts)]
    mediana = mes.groupby(mes.index.dayofweek).transform("median")
    razon = mes / mediana.where(mediana > 0)
    bajo, alto = RAZON_VOLUMEN
    fuera = (razon <= bajo) | (razon >= alto)
    ref_inicio = pd.Timestamp(CONFIG_P1.train_inicio)
    ref_fin = ultimo_dia_conocido(pd.Timestamp(CONFIG_P1.test_inicio))
    ref = por_dia.loc[(por_dia.index >= ref_inicio) & (por_dia.index <= ref_fin)]
    p1, p50, p99 = (float(v) for v in ref.quantile([0.01, 0.5, 0.99]))
    dias = [
        {
            "fecha": str(d.date()),
            "viajes": int(mes[d]),
            "mediana_mismo_dia_semana": float(mediana[d]),
            "razon": round(float(razon[d]), 4),
        }
        for d in mes.index[fuera.to_numpy()]
    ]
    extremos = {}
    if razon.notna().any():
        for nombre, dia in (("razon_min", razon.idxmin()), ("razon_max", razon.idxmax())):
            extremos[nombre] = {"fecha": str(dia.date()), "razon": round(float(razon[dia]), 4)}
    detalle = {
        "dias_fuera": dias,
        **extremos,
        "mes": {
            "dias": len(mes),
            "minimo": int(mes.min()),
            "mediana": float(mes.median()),
            "maximo": int(mes.max()),
        },
        "referencia_del_campeon": {
            "dias": len(ref),
            "p1": p1,
            "mediana": p50,
            "p99": p99,
            "dias_del_mes_sobre_p99": int((mes > p99).sum()),
            "dias_del_mes_bajo_p1": int((mes < p1).sum()),
        },
    }
    if dias:
        lista = ", ".join(f"{d['fecha']} ({d['viajes']} viajes, {d['razon']:.2f})" for d in dias)
        detalle["explicacion"] = (
            f"{len(dias)} días con viajes válidos fuera de ({bajo:g}, {alto:g}) veces la mediana "
            f"de su día de la semana en el mes: {lista}. Un día cargado dos veces ronda 2 y uno "
            "incompleto queda cerca de 0.5 o menos; revisar esos días junto con viaje_id_unico y "
            "viajes_repetidos_con_otro_viaje_id."
        )
    return _comprobacion(
        "volumen_diario_coherente_en_el_mes",
        "actual",
        not dias,
        len(dias),
        f"0 días con viajes válidos fuera de ({bajo:g}, {alto:g}) veces la mediana del mismo "
        "día de la semana en el mes (advertencia)",
        detalle,
        critica=False,
    )


def comprobar_integridad(
    pob: Poblaciones,
    inicio: str,
    fin_exclusivo: str,
    panel_campeon: pd.DataFrame | None = None,
    viajes: pd.DataFrame | None = None,
) -> list[dict]:
    """Todas las comprobaciones de integridad de la referencia y del mes actual.

    ``viajes`` es el archivo del mes leído con ``cargar_viajes``; sin él se omiten las
    comprobaciones a nivel viaje (``Viaje_Id``, repetidos, campos legibles y conciliación).
    """
    resultado = [comprobar_referencia(pob.metadatos)]
    if panel_campeon is not None:
        resultado.append(comprobar_panel_campeon(pob.panel, panel_campeon))
    if viajes is not None:
        resultado += [
            comprobar_viaje_id(viajes),
            comprobar_campos_legibles(viajes),
            comprobar_viajes_repetidos(viajes, pob.panel),
            comprobar_conciliacion(viajes, pob.panel, inicio, fin_exclusivo),
        ]
    resultado.append(comprobar_volumen_diario(pob.panel, inicio, fin_exclusivo))
    for nombre, frame in (("referencia", pob.referencia), ("actual", pob.actual)):
        resultado.append(comprobar_llave(frame, nombre))
        resultado += comprobar_calendario(frame, nombre)
        resultado.append(comprobar_conteos(frame, nombre))
        resultado += comprobar_consistencia_temporal(frame, pob.panel, nombre)
    resultado.append(
        comprobar_cobertura(pob.actual, pob.mes, pob.dias_con_viajes, inicio, fin_exclusivo)
    )
    resultado.append(comprobar_columnas_constantes(pob.actual, inicio, fin_exclusivo))
    return resultado


# --- Reporte de Evidently ---


def rangos_referencia(referencia: pd.DataFrame) -> dict[str, list[float]]:
    """Mínimo y máximo de objetivo y de cada variable numérica en la referencia."""
    return {
        c: [float(referencia[c].min()), float(referencia[c].max())] for c in (OBJETIVO, *NUMERICAS)
    }


def detallar_fuera_de_rango(
    actual: pd.DataFrame, rangos: dict[str, list[float]]
) -> dict[str, dict]:
    """Filas del mes fuera del rango de la referencia, por variable, con estaciones y días.

    Usa el criterio de ``OutRangeValueCount`` de Evidently (fuera de [mínimo, máximo], con los
    extremos dentro) para que los conteos coincidan; solo lista las variables con alguna fila.
    """
    detalle = {}
    for columna, (bajo, alto) in rangos.items():
        valores = actual[columna]
        fuera = valores.notna() & ~valores.between(bajo, alto)
        if not fuera.any():
            continue
        filas = actual.loc[fuera, ["estacion", "fecha", columna]]
        por_estacion = filas.groupby("estacion").size().sort_values(ascending=False, kind="stable")
        maximo = float(valores.max())
        detalle[columna] = {
            "filas": int(fuera.sum()),
            "proporcion": int(fuera.sum()) / int(valores.notna().sum()),
            "rango_referencia": [bajo, alto],
            "minimo_actual": float(valores.min()),
            "maximo_actual": maximo,
            "maximo_sobre_maximo_referencia": maximo / alto if alto > 0 else None,
            "filas_por_estacion": {str(int(e)): int(n) for e, n in por_estacion.items()},
            "dias": sorted(str(pd.Timestamp(d).date()) for d in filas["fecha"].unique()),
            "ejemplos": _registros(filas.sort_values(columna, ascending=False), 5),
        }
    return detalle


def _para_evidently(frame: pd.DataFrame) -> pd.DataFrame:
    """Copia con las categóricas como texto, igual que en el reporte de drift."""
    frame = frame[COLUMNAS].copy()
    for columna in CATEGORICAS:
        frame[columna] = frame[columna].map(lambda v: str(v) if pd.notna(v) else None)
    return frame


def armar_reporte(rangos: dict[str, list[float]]) -> Report:
    """Report de Evidently con pruebas explícitas; ninguna prueba se genera por omisión."""
    # Con include_tests=False en cada preset solo quedan las pruebas que se piden; las demás
    # listas van vacías para que el preset y su vista usen las mismas métricas
    sin_pruebas = {
        nombre: []
        for nombre in (
            "row_count_tests",
            "mean_tests",
            "std_tests",
            "q25_tests",
            "q50_tests",
            "q75_tests",
        )
    }
    metricas = [
        DatasetStats(
            row_count_tests=[gte(1)],
            column_count_tests=[eq(len(COLUMNAS))],
            duplicated_row_count_tests=[eq(0)],
            duplicated_column_count_tests=[eq(0, is_critical=False)],
            almost_duplicated_column_count_tests=[eq(0, is_critical=False)],
            almost_constant_column_count_tests=[
                lte(len(BANDERAS_CALENDARIO)),
                eq(0, is_critical=False),
            ],
            empty_row_count_tests=[eq(0)],
            empty_column_count_tests=[eq(0)],
            constant_columns_count_tests=[
                lte(len(BANDERAS_CALENDARIO)),
                eq(0, is_critical=False),
            ],
            dataset_missing_value_count_tests=[eq(0)],
            include_tests=False,
        ),
        MissingValueCount(column="estacion", tests=[eq(0)]),
    ]
    # objetivo y los rezagos son conteos de una estación-día: además del mínimo llevan un tope
    # crítico, FACTOR_MAXIMO veces el máximo de la referencia
    metricas += [
        ValueStats(
            columna,
            missing_values_count_tests=[eq(0)],
            min_tests=[gte(0)],
            max_tests=[lte(FACTOR_MAXIMO * rangos[columna][1])] if columna in CONTEOS else [],
            include_tests=False,
            **sin_pruebas,
        )
        for columna in (OBJETIVO, *NUMERICAS)
    ]
    metricas += [
        ValueStats(
            columna, missing_values_count_tests=[eq(0)], row_count_tests=[], include_tests=False
        )
        for columna in ("fecha", *CATEGORICAS)
    ]
    metricas += [
        OutListValueCount(column=columna, values=DOMINIOS[columna], tests=[eq(0)])
        for columna in CATEGORICAS
    ]
    # Fuera del rango de la referencia: un solo valor ya advierte (conteo) y más de
    # UMBRAL_FUERA_DE_RANGO de las filas falla (proporción)
    metricas += [
        OutRangeValueCount(
            column=columna,
            left=bajo,
            right=alto,
            tests=[eq(0, is_critical=False)],
            share_tests=[lte(UMBRAL_FUERA_DE_RANGO)],
        )
        for columna, (bajo, alto) in rangos.items()
    ]
    return Report(metricas, include_tests=True)


def generar_reporte(
    referencia: pd.DataFrame, actual: pd.DataFrame, html_path: Path, json_path: Path
) -> dict:
    """Corre el reporte de calidad de Evidently, escribe HTML y JSON y regresa el JSON leído."""
    definicion = DataDefinition(
        id_column="estacion",
        timestamp="fecha",
        numerical_columns=[OBJETIVO, *NUMERICAS],
        categorical_columns=list(CATEGORICAS),
    )
    ref = Dataset.from_pandas(_para_evidently(referencia), data_definition=definicion)
    act = Dataset.from_pandas(_para_evidently(actual), data_definition=definicion)
    snapshot = armar_reporte(rangos_referencia(referencia)).run(
        current_data=act, reference_data=ref
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))
    contenido = snapshot.json()
    json_path.write_text(contenido, encoding="utf-8")
    return json.loads(contenido)


def pruebas_evidently(resultado: dict) -> list[dict]:
    """Una entrada por prueba de Evidently: resultado, valor observado y criterio."""
    valores = {m["id"]: m["value"] for m in resultado["metrics"]}
    pruebas = []
    for prueba in resultado["tests"]:
        config = prueba["test_config"]
        metrica = prueba["metric_config"]["params"]
        es_conteo = prueba.get("bound_test", {}).get("is_count")
        valor = valores.get(prueba["metric_config"]["metric_id"])
        if isinstance(valor, dict):
            valor = valor.get("count" if es_conteo is not False else "share")
        tipo = config["type"].split(":")[-1]
        umbral = config.get("expected", config.get("threshold"))
        campo = {True: "conteo", False: "proporción"}.get(es_conteo, "valor")
        critica = bool(config.get("is_critical", True))
        pruebas.append(
            {
                "nombre": prueba["name"],
                "metrica": metrica["type"].split(":")[-1],
                "columna": metrica.get("column"),
                "resultado": ESTADOS.get(prueba["status"], prueba["status"].lower()),
                "valor_observado": valor,
                "criterio": f"{campo} {OPERADORES.get(tipo, tipo)} {umbral}",
                "critica": critica,
                "descripcion": prueba["description"],
            }
        )
    return pruebas


def _cuantas(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"


def _explicar_fuera_de_rango(columna: str, rango: list[float], info: dict | None) -> str:
    """Cuántas filas, en qué estaciones y días, y qué tan lejos del máximo de la referencia."""
    bajo, alto = rango
    if not info:
        texto = f"Hay valores de {columna} fuera del rango de la referencia [{bajo:g}, {alto:g}]."
    else:
        estaciones = list(info["filas_por_estacion"].items())
        lista = ", ".join(f"{e} ({n})" for e, n in estaciones[:8])
        if len(estaciones) > 8:
            lista += f" y {len(estaciones) - 8} más"
        dias = info["dias"]
        cuando = (
            f"el {dias[0]}" if len(dias) == 1 else f"{len(dias)} días, de {dias[0]} a {dias[-1]}"
        )
        texto = (
            f"{_cuantas(info['filas'], 'fila', 'filas')} ({info['proporcion']:.2%}) de {columna} "
            f"fuera del rango de la referencia [{bajo:g}, {alto:g}], en "
            f"{_cuantas(len(estaciones), 'estación', 'estaciones')} (filas por estación: "
            f"{lista}), {cuando}. Máximo del mes: {info['maximo_actual']:g}"
        )
        razon = info["maximo_sobre_maximo_referencia"]
        texto += f", {razon:.2f} veces el de la referencia." if razon is not None else "."
        if columna in CONTEOS and info["maximo_actual"] <= FACTOR_MAXIMO * alto:
            texto += f" Queda por debajo del tope crítico (MaxValue) de {FACTOR_MAXIMO} veces ese máximo."
    return (
        f"{texto} El campeón no vio esos valores al entrenar y extrapola en ellos; su "
        "distribución se mide en el reporte de drift."
    )


def explicar_advertencias(
    pruebas: list[dict],
    integridad: list[dict],
    rangos: dict[str, list[float]],
    fuera_de_rango: dict[str, dict] | None = None,
) -> list[dict]:
    """Explica cada advertencia con lo que encontraron las comprobaciones de integridad."""
    constantes = next(
        (c for c in integridad if c["nombre"] == "columnas_constantes_o_duplicadas_explicables"),
        {"resultado": "pass", "valor_observado": {}, "detalle": {}},
    )
    hallazgos = {
        clave: constantes["valor_observado"].get(clave, [])
        for clave in ("constantes", "casi_constantes", "duplicadas", "casi_duplicadas")
    }
    explicaciones = constantes["detalle"].get("explicaciones", {})
    por_metrica = {
        "ConstantColumnsCount": (
            f"Columnas constantes: {hallazgos['constantes']}.",
            explicaciones.get("constantes", ""),
        ),
        "AlmostConstantColumnsCount": (
            f"Columnas con un valor en ≥ 95 % de las filas: {hallazgos['casi_constantes']}.",
            explicaciones.get("casi_constantes", ""),
        ),
        "DuplicatedColumnsCount": (
            f"Columnas idénticas: {hallazgos['duplicadas']}.",
            explicaciones.get("duplicadas", ""),
        ),
        "AlmostDuplicatedColumnsCount": (
            f"Columnas iguales en ≥ 95 % de las filas: {hallazgos['casi_duplicadas']}.",
            explicaciones.get("duplicadas", ""),
        ),
    }
    advertencias = []
    for prueba in pruebas:
        if prueba["resultado"] != "warning":
            continue
        if prueba["metrica"] in por_metrica:
            hallazgo, porque = por_metrica[prueba["metrica"]]
            texto = f"{hallazgo} {porque or 'Sin explicación automática; revisar a mano.'}"
            if constantes["resultado"] == "fail":
                texto += f" No explicadas: {constantes['detalle']['no_explicadas']}."
        elif prueba["metrica"] == "OutRangeValueCount":
            columna = prueba["columna"]
            texto = _explicar_fuera_de_rango(
                columna, rangos[columna], (fuera_de_rango or {}).get(columna)
            )
        else:
            texto = "Sin explicación automática; revisar a mano."
        advertencias.append(
            {"origen": "evidently", "prueba": prueba["nombre"], "explicacion": texto.strip()}
        )
    for comprobacion in integridad:
        if comprobacion["resultado"] == "warning":
            advertencias.append(
                {
                    "origen": "integridad",
                    "prueba": comprobacion["nombre"],
                    "explicacion": comprobacion["detalle"].get("explicacion", ""),
                }
            )
    return advertencias


def totales(entradas: list[dict]) -> dict[str, int]:
    conteo = {"pass": 0, "fail": 0, "warning": 0, "error": 0}
    for entrada in entradas:
        conteo[entrada["resultado"]] = conteo.get(entrada["resultado"], 0) + 1
    return conteo | {"total": len(entradas)}


def resultado_global(*conteos: dict[str, int]) -> str:
    if any(c["fail"] or c["error"] for c in conteos):
        return "fail"
    return "warning" if any(c["warning"] for c in conteos) else "pass"


def run(
    historical_trips: Path,
    current_trips: Path,
    output_dir: Path,
    current_start: str = "2026-10-01",
    current_end_exclusive: str = "2026-11-01",
    panel_campeon: Path | None = None,
) -> dict:
    """Arma las poblaciones, corre Evidently y la integridad y escribe las tres salidas."""
    from mibici.tracking import git_commit

    month_bounds(current_start, current_end_exclusive)
    for ruta in (historical_trips, current_trips, panel_campeon):
        if ruta is not None and not ruta.is_file():
            raise FileNotFoundError(f"No existe el archivo de entrada: {ruta}")

    pob = construir_poblaciones(
        cargar_salidas(historical_trips),
        cargar_salidas(current_trips),
        current_start,
        current_end_exclusive,
    )
    panel_ref = pd.read_parquet(panel_campeon) if panel_campeon is not None else None
    integridad = comprobar_integridad(
        pob, current_start, current_end_exclusive, panel_ref, cargar_viajes(current_trips)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "p1_data_quality.html"
    json_path = output_dir / "p1_data_quality.json"
    resumen_path = output_dir / "p1_calidad_resumen.json"
    rangos = rangos_referencia(pob.referencia)
    fuera_de_rango = detallar_fuera_de_rango(pob.actual, rangos)
    pruebas = pruebas_evidently(generar_reporte(pob.referencia, pob.actual, html_path, json_path))
    total_evidently, total_integridad = totales(pruebas), totales(integridad)
    entradas = {
        "historical_trips": historical_trips,
        "current_trips": current_trips,
    } | ({"panel_campeon": panel_campeon} if panel_campeon is not None else {})
    commit, con_cambios = git_commit()
    modulo = Path(__file__)
    resumen = {
        "problema": "P1",
        "reporte": "calidad de datos del mes actual (Evidently + integridad con pandas)",
        "generado": datetime.now().astimezone().isoformat(timespec="seconds"),
        "resultado_global": resultado_global(total_evidently, total_integridad),
        "totales": {
            "evidently": total_evidently,
            "integridad": total_integridad,
            "todas": {k: total_evidently[k] + total_integridad[k] for k in total_evidently},
        },
        "advertencias_explicadas": explicar_advertencias(
            pruebas, integridad, rangos, fuera_de_rango
        ),
        "pruebas_evidently": pruebas,
        "integridad": integridad,
        "poblaciones": pob.metadatos,
        "rangos_referencia": rangos,
        "fuera_de_rango": fuera_de_rango,
        "entradas": {
            nombre: {"ruta": str(ruta), "sha256": file_sha256(ruta)}
            for nombre, ruta in entradas.items()
        },
        "versiones": {
            "python": platform.python_version(),
            **{p: version(p) for p in ("evidently", "pandas", "numpy")},
        },
        "codigo": {
            "modulo": "mibici/monitoring/p1_calidad.py",
            "sha256": file_sha256(modulo),
            "git_commit": commit,
            "arbol_con_cambios": con_cambios,
        },
        "salidas": {
            "evidently_html": html_path.name,
            "evidently_json": json_path.name,
            "resumen": resumen_path.name,
        },
        "notas": [
            (
                "Evidently prueba solo el periodo actual; la referencia se muestra al lado y da "
                "los rangos de las pruebas de valores fuera de rango."
            ),
            "Las filas repiten estaciones y días: no son observaciones independientes.",
            (
                "llave_estacion_fecha_unica y DuplicatedRowCount son controles estructurales: "
                "la tabla estación-día es única por construcción. Los viajes repetidos o "
                "recargados se revisan en el archivo del mes (viaje_id_unico, "
                "viajes_repetidos_con_otro_viaje_id, salidas_del_panel_igual_a_viajes_validos y "
                "volumen_diario_coherente_en_el_mes) y los conteos imposibles con el tope "
                "MaxValue de objetivo y los rezagos."
            ),
            (
                "objetivo del mes actual se conoce después de cerrar cada día; se revisa porque "
                "el A/B lo usa como etiqueta, no porque sea una entrada del modelo."
            ),
        ],
    }
    resumen_path.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumen


def registrar_resumen(resumen: dict) -> None:
    """Escribe en el log los totales, las fallas y las advertencias del resumen."""
    for origen in ("evidently", "integridad"):
        t = resumen["totales"][origen]
        logger.info(
            f"{origen}: {t['total']} pruebas | pass {t['pass']} | fail {t['fail']} | "
            f"warning {t['warning']} | error {t['error']}"
        )
    entradas = [("evidently", p) for p in resumen["pruebas_evidently"]] + [
        ("integridad", c) for c in resumen["integridad"]
    ]
    for origen, entrada in entradas:
        if entrada["resultado"] in ("fail", "error"):
            logger.error(
                f"FALLA [{origen}] {entrada['nombre']} ({entrada.get('poblacion', 'actual')}): "
                f"observado {entrada['valor_observado']} | criterio {entrada['criterio']}"
            )
    for advertencia in resumen["advertencias_explicadas"]:
        logger.warning(
            f"ADVERTENCIA [{advertencia['origen']}] {advertencia['prueba']}: "
            f"{advertencia['explicacion']}"
        )


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
        Path | None, typer.Option(help="Carpeta de salida (por omisión reports/monitoring/...).")
    ] = None,
    panel_campeon: Annotated[
        Path | None, typer.Option(help="Panel con que se entrenó el campeón, para compararlo.")
    ] = None,
    estricto: Annotated[
        bool, typer.Option(help="Terminar con código 1 si alguna prueba crítica falla.")
    ] = False,
):
    """Genera el reporte de calidad de datos de P1 del mes actual."""
    year, month = month_bounds(current_start, current_end_exclusive)
    output = output or CONFIG.reports_dir / "monitoring" / f"{year}_{month:02d}" / "p1"
    current_trips = current_trips or (
        CONFIG.raw_data_dir / f"datos_abiertos_{year}_{month:02d}.csv"
    )
    resumen = run(
        historical_trips,
        current_trips,
        output,
        current_start,
        current_end_exclusive,
        panel_campeon,
    )
    pob = resumen["poblaciones"]
    logger.info(
        f"Referencia: {pob['referencia']['filas']} filas de {pob['referencia']['estaciones']} "
        f"estaciones | actual: {pob['actual']['filas']} filas de "
        f"{pob['actual']['estaciones']} estaciones"
    )
    registrar_resumen(resumen)
    logger.success(f"Calidad de datos de P1: {resumen['resultado_global']} | {output}")
    if estricto and resumen["resultado_global"] == "fail":
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
