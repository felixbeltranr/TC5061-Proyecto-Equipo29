"""Features de P1: demanda diaria de salidas por estacion (regresion).

Pregunta P1: cuantas bicicletas saldran de cada estacion el dia D, pronosticado a las
23:00 del dia D-1. La unidad es la estacion-dia y el objetivo es el conteo de salidas
(viajes cuyo origen es la estacion) segun la fecha local de Inicio_del_viaje.

Regla contra la fuga de informacion: a las 23:00 de D-1 el dia D-1 todavia no cierra, asi
que el ultimo conteo completo es el de D-2. Ninguna variable usa datos de D-1 ni de D en
adelante; solo el calendario del dia objetivo, que se conoce de antemano.

Se construye sobre el dataset combinado que genera mibici.dataset (CONFIG.output_filename,
parquet o CSV) y solo lee la fecha de inicio y la estacion de origen; no se cargan datos de
usuario.
"""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici.config import CONFIG

app = typer.Typer()

# --- Que se sabe cuando se emite el pronostico ---
HORA_EMISION = 23  # el pronostico del dia D se emite a las 23:00 del dia D-1
RETRASO_CONTEO_MIN = 60  # el conteo de un dia queda disponible 60 min despues de cerrar

LAGS_SEMANALES = (7, 14, 21, 28)
VENTANA_CORTA = 7
VENTANA_LARGA = 28
COLUMNAS_VIAJES = ["Inicio_del_viaje", "Origen_Id"]
MAX_FRACCION_FECHAS_ILEGIBLES = (
    0.01  # mas que esto al cargar indica un formato distinto al esperado
)

FEATURES = [
    "dia_semana",
    "festivo",
    "festivo_lag7",
    "lag_2",
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
    "media_7d",
    "media_28d",
    "media_misma_dsem_4",
]


def emision(objetivo: pd.Timestamp | str) -> pd.Timestamp:
    """Momento en que se emite el pronostico del dia objetivo: 23:00 del dia anterior."""
    dia = pd.Timestamp(objetivo).normalize()
    return dia - pd.Timedelta(days=1) + pd.Timedelta(hours=HORA_EMISION)


def disponible_desde(dia: pd.Timestamp | str) -> pd.Timestamp:
    """Momento desde el que se conoce el conteo completo de un dia (cierre mas retraso)."""
    cierre = pd.Timestamp(dia).normalize() + pd.Timedelta(days=1)
    return cierre + pd.Timedelta(minutes=RETRASO_CONTEO_MIN)


def _lag_minimo() -> int:
    """Menor numero de dias hacia atras cuyo conteo ya existe en el momento de emitir."""
    referencia = pd.Timestamp("2000-01-10")
    lag = 1
    while disponible_desde(referencia - pd.Timedelta(days=lag)) > emision(referencia):
        lag += 1
    return lag


# Con emision a las 23:00 de D-1 y retraso de 60 min, D-1 aun no esta completo: el lag es 2
LAG_MINIMO = _lag_minimo()


def ultimo_dia_conocido(objetivo: pd.Timestamp | str) -> pd.Timestamp:
    """Ultimo dia cuyo conteo ya existe cuando se emite el pronostico del dia objetivo.

    Tambien es el ultimo dia que puede entrenar un modelo que se usa desde esa fecha.
    """
    return pd.Timestamp(objetivo).normalize() - pd.Timedelta(days=LAG_MINIMO)


def pascua(anio: int) -> date:
    """Domingo de Pascua del anio dado (algoritmo gregoriano anonimo)."""
    a = anio % 19
    b, c = divmod(anio, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    mes, dia = divmod(h + ll - 7 * m + 114, 31)
    return date(anio, mes, dia + 1)


def _lunes_n(anio: int, mes: int, n: int) -> pd.Timestamp:
    """El n-esimo lunes del mes."""
    primero = pd.Timestamp(anio, mes, 1)
    return primero + pd.Timedelta(days=(7 - primero.dayofweek) % 7 + 7 * (n - 1))


def festivos(anios: range | list[int]) -> pd.DatetimeIndex:
    """Dias festivos de Mexico: descanso obligatorio (LFT art. 74) y Jueves y Viernes Santo."""
    dias: list[pd.Timestamp] = []
    for anio in anios:
        semana_santa = pd.Timestamp(pascua(anio))
        dias += [
            pd.Timestamp(anio, 1, 1),
            _lunes_n(anio, 2, 1),
            _lunes_n(anio, 3, 3),
            semana_santa - pd.Timedelta(days=3),
            semana_santa - pd.Timedelta(days=2),
            pd.Timestamp(anio, 5, 1),
            pd.Timestamp(anio, 9, 16),
            _lunes_n(anio, 11, 3),
            pd.Timestamp(anio, 12, 25),
        ]
    return pd.DatetimeIndex(sorted(dias))


def rango_configurado() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Primer y ultimo dia del rango de descarga acordado (inclusivo), tomado de CONFIG."""
    inicio = pd.Timestamp(CONFIG.download_data_start_year, CONFIG.download_data_start_month, 1)
    fin = pd.Timestamp(CONFIG.download_data_end_year, CONFIG.download_data_end_month, 1)
    return inicio, fin + pd.offsets.MonthEnd(0)


def cargar_salidas(ruta: Path) -> pd.DataFrame:
    """Lee del dataset combinado solo fecha de inicio y estacion de origen (sin datos de usuario).

    El dataset combinado puede ser parquet o CSV (CONFIG.output_filename). Si las fechas llegan
    como texto se interpretan como ISO 8601 fila por fila, no con el formato de la primera: de
    otro modo pandas descarta en silencio las que difieren (por ejemplo, con fracciones de
    segundo). Las fechas que aun asi no se pueden leer se avisan y, si pasan de
    MAX_FRACCION_FECHAS_ILEGIBLES, se detiene la carga para no sesgar los conteos.
    """
    if ruta.suffix == ".parquet":
        viajes = pd.read_parquet(ruta, columns=COLUMNAS_VIAJES)
    else:
        viajes = pd.read_csv(ruta, usecols=COLUMNAS_VIAJES)
    presentes = viajes["Inicio_del_viaje"].notna()
    viajes["Inicio_del_viaje"] = pd.to_datetime(
        viajes["Inicio_del_viaje"], format="ISO8601", errors="coerce"
    )
    ilegibles = int((presentes & viajes["Inicio_del_viaje"].isna()).sum())
    if ilegibles:
        logger.warning(
            "{} de {} fechas de inicio no se pudieron interpretar y no se contaran",
            ilegibles,
            int(presentes.sum()),
        )
        if ilegibles > MAX_FRACCION_FECHAS_ILEGIBLES * presentes.sum():
            raise ValueError(
                f"{ilegibles} fechas de inicio ilegibles en {ruta.name}: revisa su formato"
            )
    viajes["Origen_Id"] = pd.to_numeric(viajes["Origen_Id"], errors="coerce")
    return viajes


def construir_panel_diario(
    viajes: pd.DataFrame,
    inicio: pd.Timestamp | str | None = None,
    fin: pd.Timestamp | str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Panel estacion-dia con el numero de salidas segun la fecha local de inicio del viaje.

    Se cuentan todos los registros publicados con fecha y estacion interpretables, sin
    filtrar por duracion ni por destino. Cada estacion entra al panel el dia de su primera
    salida y desde ahi se rellenan con cero los dias sin viajes; antes de ese dia no se
    inventan filas. Los timestamps son hora local sin offset (Mexico sin horario de verano
    desde 2022), asi que no hay conversiones de zona horaria. Los viajes fuera de
    [inicio, fin] se descartan y se cuentan en el resumen; si fin llega mas alla del ultimo
    dia con viajes, el panel termina en ese ultimo dia para no crear ceros falsos.
    """
    resumen: dict = {"viajes_totales": len(viajes)}
    validos = viajes.dropna(subset=COLUMNAS_VIAJES)
    resumen["viajes_sin_fecha_o_estacion"] = len(viajes) - len(validos)
    fecha = validos["Inicio_del_viaje"].dt.normalize()
    inicio = pd.Timestamp(inicio) if inicio is not None else fecha.min()
    fin = pd.Timestamp(fin) if fin is not None else fecha.max()
    en_rango = (fecha >= inicio) & (fecha <= fin)
    resumen["viajes_fuera_de_rango"] = int((~en_rango).sum())
    if not en_rango.any():
        raise ValueError("No hay salidas dentro del rango pedido")

    estacion = validos.loc[en_rango, "Origen_Id"].astype("int64")
    fecha = fecha[en_rango]
    inicio, fin = max(inicio, fecha.min()), min(fin, fecha.max())
    dias = pd.date_range(inicio, fin, freq="D")
    conteo = pd.DataFrame({"estacion": estacion, "fecha": fecha}).value_counts().reset_index()
    estaciones = np.sort(conteo["estacion"].unique())
    matriz = np.zeros((len(estaciones), len(dias)), dtype="int32")
    fila = np.searchsorted(estaciones, conteo["estacion"].to_numpy())
    columna = ((conteo["fecha"] - inicio) / pd.Timedelta(days=1)).to_numpy().astype(int)
    matriz[fila, columna] = conteo["count"].to_numpy()

    primer_idx = (matriz > 0).argmax(axis=1)
    desde_primera = np.arange(len(dias))[None, :] >= primer_idx[:, None]
    i, j = np.nonzero(desde_primera)
    panel = pd.DataFrame(
        {
            "estacion": estaciones[i].astype("int32"),
            "fecha": dias[j],
            "salidas": matriz[i, j],
            "primer_dia": dias[primer_idx[i]],
        }
    )
    resumen |= {
        "primer_dia_panel": str(dias[0].date()),
        "ultimo_dia_panel": str(dias[-1].date()),
        "estaciones": len(estaciones),
        "filas_panel": len(panel),
        "salidas_contadas": int(matriz.sum()),
        "dias_con_cero_rellenados": int((panel["salidas"] == 0).sum()),
    }
    return panel, resumen


def construir_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Features con informacion hasta D-2; objetivo = salidas de la estacion el dia D.

    El pronostico del dia D se emite a las 23:00 de D-1, cuando D-1 aun no cierra, asi que
    ninguna variable mira D-1 ni D en adelante. Variables y por que sirven:
    - dia_semana (0=lunes ... 6=domingo): del dia objetivo, que se conoce de antemano; la
      demanda sigue el patron semanal de traslados al trabajo y a la escuela.
    - festivo, festivo_lag7: el dia objetivo es festivo, o lo fue el mismo dia de la semana
      pasada (ese dato atipico distorsiona el rezago semanal).
    - lag_2: salidas en D-2, el ultimo dia completo; tambien es el baseline de persistencia.
    - lag_7, lag_14, lag_21, lag_28: salidas el mismo dia de la semana hace 1 a 4 semanas;
      capturan la estacionalidad semanal. lag_7 es el baseline de semana previa.
    - media_7d y media_28d: media diaria de los 7 y 28 dias mas recientes disponibles
      (D-8 a D-2 y D-29 a D-2); dan el nivel actual de la estacion y suavizan el ruido.
    - media_misma_dsem_4: media de lag_7 a lag_28.
    Un dia es elegible si la estacion tiene historia suficiente (29 dias para media_28d) y
    registro alguna salida en los ultimos 28 dias disponibles; esa regla solo usa datos que
    existen al emitir, asi que no depende del futuro. Devuelve todas las filas con las
    banderas sin_historia, inactiva y elegible.
    """
    df = panel.sort_values(["estacion", "fecha"]).reset_index(drop=True)
    pasos = df.groupby("estacion", sort=False)["fecha"].diff().dropna()
    if not (pasos == pd.Timedelta(days=1)).all():
        raise ValueError("El panel debe tener dias consecutivos por estacion")

    g = df.groupby("estacion", sort=False)["salidas"]
    df["objetivo"] = df["salidas"].astype("float64")
    for k in (LAG_MINIMO, *LAGS_SEMANALES):
        df[f"lag_{k}"] = g.shift(k)
    df["media_7d"] = g.transform(lambda s: s.rolling(VENTANA_CORTA).mean().shift(LAG_MINIMO))
    df["media_28d"] = g.transform(lambda s: s.rolling(VENTANA_LARGA).mean().shift(LAG_MINIMO))
    lags_semana = [f"lag_{k}" for k in LAGS_SEMANALES]
    df["media_misma_dsem_4"] = df[lags_semana].mean(axis=1, skipna=False)

    df["dia_semana"] = df["fecha"].dt.dayofweek.astype("int8")
    anios = range(int(df["fecha"].dt.year.min()) - 1, int(df["fecha"].dt.year.max()) + 1)
    dias_festivos = festivos(anios)
    df["festivo"] = df["fecha"].isin(dias_festivos).astype("int8")
    df["festivo_lag7"] = (df["fecha"] - pd.Timedelta(days=7)).isin(dias_festivos).astype("int8")

    df["baseline_semana_previa"] = df["lag_7"]
    df["baseline_persistencia"] = df[f"lag_{LAG_MINIMO}"]
    df["sin_historia"] = df["media_28d"].isna()
    df["inactiva"] = ~df["sin_historia"] & (df["media_28d"] <= 0)
    df["elegible"] = ~df["sin_historia"] & ~df["inactiva"]
    return df


@app.command()
def main(
    entrada: Path = CONFIG.processed_data_dir / CONFIG.output_filename,
    salida: Path = CONFIG.interim_data_dir / "p1_panel_diario.parquet",
    inicio: str | None = None,
    fin: str | None = None,
) -> None:
    """Genera el panel estacion-dia de salidas para P1 (rango de CONFIG si no se indica)."""
    por_defecto = rango_configurado()
    viajes = cargar_salidas(entrada)
    panel, resumen = construir_panel_diario(
        viajes, inicio or por_defecto[0], fin or por_defecto[1]
    )
    salida.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(salida, index=False)
    salida.with_suffix(".json").write_text(json.dumps(resumen, indent=2))
    logger.info("Panel P1: {}", resumen)


if __name__ == "__main__":
    app()
