"""Features y etiquetas para P2: riesgo de estacion sin bicicletas o sin anclajes.

Los datos abiertos no traen inventario por estacion, asi que aproximamos la
ocupacion con el flujo neto (llegadas menos salidas) acumulado por dia de
servicio desde un nivel inicial supuesto, recortado a [0, capacidad].
Se construye sobre combined_dataset.parquet generado por mibici.dataset.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici.config import CONFIG, CONFIG_P2

app = typer.Typer()

CLASES = ["segura", "sin_bicicletas", "sin_anclajes"]

FEATURES = [
    "hora",
    "dia_semana",
    "fin_semana",
    "capacidad",
    "ocupacion_actual",
    "neto_lag1",
    "neto_lag24",
    "neto_lag168",
    "salidas_lag1",
    "llegadas_lag1",
    "neto_media3",
    "neto_media24",
    "neto_media168_misma_hora",
    "ocupacion_lag24",
    "ocupacion_lag168",
    "est_tasa_sin_bicis",
    "est_tasa_sin_anclajes",
    "est_neto_medio_hora",
]


def etiquetar_ocupacion(
    ocupacion: np.ndarray | pd.Series,
    capacidad: np.ndarray | pd.Series,
    umbral_bajo: float = CONFIG_P2.umbral_bajo,
    umbral_alto: float = CONFIG_P2.umbral_alto,
) -> np.ndarray:
    """Clase por fila: sin_bicicletas (< umbral_bajo), sin_anclajes (> umbral_alto) o segura."""
    ocupacion = np.asarray(ocupacion, dtype=float)
    capacidad = np.asarray(capacidad, dtype=float)
    if np.any(capacidad <= 0):
        raise ValueError("La capacidad debe ser positiva")
    if np.any(ocupacion < 0) or np.any(ocupacion > capacidad):
        raise ValueError("La ocupacion debe estar entre 0 y la capacidad")
    razon = ocupacion / capacidad
    etiqueta = np.full(razon.shape, "segura", dtype=object)
    etiqueta[razon < umbral_bajo] = "sin_bicicletas"
    etiqueta[razon > umbral_alto] = "sin_anclajes"
    return etiqueta


def acumular_ocupacion(
    neto: np.ndarray, capacidad: np.ndarray, nivel_inicial: float
) -> np.ndarray:
    """Ocupacion al cierre de cada hora para una matriz (dias_estacion, 24) de flujo neto.

    Cada fila es un dia de servicio de una estacion. Se parte de nivel_inicial * capacidad
    y en cada hora se suma el neto y se recorta a [0, capacidad], de modo que nunca hay mas
    bicicletas que anclajes.
    """
    neto = np.asarray(neto, dtype=float)
    capacidad = np.asarray(capacidad, dtype=float).reshape(-1)
    if neto.ndim != 2 or neto.shape[0] != capacidad.shape[0]:
        raise ValueError("neto debe ser (n, horas) y capacidad (n,)")
    ocupacion = np.empty_like(neto)
    actual = capacidad * nivel_inicial
    for h in range(neto.shape[1]):
        actual = np.clip(actual + neto[:, h], 0, capacidad)
        ocupacion[:, h] = actual
    return ocupacion


def cargar_viajes(ruta: Path) -> pd.DataFrame:
    """Lee del dataset combinado solo fechas y estaciones; no carga datos de usuario."""
    columnas = ["Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"]
    # Parquet es la salida de mibici.dataset; el CSV queda por si alguien trae uno anterior
    if ruta.suffix == ".parquet":
        viajes = pd.read_parquet(ruta, columns=columnas)
    else:
        viajes = pd.read_csv(ruta, usecols=columnas)
    viajes["Inicio_del_viaje"] = pd.to_datetime(viajes["Inicio_del_viaje"], errors="coerce")
    viajes["Fin_del_viaje"] = pd.to_datetime(viajes["Fin_del_viaje"], errors="coerce")
    return viajes


def tabla_estaciones(nomenclatura: pd.DataFrame) -> pd.DataFrame:
    """Id y capacidad de cada estacion a partir de la nomenclatura."""
    est = nomenclatura.loc[:, ["id", "dpcapacity"]].rename(
        columns={"id": "estacion", "dpcapacity": "capacidad"}
    )
    est["estacion"] = pd.to_numeric(est["estacion"], errors="coerce").astype("Int64")
    return est.dropna().drop_duplicates("estacion")


def construir_panel(
    viajes: pd.DataFrame, estaciones: pd.DataFrame, fin: str = CONFIG_P2.panel_fin
) -> tuple[pd.DataFrame, dict]:
    """Panel estacion-hora con salidas, llegadas y flujo neto.

    Excluye estaciones sin capacidad y las que no aparecen en la nomenclatura.
    Cada estacion entra al panel desde el primer dia en que registra viajes.
    """
    resumen: dict = {"viajes_totales": len(viajes)}
    viajes = viajes.dropna(subset=["Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"])
    resumen["viajes_sin_fecha_o_estacion"] = resumen["viajes_totales"] - len(viajes)

    salidas = (
        viajes.assign(
            estacion=viajes["Origen_Id"].astype(int), ts=viajes["Inicio_del_viaje"].dt.floor("h")
        )
        .groupby(["estacion", "ts"])
        .size()
        .rename("salidas")
    )
    llegadas = (
        viajes.assign(
            estacion=viajes["Destino_Id"].astype(int), ts=viajes["Fin_del_viaje"].dt.floor("h")
        )
        .groupby(["estacion", "ts"])
        .size()
        .rename("llegadas")
    )
    flujo = pd.concat([salidas, llegadas], axis=1).fillna(0).reset_index()

    ids_viajes = set(flujo["estacion"].unique())
    cap = dict(zip(estaciones["estacion"].astype(int), estaciones["capacidad"].astype(float)))
    faltantes = sorted(ids_viajes - set(cap))
    cap_cero = sorted(e for e in ids_viajes if cap.get(e, 1) <= 0)
    resumen["estaciones_en_viajes"] = len(ids_viajes)
    resumen["estaciones_sin_nomenclatura"] = len(faltantes)
    resumen["estaciones_capacidad_cero"] = len(cap_cero)
    excluir = set(faltantes) | set(cap_cero)
    resumen["viajes_que_tocan_estacion_excluida"] = int(
        (viajes["Origen_Id"].isin(excluir) | viajes["Destino_Id"].isin(excluir)).sum()
    )
    flujo = flujo[~flujo["estacion"].isin(excluir)]

    fin_ts = pd.Timestamp(fin)
    piezas = []
    for est_id, grupo in flujo.groupby("estacion"):
        inicio = (
            grupo["ts"].min() - pd.Timedelta(hours=CONFIG_P2.hora_inicio_servicio)
        ).normalize()
        inicio += pd.Timedelta(hours=CONFIG_P2.hora_inicio_servicio)
        horas = pd.date_range(
            inicio,
            fin_ts.normalize() + pd.Timedelta(hours=24 + CONFIG_P2.hora_inicio_servicio - 1),
            freq="h",
        )
        g = grupo.set_index("ts")[["salidas", "llegadas"]].reindex(horas, fill_value=0)
        g.index.name = "ts"
        g["estacion"] = est_id
        piezas.append(g.reset_index())
    panel = pd.concat(piezas, ignore_index=True)
    panel["capacidad"] = panel["estacion"].map(cap).astype("float32")
    panel["salidas"] = panel["salidas"].astype("float32")
    panel["llegadas"] = panel["llegadas"].astype("float32")
    panel["neto"] = panel["llegadas"] - panel["salidas"]
    panel["primer_dia"] = panel.groupby("estacion")["ts"].transform("min")
    resumen["estaciones_modeladas"] = int(panel["estacion"].nunique())
    resumen["filas_panel"] = len(panel)
    return panel, resumen


def agregar_ocupacion(panel: pd.DataFrame, nivel_inicial: float) -> pd.DataFrame:
    """Ocupacion estimada y su clase al cierre de cada hora (panel ordenado, dias completos)."""
    panel = panel.sort_values(["estacion", "ts"]).reset_index(drop=True)
    n = len(panel)
    if n % 24:
        raise ValueError("El panel debe tener dias de servicio completos")
    neto = panel["neto"].to_numpy().reshape(-1, 24)
    cap = panel["capacidad"].to_numpy().reshape(-1, 24)[:, 0]
    occ = acumular_ocupacion(neto, cap, nivel_inicial).reshape(-1)
    panel["ocupacion"] = occ.astype("float32")
    panel["clase_actual"] = etiquetar_ocupacion(occ, panel["capacidad"].to_numpy())
    return panel


def construir_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Features con informacion hasta el cierre de la hora t; objetivo = clase al cierre de t+1.

    Variables y por que sirven para anticipar el riesgo:
    - hora, dia_semana, fin_semana: del instante objetivo (conocido de antemano); capturan
      los picos de entrada y salida laborales.
    - capacidad: con pocos anclajes la estacion pasa mas rapido a vacia o llena.
    - ocupacion_actual: ocupacion estimada al cierre de t entre capacidad; punto de partida
      del estado en t+1.
    - neto_lag1, salidas_lag1, llegadas_lag1: movimiento de la ultima hora (tendencia).
    - neto_lag24, neto_lag168: flujo neto en el mismo instante del objetivo hace un dia y
      una semana (shift de 23 y 167 filas desde t).
    - neto_media3, neto_media24: medias moviles hasta t, suavizan el ruido horario.
    - neto_media168_misma_hora: media del flujo neto en el mismo instante del objetivo hace
      1, 2, 3 y 4 semanas (shift de 167, 335, 503 y 671 filas desde t).
    - ocupacion_lag24, ocupacion_lag168: estado estimado en ese instante hace un dia y una semana.
    - baseline_semana: clase en ese instante una semana antes (solo para el baseline).
    Las estadisticas por estacion se agregan aparte con stats_estacion (solo train).
    """
    df = panel.sort_values(["estacion", "ts"]).reset_index(drop=True)
    g = df.groupby("estacion", sort=False)
    df["objetivo"] = g["clase_actual"].shift(-1)
    df["ts_objetivo"] = df["ts"] + pd.Timedelta(hours=1)
    df["hora"] = df["ts_objetivo"].dt.hour.astype("int8")
    df["dia_semana"] = df["ts_objetivo"].dt.dayofweek.astype("int8")
    df["fin_semana"] = (df["dia_semana"] >= 5).astype("int8")
    df["ocupacion_actual"] = (df["ocupacion"] / df["capacidad"]).astype("float32")
    df["neto_lag1"] = df["neto"]
    # mismo instante del objetivo hace 24 h y 168 h
    df["neto_lag24"] = g["neto"].shift(23)
    df["neto_lag168"] = g["neto"].shift(167)
    df["salidas_lag1"] = df["salidas"]
    df["llegadas_lag1"] = df["llegadas"]
    df["neto_media3"] = g["neto"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["neto_media24"] = g["neto"].transform(lambda s: s.rolling(24, min_periods=1).mean())
    # mismo instante del objetivo hace k semanas (k = 1..4); k = 1 coincide con neto_lag168
    sem = [g["neto"].shift(168 * k - 1) for k in range(1, 5)]
    df["neto_media168_misma_hora"] = pd.concat(sem, axis=1).mean(axis=1)
    razon = df["ocupacion"] / df["capacidad"]
    df["ocupacion_lag24"] = razon.groupby(df["estacion"]).shift(23)
    df["ocupacion_lag168"] = razon.groupby(df["estacion"]).shift(167)
    # baseline: clase del mismo instante una semana antes
    df["baseline_semana"] = g["clase_actual"].shift(167)
    df = df[df["objetivo"].notna() & df["hora"].isin(list(CONFIG_P2.horas_etiqueta))]
    return df.reset_index(drop=True)


def stats_estacion(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tasas por estacion calculadas solo con el periodo de entrenamiento."""
    tasas = (
        train.groupby("estacion")["objetivo"].value_counts(normalize=True).unstack(fill_value=0)
    )
    tasas = tasas.reindex(columns=CLASES, fill_value=0)
    por_est = pd.DataFrame(
        {
            "est_tasa_sin_bicis": tasas["sin_bicicletas"],
            "est_tasa_sin_anclajes": tasas["sin_anclajes"],
        }
    )
    por_hora = (
        train.groupby(["estacion", "hora"])["neto_lag1"]
        .mean()
        .rename("est_neto_medio_hora")
        .reset_index()
    )
    return por_est.reset_index(), por_hora


def unir_stats(df: pd.DataFrame, por_est: pd.DataFrame, por_hora: pd.DataFrame) -> pd.DataFrame:
    """Agrega a df las estadisticas por estacion calculadas con stats_estacion."""
    df = df.drop(
        columns=[
            c
            for c in ["est_tasa_sin_bicis", "est_tasa_sin_anclajes", "est_neto_medio_hora"]
            if c in df
        ]
    )
    df = df.merge(por_est, on="estacion", how="left")
    return df.merge(por_hora, on=["estacion", "hora"], how="left")


def manifiesto_datos(archivos: list[Path], contar_filas: bool = True) -> dict:
    """Hash sha256 y numero de registros por archivo; el hash global sirve como version."""
    entradas = []
    for ruta in sorted(archivos):
        h = hashlib.sha256()
        filas = 0
        with open(ruta, "rb") as f:
            for bloque in iter(lambda: f.read(1 << 20), b""):
                h.update(bloque)
                if contar_filas:
                    filas += bloque.count(b"\n")
        entradas.append(
            {"archivo": ruta.name, "sha256": h.hexdigest(), "registros": max(filas - 1, 0)}
        )
    global_hash = hashlib.sha256(json.dumps(entradas, sort_keys=True).encode()).hexdigest()
    return {"version": global_hash[:12], "sha256_global": global_hash, "archivos": entradas}


@app.command()
def main(
    entrada: Path = CONFIG.processed_data_dir / CONFIG.output_filename,
    nomenclatura: Path = CONFIG_P2.nomenclatura_path,
    salida: Path = CONFIG_P2.panel_path,
) -> None:
    """Genera el panel estacion-hora de flujos para P2."""
    viajes = cargar_viajes(entrada)
    panel, resumen = construir_panel(viajes, tabla_estaciones(pd.read_csv(nomenclatura)))
    salida.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(salida, index=False)
    salida.with_suffix(".json").write_text(json.dumps(resumen, indent=2))
    logger.info("Panel P2: {}", resumen)


if __name__ == "__main__":
    app()
