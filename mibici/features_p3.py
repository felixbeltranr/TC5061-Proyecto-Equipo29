"""Variables de uso por estación para la pregunta 3 (P3).

Entrada: el dataset combinado que genera ``python -m mibici.dataset`` (ingesta del equipo) en
``data/processed`` (CSV o Parquet, según ``CONFIG.output_filename``) y la nomenclatura más
reciente que ese mismo comando guarda en ``data/raw``.

Flujo:

1. ``leer_viajes``: lee del dataset combinado solo las 5 columnas que usa P3; los datos
   personales nunca se cargan.
2. ``limpiar_viajes``: limpieza mínima propia de P3 (duplicados exactos, fechas o ids inválidos
   y viajes de menos de 1 minuto), con una bitácora por mes.
3. ``flujos_horarios``: resume los viajes en salidas y llegadas por estación y hora.
   ``cargar_flujos`` guarda ese resumen en ``data/interim`` y lo reutiliza si el dataset no
   cambió.
4. ``perfil_estaciones``: calcula las 12 variables candidatas de cada estación en un periodo.
5. ``seleccionar_variables``: quita las redundantes con el factor de inflación de la varianza
   (VIF < 5), conservando las de mayor prioridad.

``construir_dataset`` encadena todo; ``python -m mibici.features_p3`` guarda los perfiles de
entrenamiento y evaluación en ``data/interim`` para que los lea ``mibici.modeling.train_p3``.
Los parámetros (fechas, franjas, umbrales) están en ``CONFIG_P3`` (``mibici/config.py``).

Uso desde la terminal (raíz del repositorio), después de ``python -m mibici.dataset``::

    python -m mibici.features_p3
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import typer

from mibici.config import CONFIG, CONFIG_P3, P3Config

app = typer.Typer()

COLUMNAS = ["Viaje_Id", "Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"]
COLUMNAS_PERSONALES = {"Usuario_Id", "Genero", "Año_de_nacimiento"}
CANDIDATAS = [
    "log_trips_day",
    "weekend_share",
    "dep_am",
    "dep_mid",
    "dep_pm",
    "dep_night",
    "net_am",
    "net_pm",
    "arr_am",
    "arr_mid",
    "arr_pm",
    "arr_night",
]
CONTEOS_BITACORA = [
    "filas_crudas",
    "duplicados_exactos",
    "fechas_o_ids_invalidos",
    "duracion_negativa",
    "duracion_menor_1min",
    "mismo_origen_destino",
    "viajes_validos",
]


# ---------------------------------------------------------------- lectura y limpieza mínima
def sha256_archivo(ruta: Path, bloque: int = 1 << 20) -> str:
    """Huella SHA-256 de un archivo: identifica exactamente qué datos se usaron."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(bloque), b""):
            h.update(trozo)
    return h.hexdigest()


def leer_viajes(ruta: Path) -> pd.DataFrame:
    """Lee del dataset combinado solo las columnas de P3 (sin datos personales).

    Acepta el CSV que genera ``mibici.dataset`` en ``main`` y el Parquet del PR #2.
    """
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Genera primero el dataset combinado con: python -m mibici.dataset"
        )
    if ruta.suffix == ".parquet":
        return pd.read_parquet(ruta, columns=COLUMNAS)
    return pd.read_csv(ruta, usecols=COLUMNAS, dtype={c: str for c in COLUMNAS[1:3]})


def limpiar_viajes(
    v: pd.DataFrame, duracion_min_seg: int = CONFIG_P3.duracion_min_seg
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica la limpieza mínima de P3 y regresa los viajes válidos y una bitácora por mes.

    Se quitan, en este orden: duplicados exactos, viajes con fechas o ids inválidos y viajes
    con duración negativa o de menos de ``duracion_min_seg`` segundos (anclajes fallidos).
    Los viajes de ida y vuelta a la misma estación se conservan; solo se cuentan.
    """
    v = v.copy()
    for col in ["Inicio_del_viaje", "Fin_del_viaje"]:
        v[col] = pd.to_datetime(v[col], format="ISO8601", errors="coerce")
    for col in ["Origen_Id", "Destino_Id"]:
        v[col] = pd.to_numeric(v[col], errors="coerce")
    mes = v["Inicio_del_viaje"].dt.strftime("%Y_%m").fillna("sin_fecha")
    duplicado = v.duplicated()
    invalido = ~duplicado & v[COLUMNAS[1:]].isna().any(axis=1)
    dur = (v["Fin_del_viaje"] - v["Inicio_del_viaje"]).dt.total_seconds()
    vivo = ~duplicado & ~invalido
    marcas = pd.DataFrame(
        {
            "filas_crudas": True,
            "duplicados_exactos": duplicado,
            "fechas_o_ids_invalidos": invalido,
            "duracion_negativa": vivo & (dur < 0),
            "duracion_menor_1min": vivo & (dur >= 0) & (dur < duracion_min_seg),
            "mismo_origen_destino": vivo & (v["Origen_Id"] == v["Destino_Id"]),
            "viajes_validos": vivo & (dur >= duracion_min_seg),
        },
        index=v.index,
    )
    bitacora = marcas.groupby(mes).sum().astype(int).rename_axis("mes").reset_index()
    validos = v[marcas["viajes_validos"]].astype({"Origen_Id": "int64", "Destino_Id": "int64"})
    return validos, bitacora


def flujos_horarios(v: pd.DataFrame) -> pd.DataFrame:
    """Salidas y llegadas por estación y hora (la llegada cuenta en la hora de fin del viaje)."""
    sal = v.groupby(
        [v["Origen_Id"].rename("estacion"), v["Inicio_del_viaje"].dt.floor("h").rename("hora")]
    ).size()
    lle = v.groupby(
        [v["Destino_Id"].rename("estacion"), v["Fin_del_viaje"].dt.floor("h").rename("hora")]
    ).size()
    fl = pd.concat([sal.rename("salidas"), lle.rename("llegadas")], axis=1).fillna(0)
    return fl.astype("int32").reset_index()


def cargar_flujos(
    ruta: Path, cache_dir: Path = CONFIG.interim_data_dir / CONFIG_P3.cache_subdir
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Flujos horarios y bitácora del dataset combinado, con su huella SHA-256.

    El resumen se guarda en ``cache_dir`` con la huella en el nombre: si el dataset no cambió,
    se reutiliza en lugar de volver a leer los millones de viajes.
    """
    huella = sha256_archivo(ruta)
    f_flujos = cache_dir / f"p3_flujos_{huella[:12]}.csv.gz"
    f_bit = cache_dir / f"p3_bitacora_{huella[:12]}.csv"
    if f_flujos.exists() and f_bit.exists():
        logger.info(f"Usando el resumen guardado en {cache_dir.name}/{f_flujos.name}")
        return pd.read_csv(f_flujos, parse_dates=["hora"]), pd.read_csv(f_bit), huella
    logger.info(f"Leyendo {ruta.name} (la primera vez tarda unos minutos)")
    validos, bitacora = limpiar_viajes(leer_viajes(ruta))
    fl = flujos_horarios(validos)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fl.to_csv(f_flujos, index=False)
    bitacora.to_csv(f_bit, index=False)
    return fl, bitacora, huella


def nomenclatura_mas_reciente(raw_dir: Path = CONFIG.raw_data_dir) -> Path:
    """Ruta de la nomenclatura más reciente (nomenclatura_AAAA_MM.csv) en ``raw_dir``."""
    rutas = sorted(raw_dir.glob("nomenclatura_*.csv"))
    if not rutas:
        raise FileNotFoundError(
            f"No hay nomenclatura en {raw_dir}. Genérala con: python -m mibici.dataset"
        )
    return rutas[-1]


def leer_nomenclatura(ruta: Path) -> pd.DataFrame:
    """Nomenclatura de estaciones indexada por id, con el municipio tomado del nombre."""
    try:
        nom = pd.read_csv(ruta, encoding="utf-8-sig")
    except UnicodeDecodeError:
        nom = pd.read_csv(ruta, encoding="cp1252")
    nom = nom.set_index("id")
    nom["municipio"] = nom["name"].str.extract(r"\((GDL|ZPN|TLQ)-")[0].fillna("otro")
    return nom


# ---------------------------------------------------------------- variables por estación
def filtrar_periodo(fl: pd.DataFrame, inicio: str, fin: str) -> pd.DataFrame:
    """Flujos horarios entre ``inicio`` y ``fin`` (ambas fechas incluidas)."""
    fin_excl = pd.Timestamp(fin) + pd.Timedelta(days=1)
    return fl[(fl.hora >= pd.Timestamp(inicio)) & (fl.hora < fin_excl)]


def matrices_horarias(fl: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Salidas y llegadas de día hábil por estación (filas) y hora del día (columnas 0 a 23)."""
    hab = fl[fl.hora.dt.dayofweek < 5].assign(h=lambda d: d.hora.dt.hour)

    def piv(col):
        return hab.pivot_table(
            index="estacion", columns="h", values=col, aggfunc="sum", fill_value=0
        ).reindex(columns=range(24), fill_value=0)

    return piv("salidas"), piv("llegadas")


def perfil_estaciones(fl: pd.DataFrame, franjas: dict = CONFIG_P3.franjas) -> pd.DataFrame:
    """Variables candidatas de uso por estación, calculadas solo con los flujos ``fl``.

    - ``trips_day`` y ``log_trips_day``: viajes por día activo (y su logaritmo).
    - ``weekend_share``: proporción de salidas en fin de semana.
    - ``dep_<franja>`` y ``arr_<franja>``: proporción de salidas y llegadas del día hábil en
      cada franja.
    - ``net_am`` y ``net_pm``: (llegadas - salidas) / total en la hora pico de la mañana y de
      la tarde; negativo = la estación se vacía, positivo = se llena.
    """
    fl = fl.assign(fecha=fl.hora.dt.normalize(), finde=fl.hora.dt.dayofweek >= 5)
    g = fl.groupby("estacion")
    p = pd.DataFrame(
        {
            "dias_activos": g.fecha.nunique(),
            "salidas": g.salidas.sum(),
            "llegadas": g.llegadas.sum(),
        }
    )
    p["trips_day"] = (p.salidas + p.llegadas) / 2 / p.dias_activos
    p["log_trips_day"] = np.log1p(p.trips_day)
    finde = fl[fl.finde].groupby("estacion").salidas.sum().reindex(p.index, fill_value=0)
    p["weekend_share"] = finde / p.salidas
    sal, lle = (m.reindex(p.index, fill_value=0) for m in matrices_horarias(fl))
    p["salidas_habiles"] = sal.sum(axis=1)
    for nombre, horas in franjas.items():
        p[f"dep_{nombre}"] = sal[list(horas)].sum(axis=1) / sal.sum(axis=1)
        p[f"arr_{nombre}"] = lle[list(horas)].sum(axis=1) / lle.sum(axis=1)
    for nombre in ["am", "pm"]:
        s, ll = sal[list(franjas[nombre])].sum(axis=1), lle[list(franjas[nombre])].sum(axis=1)
        p[f"net_{nombre}"] = (ll - s) / (ll + s)
    return p.replace([np.inf, -np.inf], np.nan)


def elegibles(
    p: pd.DataFrame, ids_con_ubicacion, min_dias: int = CONFIG_P3.min_dias_activos
) -> pd.DataFrame:
    """Estaciones con ubicación conocida y al menos ``min_dias`` días con viajes."""
    ok = p.index.isin(ids_con_ubicacion) & (p.dias_activos >= min_dias)
    return p[ok].dropna(subset=CANDIDATAS)


def vif(df: pd.DataFrame) -> pd.Series:
    """Factor de inflación de la varianza de cada columna: 1 / (1 - R²) contra las demás."""
    r2 = {
        c: LinearRegression().fit(df.drop(columns=c), df[c]).score(df.drop(columns=c), df[c])
        for c in df
    }
    return pd.Series({c: 1 / (1 - r) if r < 1 else np.inf for c, r in r2.items()})


def seleccionar_variables(
    df: pd.DataFrame, prioridad=CONFIG_P3.prioridad_variables, umbral: float = CONFIG_P3.umbral_vif
) -> list[str]:
    """Quita una a una las variables con VIF >= ``umbral``, empezando por la de menor prioridad."""
    seleccion = list(prioridad)
    while (v := vif(df[seleccion])).max() >= umbral:
        seleccion.remove([c for c in seleccion if v[c] >= umbral][-1])
    return seleccion


def a_km(nom: pd.DataFrame, indices, lat_ref: float = CONFIG_P3.lat_ref) -> np.ndarray:
    """Ubicación de las estaciones en km (proyección equirectangular local)."""
    lat = nom.loc[indices, "latitude"].to_numpy()
    lon = nom.loc[indices, "longitude"].to_numpy()
    return np.c_[lon * 111.32 * np.cos(np.radians(lat_ref)), lat * 111.32]


# ---------------------------------------------------------------- versión de los datos
def version_manifiesto(
    manifiesto_path: Path = CONFIG_P3.manifiesto_path,
    raw_dir: Path = CONFIG.raw_data_dir,
) -> str:
    """``data_version`` del proyecto: la "version" del manifiesto de datos en ``references/``.

    Es el mismo manifiesto que usa P2 (sha256 de cada CSV crudo y de la nomenclatura), así que
    las dos preguntas registran la misma versión. Si los archivos de ``raw_dir`` no coinciden
    con el manifiesto, se avisa en el log.
    """
    if not manifiesto_path.exists():
        raise FileNotFoundError(f"No existe el manifiesto de datos {manifiesto_path}")
    manifiesto = json.loads(manifiesto_path.read_text())
    distintos = [
        a["archivo"]
        for a in manifiesto["archivos"]
        if not (raw_dir / a["archivo"]).exists()
        or sha256_archivo(raw_dir / a["archivo"]) != a["sha256"]
    ]
    if distintos:
        logger.warning(
            f"{len(distintos)} archivos de {raw_dir.name} faltan o no coinciden con el manifiesto "
            f"{manifiesto_path.name}: {distintos[:3]}"
        )
    return manifiesto["version"]


# ---------------------------------------------------------------- dataset completo de P3
@dataclass
class DatasetP3:
    """Todo lo que necesita el entrenamiento de P3."""

    bitacora: pd.DataFrame
    nom: pd.DataFrame
    entrenamiento: pd.DataFrame  # perfil de las estaciones elegibles del periodo de entrenamiento
    evaluacion: pd.DataFrame  # perfil de las estaciones elegibles del periodo de evaluación
    variables: list[str]
    nuevas: pd.Index  # estaciones con ubicación cuyo primer viaje es del periodo de evaluación
    data_version: str
    nomenclatura: str  # nombre del archivo de nomenclatura usado


def construir_dataset(
    dataset_path: Path = CONFIG.processed_data_dir / CONFIG.output_filename,
    raw_dir: Path = CONFIG.raw_data_dir,
    cfg: P3Config = CONFIG_P3,
) -> DatasetP3:
    """Lee el dataset combinado, arma los perfiles por periodo y selecciona las variables."""
    flujos, bitacora, _ = cargar_flujos(dataset_path, CONFIG.interim_data_dir / cfg.cache_subdir)
    flujos = filtrar_periodo(flujos, f"{cfg.meses_inicio}-01", cfg.fecha_fin_evaluacion)
    meses = pd.period_range(cfg.meses_inicio, cfg.meses_fin, freq="M").strftime("%Y_%m")
    faltantes = sorted(set(meses) - set(bitacora.loc[bitacora.viajes_validos > 0, "mes"]))
    if faltantes:
        raise ValueError(
            f"Al dataset combinado le faltan meses: {faltantes}. Vuelve a correr "
            "python -m mibici.dataset con el rango de P3 (enero 2025 a septiembre 2026)."
        )
    bitacora = bitacora[bitacora.mes.isin(meses)].reset_index(drop=True)
    for mes_ in bitacora.itertuples():
        logger.info(f"{mes_.mes}: {mes_.viajes_validos:,} viajes válidos")

    f_nom = nomenclatura_mas_reciente(raw_dir)
    nom = leer_nomenclatura(f_nom)

    fl_en = filtrar_periodo(flujos, cfg.fecha_inicio_entrenamiento, cfg.fecha_fin_entrenamiento)
    fl_ev = filtrar_periodo(flujos, cfg.fecha_inicio_evaluacion, cfg.fecha_fin_evaluacion)
    en = elegibles(perfil_estaciones(fl_en, cfg.franjas), nom.index, cfg.min_dias_activos)
    ev = elegibles(perfil_estaciones(fl_ev, cfg.franjas), nom.index, cfg.min_dias_activos)
    # La selección de variables usa SOLO el periodo de entrenamiento
    variables = seleccionar_variables(en, cfg.prioridad_variables, cfg.umbral_vif)

    primera = flujos.groupby("estacion").hora.min()
    nuevas = primera.index[primera >= pd.Timestamp(cfg.fecha_inicio_evaluacion)]
    nuevas = nuevas[nuevas.isin(nom.index)]
    data_version = version_manifiesto(cfg.manifiesto_path, raw_dir)
    return DatasetP3(bitacora, nom, en, ev, variables, nuevas, data_version, f_nom.name)


def guardar_dataset(ds: DatasetP3, salida_dir: Path = CONFIG.interim_data_dir) -> None:
    """Guarda los perfiles y su resumen en ``salida_dir`` para que los lea ``train_p3``."""
    salida_dir.mkdir(parents=True, exist_ok=True)
    ds.entrenamiento.to_parquet(salida_dir / "p3_perfiles_entrenamiento.parquet")
    ds.evaluacion.to_parquet(salida_dir / "p3_perfiles_evaluacion.parquet")
    ds.bitacora.to_csv(salida_dir / "p3_bitacora_limpieza.csv", index=False)
    resumen = {
        "data_version": ds.data_version,
        "nomenclatura": ds.nomenclatura,
        "variables": ds.variables,
        "nuevas": [int(e) for e in ds.nuevas],
    }
    (salida_dir / "p3_resumen.json").write_text(json.dumps(resumen, indent=2) + "\n")


def cargar_dataset(
    salida_dir: Path = CONFIG.interim_data_dir, raw_dir: Path = CONFIG.raw_data_dir
) -> DatasetP3:
    """Lee lo que dejó ``python -m mibici.features_p3`` en ``salida_dir``."""
    f_resumen = salida_dir / "p3_resumen.json"
    if not f_resumen.exists():
        raise FileNotFoundError(
            f"No existe {f_resumen}. Genera primero los perfiles con: python -m mibici.features_p3"
        )
    resumen = json.loads(f_resumen.read_text())
    return DatasetP3(
        bitacora=pd.read_csv(salida_dir / "p3_bitacora_limpieza.csv"),
        nom=leer_nomenclatura(raw_dir / resumen["nomenclatura"]),
        entrenamiento=pd.read_parquet(salida_dir / "p3_perfiles_entrenamiento.parquet"),
        evaluacion=pd.read_parquet(salida_dir / "p3_perfiles_evaluacion.parquet"),
        variables=resumen["variables"],
        nuevas=pd.Index(resumen["nuevas"], name="estacion"),
        data_version=resumen["data_version"],
        nomenclatura=resumen["nomenclatura"],
    )


@app.command()
def main(
    dataset_path: Path = CONFIG.processed_data_dir / CONFIG.output_filename,
    salida_dir: Path = CONFIG.interim_data_dir,
):
    """Genera los perfiles de uso de P3 y los guarda en data/interim."""
    ds = construir_dataset(dataset_path)
    guardar_dataset(ds, salida_dir)
    logger.success(
        f"data_version {ds.data_version} | entrenamiento {len(ds.entrenamiento)} estaciones | "
        f"evaluación {len(ds.evaluacion)} | variables: {ds.variables} | "
        f"guardado en {salida_dir.name}/"
    )


if __name__ == "__main__":
    app()
