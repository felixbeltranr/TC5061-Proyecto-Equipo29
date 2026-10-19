from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")


@dataclass(frozen=True)
class ProjectConfig:
    data_dir: Path = PROJ_ROOT / "data"
    raw_data_dir: Path = PROJ_ROOT / "data" / "raw"
    interim_data_dir: Path = PROJ_ROOT / "data" / "interim"
    processed_data_dir: Path = PROJ_ROOT / "data" / "processed"
    external_data_dir: Path = PROJ_ROOT / "data" / "external"
    output_filename: str = "combined_dataset.parquet"

    models_dir: Path = PROJ_ROOT / "models"

    reports_dir: Path = PROJ_ROOT / "reports"

    target_column: str = "ToDo"

    # MiBici open-data download range (inclusive), the one agreed by the team
    download_data_start_year: int = 2025
    download_data_start_month: int = 1
    download_data_end_year: int = 2026
    download_data_end_month: int = 9

    train_size: float = 0.70
    validation_size: float = 0.15
    test_size: float = 0.15

    mlflow_tracking_uri: str = "http://127.0.0.1:5000"


CONFIG = ProjectConfig()


# ---------------------------------------------------------------------------
# P3 · clustering de estaciones (Planeación urbana)
# Todo lo particular de P3 está aquí; features_p3.py y modeling/train_p3.py lo leen de CONFIG_P3.
# ---------------------------------------------------------------------------
def _franjas_p3() -> dict[str, tuple[int, ...]]:
    # Franjas del día hábil, definidas con la curva horaria del sistema (notebook de P3, 4.2)
    return {
        "am": (6, 7, 8, 9),
        "mid": (11, 12, 13, 14, 15),
        "pm": (17, 18, 19, 20),
        "night": (21, 22, 23, 0, 1, 2, 3, 4, 5),
    }


@dataclass(frozen=True)
class P3Config:
    # Datos. Las carpetas son las de CONFIG: entrada en processed_data_dir (dataset combinado
    # de python -m mibici.dataset) y raw_data_dir (nomenclatura); salidas en interim_data_dir.
    manifiesto_path: Path = PROJ_ROOT / "references" / "manifiesto_datos_p2.json"  # data_version
    cache_subdir: str = "p3_flujos_horarios"  # subcarpeta de interim_data_dir para el resumen
    meses_inicio: str = "2025-01"  # meses que debe traer el dataset combinado
    meses_fin: str = "2026-09"

    # Partición temporal: entrenamiento con la red estable, evaluación con la red ampliada
    fecha_inicio_entrenamiento: str = "2025-01-01"
    fecha_fin_entrenamiento: str = "2026-04-30"
    fecha_inicio_evaluacion: str = "2026-05-01"
    fecha_fin_evaluacion: str = "2026-09-30"

    # Limpieza y elegibilidad
    duracion_min_seg: int = 60  # viajes de menos de 1 minuto se descartan
    min_dias_activos: int = 60  # días con viajes para que el perfil de una estación sea confiable

    # Variables de uso
    franjas: dict[str, tuple[int, ...]] = field(default_factory=_franjas_p3)
    prioridad_variables: tuple[str, ...] = (
        "log_trips_day",
        "weekend_share",
        "net_am",
        "net_pm",
        "dep_am",
        "dep_pm",
        "dep_mid",
        "dep_night",
        "arr_am",
        "arr_pm",
        "arr_mid",
        "arr_night",
    )  # orden en que se conservan las variables al quitar redundancia
    umbral_vif: float = 5.0

    # Modelo
    seed: int = 42
    k_baseline: int = 5
    k_comparacion: tuple[int, ...] = (4, 5, 6)
    n_init: int = 20
    lat_ref: float = 20.67  # latitud de Guadalajara para proyectar a km

    # MLflow: mismo backend local y esquema de etiquetas que P2. No se usa
    # CONFIG.mlflow_tracking_uri porque apunta a un servidor (http://127.0.0.1:5000) que ni P2
    # ni P3 usan; queda pendiente acordarlo en equipo.
    experimento: str = "mibici-equipo29"
    tracking_uri: str = f"sqlite:///{(PROJ_ROOT / 'mlflow.db').as_posix()}"
    artifact_location: str = (PROJ_ROOT / "mlartifacts").as_uri()
    team: str = "29"
    author: str = "felixbeltran"


CONFIG_P3 = P3Config()

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


# P1 · demanda diaria de salidas por estacion (regresion). Parametros del experimento de P1:
# rutas, momento del pronostico, cortes temporales, semilla, rejillas y MLflow. Los leen
# features.py, modeling/train_p1.py y tracking.py. Las definiciones de las variables (rezagos,
# ventanas y nombres de columnas) se quedan en features.py porque los nombres de las variables
# dependen de ellas.
@dataclass(frozen=True)
class P1Config:
    # Datos: panel estacion-dia que genera python -m mibici.features y manifiesto de los CSV
    panel_path: Path = CONFIG.interim_data_dir / "p1_panel_diario.parquet"
    manifiesto_path: Path = PROJ_ROOT / "references" / "manifiesto_datos_p2.json"

    # Momento del pronostico: el dia D se emite a las 23:00 de D-1 y el conteo de un dia queda
    # disponible 60 min despues de cerrar
    hora_emision: int = 23
    retraso_conteo_min: int = 60

    # Particion por fecha objetivo (test_fin es exclusivo); los mismos meses que train_p2.CORTES
    train_inicio: str = "2025-01-01"
    val_inicio: str = "2026-03-01"
    test_inicio: str = "2026-05-01"
    test_fin: str = "2026-10-01"
    semilla: int = 29

    # Rejillas de los candidatos; la configuracion se elige con el MAE de validacion
    alphas_poisson: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    rejilla_hgb: tuple[dict, ...] = (
        {"learning_rate": 0.1, "max_leaf_nodes": 15, "max_iter": 200, "l2_regularization": 0.0},
        {"learning_rate": 0.05, "max_leaf_nodes": 31, "max_iter": 300, "l2_regularization": 1.0},
        {"learning_rate": 0.05, "max_leaf_nodes": 63, "max_iter": 400, "l2_regularization": 1.0},
    )

    # MLflow: almacen local de P1 y experimento del equipo (mismo esquema que CONFIG_P3)
    experimento: str = "mibici-equipo29"
    equipo: str = "29"
    tracking_uri: str = f"sqlite:///{PROJ_ROOT / 'mlflow.db'}"
    artefactos_dir: Path = PROJ_ROOT / "mlartifacts"


CONFIG_P1 = P1Config()
