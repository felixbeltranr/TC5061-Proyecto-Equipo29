from dataclasses import dataclass
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
    output_filename: str = "combined_dataset.csv"

    models_dir: Path = PROJ_ROOT / "models"

    reports_dir: Path = PROJ_ROOT / "reports"

    target_column: str = "ToDo"

    # MiBici open-data download range (inclusive)
    download_data_start_year: int = 2014
    download_data_start_month: int = 1
    download_data_end_year: int = 2026
    download_data_end_month: int = 12

    train_size: float = 0.70
    validation_size: float = 0.15
    test_size: float = 0.15

    mlflow_tracking_uri: str = "http://127.0.0.1:5000"

CONFIG = ProjectConfig()

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
