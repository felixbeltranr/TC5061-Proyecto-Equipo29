from __future__ import annotations

from collections.abc import Iterable
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

from loguru import logger
import pandas as pd
import typer

from mibici.config import CONFIG

DATA_PAGE_URL = "https://mibici.net/es/datos-abiertos/"
DATASET_FILENAME_RE = re.compile(r"^datos_abiertos_(\d{4})_(\d{2})\.csv$", re.IGNORECASE)
NOMENCLATURE_FILENAME_RE = re.compile(r"^nomenclatura_(\d{4})_(\d{2})\.csv$", re.IGNORECASE)
DOWNLOAD_TIMEOUT_SECONDS = 60
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MiBiciDataDownloader/0.1)"}

app = typer.Typer()


class _DatasetLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href is not None:
                self.hrefs.append(href)


def _dataset_sort_key(url: str) -> tuple[int, int]:
    filename = Path(urlsplit(url).path).name
    match = DATASET_FILENAME_RE.fullmatch(filename)
    if match is None:
        raise ValueError(f"Not a MiBici monthly dataset URL: {url}")
    year, month = int(match.group(1)), int(match.group(2))
    if not 1 <= month <= 12:
        raise ValueError(f"Not a valid monthly MiBici dataset URL: {url}")
    return year, month


def _nomenclature_sort_key(url: str) -> tuple[int, int]:
    filename = Path(urlsplit(url).path).name
    match = NOMENCLATURE_FILENAME_RE.fullmatch(filename)
    if match is None:
        raise ValueError(f"Not a MiBici nomenclature URL: {url}")
    year, month = int(match.group(1)), int(match.group(2))
    if not 1 <= month <= 12:
        raise ValueError(f"Not a valid MiBici nomenclature URL: {url}")
    return year, month


def _discover_page_links(page_url: str) -> list[str]:
    request = Request(page_url, headers=REQUEST_HEADERS)
    with urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        page_html = response.read().decode(charset)

    parser = _DatasetLinkParser()
    parser.feed(page_html)
    return [urljoin(page_url, href) for href in parser.hrefs]


def discover_dataset_urls(page_url: str = DATA_PAGE_URL) -> list[str]:
    """Find monthly CSV download links on the MiBici open-data page."""

    urls_by_filename: dict[str, str] = {}
    for url in _discover_page_links(page_url):
        filename = Path(urlsplit(url).path).name
        match = DATASET_FILENAME_RE.fullmatch(filename)
        if match is None:
            continue
        month = int(match.group(2))
        if not 1 <= month <= 12:
            continue
        urls_by_filename.setdefault(filename, url)

    if not urls_by_filename:
        raise ValueError(f"No monthly MiBici CSV links found on {page_url}")

    return sorted(urls_by_filename.values(), key=_dataset_sort_key)


def discover_latest_nomenclature_url(page_url: str = DATA_PAGE_URL) -> str:
    """Find the most recent nomenclature CSV linked on the MiBici open-data page."""
    nomenclature_urls: list[str] = []
    for url in _discover_page_links(page_url):
        filename = Path(urlsplit(url).path).name
        match = NOMENCLATURE_FILENAME_RE.fullmatch(filename)
        if match is not None and 1 <= int(match.group(2)) <= 12:
            nomenclature_urls.append(url)

    if not nomenclature_urls:
        raise ValueError(f"No monthly MiBici nomenclature CSV found on {page_url}")

    return max(nomenclature_urls, key=_nomenclature_sort_key)


def _filter_dataset_urls(
    dataset_urls: list[str],
    start_year: int | None,
    start_month: int | None,
    end_year: int | None,
    end_month: int | None,
) -> list[str]:
    if all(value is None for value in (start_year, start_month, end_year, end_month)):
        return dataset_urls
    if start_year is None or start_month is None or end_year is None or end_month is None:
        raise ValueError(
            "Provide all four values: start_year, start_month, end_year, and end_month"
        )
    if not 1 <= start_year <= 9999 or not 1 <= end_year <= 9999:
        raise ValueError("Start and end years must be between 1 and 9999")
    if not 1 <= start_month <= 12 or not 1 <= end_month <= 12:
        raise ValueError("Start and end months must be between 1 and 12")

    start = (start_year, start_month)
    end = (end_year, end_month)
    if start > end:
        raise ValueError("The start year and month must be before or equal to the end")

    filtered_urls = [url for url in dataset_urls if start <= _dataset_sort_key(url) <= end]
    if not filtered_urls:
        raise ValueError(
            f"No MiBici datasets found between {start_year}-{start_month:02} "
            f"and {end_year}-{end_month:02}"
        )
    return filtered_urls


def _download_file(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlsplit(url).path).name
    destination = output_dir / filename

    temporary_path: Path | None = None
    try:
        request = Request(url, headers=REQUEST_HEADERS)
        with (
            urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
            tempfile.NamedTemporaryFile(
                dir=output_dir,
                prefix=f".{filename}.",
                suffix=".part",
                delete=False,
            ) as temporary_file,
        ):
            temporary_path = Path(temporary_file.name)
            shutil.copyfileobj(response, temporary_file)
        if temporary_path is None:
            raise RuntimeError("Temporary download file was not created")
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return destination


def download_datasets(
    dataset_urls: Iterable[str], output_dir: Path = CONFIG.raw_data_dir
) -> list[Path]:
    """Download the given monthly CSV URLs and return their local paths."""
    paths = []
    for url in dataset_urls:
        _dataset_sort_key(url)
        paths.append(_download_file(url, output_dir))
    return paths


def _read_dataset(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252")


def download_and_load_latest_nomenclature(
    page_url: str = DATA_PAGE_URL, output_dir: Path = CONFIG.raw_data_dir
) -> pd.DataFrame:
    """Download and load the latest nomenclature CSV as a separate DataFrame."""
    url = discover_latest_nomenclature_url(page_url)
    path = _download_file(url, output_dir)
    return _read_dataset(path)


def load_datasets(data_dir: Path = CONFIG.raw_data_dir) -> dict[str, pd.DataFrame]:
    """Load each monthly CSV in a directory into a DataFrame keyed by its filename stem."""
    paths = sorted(
        (
            path
            for path in data_dir.glob("*.csv")
            if DATASET_FILENAME_RE.fullmatch(path.name) is not None
            and 1 <= int(path.stem[-2:]) <= 12
        ),
        key=lambda path: _dataset_sort_key(path.name),
    )
    if not paths:
        raise FileNotFoundError(f"No monthly MiBici CSV files found in {data_dir}")
    return {path.stem: _read_dataset(path) for path in paths}


def download_and_load_datasets(
    page_url: str = DATA_PAGE_URL,
    output_dir: Path = CONFIG.raw_data_dir,
    start_year: int | None = CONFIG.download_data_start_year,
    start_month: int | None = CONFIG.download_data_start_month,
    end_year: int | None = CONFIG.download_data_end_year,
    end_month: int | None = CONFIG.download_data_end_month,
) -> dict[str, pd.DataFrame]:
    """Download monthly datasets in an inclusive optional date range and load each."""
    dataset_urls = _filter_dataset_urls(
        discover_dataset_urls(page_url),
        start_year,
        start_month,
        end_year,
        end_month,
    )
    paths = download_datasets(dataset_urls, output_dir)
    return {path.stem: _read_dataset(path) for path in paths}


def merge_dataset_define_nomenclature(
    datasets: dict[str, pd.DataFrame], nomenclature: pd.DataFrame
) -> pd.DataFrame:
    """Combine monthly trips and add station details for their origins and destinations."""
    if not datasets:
        raise ValueError("At least one monthly dataset is required")

    trips = pd.concat(datasets.values(), ignore_index=True)
    required_trip_columns = {"Origen_Id", "Destino_Id"}
    missing_trip_columns = required_trip_columns.difference(trips.columns)
    if missing_trip_columns:
        raise ValueError(
            f"Trip datasets are missing required columns: {sorted(missing_trip_columns)}"
        )

    station_columns = {"id", "name", "latitude", "longitude", "dpcapacity"}
    missing_station_columns = station_columns.difference(nomenclature.columns)
    if missing_station_columns:
        raise ValueError(
            f"Nomenclature is missing required columns: {sorted(missing_station_columns)}"
        )

    trips["Origen_Id"] = pd.to_numeric(trips["Origen_Id"], errors="coerce").astype("Int64")
    trips["Destino_Id"] = pd.to_numeric(trips["Destino_Id"], errors="coerce").astype("Int64")
    stations = nomenclature.loc[:, ["id", "name", "latitude", "longitude", "dpcapacity"]].copy()
    stations["id"] = pd.to_numeric(stations["id"], errors="coerce").astype("Int64")

    for role, trip_id_column in (("origen", "Origen_Id"), ("destino", "Destino_Id")):
        station_lookup = stations.rename(
            columns={
                "id": trip_id_column,
                "name": f"{role}_nombre",
                "latitude": f"{role}_latitud",
                "longitude": f"{role}_longitud",
                "dpcapacity": f"{role}_dpcapacidad",
            }
        )
        trips = trips.merge(
            station_lookup,
            on=trip_id_column,
            how="left",
            validate="many_to_one",
        )

    return trips


@app.command()
def main(
    page_url: str = DATA_PAGE_URL,
    output_raw_dir: Path = CONFIG.raw_data_dir,
    output_processed_dir: Path = CONFIG.processed_data_dir,
    start_year: int | None = CONFIG.download_data_start_year,
    start_month: int | None = CONFIG.download_data_start_month,
    end_year: int | None = CONFIG.download_data_end_year,
    end_month: int | None = CONFIG.download_data_end_month,
) -> pd.DataFrame:
    datasets = download_and_load_datasets(
        page_url,
        output_raw_dir,
        start_year,
        start_month,
        end_year,
        end_month,
    )
    for name, dataframe in datasets.items():
        logger.info(
            "Loaded {}: {} rows, {} columns",
            name,
            len(dataframe),
            len(dataframe.columns),
        )
    nomenclature = download_and_load_latest_nomenclature(page_url, output_raw_dir)
    logger.info(
        "Loaded latest station nomenclature: {} rows, {} columns",
        len(nomenclature),
        len(nomenclature.columns),
    )

    combined_dataset = merge_dataset_define_nomenclature(datasets, nomenclature)
    logger.info(
        "Combined trips with station nomenclature: {} rows, {} columns",
        len(combined_dataset),
        len(combined_dataset.columns),
    )

    # data/ no se versiona, en un clon limpio la carpeta no existe
    output_processed_dir.mkdir(parents=True, exist_ok=True)
    combined_dataset.to_csv(output_processed_dir / CONFIG.output_filename, index=False)
    logger.info(
        "Saved combined dataset to {}",
        output_processed_dir / CONFIG.output_filename,
    )

    return combined_dataset


if __name__ == "__main__":
    app()
