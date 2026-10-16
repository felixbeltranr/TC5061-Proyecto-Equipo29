# MiBici

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Diseñar, como equipo, una propuesta de solución de Machine Learning para un conjunto de preguntas sobre el sistema de bicicleta pública MiBici, que combine modelos de regresión, clasificación y clustering.

## Descarga de datos abiertos

Para descargar los CSV mensuales publicados en [Datos abiertos MiBici](https://mibici.net/es/datos-abiertos/) y cargar cada archivo como un DataFrame independiente:

```python
from mibici.dataset import (
    download_and_load_datasets,
    download_and_load_latest_nomenclature,
    merge_dataset_nomenclature,
)

datasets = download_and_load_datasets()
enero_2024 = datasets["datos_abiertos_2024_01"]
nomenclature = download_and_load_latest_nomenclature()
all_trips = merge_dataset_nomenclature(datasets, nomenclature)

# Opcional: reemplaza el rango configurado para esta llamada
datasets = download_and_load_datasets(
    start_year=2023,
    start_month=12,
    end_year=2024,
    end_month=1,
)
```

El rango inclusivo predeterminado se define en `mibici/config.py` mediante `DATA_START_YEAR`, `DATA_START_MONTH`, `DATA_END_YEAR` y `DATA_END_MONTH`. Actualmente cubre enero-diciembre de 2024. Los archivos originales se guardan en `data/raw/`. `datasets` es un diccionario cuyas claves son los nombres de archivo sin `.csv` y cuyos valores son los DataFrames mensuales. `nomenclature` contiene el DataFrame separado de la nomenclatura más reciente publicada; su CSV se guarda también en `data/raw/` con su nombre original. La función busca el año y mes más recientes enlazados en la página, independientemente del rango de viajes configurado. `merge_dataset_nomenclature` concatena los DataFrames mensuales y devuelve un único DataFrame con las columnas de nombre, latitud, longitud y capacidad para origen y destino; conserva también `Origen_Id` y `Destino_Id`. También se puede ejecutar `python -m mibici.dataset` para descargar el rango configurado, la nomenclatura más reciente, combinarlos y registrar sus dimensiones.

Para reemplazar el rango desde la línea de comandos, especifica los cuatro valores:

```powershell
python -m mibici.dataset --start-year 2024 --start-month 12 --end-year 2025 --end-month 1
```

El rango incluye los meses inicial y final. Si omites las opciones, se usa el rango de `mibici/config.py`. Debes proporcionar los cuatro valores juntos; los meses deben estar entre 1 y 12 y el inicio no puede ser posterior al final.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mibici and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mibici   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mibici a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------
