# MiBici · Equipo 29

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Diseñar, como equipo, una propuesta de solución de Machine Learning para un conjunto de preguntas sobre el sistema de bicicleta pública MiBici, que combine modelos de regresión, clasificación y clustering.

## Preguntas del proyecto

| # | Pregunta de negocio | Tarea | Planteamiento del equipo | Código |
| --- | --- | --- | --- | --- |
| P1 | ¿Cuántos viajes se iniciarán en una estación o zona en un periodo futuro? | Regresión | Salidas de cada estación el día D, pronosticadas a las 23:00 de D-1 (estación-día, horizonte de un día). | `mibici/features.py`, `mibici/modeling/train_p1.py` |
| P2 | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles en un periodo futuro? | Clasificación | Riesgo por estación y hora, aproximado con la ocupación estimada a partir de los viajes y la capacidad. | `mibici/features_p2.py`, `mibici/modeling/train_p2.py` |
| P3 | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad? | Clustering | Grupos de estaciones con perfiles de uso parecidos y su ubicación. | `mibici/features_p3.py`, `mibici/modeling/train_p3.py` |

Los ML Canvas de cada pregunta están en `references/`.

## Datos

- Fuente: [Datos abiertos MiBici](https://mibici.net/es/datos-abiertos/): viajes mensuales (`datos_abiertos_AAAA_MM.csv`) y nomenclatura de estaciones (`nomenclatura_AAAA_MM.csv`). Las condiciones de uso son las publicadas en ese portal.
- Rango acordado: enero de 2025 a septiembre de 2026 (21 meses) y la nomenclatura 2026-09; 7,894,097 viajes.
- Versión: `references/manifiesto_datos_p2.json` guarda el SHA256 y el número de registros de cada archivo; su campo `version` (`66dd3d190bba`) se registra en cada run de MLflow como `data_version`.
- Privacidad: `data/` no se versiona. Cada pregunta lee sólo las columnas que necesita (P1: `Inicio_del_viaje` y `Origen_Id`) y en MLflow sólo se registran métricas agregadas, nunca columnas de usuario.

## Instalación

Requiere Python 3.14 (`requires-python = "~=3.14.0"`). Desde la raíz del repositorio:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; en Linux o macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` fija las versiones de todas las dependencias e instala el paquete `mibici` en modo editable.

## Descarga de datos abiertos

Para descargar los CSV mensuales publicados en [Datos abiertos MiBici](https://mibici.net/es/datos-abiertos/) y cargar cada archivo como un DataFrame independiente:

```python
from mibici.dataset import (
    download_and_load_datasets,
    download_and_load_latest_nomenclature,
    merge_dataset_define_nomenclature,
)

datasets = download_and_load_datasets()
enero_2025 = datasets["datos_abiertos_2025_01"]
nomenclature = download_and_load_latest_nomenclature()
all_trips = merge_dataset_define_nomenclature(datasets, nomenclature)

# Opcional: reemplaza el rango configurado para esta llamada
datasets = download_and_load_datasets(
    start_year=2023,
    start_month=12,
    end_year=2024,
    end_month=1,
)
```

El rango inclusivo predeterminado se define en `CONFIG` dentro de `mibici/config.py` mediante `download_data_start_year`, `download_data_start_month`, `download_data_end_year` y `download_data_end_month`. Actualmente cubre de enero de 2025 a septiembre de 2026, el rango acordado por el equipo. Los archivos originales se guardan en `data/raw/`. `datasets` es un diccionario cuyas claves son los nombres de archivo sin `.csv` y cuyos valores son los DataFrames mensuales. `nomenclature` contiene el DataFrame separado de la nomenclatura más reciente publicada; su CSV se guarda también en `data/raw/` con su nombre original. La función busca el año y mes más recientes enlazados en la página, independientemente del rango de viajes configurado. Al cargar cada CSV se normalizan los nombres de columnas (espacios, comillas y acentos) para que coincidan entre meses. `merge_dataset_define_nomenclature` concatena los DataFrames mensuales, convierte `Inicio_del_viaje` y `Fin_del_viaje` a fechas y devuelve un único DataFrame con las columnas de nombre, latitud, longitud y capacidad para origen y destino; conserva también `Origen_Id` y `Destino_Id`. También se puede ejecutar `python -m mibici.dataset` para descargar el rango configurado, la nomenclatura más reciente, combinarlos, registrar sus dimensiones y guardar el resultado en `data/processed/combined_dataset.parquet` (la carpeta se crea si no existe).

Para reemplazar el rango desde la línea de comandos, especifica los cuatro valores:

```powershell
python -m mibici.dataset --start-year 2024 --start-month 12 --end-year 2025 --end-month 1
```

El rango incluye los meses inicial y final. Si omites las opciones, se usa el rango de `mibici/config.py`. Debes proporcionar los cuatro valores juntos; los meses deben estar entre 1 y 12 y el inicio no puede ser posterior al final.

`combined_dataset.parquet` es la entrada común de las tres preguntas: conserva todos los viajes y sus identificadores (`Viaje_Id`, `Origen_Id`, `Destino_Id`). Cada pregunta aplica después sus propios filtros.

## Configuración

`mibici/config.py` reúne los parámetros de los experimentos:

| Bloque | Contenido |
| --- | --- |
| `CONFIG` | Carpetas de datos, nombre del parquet combinado y rango de meses de la descarga. |
| `CONFIG_P1` | Ruta del panel y del manifiesto, hora de emisión y retraso del conteo, cortes temporales, semilla, rejillas de Poisson y HistGradientBoosting, experimento y almacén de MLflow. |
| `CONFIG_P3` | Fechas, franjas, umbrales, semilla, valores de k y MLflow de P3. |

Se quedan en su módulo las constantes internas que forman parte de la definición del código: columnas que lee cada pregunta, nombres de variables, rezagos y ventanas de P1 (los nombres `lag_7`, `media_28d`, ... dependen de ellos) y el umbral de fechas ilegibles al cargar. Todos los parámetros que cambian un run (rango de fechas, versión de datos, semilla e hiperparámetros) se registran además en MLflow.

## Experimento base

### P1 · demanda diaria por estación

```bash
python -m mibici.dataset
python -m mibici.features
python -m mibici.modeling.train_p1 --autor "Nombre Apellido"
```

1. `mibici.dataset` descarga los 21 meses y la nomenclatura y guarda `data/processed/combined_dataset.parquet`.
2. `mibici.features` construye el panel estación-día de salidas en `data/interim/p1_panel_diario.parquet` y su resumen `p1_panel_diario.json`.
3. `mibici.modeling.train_p1` entrena los baselines (semana previa y persistencia), la regresión de Poisson regularizada y HistGradientBoosting con pérdida de Poisson; registra cuatro runs en MLflow y guarda sus figuras y `p1_resultados.json` en `reports/figures/`.

Partición por fecha objetivo: FIT hasta 2026-02-27, VAL de 2026-03-01 a 2026-04-29, entrenamiento final con las filas elegibles hasta 2026-04-29 y TEST de 2026-05-01 a 2026-09-30. Las variables usan información hasta D-2 porque el día D-1 aún no cierra al emitir el pronóstico. Los cortes, la semilla (29), las rejillas de hiperparámetros y las rutas de P1 están en `CONFIG_P1` de `mibici/config.py`.

### P2 · riesgo de estación

```bash
python -m mibici.features_p2
python -m mibici.modeling.train_p2
```

### P3 · grupos de estaciones

```bash
python -m mibici.features_p3
python -m mibici.modeling.train_p3
```

1. `mibici.features_p3` lee del parquet combinado sólo las cinco columnas que usa P3, aplica su limpieza mínima y guarda los perfiles de uso de entrenamiento y evaluación en `data/interim`.
2. `mibici.modeling.train_p3` entrena el baseline KMeans geográfico con k = 4, 5 y 6 (`--k` cambia los valores) y registra los runs en MLflow.

Sus fechas, franjas, umbrales y semilla están en `CONFIG_P3` de `mibici/config.py`.

### Pruebas, lint y formato

```bash
python -m pytest
ruff check .
ruff format --check .
```

La configuración de Ruff en `pyproject.toml` revisa `mibici/`.

## Seguimiento en MLflow

Todos los runs usan el experimento `mibici-equipo29`, con nombre descriptivo y etiquetas `question` (P1, P2 o P3), `team`, `author`, `data_range`, `seed`, `data_version`, `git_commit` y `git_dirty`. Las métricas indican el conjunto en que se calculan (`val_mae`, `test_mae`, ...). El almacén es local: `mlflow.db` y `mlartifacts/` en la raíz del repositorio, ambos fuera de git. Para abrir la interfaz:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Usar el mismo nombre de experimento en otro archivo SQLite no junta las corridas: para compararlas deben registrarse en el mismo almacén.

## Flujo de trabajo con Git

Seguimos GitHub Flow:

- `main` está protegida: todo cambio entra por Pull Request revisado por otro integrante.
- Ramas: `<versión>_<iniciales>-<descripción>`, por ejemplo `2.0_cg-p2-clasificacion` o `4.0_ab-p1-regresion`.
- Commits: [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/) con la forma `tipo(alcance): descripción` (`feat`, `fix`, `docs`, `test`, `refactor`, `build`, `style`).
- Cada integrante abre al menos un Pull Request y revisa al menos uno de otro integrante.
- No se versionan datos de MiBici, `mlruns/`, `mlartifacts/`, `mlflow.db`, archivos `.env` ni modelos serializados (`.gitignore`).
- Los notebooks son sólo de exploración y siguen la convención `número-iniciales-descripción.ipynb`; la lógica final vive en el módulo `mibici`.

## Integrantes

| Integrante | Usuario de GitHub | Rol |
| --- | --- | --- |
| Ulises Orozco | xAlucardzx | Ingesta, preparación de datos y exploración inicial |
| Alan Alfonso Boldo Reyes | a01840556 | P1: regresión de demanda diaria por estación |
| Carlos Galván | carloshgalvan95 | P2: clasificación de riesgo por estación |
| Félix Beltrán | felixbeltranr | P3: clustering de estaciones; estructura inicial con ccds |

## Organización del proyecto

```
├── Makefile           <- Comandos de conveniencia
├── README.md
├── data               <- No se versiona
│   ├── external
│   ├── interim        <- Paneles intermedios (p. ej. p1_panel_diario.parquet)
│   ├── processed      <- combined_dataset.parquet
│   └── raw            <- CSV originales de MiBici
├── docs
├── models             <- Modelos serializados (fuera de git)
├── notebooks          <- Exploración: número-iniciales-descripción.ipynb
├── pyproject.toml     <- Metadatos del paquete y configuración de Ruff
├── references         <- ML Canvas y manifiesto de datos
├── reports
│   └── figures        <- Figuras y resultados del experimento base
├── requirements.txt   <- Dependencias con versión fija
├── tests              <- Pruebas con pytest
└── mibici             <- Código del proyecto
    ├── config.py      <- Rutas, rango de datos, cortes y semillas
    ├── dataset.py     <- Ingesta: descarga, normalización y parquet combinado
    ├── features.py    <- P1: panel estación-día y variables
    ├── features_p2.py <- P2: panel horario y etiqueta de riesgo
    ├── features_p3.py <- P3: perfiles de uso por estación
    ├── manifiesto_p2.py <- Manifiesto con SHA256 y registros de cada archivo
    ├── tracking.py    <- Convención de MLflow de P1
    ├── plots.py
    └── modeling
        ├── train_p1.py
        ├── train_p2.py
        ├── train_p3.py
        ├── train.py
        └── predict.py
```
