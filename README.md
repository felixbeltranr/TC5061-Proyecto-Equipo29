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
- Versión: `references/manifiesto_datos_p2.json` guarda el SHA256 y el número de registros de cada archivo; su campo `version` (`66dd3d190bba`) se registra en cada run de MLflow como `data_version`. Para regenerarlo después de la descarga: `python -m mibici.manifiesto_p2`.
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

## MLflow Project

El `MLproject` de la raíz define dos entry points. El entorno usa las dependencias de `requirements.txt`; los defaults de rutas, fechas, semillas y parámetros del modelo se leen desde `mibici/config.py`:

```powershell
$raiz = (Resolve-Path .).Path.Replace('\', '/')
$env:MIBICI_TRACKING_URI = "sqlite:///$raiz/mlflow.db"
$env:MLFLOW_TRACKING_URI = $env:MIBICI_TRACKING_URI
mlflow run . -e ingest --env-manager local
mlflow run . -e train -P question=p1 -P model=auto -P 'hyperparameters={}' --env-manager local
mlflow run . -e train -P question=p2 -P model=hist_gradient_boosting -P hyperparameters=max_iter=300,max_leaf_nodes=63 --env-manager local
mlflow run . -e train -P question=p3 -P model=seleccion --env-manager local
```

Ejecuta `ingest` antes de `train`: descarga y combina los datos y después genera el manifiesto común de versión en `references/manifiesto_datos_p2.json`. El entry point `train` genera las features del problema indicado (`p1`, `p2` o `p3`) y después llama su entrenador configurado. Los modelos disponibles actualmente son P1 `auto`, `poisson`, `hgb_poisson` y `hgb_absoluto`; P2 `hist_gradient_boosting`; y P3 `kmeans_geo`, `kmeans_uso`, `gmm_uso` y `seleccion` (flujo completo, ver la sección de P3). `auto` usa la selección configurada para P1. `hyperparameters` acepta pares separados por coma: P1 Poisson admite `alpha`; P1 HGB Poisson, HGB absoluto y P2 admiten `learning_rate`, `max_leaf_nodes`, `max_iter` y `l2_regularization`; P3 admite `k` y `n_init` (`kmeans_geo` y `kmeans_uso`), `n_components`, `covariance_type` y `n_init` (`gmm_uso`), o `buscar=1` para la búsqueda con runs anidados. Por ejemplo, Random Forest y `n_estimators` no son compatibles como opciones individuales del entry point.

Los ejemplos incluyen `--env-manager local` para reutilizar el entorno virtual activo. Si no vas a usar un entorno virtual ya activado y quieres que MLflow gestione el entorno del proyecto, puedes omitir esa opción. La corrida de verificación del cierre se hizo con `--env-manager local`, en un entorno existente con Python 3.14.8 y las versiones fijas de `requirements.txt`. Sin esa opción, MLflow crea el entorno de `python_env.yaml` con virtualenv y pyenv; ese camino no se probó.

La sensibilidad de P2 a la ocupación inicial (niveles 0.3 y 0.7, además de 0.5) no está expuesta como parámetro del `MLproject`: `train` con `question=p2` o `question=all` solo corre el nivel 0.5. Esa comparación se corre con `python -m mibici.modeling.train_p2 --niveles 0.5 0.3 0.7`. Si el almacén lo creó el servidor de tracking, agrega `--tracking-uri http://127.0.0.1:5000` con el servidor encendido, porque sus artefactos usan direcciones `mlflow-artifacts:/` que solo resuelve el servidor.


El almacén del proyecto se elige con `MIBICI_TRACKING_URI`; si no está definida, se usa una URI SQLite absoluta a `mlflow.db` en la raíz del repositorio. Las rutas SQLite relativas de esa variable se resuelven contra la raíz del repositorio, independientemente del directorio desde el que se invoque. MLflow inyecta `MLFLOW_TRACKING_URI` en sus subprocesos incluso sin una elección explícita del usuario; por ello su presencia no sustituye esta configuración del proyecto.

Antes de `mlflow run`, desde la raíz del repositorio, configura ambas variables con la misma URI absoluta para que el run exterior y los entrenamientos usen el mismo almacén:

```powershell
$raiz = (Resolve-Path .).Path.Replace('\', '/')
$env:MIBICI_TRACKING_URI = "sqlite:///$raiz/mlflow.db"
$env:MLFLOW_TRACKING_URI = $env:MIBICI_TRACKING_URI
mlflow run . -e train -P question=all -P 'hyperparameters={}' --env-manager local
```

`question=all` ejecuta P1 con `auto`, P2 con su modelo configurado y P3 con `seleccion`; no acepta un modelo ni hiperparámetros compartidos. El entry point comprueba la URI inyectada del run exterior antes de entrenar: si difiere del almacén elegido o sigue siendo una ruta SQLite relativa ambigua, detiene la ejecución y pide configurar ambas variables antes de relanzar. Cambiar el almacén dentro del subproceso no puede trasladar un run exterior ya creado. En una invocación directa sin run exterior, el almacén del proyecto prevalece sobre un `MLFLOW_TRACKING_URI` incidental. Al terminar, el despachador restaura el entorno previo.

`combined_dataset.parquet` es la entrada común de las tres preguntas: conserva todos los viajes y sus identificadores (`Viaje_Id`, `Origen_Id`, `Destino_Id`). Cada pregunta aplica después sus propios filtros.

## Configuración

`mibici/config.py` reúne los parámetros de los experimentos:

| Bloque | Contenido |
| --- | --- |
| `CONFIG` | Carpetas de datos, nombre del parquet combinado y rango de meses de la descarga. |
| `CONFIG_P1` | Ruta del panel y del manifiesto, hora de emisión y retraso del conteo, cortes temporales, semilla, rejillas de Poisson y HistGradientBoosting, experimento y almacén de MLflow. |
| `CONFIG_P2` | Cortes temporales, umbrales de riesgo, horizonte, limpieza de viajes, semilla, rejilla de HistGradientBoosting, experimento y almacén de MLflow de P2. |
| `CONFIG_P3` | Fechas, franjas, umbrales, semilla, valores de k y MLflow de P3. |

Se quedan en su módulo las constantes internas que forman parte de la definición del código: columnas que lee cada pregunta, nombres de variables, rezagos y ventanas de P1 (los nombres `lag_7`, `media_28d`, ... dependen de ellos) y el umbral de fechas ilegibles al cargar. Todos los parámetros que cambian un run (rango de fechas, versión de datos, semilla e hiperparámetros) se registran además en MLflow.

### Transformaciones compartidas

`mibici/features_pipeline.py` centraliza la construcción de pipelines tabulares de scikit-learn y un helper para ajustar transformadores exclusivamente con el conjunto de entrenamiento. No centraliza la creación de variables ni impone las mismas transformaciones a todas las preguntas: cada tarea conserva sus columnas, etiquetas y decisiones de modelado.

- P1 usa el constructor compartido para la regresión de Poisson: aplica `log1p` y escalado a los conteos, codifica el día de la semana y deja pasar las variables binarias. HistGradientBoosting recibe sus features directamente, pues maneja el día como categórico y no necesita esa transformación.
- P2 lo usa para la regresión logística (imputación y escalado numérico, one-hot para hora y día) y Random Forest (imputación). HistGradientBoosting continúa recibiendo las variables originales, ya que no requiere escalado y maneja valores faltantes.
- P3 usa el helper de ajuste train-only para escalar el espacio de uso que alimenta las métricas. El baseline geográfico usa coordenadas en kilómetros; los modelos de uso conservan sus pipelines de selección por VIF, escalado y clustering sobre variables de uso.

Al añadir un modelo, elige transformaciones según el algoritmo y el significado de sus features. Ajusta imputadores, escaladores y codificadores solo con `fit`/`fit_transform` sobre train; usa luego `transform` en validación, prueba y producción. Mantén juntas las transformaciones y el estimador en un `Pipeline` cuando formen parte de la predicción, para evitar diferencias entre entrenamiento e inferencia. No escales por defecto modelos de árboles ni apliques transformaciones pensadas para otro problema.

## Experimento base

### P1 · demanda diaria por estación

```powershell
mlflow run . -e ingest --env-manager local
mlflow run . -e train -P question=p1 -P model=auto -P 'hyperparameters={}' --env-manager local
```

1. `mibici.dataset` descarga los 21 meses y la nomenclatura y guarda `data/processed/combined_dataset.parquet`.
2. `mibici.features` construye el panel estación-día de salidas en `data/interim/p1_panel_diario.parquet` y su resumen `p1_panel_diario.json`.
3. `mibici.modeling.train_p1` entrena los baselines (semana previa y persistencia) y tres candidatas: la regresión de Poisson regularizada (`poisson`), HistGradientBoosting con pérdida de Poisson (`hgb_poisson`) y HistGradientBoosting con pérdida absoluta (`hgb_absoluto`). Las dos de HistGradientBoosting recorren la misma rejilla, así que entre ellas sólo cambia la pérdida: la de Poisson estima la media de un conteo y la absoluta la mediana, que es lo que minimiza el MAE con el que se elige. La búsqueda registra un run padre por candidata y un hijo anidado por cada combinación de hiperparámetros; después registra las candidatas finales y sus artefactos. Con `mlflow run`, `-P model=` también acepta `hgb_absoluto`, con los mismos hiperparámetros que `hgb_poisson`.

Partición por fecha objetivo: FIT hasta 2026-02-27, VAL de 2026-03-01 a 2026-04-29, entrenamiento final con las filas elegibles hasta 2026-04-29 y TEST de 2026-05-01 a 2026-09-30. Las variables usan información hasta D-2 porque el día D-1 aún no cierra al emitir el pronóstico. Los cortes, la semilla (29), las rejillas de hiperparámetros y las rutas de P1 están en `CONFIG_P1` de `mibici/config.py`.

Cada hiperparámetro se elige por menor `val_mae`, sin consultar prueba. La selección parte de la baseline de semana previa y sólo la reemplaza una candidata que también cumpla la política de dominio descrita abajo; si dos empatan, se queda la primera en el orden `poisson`, `hgb_poisson`, `hgb_absoluto`. Persistencia se mide y se registra como baseline, pero no es elegible para sustituir a semana previa bajo la política vigente. Cambiar esa elegibilidad requiere una decisión del equipo, no sólo una diferencia numérica observada en prueba.

#### Politica de dominio para esta iteracion

Antes de comparar MAE, una candidata solo es elegible si todas sus predicciones de validacion son numericas, finitas y mayores o iguales a cero, como corresponde a una demanda de viajes. La seleccion conserva la regla de menor `val_mae` y no consulta prueba. La candidata que no cumpla el dominio conserva su run, metricas y artefactos; no se recorta su salida ni se ajusta de nuevo.

Esta regla se introdujo despues de la auditoria local en la que ya se habian observado resultados de prueba. Por tanto, esta iteracion no se presenta como una evaluacion con prueba intacta. La regla se aplica exclusivamente con validacion y no autoriza a retocar modelos o hiperparametros usando prueba.

Cada candidata registra `val_pred_min` cuando existe al menos una prediccion finita, los conteos de negativos y no finitos y `val_domain_eligible`, ademas de etiquetas con el motivo de elegibilidad. Las tres candidatas siguen visibles y comparables aunque alguna sea inelegible. El humo de la version inmutable exige una prediccion numerica, finita y no negativa antes de mover el alias.

El Registry solo puede mover `champion` en una ejecucion `model=auto`, despues de comparar las candidatas. Una ejecucion individual registra evidencia pero no promueve. Si no hay candidatas elegibles, no se promueve ninguna y se conserva el campeon previo.

#### Tracking avanzado y Registry de P1

- Los runs llevan `question`, `team`, `author`, `data_range`, `seed`, `data_version`, `git_commit`, `git_dirty`, `model_family` y `stage_of_work`; esta última distingue `baseline`, `search` y `candidate`.
- Con `model=auto`, las rejillas configuradas producen cinco hijos de búsqueda para Poisson y tres para cada HistGradientBoosting (once en total), y tres runs `candidate` con artefactos completos, uno por candidata.
- Cada candidato final guarda el modelo en formato skops mediante `log_model`, con firma y ejemplo de entrada. También registra muestras acotadas de entrenamiento, validación y prueba mediante `log_input`; se guardan esquema, perfil, digest, versión y número total de filas, no las tablas completas.
- Los artefactos se organizan en `plots/` (real contra pronóstico y, en cada candidata, residuales de prueba), `data/` (manifiesto, resumen y, cuando aplica, coeficientes) y `config/` (configuración y rejilla). `p1_resultados.json` conserva la comparación de baselines y candidatos.
- Diagnóstico de prueba del modelo ya elegido, sin intervenir en la selección ni en los hiperparámetros: cada run final registra `test_mae_AAAA_MM` de mayo a septiembre de 2026 y `test_mae_mensual_desv` (desviación del MAE entre meses), y cada candidata guarda `plots/p1_residuales_<candidata>.png` con los residuales (pronóstico menos real) contra el pronóstico y su distribución con el sesgo medio.
- El ganador elegible se registra como `mibici-equipo29-p1-demanda`. La versión incluye linaje al experimento, run de origen, commit, estado `git_dirty`, versión de datos y diagnóstico de dominio. Primero se carga la URI inmutable `models:/mibici-equipo29-p1-demanda/<version>` y se exige una predicción numérica, finita y no negativa; sólo entonces se mueve el alias `champion`. Si la carga o la predicción falla, la versión queda disponible para diagnóstico y el alias previo se conserva.
- El Registry sólo promueve el modelo elegido en una comparación `model=auto`. Si la política vigente seleccionara semana previa o no hubiera candidatas elegibles, el flujo no promueve silenciosamente otra candidata y conserva el alias previo.

Para cargar el campeón validado:

```python
import mlflow
import os

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
modelo = mlflow.pyfunc.load_model("models:/mibici-equipo29-p1-demanda@champion")
demanda = modelo.predict(entrada)  # una predicción numérica por fila estación-día
```

### P2 · riesgo de estación

P2 clasifica el riesgo horario de que una estación se quede sin bicicletas o sin anclajes. La ocupación se estima a partir del flujo neto acumulado de viajes y la capacidad de la estación; no proviene de mediciones directas de inventario. El entrenamiento compara baselines con `HistGradientBoostingClassifier` balanceado y evalúa macro F1, métricas por clase y estaciones nuevas.

```powershell
mlflow run . -e train -P question=p2 -P 'hyperparameters={}' --env-manager local
```

`features_p2` genera `data/interim/p2_panel_horario.parquet` y su resumen JSON. `train_p2` busca los hiperparámetros del HGB en la rejilla de `CONFIG_P2` como corridas anidadas (un run padre `p2_busqueda_<nivel>` y un hijo por combinación), entrena tres baselines y tres candidatas (regresión logística balanceada, HGB balanceado con la mejor combinación y RandomForest balanceado, con los parámetros de `CONFIG_P2`), registra los runs con las etiquetas `model_family` y `stage_of_work`, métricas de validación y de prueba, `fit_seconds` y `model_size_mb` y guarda matrices de confusión, importancia por permutación y `p2_resultados.json` en `reports/runs_p2/` (ignorada por git, configurable con `--figuras`; `reports/figures/` conserva las figuras del reporte). Los artefactos de MLflow van a `mlartifacts/` o a la carpeta de `--artefactos` si el experimento aún no existe. Los cortes temporales, umbrales de riesgo, nivel inicial de ocupación, semilla y rejilla están en `CONFIG_P2` de `mibici/config.py`.

#### Campeón de P2 en el Model Registry

Al terminar, `train_p2` registra **solo al campeón** en el modelo `mibici-equipo29-p2-riesgo` (una pregunta, un modelo registrado). Antes de mover `champion`, carga la nueva versión por su URI inmutable y exige una clase válida por fila; si falla, la versión queda marcada y el alias anterior se conserva. Después mantiene la comprobación final por alias. Cada versión nueva lleva su descripción y las etiquetas de linaje `run_id`, `git_commit`, `data_version`, `experimento`, `nivel_inicial`, `criterio_seleccion` y `candidata`. La opción `--no-registrar` omite el registro; por omisión está activo. El entry point propaga modelo, hiperparámetros y URI común; con `mlflow run` el registro está activo.

El campeón se elige con una regla explícita (`mibici/registro.py`, `seleccionar_campeon`): entre las candidatas balanceadas (`CONFIG_P2.candidatas_campeon`), la de mayor recall medio de `sin_bicicletas` y `sin_anclajes` en **validación** al primer nivel inicial de la corrida; si dos candidatas difieren en menos de 0.005 de ese recall se consideran empatadas y decide el PR AUC medio de las mismas clases. No se usa la prueba para elegir y tampoco el macro F1. En la corrida real con nivel inicial 0.5 el campeón es HistGradientBoosting balanceado (recall medio de validación 0.952 contra 0.950 de la regresión logística y 0.942 del RandomForest, con mayor PR AUC).

Limitaciones de la elección:

- Es un **supuesto operativo**: quedarse sin aviso cuando una estación se vacía o se llena cuesta más que una falsa alarma, por eso se prioriza el recall de las clases de riesgo. Si el supuesto cambia, la regla y el campeón pueden cambiar.
- Por **macro F1 de prueba la baseline de persistencia es mejor** que las tres candidatas en los tres niveles iniciales (con nivel 0.5, 0.804 contra 0.741 del HGB balanceado). Las candidatas balanceadas ganan en recall de riesgo (alrededor de 0.94 y 0.96 contra cerca de 0.70 de la persistencia) pero quedan por debajo en macro F1.
- Con 0.005 de margen el HGB gana en los tres niveles iniciales; sin margen, en el nivel 0.3 la regresión logística supera al HGB por 0.002 de recall medio de validación. La diferencia entre las dos es pequeña y no se probó que sea estadísticamente distinta.

Cómo cargarlo y qué necesita quien lo use:

```python
import mlflow

import os
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
modelo = mlflow.pyfunc.load_model("models:/mibici-equipo29-p2-riesgo@champion")
clases = modelo.predict(entrada)  # una clase por fila: segura, sin_bicicletas o sin_anclajes
```

- `entrada` es un `DataFrame` con las columnas de `features_p2.FEATURES`, en ese orden y como `float64` (la firma del modelo y `config/config_p2.json` del run las listan). `hora`, `dia_semana` y `fin_semana` son del instante que se pronostica, una hora después del corte.
- Las tres últimas (`est_tasa_sin_bicis`, `est_tasa_sin_anclajes` y `est_neto_medio_hora`) son estadísticas de entrenamiento por estación y por estación y hora. Se guardan en `data/stats_estacion.csv` y `data/stats_estacion_hora.csv` del mismo run y se unen al panel con `features_p2.unir_stats`. Una estación que no estaba en entrenamiento queda con esas columnas vacías, y los modelos las aceptan como valores faltantes.
- Para obtener probabilidades, `mlflow.sklearn.load_model` sobre el mismo URI y `predict_proba`; las columnas siguen el orden de `classes_` (alfabético).
- El modelo se guarda en formato skops, que solo carga tipos de scikit-learn y NumPy declarados como confiables al registrarlo.

Cada candidata guarda en su run `plots/` (matrices de confusión, curvas precision-recall de las dos clases de riesgo e importancia por permutación del HGB), `data/` (manifiesto de datos, resumen del panel y estadísticas de entrenamiento) y `config/` (configuración del run y rejilla de la búsqueda), además del modelo con firma y ejemplo de entrada. Train, validación y prueba se registran como entradas del run (`log_input`) con una muestra de 2000 filas y la versión de los datos; no se sube ninguna tabla de datos ni se versiona `mlflow.db`, `mlruns` ni `mlartifacts`.

Con MLflow Project, después de la ingesta y de generar el manifiesto, se pueden ejecutar las features y el entrenamiento con:

```powershell
mlflow run . -e train -P question=p2 --env-manager local
mlflow run . -e train -P question=p2 -P model=hist_gradient_boosting -P hyperparameters=max_iter=300,max_leaf_nodes=63 --env-manager local
```

### P3 · grupos de estaciones

P3 agrupa estaciones con perfiles de uso similares y los compara con su ubicación. `features_p3` lee el parquet combinado, construye los perfiles de uso por estación del periodo de entrenamiento (enero 2025 a abril 2026) y del de evaluación (mayo a septiembre 2026) y los guarda en `data/interim`. Todo lo que se elige (variables, hiperparámetros y campeón) usa solo el periodo de entrenamiento; el de evaluación solo se reporta.

Reproducir el modelo elegido desde un clon limpio, después de `ingest` y de generar el manifiesto:

```powershell
mlflow run . -e train -P question=p3 -P model=seleccion --env-manager local
```

Ese comando corre todo el flujo de `train_p3.seleccionar`:

1. **Baseline** `kmeans_geo`: KMeans sobre la ubicación (k = 5).
2. **Búsquedas con runs anidados** (un run padre `p3-busqueda-<modelo>` y un hijo por combinación): `kmeans_uso` con k de 3 a 8 y `gmm_uso` con 3 a 8 componentes y cuatro tipos de covarianza.
3. **Candidatas**: C1 KMeans de uso con k = 5 (referencia de la entrega 1), C2 el mejor KMeans de la búsqueda y C3 la mejor GaussianMixture, reentrenadas con todo el periodo de entrenamiento.
4. **Selección y registro** del campeón y del retador en el Model Registry.

Los modelos de uso (`mibici/modeling/modelos_p3.py`) son pipelines de scikit-learn que reciben las 12 variables de uso, eligen por VIF las que no son redundantes, las estandarizan y agrupan; el grupo 0 es el de menor uso.

**Validación por estaciones** (`busqueda_p3.py`): 5 particiones reproducibles de las estaciones de entrenamiento; en cada una se entrena con el 80 % y se valida con el 20 % reservado, con la selección por VIF y el escalado aprendidos solo con el 80 %. Las métricas `val_` son el promedio de las particiones (silueta, Davies-Bouldin y Calinski-Harabasz en el espacio de uso) y `val_ari_estabilidad` es el ARI promedio entre las asignaciones de los cinco modelos. Limitación: la validación separa estaciones, no periodos, así que mide si los grupos se sostienen en estaciones no vistas del mismo periodo.

**Selección del campeón** (`seleccion_p3.py`): ranking por mayor silueta de validación; a menos de 0.01 de la mejor decide la estabilidad (ARI), luego Davies-Bouldin y luego la menor complejidad. El campeón es la primera del ranking que cumple los criterios de interpretabilidad fijados antes de correr: cada grupo con al menos 5 % de las estaciones (I1), grupos distinguibles por al menos 0.5 desviaciones estándar (I2), nombrables sin repetir nombre (I3) y al menos un grupo que se vacía y otro que se llena en la mañana (U1). La tabla queda en `data/criterios_interpretabilidad.csv` de cada candidata.

**Artefactos de las candidatas**: `model/` (pipeline con firma y ejemplo de entrada), `data/` (manifiesto, asignación de estaciones, perfil de uso por grupo, bitácora de limpieza y criterios), `plots/` (mapa y perfiles de los grupos) y `config/config_corrida.json`; los perfiles se registran con `log_input`. Solo hay datos por estación, ninguno de las personas usuarias.

#### Campeón de P3 en el Model Registry

El modelo registrado es `mibici-equipo29-p3-grupos`. Usamos **aliases** en lugar de etapas, porque las etapas están en desuso en MLflow y un alias se mueve sin crear estados intermedios: `champion` apunta a la versión en producción y `challenger` a la siguiente candidata que cumple los criterios, lista para reemplazarla. Cada versión lleva su descripción y su linaje en etiquetas: run de origen, `git_commit`, `data_version`, experimento, candidata, posición en el ranking y motivo. Cada corrida de `model=seleccion` registra versiones nuevas, carga cada una por `models:/mibici-equipo29-p3-grupos/<versión>` y comprueba su predicción, y solo entonces mueve ambos aliases; si la comprobación falla, los aliases se quedan en las versiones anteriores y las nuevas quedan marcadas con `comprobacion = fallida`.

```python
import mlflow

import os
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
modelo = mlflow.pyfunc.load_model("models:/mibici-equipo29-p3-grupos@champion")
grupos = modelo.predict(perfiles)  # un grupo por estación
```

`perfiles` es un `DataFrame` con las 12 variables de `CONFIG_P3.prioridad_variables` como `float64` (la firma las lista). El modelo usa clases de `mibici.modeling.modelos_p3`, así que requiere el paquete `mibici` instalado (`requirements.txt` lo instala con `-e .`).

Otros comandos (por separado, con `python -m mibici.modeling.train_p3` o con `mlflow run`):

```powershell
mlflow run . -e train -P question=p3 --env-manager local                                      # baseline k = 4, 5 y 6
mlflow run . -e train -P question=p3 -P model=kmeans_uso -P hyperparameters=k=4 --env-manager local
mlflow run . -e train -P question=p3 -P model=gmm_uso -P hyperparameters=n_components=4,covariance_type=diag --env-manager local
mlflow run . -e train -P question=p3 -P model=kmeans_uso -P hyperparameters=buscar=1 --env-manager local
```

Las fechas, franjas horarias, umbrales de limpieza y de VIF, semilla, rejillas de búsqueda, particiones, criterios y nombre del modelo registrado están en `CONFIG_P3` de `mibici/config.py`.

### Pruebas, lint y formato

```bash
python -m pytest
ruff check .
ruff format --check .
```

La configuración de Ruff en `pyproject.toml` revisa `mibici/`.

## Seguimiento en MLflow

Todos los runs usan el experimento `mibici-equipo29`, con nombre descriptivo y etiquetas `question` (P1, P2 o P3), `team`, `author`, `data_range`, `seed`, `data_version`, `git_commit` y `git_dirty`. Las métricas indican el conjunto en que se calculan (`val_mae`, `test_mae`, ...). El almacén es local: `mlflow.db` y `mlartifacts/` en la raíz del repositorio, ambos fuera de git.

### Servidor de tracking

El Model Registry necesita un almacén con base de datos. Desde la raíz del repositorio, para que la base sea el mismo `mlflow.db` que usa el proyecto por omisión:

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000 --workers 1
```

En Windows, el servidor con varios workers deja de aceptar conexiones en algunos de ellos (`WinError 10022`); `--workers 1` lo evita. En Linux y macOS se puede omitir. En otra terminal, también desde la raíz, apunta ambas variables al servidor y lanza el proyecto:

```powershell
$env:MIBICI_TRACKING_URI = "http://127.0.0.1:5000"
$env:MLFLOW_TRACKING_URI = $env:MIBICI_TRACKING_URI
mlflow run . -e ingest --env-manager local --experiment-name mibici-equipo29
mlflow run . -e train -P question=all -P 'hyperparameters={}' --env-manager local --experiment-name mibici-equipo29
```

La interfaz queda en http://127.0.0.1:5000 y los artefactos que sube el servidor en `mlartifacts/`. Las dos corridas de proyecto que crea `mlflow run` (`ingest` y `train`) llevan un nombre automático y no tienen la etiqueta `question`; las corridas de entrenamiento sí. No mezcles en una misma sesión el servidor y una URI `sqlite:///` directa: cada una debe apuntar al mismo `mlflow.db` de la raíz. Sin servidor, la interfaz se abre con:

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
├── references         <- ML Canvas del equipo, diagrama de arquitectura y manifiesto de datos
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
