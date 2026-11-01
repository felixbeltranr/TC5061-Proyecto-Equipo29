# Chequeos de seguridad básicos de P1

Revisión del 31 de octubre de 2026 con `bandit` 1.9.4 (análisis estático del código) y
`pip-audit` 2.10.1 (dependencias con vulnerabilidades publicadas). Las bases de datos de
vulnerabilidades cambian: una corrida posterior puede dar otro resultado.

## Entorno de las herramientas

`bandit` y `pip-audit` se instalaron en un entorno aparte con Python 3.14.8, creado fuera del
repositorio. No se agregan a `requirements.txt` ni al entorno del proyecto o al de DeepChecks:
`pip-audit` lee los paquetes instalados de esos entornos sin modificarlos.

```bash
python -m venv <fuera-del-repo>/venv-seguridad
<fuera-del-repo>/venv-seguridad/Scripts/activate    # Linux o macOS: .../bin/activate
pip install bandit==1.9.4 pip-audit==2.10.1
```

## bandit sobre el código de P1

```bash
bandit mibici/config.py mibici/features.py mibici/features_pipeline.py mibici/registro_p1.py mibici/tracking.py mibici/modeling/train_p1.py scripts/p1/validar_panel_p1.py scripts/p1/exportar_features_p1.py scripts/p1/medir_drift_p1.py
```

2,195 líneas revisadas: los módulos que usa el entrenamiento de P1 y los scripts de validación
de datos. Hay doce hallazgos, once de severidad baja, uno de media y ninguno de alta.
`config.py`, `features.py`, `features_pipeline.py`, `registro_p1.py` y `train_p1.py` no tienen
hallazgos.

| Regla | Severidad | Dónde | Decisión |
|---|---|---|---|
| B404 | baja | `tracking.py:14` y `exportar_features_p1.py:19` (`import subprocess`) | Se usa sólo para llamar a `git`. |
| B603 (3) y B607 (3) | baja | `tracking.py:29, 33 y 45` | `git rev-parse`, `git status` y `git config user.name` con argumentos fijos y sin shell, para las etiquetas `git_commit`, `git_dirty` y `author` de los runs. B607 avisa que `git` se busca en el `PATH`; si no está, las etiquetas quedan como "desconocido". |
| B603 (2) | baja | `exportar_features_p1.py:249-250` | Argumentos fijos y sin shell; la única entrada es la ruta `--repo`, que va en `-c safe.directory` y en `git -C`. |
| B102 | media | `exportar_features_p1.py:106` (`exec`) | Compila funciones elegidas por AST, en los dos modos: `native` sólo `particionar` de `train_p1.py` (para no importar ese módulo, que importa MLflow) e `isolated-source` además ocho funciones de `features.py`. Antes comprueba sus hashes contra main `d2bec05` y rechaza funciones decoradas. |
| B105 | baja | `medir_drift_p1.py:126` | Falso positivo: la clave `quality_pass` contiene "pass" y su valor es `None`. |

No se marcan con `# nosec`: quedan visibles en el reporte con la decisión de esta tabla.

## pip-audit sobre las dependencias

`--no-deps --disable-pip` sólo acepta versiones fijas, así que `requirements.txt` se audita
desde una copia sin las líneas `pip` y `-e .`; `requirements-deepchecks.txt` se audita tal cual.

```bash
pip-audit -r <copia-de-requirements.txt-sin-pip-ni-editable> --no-deps --disable-pip
pip-audit -r requirements-deepchecks.txt --no-deps --disable-pip
pip-audit --path <entorno>/Lib/site-packages               # todo lo instalado en un entorno
```

| Origen | Paquetes | Con avisos | Avisos |
|---|---|---|---|
| Pines de `requirements.txt` | 19 | 1 | `mlflow` 3.16.1: CVE-2026-53995 (recorrido de rutas en el backend FileStore; corregido en 3.17.2), CVE-2026-115886 (endpoint del servidor de tracking sin autenticación; 3.17.0) y CVE-2026-116691 (claves de CrewAI en trazas de `mlflow.crewai.autolog`; 3.17.0) |
| Entorno del proyecto instalado | 176 (177 con `mibici`) | 4 | `mlflow` (los mismos tres), `certifi` 2026.7.22 (vía `requests` y `httpx`), `pyasn1` 0.6.4 (vía `google-auth` y `databricks-sdk`, que trae MLflow) y `tornado` 6.5.10 (vía Jupyter) |
| Pines de `requirements-deepchecks.txt` | 8 | 1 | `setuptools` 80.10.2: PYSEC-2026-3447 (sdist con `MANIFEST.in`; 83.0.0) y GHSA-xmj3-9gf7-h52w (ruta heredada de instalación de wheels en Windows; 84.1.0) |
| Entorno de DeepChecks instalado | 118 | 2 | `setuptools` (los mismos) y el `pip` 24.0 del entorno, que no fija ningún archivo de requisitos |

`mibici` no está en PyPI, así que `pip-audit` no lo audita.

Exposición en P1:

- `mlflow`: P1 guarda el tracking en SQLite (`mlflow.db`), no en el backend FileStore, y el
  servidor de tracking del README se lanza sin `--host`, así que escucha sólo en `127.0.0.1`
  (el valor por omisión de `mlflow server`). P1 no usa CrewAI. Subir `mlflow`,
  `mlflow-skinny` y `mlflow-tracing` a 3.17.2 (van fijados a la misma versión) corrige los tres
  avisos, pero `requirements.txt` lo comparten P1, P2 y P3: es una decisión del equipo y obliga
  a volver a correr la suite y los entrenamientos.
- `setuptools` 80.10.2 está fijado porque deepchecks 0.19.1 importa `pkg_resources`, que las
  versiones corregidas (83.0.0 y 84.1.0) ya no incluyen. Ese entorno sólo corre validaciones
  sobre datos locales y no construye ni instala paquetes de terceros después de crearse.
- `pip` 24.0 del entorno de DeepChecks se corrige con `python -m pip install --upgrade pip` al
  crearlo.
- `certifi`, `pyasn1` y `tornado` son dependencias transitivas del entorno instalado (cliente
  HTTP, autenticación de Databricks y Jupyter); P1 no las usa directamente. Una instalación
  nueva puede resolver versiones distintas.

## Secretos, `.gitignore` y variables de entorno

- `.gitignore` excluye `.env`, `mlflow.db`, `mlartifacts/`, `/data/`, `models/*.pkl`,
  `.venv` y `.venv-deepchecks/`; `reports/deepchecks/` tiene su propio `.gitignore`.
- No hay archivos versionados con aspecto de secreto, base de datos o datos (`.env`, `.db`,
  `.parquet`, `.pkl`, llaves), ni un `.env` en el historial.
- No hay claves, tokens ni contraseñas en el código de P1 ni en sus pruebas.
- `mibici/config.py` carga un `.env` opcional con `python-dotenv`; el almacén de tracking se
  elige con `MIBICI_TRACKING_URI` y `MLFLOW_TRACKING_URI`. El exportador de `scripts/p1/`
  desactiva la carga del `.env` (`PYTHON_DOTENV_DISABLED`) y los scripts de DeepChecks, su
  telemetría.
