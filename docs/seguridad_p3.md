# Chequeos de seguridad básicos de P3

Revisión del 1 de noviembre de 2026 con `bandit` 1.9.4 y `pip-audit` 2.10.1, instalados en un
entorno aparte fuera del repositorio, igual que en `docs/seguridad_p1.md` y
`docs/seguridad_p2.md` (no se agregan a `requirements.txt`). Las bases de datos de
vulnerabilidades cambian: una corrida posterior puede dar otro resultado.

## bandit sobre el código de P3

```bash
bandit mibici/features_p3.py mibici/modeling/modelos_p3.py mibici/modeling/busqueda_p3.py mibici/modeling/seleccion_p3.py mibici/modeling/artefactos_p3.py mibici/modeling/train_p3.py
```

1,596 líneas revisadas: los módulos exclusivos de P3 (perfiles de estaciones, pipelines,
validación por estaciones, selección del campeón, artefactos y entrenamiento). Hay tres
hallazgos, todos de severidad baja y todos en `train_p3.py`; ninguno de severidad media o
alta. `features_p3.py`, `modelos_p3.py`, `busqueda_p3.py`, `seleccion_p3.py` y
`artefactos_p3.py` no tienen hallazgos.

| Regla | Severidad | Dónde | Decisión |
|---|---|---|---|
| B404 | baja | `train_p3.py:35` (`import subprocess`) | Se usa sólo para llamar a `git`. |
| B603 y B607 | baja | `train_p3.py:139` (`_git`) | `git rev-parse HEAD` y `git status --porcelain --untracked-files=no` con argumentos fijos, sin shell y con `cwd` en la raíz del proyecto, para las etiquetas `git_commit` y `git_dirty` de los runs. B607 avisa que `git` se busca en el `PATH` en vez de una ruta completa; el riesgo sería un `git` falso puesto antes en el `PATH` de la propia máquina. |

Sobre las pruebas de P3 (`tests/test_features_p3.py`, `tests/modeling/test_*_p3.py` y
`tests/datos/test_deepchecks_p3.py`) `bandit` sólo reporta B101 (uso de `assert`), que es la
forma normal de escribir pruebas con pytest.

No se marcan con `# nosec` y no se cambió código: ningún hallazgo es explotable con las
entradas que recibe P3.

### Serialización de modelos

P3 guarda sus modelos en MLflow con formato `skops`, no con `pickle`. Antes de registrar,
`artefactos_p3.tipos_confiables` revisa los tipos que `skops` pediría confiar al cargar y
rechaza cualquiera fuera de scikit-learn, NumPy y `mibici.modeling.modelos_p3`
(`PREFIJOS_CONFIABLES`); lo comprueba la prueba
`test_tipos_confiables_acepta_solo_sklearn_numpy_y_modelos_p3`. `seleccion_p3` carga cada
versión con `mlflow.pyfunc.load_model`, sólo desde el Model Registry local del equipo.

## pip-audit sobre las dependencias

P3 no agrega dependencias propias: usa los mismos `requirements.txt` y
`requirements-deepchecks.txt` que P1 y P2. Se volvió a correr para esta revisión:

```bash
pip-audit -r <copia-de-requirements.txt-sin-pip-ni-editable> --no-deps --disable-pip
pip-audit -r requirements-deepchecks.txt --no-deps --disable-pip
```

| Origen | Paquetes | Con avisos | Avisos |
|---|---|---|---|
| Pines de `requirements.txt` | 19 | 1 | `mlflow` 3.16.1: CVE-2026-53995 (corregido en 3.17.2), CVE-2026-115886 y CVE-2026-116691 (3.17.0) |
| Pines de `requirements-deepchecks.txt` | 8 | 1 | `setuptools` 80.10.2: PYSEC-2026-3447 (83.0.0) y GHSA-xmj3-9gf7-h52w (84.1.0) |

El resultado coincide con el de P1 y P2. Lo que toca a P3:

- `mlflow`: P3 guarda el tracking en SQLite (`mlflow.db`, valor de `CONFIG_P3.tracking_uri`) o
  en el servidor común del README, que se lanza sin `--host` y escucha sólo en `127.0.0.1`; no
  usa el backend FileStore del CVE de recorrido de rutas ni CrewAI. Subir `mlflow` a 3.17.2
  queda como decisión del equipo, porque `requirements.txt` es compartido.
- `setuptools`: el entorno de DeepChecks corre `tests/datos/test_deepchecks_p3.py` sobre los
  perfiles locales y no construye ni instala paquetes de terceros. La razón del pin está en
  `docs/seguridad_p1.md`.

## Secretos, datos personales y datos versionados

- No hay claves, tokens ni contraseñas en los módulos de P3 ni en sus pruebas, y P3 no lee
  variables de entorno propias; el `.env` opcional lo carga `mibici/config.py` con
  `python-dotenv` y no se versiona.
- P3 lee sólo cinco columnas de los viajes (`Viaje_Id`, `Inicio_del_viaje`, `Fin_del_viaje`,
  `Origen_Id` y `Destino_Id`) y nunca carga `Usuario_Id`, `Genero` ni `Año_de_nacimiento`
  (`COLUMNAS_PERSONALES` en `features_p3.py`). Lo comprueban
  `test_leer_viajes_no_carga_datos_personales` y
  `test_asignacion_sin_datos_personales_y_marca_nuevas`; los artefactos de MLflow son agregados
  por estación.
- `.gitignore` excluye `/data/` (donde quedan los perfiles `p3_perfiles_*.parquet` y la caché
  `p3_flujos_*.csv.gz` / `p3_bitacora_*.csv`), `mlflow.db`, `mlartifacts/` y `.env`;
  `reports/deepchecks/` tiene su propio `.gitignore`.
- Los archivos de P3 versionados que no son código son el notebook de exploración y once
  figuras en `reports/figures/`, todas agregadas por estación, hora o mes. En el notebook los
  nombres de las columnas personales aparecen sólo en la lista de columnas que se excluyen.
  No hay CSV ni Parquet de P3 en el repositorio, y en el historial no se agregó nunca un
  `.env`, una base de datos ni llaves.
