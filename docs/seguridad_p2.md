# Chequeos de seguridad básicos de P2

Revisión del 31 de octubre de 2026 con `bandit` 1.9.4 y `pip-audit` 2.10.1, instalados en un
entorno aparte fuera del repositorio, igual que en `docs/seguridad_p1.md` (no se agregan a
`requirements.txt`). Las bases de datos de vulnerabilidades cambian: una corrida posterior
puede dar otro resultado.

## bandit sobre el código de P2

```bash
bandit mibici/features_p2.py mibici/manifiesto_p2.py mibici/registro.py mibici/modeling/train_p2.py mibici/modeling/horizontes_p2.py mibici/modeling/exportar_p2.py
```

1,552 líneas revisadas: los módulos exclusivos de P2 (features, manifiesto de datos,
entrenamiento, horizontes y exportación de tablas) y `registro.py`, que nació para el campeón
de P2 y ahora también lo usa P3. Hay seis hallazgos, todos de severidad baja y todos en
`train_p2.py`; ninguno de severidad media o alta. `features_p2.py`, `manifiesto_p2.py`,
`registro.py`, `horizontes_p2.py` y `exportar_p2.py` no tienen hallazgos.

| Regla | Severidad | Dónde | Decisión |
|---|---|---|---|
| B403 | baja | `train_p2.py:27` (`import pickle`) | Sólo se usa `pickle.dumps` (línea 305) para medir el tamaño del modelo recién entrenado en MB. No se deserializa nada con `pickle`, así que no hay carga de datos no confiables. |
| B404 | baja | `train_p2.py:28` (`import subprocess`) | Se usa sólo para llamar a `git`. |
| B603 (2) y B607 (2) | baja | `train_p2.py:99 y 103` | `git rev-parse HEAD` y `git status --porcelain --untracked-files=no` con argumentos fijos y sin shell, para las etiquetas de commit y de árbol sucio de los runs. B607 avisa que `git` se busca en el `PATH`. |

No se marcan con `# nosec` y no se cambió código: ningún hallazgo es explotable con las
entradas que recibe P2.

## pip-audit sobre las dependencias

P2 no agrega dependencias propias: usa los mismos `requirements.txt` y
`requirements-deepchecks.txt` que P1, así que la auditoría es la misma que la de
`docs/seguridad_p1.md`. Se volvió a correr para esta revisión:

```bash
pip-audit -r <copia-de-requirements.txt-sin-pip-ni-editable> --no-deps --disable-pip
pip-audit -r requirements-deepchecks.txt --no-deps --disable-pip
```

| Origen | Paquetes | Con avisos | Avisos |
|---|---|---|---|
| Pines de `requirements.txt` | 19 | 1 | `mlflow` 3.16.1: CVE-2026-53995 (corregido en 3.17.2), CVE-2026-115886 y CVE-2026-116691 (3.17.0) |
| Pines de `requirements-deepchecks.txt` | 8 | 1 | `setuptools` 80.10.2: PYSEC-2026-3447 (83.0.0) y GHSA-xmj3-9gf7-h52w (84.1.0) |

El resultado coincide con el de P1. Lo que toca a P2:

- `mlflow`: `train_p2` y `horizontes_p2` guardan el tracking en SQLite (`mlflow.db`, valor de
  `CONFIG_P2.tracking_uri`), no en el backend FileStore del CVE de recorrido de rutas. El
  servidor de tracking del README se lanza sin `--host`, así que escucha sólo en `127.0.0.1`.
  P2 no usa CrewAI. `registro.py` carga el campeón con `mlflow.pyfunc.load_model`, que
  deserializa el modelo, pero sólo desde el Model Registry local del equipo. Subir `mlflow` a
  3.17.2 queda como decisión del equipo, porque `requirements.txt` es compartido.
- `setuptools`: el entorno de DeepChecks corre las pruebas de `tests/datos/test_deepchecks_p2.py`
  sobre tablas locales y no construye ni instala paquetes de terceros. La razón del pin ya está
  en `docs/seguridad_p1.md`.

## Secretos, `.gitignore` y datos versionados

- No hay claves, tokens ni contraseñas en los módulos de P2 ni en sus pruebas
  (`tests/test_features_p2.py`, `tests/test_registro.py`, `tests/modeling/*p2*` y
  `tests/datos/test_deepchecks_p2.py`).
- En el historial de la rama no se agregó nunca un `.env`, una base de datos, un parquet, un
  pickle ni llaves, y los diffs de sus commits no contienen asignaciones de claves o tokens.
- `.gitignore` excluye `/data/` (donde quedan `p2_panel_horario.parquet`, `p2_train.parquet` y
  `p2_test.parquet`), `mlflow.db`, `mlruns/`, `mlartifacts/`, `reports/runs_p2/` (salida de
  `train_p2`), `models/*.pkl`, `models/*.joblib` y `.env`.
- Los únicos archivos de datos de P2 versionados son cinco CSV en `reports/figures/`: error por
  estación y hora, métricas por horizonte e importancia por permutación. Son agregados de datos
  abiertos de MiBici con identificadores de estación, sin información de usuarios.
- P2 no lee variables de entorno propias; el `.env` opcional lo carga `mibici/config.py` con
  `python-dotenv` y no se versiona.
