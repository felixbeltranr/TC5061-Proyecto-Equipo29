# Validación de datos y drift descriptivo P1

## Dependencia y separación de entornos

Estos auxiliares usan el entorno de `requirements-deepchecks.txt`, que agregó el PR #13
(head `2438e1b207564fc1283a6cf98920c3499c90c8ed`, ya en main). No se duplica ni se cambia esa
infraestructura, ni `requirements.txt` o el entorno principal.

- DeepChecks: Python 3.11; deepchecks 0.19.1, numpy 1.26.4, pandas 2.3.3,
  scikit-learn 1.5.2, setuptools 80.10.2, plotly 5.24.1, pyarrow 25.0.1, pytest 9.1.1.
- Exportación `native`: entorno principal Python 3.14 ya disponible.
- Exportación `isolated-source`: diagnóstico adicional posible en 3.11; ejecuta
  sólo funciones puras seleccionadas del código original, no importa el paquete
  principal completo y no acredita su compatibilidad en 3.11.

No se propone instalar el proyecto editable en el entorno DeepChecks.

## Integridad del panel completo

`validar_panel_p1.py` lee sólo `estacion`, `fecha`, `salidas`, `primer_dia`.
Comprueba tipos, finitud, conteos enteros no negativos, fechas diarias consecutivas,
claves estación-fecha únicas y fechas locales sin zona horaria. Exige que
`primer_dia` corresponda al inicio de cada estación en este **panel completo**.
Una ventana recortada puede conservar legítimamente un `primer_dia` anterior:
ese rechazo significa que no satisface este contrato, no que el origen esté corrupto.

Después del contrato ejecuta sobre todas las filas dos checks explícitos:
`PercentOfNulls <= 0` y `DataDuplicates` sobre las claves con `max_ratio=0`.
Conteos repetidos y días con cero son válidos; no se aplican automáticamente
reglas genéricas de correlación o variables constantes. Datos ausentes, check no
ejecutado o contrato fallido no se presentan como PASS/SKIP satisfactorio.

Los dos checks repiten, con DeepChecks, reglas que el contrato ya exige (sin nulos y sin
claves repetidas) y sólo corren si el contrato pasa; por eso, en esta herramienta su PASS
confirma el contrato y no agrega un criterio nuevo. Las pruebas los ejecutan también
directamente sobre paneles con nulos y con claves repetidas para comprobar que fallan.

## Exportación y drift

`exportar_features_p1.py` verifica hashes normalizados CRLF/LF de `mibici/features.py`,
`mibici/modeling/train_p1.py` y, de `mibici/config.py`, del texto de la clase `P1Config`
(con su decorador) y de la instancia `CONFIG_P1`, contra las de main `d2bec05`. El resto
de `config.py` lo comparten P2 y P3: un cambio suyo no invalida el control de P1. El
exportador usa los valores de la clase y exige que `CONFIG_P1 = P1Config()` se defina una
sola vez y sin argumentos. Registra el HEAD con `git rev-parse HEAD`, sin
fetch/show/checkout: requiere git y que `--repo` sea la raíz de un checkout con su propio
`.git`; si no, se detiene, y las pruebas que exportan se saltan. Si cambia el código de
P1, hay que revisar los hashes, no actualizarlos ciegamente.

Reutiliza `construir_features` y `particionar`: elegibilidad original, entrenamiento
hasta 2026-04-29, prueba desde 2026-05-01 hasta antes de 2026-10-01, embargo D-2.
Rechaza la exportación si la separación entre ambos conjuntos es menor al rezago mínimo
de `mibici.features` (2 días) y lo registra en el manifiesto (`embargo_days`).
Exporta sólo claves y las 11 features, sin etiqueta ni columnas de usuario; no
entrena modelos ni llama `main`, MLflow o el registro.

`medir_drift_p1.py` verifica hashes/esquema/claves y separa dos poblaciones:
todas las estaciones y únicamente estaciones presentes en ambos periodos.
Las claves no se usan como variables de drift. Usa KS para ocho variables
numéricas y Cramér V para `dia_semana`, `festivo`, `festivo_lag7`.
No comparar ambos tipos de puntuación como si fueran la misma escala operativa.

Muestreo determinista, semilla 29 y máximo 100,000 filas por periodo; filas,
cohortes, nulos y claves se verifican sobre la población completa. KS excluye
nulos numéricos, que se reportan aparte; los nulos categóricos se conservan.
No hay umbrales arbitrarios: `quality_pass: null`. Exit 0 significa cálculo
completo, no calidad aprobada ni recomendación de reentrenar. Dependencia temporal
y composición de estaciones limitan la interpretación causal.

## Comandos (Bash; rutas propias fuera del repo)

Los intérpretes deben existir. No ejecutar estos comandos con rutas copiadas de
otra máquina. `SALIDA` debe ser una carpeta externa nueva; las entradas no se editan.

```bash
REPO=/ruta/al/checkout
PANEL=/ruta/al/p1_panel_diario.parquet
SALIDA=/ruta/privada/nueva/p1
PY311=/ruta/al/venv-deepchecks/bin/python
PY314=/ruta/al/venv-principal/bin/python
export PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=2
"$PY311" "$REPO/scripts/p1/validar_panel_p1.py" --repo "$REPO" --input "$PANEL" --output "$SALIDA/integridad"
"$PY314" "$REPO/scripts/p1/exportar_features_p1.py" --repo "$REPO" --input "$PANEL" --output "$SALIDA/exports" --mode native
"$PY311" "$REPO/scripts/p1/medir_drift_p1.py" --repo "$REPO" --exports "$SALIDA/exports" --output "$SALIDA/drift" --sample-size 100000 --seed 29 --populations both
```

Cada paso requiere exit 0 antes de continuar. Para el diagnóstico limitado en
3.11, sustituir sólo la línea del exportador por el intérprete 3.11 y
`--mode isolated-source`, etiquetando así los resultados. No llamarlo ejecución nativa.

QA sintética, desde la raíz y con el entorno DeepChecks existente:

```bash
"$PY311" -m pytest tests/datos/test_integridad_p1.py tests/datos/test_drift_p1.py -v -rs -p no:cacheprovider
```

Los tests usan fixtures sintéticos, no descargan datos y no prueban el modelo.
`MIBICI_REPO` permite apuntar las pruebas a otro checkout, que sólo se lee.

## Evidencia

Windows 11; DeepChecks en Python 3.11.9 con los pins de arriba; exportación `native`
en el entorno principal Python 3.14.8. Checkout: main `d2bec05` con el PR #13
(`2438e1b`) fusionado localmente y estos archivos.

- QA sintética: **40 passed, 0 omitidas** (11.9 s). Incluye controles negativos,
  hashes alterados, la huella de configuración ante cambios dentro y fuera de `P1Config`
  y `CONFIG_P1`, la instancia con argumentos o definida dos veces, funciones originales
  decoradas y una copia sin `.git`. Sin git en el `PATH`: 27 passed y 13 skipped (las que
  exportan), exit 0; con una copia sin `.git` dentro de otro repositorio git, esas mismas 13
  se saltan en vez de registrar el commit ajeno.
- Panel histórico P1 `p1_panel_diario.parquet`: 245,185 filas, 472 estaciones,
  SHA256 `8f14faf9743363836339d53e29386708fde77ce8a8a7e24384a515658349d468`.
  Contrato y dos checks de integridad PASS; entrada inalterada.
- Exportación `native`: 167,177 filas de entrenamiento y 62,426 de prueba; features
  sin nulos; separación de 2 días entre conjuntos. Drift calculado para all/common, sin
  aprobación de calidad.
- All: máximo KS `lag_28 = 0.073079`; common: `media_28d = 0.036529`.
  Cramér V máximo `festivo = 0.052547 / 0.054407`, respectivamente.
  370 estaciones comunes, 101 sólo test, una sólo train.
- La exportación `isolated-source` en Python 3.11 del mismo panel produjo las mismas
  tablas (comparadas con `DataFrame.equals`) y las mismas 22 puntuaciones de drift.
- Tras integrar main `f164de8` (#13 y #14 fusionados): el comando de QA de arriba sigue
  en **40 passed**. Con todo `tests/datos` (`python -m pytest tests/datos -v -rs`), se
  agregan 6 omitidas de P2 y P3, porque no están sus tablas en `data/interim`. La huella de
  `P1Config` y `CONFIG_P1` y los hashes de `features.py` y `train_p1.py` no cambian, así
  que la corrida con el panel real de arriba sigue vigente y no se repitió.

El panel fue generado el 18 de octubre, no regenerado con main actual.
Los JSON y logs registran el alcance; no se versionan Parquets ni HTML con datos.
Los HTML abren sin conexión: el de integridad muestra las dos condiciones en PASS y
los de drift, la puntuación y la distribución de las cinco variables con mayor drift.
La ruta `native` no tiene prueba automática, porque en el entorno principal se omite
`tests/datos`; queda cubierta por esta ejecución con el panel real.

### Separación durante la colección

Los dos módulos de datos se omiten explícitamente si el intérprete no es Python
3.11 o falta DeepChecks. Ese SKIP no acredita que los datos pasen. Con Python 3.14.8
ambos aparecen como SKIP por versión, también dentro de la suite completa; con Python
3.11 y `deepchecks` bloqueado, como SKIP por dependencia faltante. En las dos
colecciones dirigidas: 0 tests, 2 módulos omitidos y exit 5.
El exportador detecta sólo módulos MLflow que él introduzca, no falla porque otra
prueba ya hubiera importado MLflow; una prueba cubre ese caso.

### Plantilla PowerShell

Usar rutas propias a entornos existentes; no instala paquetes ni cambia el principal.
Cada intérprete es explícito. Se ejecutó con rutas reales en lugar de las de ejemplo:
los tres pasos terminaron con exit 0, y las tablas y las 22 puntuaciones coinciden con
las de los comandos Bash.

```powershell
$repo = 'C:\ruta\checkout'
$panel = 'C:\ruta\datos\p1_panel_diario.parquet'
$salida = 'C:\ruta\privada\nueva-p1'
$py311 = 'C:\ruta\venv-deepchecks\Scripts\python.exe'
$py314 = 'C:\ruta\venv-principal\Scripts\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:GIT_OPTIONAL_LOCKS = '0'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:LOKY_MAX_CPU_COUNT = '2'
& $py311 "$repo\scripts\p1\validar_panel_p1.py" --repo $repo --input $panel --output "$salida\integridad"
$codigo = $LASTEXITCODE; if ($codigo -ne 0) { throw "Integridad terminó con exit $codigo" }
& $py314 "$repo\scripts\p1\exportar_features_p1.py" --repo $repo --input $panel --output "$salida\exports" --mode native
$codigo = $LASTEXITCODE; if ($codigo -ne 0) { throw "Exportación terminó con exit $codigo" }
& $py311 "$repo\scripts\p1\medir_drift_p1.py" --repo $repo --exports "$salida\exports" --output "$salida\drift" --sample-size 100000 --seed 29 --populations both
$codigo = $LASTEXITCODE; if ($codigo -ne 0) { throw "Drift terminó con exit $codigo" }
```

`$LASTEXITCODE` se captura inmediatamente después de cada proceso, antes de otra
orden. La alternativa `isolated-source` usa `$py311` y conserva sus límites.
