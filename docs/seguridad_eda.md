# Chequeos de seguridad de EDA y módulos compartidos

Revisión del 2 de noviembre de 2026 con Bandit 1.9.4 y pip-audit 2.10.1, instalados fuera
del entorno del proyecto. Las bases de datos de vulnerabilidades cambian; una corrida
posterior puede producir resultados distintos. No se modificó código ni se actualizaron
dependencias en esta revisión, para mantener el alcance en EDA.

## Alcance de Bandit

Se revisaron los módulos usados por el flujo de EDA y los componentes compartidos que
indicó la revisión:

```powershell
bandit mibici\dataset.py mibici\features_eda.py mibici\mlflow_project.py mibici\plots.py mibici\modeling\train.py mibici\modeling\predict.py
```

Bandit analizó 621 líneas y encontró dos alertas B310, ambas de severidad media y confianza
alta. No reportó hallazgos adicionales en los otros cinco módulos.

| Regla | Severidad | Ubicación | Evaluación |
|---|---|---|---|
| B310 | Media, confianza alta | `mibici/dataset.py:65` y `mibici/dataset.py:150` | `urlopen` permite esquemas distintos de HTTP(S), incluidos `file:` y esquemas personalizados. El flujo normal parte de la URL HTTPS fija de MiBici, y los nombres/meses de archivos se validan; sin embargo, `page_url` se puede proporcionar y los enlaces descubiertos no restringen esquema, host ni redirecciones. |

El riesgo operativo es bajo cuando se usa la fuente predeterminada y confiable, pero la alerta
no se considera un falso positivo absoluto: una URL de entrada o una página manipulada podría
dirigir solicitudes a destinos no previstos. Como endurecimiento pendiente, validar HTTPS,
restringir los hosts permitidos y volver a validar el destino después de redirecciones. La
corrección no se implementa aquí porque el descargador es compartido y un cambio de ese
comportamiento requiere una revisión aparte.

## Auditoría de dependencias

Los requisitos son compartidos por EDA y el resto del proyecto; estos hallazgos no corresponden
a dependencias exclusivas de EDA. Se usó `--no-deps`, por lo que la auditoría cubre los pines
enumerados directamente en los archivos, no una resolución exhaustiva de todas las
dependencias transitivas.

DeepChecks:

```powershell
pip-audit -r requirements-deepchecks.txt --no-deps --disable-pip --format markdown
```

Para auditar `requirements.txt`, se excluyen temporalmente sus líneas `pip` y `-e .`, que
`pip-audit` no trata como pines de paquetes:

```powershell
$auditReq = Join-Path $env:TEMP 'tc5061-eda-requirements-audit.txt'
Get-Content requirements.txt |
    Where-Object { $_ -notmatch '^\s*(pip\s*(#.*)?|-e\s+\.\s*(#.*)?)\s*$' } |
    Set-Content $auditReq
pip-audit -r $auditReq --no-deps --disable-pip --format markdown
Remove-Item $auditReq
```

| Archivo | Paquete y advisory | Severidad publicada | Contexto y corrección |
|---|---|---|---|
| `requirements.txt:18-20` | `mlflow==3.16.1`, [CVE-2026-53995](https://github.com/advisories/GHSA-wqmh-qccw-pqwf) | Alta, CVSS 8.2 | Path traversal en el backend FileStore. El servidor del README usa SQLite y loopback; el advisory aplica si se configura FileStore. Corregida en MLflow 3.17.2. |
| `requirements.txt:18-20` | `mlflow==3.16.1`, [CVE-2026-115886](https://github.com/advisories/GHSA-2wqp-q3c6-ccj5) | Media, CVSS 6.5 | SSRF en endpoints de scorer de un servidor MLflow no autenticado, bajo ciertas integraciones opcionales. No forma parte del flujo EDA documentado. Corregida en 3.17.0. |
| `requirements.txt:18-20` | `mlflow==3.16.1`, [CVE-2026-116691](https://github.com/advisories/GHSA-j7qq-vr8c-xvw3) | Media, CVSS 6.5 | CrewAI autologging puede registrar claves de proveedor en trazas; el repositorio no usa esa integración en EDA. Corregida en 3.17.0. |
| `requirements-deepchecks.txt:8` | `setuptools==80.10.2`, [PYSEC-2026-3447](https://osv.dev/vulnerability/PYSEC-2026-3447) (alias CVE-2026-59890; duplicado en la salida) | Media, CVSS 6.1 | Exclusiones de `MANIFEST.in` pueden fallar con nombres Unicode normalizados al generar distribuciones en macOS. Corregida en 83.0.0. |
| `requirements-deepchecks.txt:8` | `setuptools==80.10.2`, [GHSA-xmj3-9gf7-h52w](https://github.com/advisories/GHSA-xmj3-9gf7-h52w) | Baja | En Windows, un wheel manipulado puede crear un archivo fuera del directorio de instalación por la ruta legacy `install_as_egg`. Corregida en 84.1.0. |

`pip-audit` imprime tres registros para `setuptools`, porque lista dos veces
PYSEC-2026-3447; son dos advisories distintos. Los pines son compartidos y no se actualizaron
en esta revisión. DeepChecks 0.19.1 requiere `pkg_resources`, razón del pin actual; validar
compatibilidad o actualizar DeepChecks antes de elevar setuptools. También se recomienda
evaluar la actualización de MLflow en un cambio independiente.

Las imágenes de ejecución están en
[`eda_seguridad_bandit.png`](../reports/figures/eda_seguridad_bandit.png) y
[`eda_seguridad_pip_audit.png`](../reports/figures/eda_seguridad_pip_audit.png).

## Credenciales, variables de entorno y datos

- No se identificaron claves privadas ni patrones comunes de tokens o credenciales en los
  archivos rastreados que se revisaron.
- No hay un archivo `.env` rastreado; `.gitignore` excluye `.env`, los entornos `.venv` y
  `.venv-deepchecks`, y `/data/`, donde se generan los datasets locales. La ejecución EDA no
  requiere credenciales propias.
- Si una integración futura requiere secretos, deben leerse de variables de entorno o de un
  `.env` local ignorado por Git, nunca incorporarse al código ni a los datos versionados.

Esta comprobación no equivale a un análisis forense de todo el historial de Git.
