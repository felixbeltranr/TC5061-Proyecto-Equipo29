# Model Card · Proyecto MiBici · Equipo 29

**Versión del documento:** borrador 1, 24 de noviembre de 2026
**Repositorio:** https://github.com/felixbeltranr/TC5061-Proyecto-Equipo29

## Cómo leer este documento

El proyecto entrena tres modelos independientes, uno por pregunta, y cada uno se registra y se sirve por separado. Este documento tiene por eso una **parte común**, que describe los datos y las prácticas que comparten los tres, y después **una ficha por modelo**, que se lee sola.

Todas las cifras de este documento salen de archivos del repositorio o del almacén de modelos publicado, y cada sección dice de cuál. Ninguna viene de memoria ni de documentos anteriores.

---

# Parte común

## 1. Los datos

| | |
|---|---|
| Fuente | Datos abiertos del sistema de bicicleta pública MiBici, Área Metropolitana de Guadalajara |
| Periodo | Enero de 2025 a septiembre de 2026 (21 archivos mensuales) |
| Nomenclatura de estaciones | `nomenclatura_2026_09.csv` |
| Versión de los datos | `66dd3d190bba` (`references/manifiesto_datos_p2.json`) |
| Unidad del archivo original | Un viaje por fila |

La versión de los datos es el resumen criptográfico del manifiesto que lista cada archivo con su huella SHA256. Se registra como parámetro `data_version` en cada corrida de MLflow, así que cualquier run puede rastrearse hasta el conjunto exacto de archivos con el que se entrenó.

### Calidad y limpieza

Cifras de la bitácora de limpieza de la corrida de cierre (`data/bitacora_limpieza.csv` en los artefactos del run `fd704e3c`):

| Concepto | Viajes | Porcentaje |
|---|---|---|
| Filas crudas | 7,894,097 | 100.00 % |
| Duplicados exactos (se quitan) | 758 | 0.01 % |
| Fechas o identificadores inválidos (se quitan) | 0 | 0.00 % |
| Duración negativa (se quitan) | 0 | 0.00 % |
| Duración menor a 1 minuto (se quitan) | 245,300 | 3.11 % |
| Ida y vuelta a la misma estación (**se cuentan, no se quitan**) | 487,305 | 6.17 % |
| Viajes válidos | 7,648,039 | 96.88 % |

Dos decisiones de limpieza que conviene conocer antes de interpretar cualquier resultado:

- **Los viajes de menos de un minuto se descartan** porque corresponden a anclajes fallidos: el usuario no logra asegurar la bicicleta, el viaje se cierra y se vuelve a abrir. No son desplazamientos.
- **Los viajes de ida y vuelta a la misma estación se conservan.** Son el 6.17 % y se reportan en la bitácora, pero entran al conteo. Eso importa para cualquier medida de flujo: un viaje de ida y vuelta suma una salida y una llegada en la misma estación, así que no altera el flujo neto pero sí el volumen de uso.

Los 758 duplicados exactos están todos en septiembre de 2026.

### Validación contra una fuente independiente

Los conteos mensuales se contrastaron con las fichas del Instituto de Información Estadística y Geográfica de Jalisco (IIEG), que publica las mismas cifras: https://iieg.gob.mx/ns/?page_id=25284. Las diferencias están entre 0.02 % y 0.05 %, consistentes con cortes de corte de mes.

## 2. Procedencia y condiciones de uso

**Los datos se publican como datos abiertos, sin licencia explícita.**

Lo verificado en https://mibici.net/es/datos-abiertos/ el 24 de noviembre de 2026:

- La página se titula "Datos abiertos" y los archivos se nombran `datos_abiertos_AAAA_MM.csv`, así que el rótulo lo pone el propio publicador.
- No hay licencia, ni condiciones de reutilización, ni requisito de atribución.
- La página de "Términos y condiciones" del sitio trata exclusivamente del servicio de renta (suscripciones, anclaje, facturación, reembolsos) y no menciona los datos.
- El pie del sitio lleva un aviso general de derechos reservados.

Quién publica:

- **Operado por:** BKT Bici pública S.A. de C.V.
- **Administrado por:** Agencia Metropolitana de Servicios de Infraestructura para la Movilidad del Área Metropolitana de Guadalajara.

Este documento describe lo anterior y no declara una licencia abierta, porque el publicador no otorga una. Que un organismo público (el IIEG) republique las mismas cifras respalda el uso con fines de estudio, pero no sustituye una licencia.

## 3. Datos personales y minimización

Los archivos originales incluyen atributos de la persona usuaria. **El proyecto no los lee.**

La lectura está acotada en el código, no por convención. `mibici/features_p3.py` declara las columnas permitidas y, por separado, nombra las personales de forma explícita:

```python
COLUMNAS = ["Viaje_Id", "Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"]
COLUMNAS_PERSONALES = {"Usuario_Id", "Genero", "Año_de_nacimiento"}
```

La lectura se hace con `usecols=COLUMNAS`, así que las columnas personales no llegan a memoria. Nombrarlas en una constante aparte permite además que una prueba verifique que ninguna aparece entre las que se leen.

Ningún modelo de este proyecto recibe atributos de la persona usuaria, y por lo tanto ninguno puede discriminar por edad o género: esas variables no existen en el pipeline. La unidad de análisis es siempre la estación, nunca la persona.

Controles adicionales: `bandit` sobre el código y `pip-audit` sobre las dependencias, con evidencias en `docs/seguridad_p1.md`, `docs/seguridad_p2.md`, `docs/seguridad_p3.md` y `docs/seguridad_eda.md`. Los datos, la base de MLflow y los artefactos están en `.gitignore` desde el primer commit y nunca se versionan.

## 4. Gobernanza

### Quién responde por cada modelo

| Ámbito | Responsable | Usuario de GitHub |
|---|---|---|
| P1 · demanda diaria por estación | Alan Alfonso Boldo Reyes | `a01840556` |
| P2 · riesgo horario de ocupación | Carlos Galván | `carloshgalvan95` |
| P3 · grupos de estaciones por perfil de uso | Félix Beltrán | `felixbeltranr` |
| Infraestructura común de los modelos | Ulises Orozco | `xAlucardzx` |

El responsable de un modelo responde por su entrenamiento, su registro en el Model Registry, su documentación en esta ficha y su monitoreo. La infraestructura que sostiene a los tres —ingesta y preparación de datos, siembra del Registry, orquestación con Compose y canalización de integración continua— corresponde a Ulises Orozco.

La asignación coincide con la tabla de integrantes del README y con el historial de Git, así que es verificable y no solo declarativa.

### Cómo se autoriza un cambio en producción

Ninguna versión llega a producción por decisión de una sola persona. El responsable del modelo propone el cambio en un pull request, y la rama `main` está protegida y exige dos aprobaciones, así que el cambio necesita al menos un integrante más. El responsable de la infraestructura común participa cuando el cambio toca la siembra del Registry o el despliegue.

Los controles técnicos que respaldan esa decisión están implementados y se pueden verificar en el código:

- **Promoción con verificación previa.** Antes de mover el alias `champion`, el código carga la versión nueva por su URI inmutable y exige una predicción válida. Si la carga o la predicción falla, el alias anterior se conserva y la versión nueva queda marcada con `comprobacion = fallida`.
- **El monitoreo no promueve.** Los módulos de A/B registran el modelo reentrenado como `challenger` con `registro.registrar_retador`, que nunca mueve `champion`. Promover es un paso aparte y deliberado.
- **La regla de decisión se fija antes de comparar.** En las tres preguntas, el criterio para reemplazar al campeón se escribió en un commit anterior a la corrida del A/B, de modo que no pueda ajustarse al resultado.
- **Alias en lugar de etapas.** Las etapas están en desuso en MLflow y un alias se mueve sin estados intermedios.
- **Control de cambios.** Rama `main` protegida, GitHub Flow, Conventional Commits y dos aprobaciones por pull request.

Lo que este documento todavía no fija es qué pasa cuando el responsable de un modelo no está disponible y hay que decidir sobre su modelo. En un equipo de cuatro personas durante un semestre no ha hecho falta, pero en operación real sería lo primero que habría que resolver.

## 5. Subgrupo de referencia para el análisis desagregado

Una métrica global puede ocultar que un modelo funciona peor para una parte del sistema. En este proyecto el subgrupo relevante es el mismo para las tres preguntas:

**Estaciones nuevas de la expansión de 2026 contra estaciones existentes.**

Entre mayo y agosto de 2026 el sistema abrió 99 estaciones (71 en Guadalajara y 28 en Zapopan), un crecimiento de alrededor del 21 % sobre las 470 que había. Son el subgrupo natural porque tienen poca historia, su demanda todavía no se estabiliza y se concentran en la periferia: justamente las condiciones en las que un modelo entrenado con el sistema consolidado puede fallar sin que el promedio lo muestre.

Cada ficha reporta su desempeño separando ese subgrupo.

---

# Ficha 1 · P1 · Demanda diaria por estación

`<POR RELLENAR>`

## 1.1 Identificación
`<POR RELLENAR>`

## 1.2 Uso previsto
`<POR RELLENAR>`

## 1.3 Fuera de alcance
`<POR RELLENAR>`

## 1.4 Datos de entrenamiento y evaluación
`<POR RELLENAR>`

## 1.5 Variables
`<POR RELLENAR>`

## 1.6 Métricas y regla de decisión
`<POR RELLENAR>`

## 1.7 Desempeño desagregado (estaciones nuevas contra existentes)
`<POR RELLENAR>`

## 1.8 Limitaciones y riesgos
`<POR RELLENAR>`

## 1.9 Monitoreo
`<POR RELLENAR>`

---

# Ficha 2 · P2 · Riesgo horario de ocupación

## 2.1 Identificación

| Campo | Valor |
|---|---|
| Modelo registrado | `mibici-equipo29-p2-riesgo` |
| Alias | `champion` |
| Versión | 1 |
| Tipo | Clasificación multiclase: `HistGradientBoostingClassifier` de scikit-learn con pesos de clase balanceados y sin paro temprano |
| Hiperparámetros | `learning_rate = 0.1`, `max_leaf_nodes = 63`, `max_iter = 300`, `l2_regularization = 1.0`, `class_weight = "balanced"`, semilla 29 |
| Run de origen | `48bb03f4818147b2a5cc9b893b3b823f`, nombre `p2_hgb_balanceado_ini50` |
| Commit | `42918951d913fd2a27949a7237a381536242b11e` (árbol limpio) |
| URI del modelo | `models:/m-979cb0f7ef5a4cdaba35c5e32d892530` |
| Huella del modelo serializado | SHA256 `8dc3a262b88e23cb6c88a2edd3a38fd0aada6b33679d28e8556f52840d5d0c75` (`model.skops`, 9.17 MiB) |
| Fecha | 25 de octubre de 2026 (corrida de cierre de la Entrega 2) |
| Código | `mibici/features_p2.py`, `mibici/modeling/train_p2.py` y `mibici/registro.py` |
| Entrada | 18 variables por estación-hora (sección 2.5) |
| Salida | Clase de riesgo de la estación al cierre de la hora siguiente: `segura`, `sin_bicicletas` o `sin_anclajes`. La API la devuelve en el campo `risk_class` |

Fuente: `scripts/p2/campeon_api_p2.json` y las etiquetas, parámetros y descripción de la versión en el almacén publicado.

**Sobre la numeración de versiones.** Esta ficha documenta la versión 1, que es la que trae el almacén publicado en el Release `almacen-entrega2-r1` y la que `registry-init` siembra al levantar el sistema. La versión 2 (`challenger`) es el mismo HGB, con los mismos hiperparámetros, reentrenado con fechas objetivo hasta el 30 de septiembre de 2026, y quedó registrada como `challenger`.

## 2.2 Uso previsto

Al cierre de cada hora t, clasificar cada estación según el riesgo que tendrá al cierre de t+1: sin bicicletas (ocupación estimada menor al 10 % de la capacidad), sin anclajes (mayor al 90 %) o segura.

El uso previsto es **operativo de corto plazo**: el equipo de rebalanceo recibe la lista de estaciones en riesgo y decide a cuáles llevar bicicletas y de cuáles retirarlas en la hora siguiente. Agregada por meses, la frecuencia de riesgo de una estación orienta ampliaciones de anclajes o de flota. El modelo se sirve en línea, una estación a la vez o en lotes horarios de hasta 1000 estaciones, y el lote también se puede encolar en Redis para que lo atienda el worker de RQ.

Usuarios previstos: el equipo de rebalanceo y quien opera el tablero de la operación. El modelo prioriza visitas; la decisión de mover bicicletas la toma una persona.

## 2.3 Fuera de alcance

- **No mide el inventario.** La clase se calcula sobre una ocupación estimada a partir de los viajes (sección 2.4), no sobre bicicletas contadas en la estación. No sirve para afirmar que una estación estuvo vacía o llena ni para auditar el servicio.
- **No pronostica a más de una hora.** El modelo publicado es de horizonte h = 1. El análisis de 2 a 4 horas de `horizontes_p2` no registra modelos.
- **No cubre la madrugada.** Las horas objetivo van de 6 a 23, y la API rechaza cualquier otra con 422.
- **No dice cuántas bicicletas mover**, solo la clase. Cuántos viajes saldrán de una estación es la pregunta de P1.
- **No se traslada a otro sistema ni a otra ciudad** sin reentrenar y volver a evaluar.
- **No dice nada sobre personas.** La unidad es la estación-hora. No hay atributos de usuario en el modelo.

## 2.4 Datos de entrenamiento y evaluación

| Conjunto | Fechas objetivo | Filas estación-hora | Uso |
|---|---|---|---|
| Ajuste para la selección | 1 ene 2025 a 28 feb 2026 | sin registro en el run | Entrenar las candidatas que se comparan en validación |
| Validación | 1 mar 2026 a 30 abr 2026 | sin registro en el run | Elegir hiperparámetros y candidata |
| Entrenamiento final | 1 ene 2025 a 30 abr 2026 | 3,224,772 | Ajustar el modelo que se registra |
| Prueba | 1 may 2026 a 30 sep 2026 | 1,185,156 | Diagnóstico del modelo ya elegido |

La partición es temporal, por fecha objetivo. La prueba empieza en mayo de 2026, el mes en que empiezan las aperturas de la expansión, así que es el primer periodo con estaciones nuevas. Los hiperparámetros y la candidata se eligen solo con validación (sección 2.6).

**La prueba no está intacta.** Los resultados de prueba de P2 se publicaron en el repositorio el 17 de octubre de 2026 y la regla que elige al campeón se escribió el 24 de octubre, así que las cifras de prueba se reportan como referencia y no como una medición sobre datos nunca vistos. Octubre de 2026 sí quedó fuera del entrenamiento de las dos versiones que se comparan en la sección 2.9.

**La etiqueta es una aproximación.** Los datos abiertos no traen inventario por estación. La ocupación se estima con el flujo neto (llegadas menos salidas) acumulado hora por hora desde un nivel supuesto del 50 % de la capacidad a las 5:00, recortado a entre cero y la capacidad. Se reinicia cada día de servicio porque el rebalanceo nocturno no aparece en los datos. La capacidad es `dpcapacity` de `nomenclatura_2026_09.csv`, una por estación para todo el periodo. Con esa ocupación, en la prueba el 89.6 % de las filas son `segura`, el 5.3 % `sin_bicicletas` y el 5.1 % `sin_anclajes`.

**El panel.** 5,881,296 filas estación-hora de 470 estaciones. P2 aplica la limpieza de `features_p3.limpiar_viajes` descrita en la parte común: parte de los 7,894,097 registros y usa 7,648,039 viajes válidos. Quedan fuera dos estaciones que no están en la nomenclatura, y con ellas 3,869 viajes que las tocan.

**Elegibilidad:** una estación entra al panel desde su primer día con viajes. Solo cuentan orígenes desde las 5:00, porque antes la ocupación todavía es la del día anterior, y horas objetivo de 6 a 23.

Versión de los datos: `66dd3d190bba`, registrada como `data_version` en cada run de P2 y en la versión del Registry.

## 2.5 Variables

**Entrada del modelo (18 variables)**, derivadas de los flujos horarios de la propia estación, de su capacidad y del calendario:

| Variable | Qué es |
|---|---|
| `hora`, `dia_semana`, `fin_semana` | Calendario del instante objetivo t+1 |
| `capacidad` | Anclajes de la estación según la nomenclatura |
| `ocupacion_actual` | Ocupación estimada al cierre de t, como proporción de la capacidad |
| `neto_lag1`, `salidas_lag1`, `llegadas_lag1` | Flujo neto, salidas y llegadas de la hora t |
| `neto_lag24`, `neto_lag168` | Flujo neto en la hora del objetivo, un día y una semana antes |
| `neto_media3`, `neto_media24` | Media móvil del flujo neto en las últimas 3 y 24 horas hasta t |
| `neto_media168_misma_hora` | Media del flujo neto en la hora del objetivo, de una a cuatro semanas antes |
| `ocupacion_lag24`, `ocupacion_lag168` | Ocupación estimada en la hora del objetivo, un día y una semana antes |
| `est_tasa_sin_bicis`, `est_tasa_sin_anclajes` | Proporción de horas de la estación en cada clase de riesgo durante el entrenamiento |
| `est_neto_medio_hora` | Flujo neto medio de la estación en esa hora durante el entrenamiento |

**Regla contra la fuga de información.** Todas las variables usan datos hasta el cierre de t; solo el calendario es del instante objetivo, que se conoce de antemano. Las tres `est_*` se calculan con el periodo de ajuste para validar y con el entrenamiento final para la prueba, y se guardan en `data/stats_estacion.csv` y `data/stats_estacion_hora.csv` del run. Una estación sin historia en el entrenamiento queda con esas tres vacías, y el modelo las trata como valores faltantes.

El modelo no recibe el identificador de la estación ni su ubicación. Tampoco clima ni eventos.

## 2.6 Métricas y regla de decisión

En una estación vacía o llena el usuario pierde el viaje, así que las métricas se centran en las dos clases de riesgo: recall (cuántas horas de riesgo detecta), precisión (cuántas alertas son correctas) y PR AUC. El macro F1 resume las tres clases.

| Modelo | Recall medio de riesgo val. | PR AUC medio de riesgo val. | Macro F1 val. | Recall `sin_bicicletas` prueba | Recall `sin_anclajes` prueba | Macro F1 prueba |
|---|---|---|---|---|---|---|
| Referencia: clase mayoritaria | 0.0000 | n/d | 0.3089 | 0.0000 | 0.0000 | 0.3150 |
| Referencia: misma hora, semana previa | 0.4153 | n/d | 0.5846 | 0.3363 | 0.4477 | 0.5712 |
| Referencia: persistencia | 0.7107 | n/d | 0.8137 | 0.6977 | 0.6932 | 0.8041 |
| Regresión logística balanceada | 0.9502 | 0.7868 | 0.7401 | 0.9386 | 0.9498 | 0.7272 |
| **HGB balanceado (champion)** | **0.9518** | **0.8475** | **0.7562** | **0.9418** | **0.9555** | **0.7405** |
| RandomForest balanceado | 0.9419 | 0.8235 | 0.7557 | 0.9259 | 0.9438 | 0.7431 |
| HGB sin pesos (no candidata) | 0.7391 | 0.8485 | 0.8319 | 0.7041 | 0.7410 | 0.8227 |

Las referencias no dan probabilidades, por eso no tienen PR AUC. La persistencia repite la clase de la hora t.

**El campeón detecta casi todas las horas de riesgo, pero cerca de la mitad de sus alertas son falsas.** En prueba su precisión es 0.4738 en `sin_bicicletas` y 0.4993 en `sin_anclajes`, contra 0.7557 y 0.7444 de la persistencia. Por eso la persistencia tiene mejor macro F1 de prueba (0.8041 contra 0.7405), aunque deja pasar alrededor del 30 % de las horas de riesgo.

### La regla de selección

- **Hiperparámetros.** Búsqueda en una rejilla de tres configuraciones del HGB, con el macro F1 de validación (0.7501, 0.7562 y 0.7541). Gana la segunda.
- **Selección.** Entre las tres candidatas balanceadas gana la de mayor recall medio de `sin_bicicletas` y `sin_anclajes` en validación. Si otra queda a menos de 0.005, se consideran empatadas y decide el PR AUC medio de las mismas clases. La prueba y el macro F1 no intervienen. El HGB sin pesos y las referencias se reportan, pero no son elegibles.
- **Promoción.** El alias `champion` solo se mueve después de cargar la versión por su URI inmutable y obtener una clase válida por fila. La versión 1 tiene la etiqueta `comprobacion = ok`.

En esta corrida la regresión logística quedó a 0.0016 del HGB en recall medio, dentro del margen, y el PR AUC medio decidió: 0.8475 contra 0.7868. El RandomForest quedó a 0.0099, fuera del empate.

La regla parte de un **supuesto operativo**: quedarse sin aviso cuando una estación se vacía o se llena cuesta más que una visita de más. Si ese supuesto cambia, la regla y el campeón pueden cambiar.

**Sensibilidad al nivel inicial.** Con nivel inicial de 30 % y de 70 % el campeón sigue siendo el HGB balanceado, con macro F1 de prueba de 0.7606 y 0.7384. En los tres niveles la persistencia tiene mayor macro F1 de prueba (0.8084, 0.8041 y 0.8048). Sin el margen de 0.005, con nivel 30 % la regresión logística superaría al HGB por 0.002 de recall medio de validación.

Fuente: métricas de los runs de la corrida de cierre en el almacén publicado; `reports/figures/p2_resultados.json` para los niveles 30 % y 70 %.

## 2.7 Desempeño desagregado (estaciones nuevas contra existentes)

En P2 una estación es nueva si su primer viaje es del 1 de mayo de 2026 en adelante. En la prueba son 99 (parámetro `estaciones_nuevas_test` del run), y no tienen estadísticas de entrenamiento: sus tres `est_*` llegan vacías. En la muestra de DeepChecks sobre la prueba, el 13.8 % de las filas tiene vacía `est_tasa_sin_bicis` por esa razón.

Periodo de prueba, mayo a septiembre de 2026, campeón:

| Grupo | Macro F1 | Recall `sin_bicicletas` | Recall `sin_anclajes` | Precisión `sin_bicicletas` | Precisión `sin_anclajes` | PR AUC `sin_bicicletas` | PR AUC `sin_anclajes` |
|---|---|---|---|---|---|---|---|
| Todas | 0.7405 | 0.9418 | 0.9555 | 0.4738 | 0.4993 | 0.8181 | 0.8343 |
| Nuevas | 0.7415 | **0.9053** | 0.9516 | 0.4705 | 0.4618 | 0.8271 | 0.8113 |

Con la persistencia, en las nuevas el macro F1 es 0.8238 y el recall es 0.7222 en `sin_bicicletas` y 0.6962 en `sin_anclajes`.

**En las nuevas el modelo deja pasar más horas sin bicicletas:** el recall baja de 0.942 a 0.905, mientras el de `sin_anclajes` casi no cambia. El macro F1 es prácticamente igual.

En el A/B de octubre de 2026 (sección 2.9), reentrenar con datos hasta septiembre, que ya incluyen a estas estaciones, **bajó su recall medio de riesgo en 0.0401**.

Lo que no está medido: las métricas de las existentes por separado (el run registra todas y nuevas, y el grupo "todas" incluye a las nuevas), el número de filas de las nuevas en la prueba y el error según las semanas desde la apertura. La consecuencia operativa: en una estación con menos de un mes de operación, una predicción `segura` merece menos confianza que en una consolidada.

Fuente: métricas `test_*` y `test_nuevas_*` de los runs del campeón (`48bb03f4`) y de la persistencia (`6dd63d632f464d65af8ff4e03a0e04a6`); sección "Hallazgos de P2 con los datos reales" del README.

## 2.8 Limitaciones y riesgos

- **Todo se mide contra una etiqueta estimada.** El error de la ocupación estimada no se puede medir porque no hay inventario contra el cual compararla. Un recall de 0.94 quiere decir que el modelo coincide con la aproximación, no que acierte el estado real de la estación.
- **El modelo depende casi por completo de `ocupacion_actual`.** En la importancia por permutación (caída en macro F1 sobre 150,000 filas de prueba) esa variable vale 0.364; ninguna otra pasa de 0.005 y 11 dan valores negativos. Cualquier error en la estimación de la ocupación pasa directo a la predicción.
- **Muchas falsas alarmas.** Cerca de la mitad de las alertas son falsas (sección 2.6). Si el equipo de rebalanceo alcanza pocas visitas por turno, la lista puede ser más larga de lo que puede atender. El HGB sin pesos da otro balance (precisión cercana a 0.78 y recall cercano a 0.72), pero no es candidato con la regla actual.
- **La persistencia gana en macro F1** en los tres niveles iniciales (sección 2.6). El modelo solo es preferible si se acepta el supuesto de que una alerta perdida cuesta más que una falsa.
- **La prueba no está intacta**, como se explica en 2.4.
- **Capacidad fija.** Se usa la capacidad de septiembre de 2026 para todo el periodo. Un cambio de anclajes en una estación antes de esa fecha no queda reflejado.
- **Error mayor en estaciones nuevas** para `sin_bicicletas`, documentado en 2.7.
- **Sin clima ni eventos.** Lluvias, eventos masivos o cierres viales quedan fuera de lo que el modelo puede anticipar.
- **Equidad territorial sin evaluar.** El error no se midió por zona ni por municipio. No se sabe si el modelo falla más en unas partes de la ciudad que en otras.
- **Quién arma las variables.** La API no calcula el panel: quien la consume debe armar las 18 variables con `mibici.features_p2` y unir las estadísticas del run con `features_p2.unir_stats`.
- **La API no da probabilidades.** El cargador compartido es pyfunc y devuelve solo la clase, así que el punto de corte no se puede ajustar al servir.
- **Evaluación fuera de línea.** No hay tráfico real. El Registry del que la API carga `champion` es la copia local que siembra `registry-init` a partir del almacén publicado.

## 2.9 Monitoreo

**Deriva de datos.** Con Evidently, comparando las 18 variables de octubre de 2026 (262,404 filas estación-hora de 471 estaciones, 100 nuevas) contra las filas de entrenamiento del campeón. Además del umbral de 0.1 de Evidently, cada variable lleva un umbral calibrado, definido como la mayor distancia de un mes del entrenamiento contra los demás. Las distancias se separan por estaciones existentes y nuevas y se comparan contra octubre de 2025 como control estacional.

Resultado de octubre de 2026: **con el umbral calibrado hay deriva en cuatro variables**: `salidas_lag1` (0.187 contra 0.085), `llegadas_lag1` (0.177 contra 0.080), `neto_media24` (0.128 contra 0.083) y `capacidad` (0.107 contra 0.002, por las estaciones nuevas). El calendario no cambia.

**A/B testing.** `p2_ab` compara el campeón (A) contra el mismo HGB balanceado reentrenado con fechas objetivo de enero de 2025 a septiembre de 2026 (B), sobre las mismas filas de octubre: recall, precisión, F1 y PR AUC por clase de riesgo, macro F1 y Brier multiclase. Las diferencias llevan un intervalo del 95 % por bootstrap pareado por estación (1,000 réplicas, semilla 29).

La regla se fijó en el commit `34fe4f8` del 13 de noviembre de 2026, antes del commit que agregó la comparación: B se puede promover si el recall medio de riesgo sube al menos 0.005 con intervalo sobre cero; si la diferencia queda por debajo de 0.005, decide el PR AUC medio, también con intervalo sobre cero. Es el mismo margen de la selección del campeón. Cumplir la regla permite proponer B; no lo promueve.

| Métrica (B − A) | Diferencia | IC del 95 % |
|---|---|---|
| Recall medio de riesgo | +0.0029 | [+0.0003, +0.0058] |
| PR AUC medio de riesgo | +0.0022 | [+0.0007, +0.0037] |

Resultado de octubre de 2026: la diferencia de recall queda dentro del margen, así que decide el PR AUC, y su intervalo queda sobre cero. **B cumple la regla.** El A/B no mueve alias: B quedó registrada como versión 2 `challenger` y `champion` sigue en la versión 1, que es la que sirve la API. Hay dos señales que la regla no mide y que conviene vigilar antes de promover: el Brier empeora +0.0026 (B está un poco peor calibrada) y en las 99 estaciones nuevas para A el recall medio de riesgo baja 0.0401.

**Qué vigilar cada mes.** El ML Canvas de P2 fija macro F1 y recall de las clases de riesgo, proporción de clases predichas y desempeño en estaciones nuevas. No hay un umbral numérico fijado para reentrenar: hoy la señal para proponer una versión nueva es la regla del A/B.

**Operación y costo.** P2 es la pregunta que más inferencias genera: con 468 estaciones y 18 horas al día son unas 256 mil de las 271 mil inferencias al mes del sistema. En la prueba de carga, a la capacidad sostenida, `/api/v1/p2/predict` respondió en 76 ms en promedio y 270 ms en p95, y un lote de 468 estaciones tardó 52 ms. En lotes, P2 baja de unas 256 mil a 547 peticiones al mes. El costo por inferencia se calcula para la API completa en la sección de FinOps del README.

Salidas del monitoreo: `reports/monitoring/2026_10/p2/` (fuera de control de versiones).

---

# Ficha 3 · P3 · Grupos de estaciones por perfil de uso

## 3.1 Identificación

| Campo | Valor |
|---|---|
| Modelo registrado | `mibici-equipo29-p3-grupos` |
| Alias | `champion` |
| Versión | 1 |
| Tipo | Agrupamiento (KMeans), k = 3 |
| Hiperparámetros | `k = 3`, `n_init = 20`, semilla 42 |
| Run de origen | `fd704e3c5ecc4bbe818a5fb17db9cb32`, nombre `p3-kmeans-uso-k3` |
| Commit | `42918951d913fd2a27949a7237a381536242b11e` (árbol limpio) |
| URI del modelo | `models:/m-578c6f69d09c4bdf834e9cd84866ba04` |
| Huella del modelo serializado | SHA256 `6b0d9ca141ab94b0070a2f1b650d5c92862f1634f4ed2e3f999d19301d33c066` |
| Candidata seleccionada | C2, primera del ranking |

Fuente: `scripts/p3/campeon_api_p3.json` y las etiquetas del run en el almacén publicado.

**Sobre la numeración de versiones.** Esta ficha documenta la versión 1, que es la que está en producción: es la que trae el almacén publicado como asset del Release `almacen-entrega2-r1` y la que `registry-init` siembra al levantar el sistema. El Registry de desarrollo que se usó para el A/B de octubre acumuló versiones adicionales, y ahí el mismo campeón figura con otro número. La identidad que no cambia entre los dos es el `run_id` y la huella SHA256 del modelo serializado.

La huella permite además que la siembra del Registry verifique que el almacén contiene esta versión y no otra: si no coincide, `registry-init` falla de forma explícita y la API no arranca con un Registry equivocado.

## 3.2 Uso previsto

Agrupar las estaciones del sistema según su perfil de uso —cuánto se usan, cuándo, y si se vacían o se llenan en cada franja del día— para poder describirlas por cómo se comportan y no solo por dónde están.

El uso previsto es **descriptivo y de planeación**: entender qué tipos de estación existen, comparar esa tipología con la geográfica, y dar un vocabulario común para hablar del sistema. La salida del modelo es una etiqueta de grupo por estación.

Usuarios previstos: el equipo del proyecto y, en un escenario real, un área de planeación de la operación.

## 3.3 Fuera de alcance

- **No predice nada.** No estima demanda, ni disponibilidad, ni riesgo. Un grupo no es un pronóstico.
- **No sirve para decidir dónde abrir estaciones nuevas.** Además de que no predice, la sección 3.7 muestra que el modelo asigna las estaciones recién abiertas casi siempre al grupo de menor uso, así que usarlo para evaluar una apertura devuelve una respuesta predecible de antemano.
- **No sirve para evaluar una estación con poca historia.** La elegibilidad exige al menos 60 días activos; por debajo de eso el perfil no es confiable.
- **No sustituye el dimensionamiento de capacidad.** Describe el uso observado, no cuántos anclajes debería tener una estación.
- **No dice nada sobre personas.** La unidad es la estación. No hay atributos de usuario en el modelo.

## 3.4 Datos de entrenamiento y evaluación

| | Entrenamiento | Evaluación |
|---|---|---|
| Periodo | 1 ene 2025 – 30 abr 2026 | 1 may 2026 – 30 sep 2026 |
| Estaciones elegibles | 370 | 453 |
| De ellas, nuevas | 0 | 84 |

La partición es temporal y no aleatoria, porque el corte tiene un motivo: las aperturas de la expansión empiezan en mayo de 2026, así que el periodo de evaluación es el primero que contiene estaciones nuevas. **Todo lo que se elige —variables, hiperparámetros y campeón— usa solo el periodo de entrenamiento.** El de evaluación únicamente se reporta.

**Elegibilidad:** una estación entra si tiene ubicación conocida en la nomenclatura y al menos 60 días con viajes en el periodo. El umbral viene de que el error estándar del perfil se estabiliza alrededor de ahí.

**Validación:** KFold de 5 particiones por estación sobre el periodo de entrenamiento, con margen de empate de 0.01 en silueta.

Versión de los datos: `66dd3d190bba`. Nomenclatura: `nomenclatura_2026_09.csv`.

## 3.5 Variables

**Entrada del modelo (12 variables)**, todas derivadas de los flujos horarios por estación:

`log_trips_day`, `weekend_share`, `net_am`, `net_pm`, `dep_am`, `dep_pm`, `dep_mid`, `dep_night`, `arr_am`, `arr_pm`, `arr_mid`, `arr_night`

Franjas horarias: mañana 6–9, media 11–15, tarde 17–20, noche 21–5. Las horas 10 y 16 quedan fuera a propósito, por ser de transición entre franjas.

**Subconjunto para evaluación (7 variables)**, el que queda tras filtrar por factor de inflación de varianza con umbral 5:

`log_trips_day`, `weekend_share`, `net_am`, `net_pm`, `dep_pm`, `dep_night`, `arr_night`

La distinción importa: el modelo consume las 12, y las métricas de calidad del agrupamiento se calculan sobre las 7 para que la colinealidad entre franjas no infle artificialmente la separación entre grupos.

Las variables se estandarizan antes del agrupamiento.

## 3.6 Métricas y regla de decisión

Las métricas son las propias del agrupamiento: silueta (separación entre grupos), Davies-Bouldin (menor es mejor), Calinski-Harabasz, y un índice Rand ajustado de estabilidad, que mide si el agrupamiento se repite al cambiar la partición.

| Métrica | Entrenamiento | Validación | Evaluación |
|---|---|---|---|
| Silueta (espacio de uso) | 0.1878 | 0.1859 ± 0.0247 | 0.1681 |
| Davies-Bouldin | 1.6207 | 1.6147 | 1.6700 |
| Calinski-Harabasz | 102.73 | 21.996 | 121.90 |
| ARI de estabilidad | — | 0.7026 | — |

Calinski-Harabasz no se compara entre periodos porque crece con el número de observaciones, y el periodo de evaluación tiene más estaciones.

### La regla de selección

El campeón **no se eligió por la mejor métrica**, sino por la primera del ranking que cumple cuatro criterios de interpretabilidad fijados antes de correr:

- **I1, tamaño:** cada grupo con al menos 5 % de las estaciones.
- **I2, distinguibles:** los grupos se separan por al menos 0.5 desviaciones estándar.
- **I3, nombrables:** cada grupo recibe un nombre por sus dos variables dominantes, y no hay dos grupos con el mismo nombre.
- **U1, utilidad:** al menos un grupo que se vacía y otro que se llena por la mañana, que es lo que haría accionable un rebalanceo.

El ranking ordena por silueta de validación; a menos de 0.01 de la mejor decide la estabilidad, luego Davies-Bouldin y luego la menor complejidad.

| Candidata | Pos. | Papel | Modelo | k | Silueta val. | ARI val. | Davies-Bouldin val. |
|---|---|---|---|---|---|---|---|
| C2 | 1 | champion | KMeans sobre uso | 3 | 0.1859 | 0.7026 | 1.6147 |
| C1 | 2 | challenger | KMeans sobre uso | 5 | 0.1669 | 0.5997 | 1.5535 |
| C3 | 3 | — | Mezcla gaussiana (tied) | 3 comp. | 0.1548 | 0.2703 | 1.8591 |

Las tres cumplen los cuatro criterios. C2 gana por silueta y, sobre todo, por estabilidad: 0.70 contra 0.60. La mezcla gaussiana tiene mejor Davies-Bouldin en un caso pero una estabilidad muy inferior, 0.27.

Fuente: `data/criterios_interpretabilidad.csv` en los artefactos del run.

### Los tres grupos

Los nombres **los genera el código**, no una persona: `seleccion_p3.py::nombre_grupo()` los arma con las dos variables dominantes del grupo y el signo del flujo.

| Grupo | Estaciones | Proporción | Nombre | Viajes/día | Flujo neto mañana |
|---|---|---|---|---|---|
| 0 | 97 | 26.2 % | se vacía en la mañana · llegadas de noche | 12.3 | −0.45 |
| 1 | 129 | 34.9 % | se vacía en la tarde · salidas de tarde | 25.5 | +0.20 |
| 2 | 144 | 38.9 % | mucho uso | 52.2 | −0.07 |

El eje que más separa a los grupos es la intensidad de uso: del grupo 0 al 2 los viajes por día se multiplican por más de cuatro.

## 3.7 Desempeño desagregado (estaciones nuevas contra existentes)

Calculado sobre `data/asignacion_estaciones.csv` de los artefactos del run, que trae una columna por estación indicando periodo y si es nueva. Periodo de evaluación, 453 estaciones de las cuales 84 son nuevas:

| Grupo | Existentes | Nuevas | % de nuevas dentro del grupo |
|---|---|---|---|
| 0 se vacía en la mañana | 95 | 57 | 37.5 % |
| 1 se vacía en la tarde | 120 | 24 | 16.7 % |
| 2 mucho uso | 154 | 3 | **1.9 %** |

Visto desde el otro lado:

| | Grupo 0 | Grupo 1 | Grupo 2 |
|---|---|---|---|
| Reparto de las 84 nuevas | 67.9 % | 28.6 % | **3.6 %** |
| Reparto de las 369 existentes | 25.7 % | 32.5 % | **41.7 %** |

**El modelo prácticamente nunca asigna una estación nueva al grupo de mucho uso:** 3 de 84, contra el 41.7 % de las existentes. Las nuevas se concentran en el grupo de menor uso.

Hay dos lecturas y ninguna se puede descartar con los datos disponibles:

1. **El modelo tiene razón.** Una estación abierta hace pocos meses efectivamente tiene poco uso, y agruparla con las de poco uso es correcto.
2. **El modelo no puede hacer otra cosa.** Con pocos meses de historia, el perfil de una estación nueva no alcanza a parecerse al de una consolidada, independientemente de hacia dónde vaya su demanda.

Para distinguirlas haría falta observar esas mismas estaciones dentro de un año. La consecuencia operativa, sin embargo, es la misma en los dos casos y está declarada en la sección 3.3: aplicado a una estación recién abierta, el modelo responde "poco uso" de manera casi determinista, así que no debe usarse para evaluar aperturas.

## 3.8 Limitaciones y riesgos

- **La silueta es baja en términos absolutos** (0.19 en entrenamiento, 0.17 en evaluación). Las estaciones no forman grupos nítidos: forman un continuo de intensidad de uso que el modelo corta en tres. Los grupos son una descripción útil, no una partición natural del sistema.
- **Caída en evaluación.** La silueta baja de 0.1878 a 0.1681 entre entrenamiento y evaluación. La parte de esa caída atribuible a las estaciones nuevas no está medida por separado.
- **Sesgo hacia el grupo de bajo uso en estaciones nuevas**, documentado en 3.7.
- **Los viajes de ida y vuelta se cuentan** (6.17 % del total, parte común sección 1). Inflan el volumen de uso de una estación sin aportar flujo neto, y afectan más a las estaciones recreativas.
- **Los nombres de los grupos son automáticos.** Describen las dos variables dominantes, no una interpretación urbana. "Mucho uso" dice lo que dice; no afirma que sean estaciones de centro.
- **Las horas 10 y 16 quedan fuera** de las franjas. Es una decisión deliberada que descarta información.
- **La nomenclatura es la de septiembre de 2026.** Una estación que no aparezca ahí no es elegible aunque tenga viajes.

## 3.9 Monitoreo

**Deriva de datos.** Con Evidently, comparando el perfil mensual de cada estación contra los meses de entrenamiento. La unidad es el perfil de un mes, no de 16, porque un perfil mensual es más ruidoso y hay que compararlo contra algo comparable. Entra una estación con ubicación conocida y al menos 15 días con viajes en el mes.

El umbral por omisión de Evidently (0.1) **no sirve en este caso**: marca deriva en 6 de los 16 meses del propio periodo de entrenamiento, o sea que señalaría cambio donde solo hay estacionalidad. Por eso cada variable lleva además un umbral calibrado, definido como la mayor distancia observada entre un mes del entrenamiento y los demás.

Resultado de octubre de 2026: Evidently marcó deriva en 10 de las 12 variables; **con el umbral calibrado quedan tres**: `log_trips_day` (0.244 contra 0.166), `net_pm` (0.214 contra 0.096) y `net_am` (0.115 contra 0.075).

**A/B testing.** `p3_ab` compara el campeón contra el mismo modelo reentrenado hasta septiembre de 2026, sobre las mismas estaciones de octubre: silueta y Davies-Bouldin con las 12 variables, estabilidad entre quincenas, e intervalos del 95 % por bootstrap pareado.

La regla, fijada antes de comparar: el retador reemplaza al campeón solo si mejora la silueta con un intervalo que no toca el cero, no es menos estable, y ningún grupo baja del 5 % de las estaciones.

Resultado de octubre de 2026: el retador **no mejora la silueta** (+0.006, IC [−0.004, +0.017]) y **es menos estable** (ARI −0.124, IC [−0.226, −0.030]). Se mantiene el campeón. El retador quedó registrado como `challenger` y el alias `champion` no se movió.

**Qué dispararía una revisión.** Deriva sostenida con umbral calibrado en `log_trips_day` o en las variables de flujo neto durante varios meses seguidos, o un retador que cumpla la regla completa. La deriva de octubre es coherente con la expansión del sistema —hay más estaciones y son distintas— y no con un cambio en el fenómeno, así que por sí sola no justifica reentrenar.

Salidas del monitoreo: `reports/monitoring/2026_10/p3/` (fuera de control de versiones).

---

## Fuentes de este documento

| Sección | Fuente |
|---|---|
| Datos, calidad y limpieza | `data/bitacora_limpieza.csv` y `config/config_corrida.json` en los artefactos del run `fd704e3c`; `mibici/features_p3.py` |
| Validación independiente | Fichas del IIEG, https://iieg.gob.mx/ns/?page_id=25284 |
| Procedencia | https://mibici.net/es/datos-abiertos/ y la página de términos del mismo sitio, consultadas el 24/11/2026 |
| Minimización de datos | `mibici/features_p3.py`, constantes `COLUMNAS` y `COLUMNAS_PERSONALES` |
| Identificación de P2 | `scripts/p2/campeon_api_p2.json`; etiquetas, parámetros y descripción de la versión en `.model-store/mlflow.db` |
| Conjuntos, panel y etiqueta de P2 | Parámetros `n_train`, `n_test` y `estaciones_nuevas_test` del run `48bb03f4`; `data/p2_panel_horario.json` y `config/config_p2.json` en sus artefactos; `mibici/features_p2.py` y `CONFIG_P2` en `mibici/config.py` |
| Métricas de P2 y desagregado | Métricas de los runs de la corrida de cierre en `.model-store/mlflow.db`; `reports/figures/p2_resultados.json` para los niveles 30 % y 70 % |
| Importancia de variables de P2 | `data/p2_importancia_permutacion_ini50.csv` del run `48bb03f4` |
| Monitoreo y operación de P2 | Secciones "P2: drift y A/B testing (octubre de 2026)", "Hallazgos de P2 con los datos reales" y "Prueba de carga y análisis FinOps" del README en `main`; `reports/carga/2026-11-20_2224/finops.json` |
| Identificación de P3 | `scripts/p3/campeon_api_p3.json`; etiquetas y parámetros del run en `.model-store/mlflow.db` |
| Métricas de P3 | Métricas del run en `.model-store/mlflow.db` |
| Candidatas y criterios | `data/criterios_interpretabilidad.csv` y `data/grupos_interpretabilidad.csv` |
| Perfil de los grupos | `data/perfil_uso_por_grupo.csv` |
| Desagregado | `data/asignacion_estaciones.csv` |
| Monitoreo de P3 | Sección "P3: drift y A/B testing" del README en `main` |
