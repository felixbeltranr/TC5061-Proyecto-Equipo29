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

## 1.1 Identificación

| Campo | Valor |
|---|---|
| Modelo registrado | `mibici-equipo29-p1-demanda` |
| Alias | `champion` |
| Versión | 1 |
| Tipo | Regresión de conteos: `HistGradientBoostingRegressor` de scikit-learn con pérdida de Poisson, `dia_semana` como variable categórica y sin paro temprano |
| Hiperparámetros | `learning_rate = 0.05`, `max_leaf_nodes = 63`, `max_iter = 400`, `l2_regularization = 1.0`, semilla 29 |
| Run de origen | `a8fea49c708d4e849bd92ec54536809f`, nombre `p1_hgb_poisson` |
| Commit | `42918951d913fd2a27949a7237a381536242b11e` (árbol limpio) |
| URI del modelo | `models:/m-27be68580c664b51ad959511e90832f5` |
| Huella del modelo serializado | SHA256 `d0a9f7e17cba6001d5f60e04d2b355c07f7ad270433abf300953cf38188f8127` (`model.skops`, 4.39 MiB) |
| Fecha | 25 de octubre de 2026 (corrida de cierre de la Entrega 2) |
| Código | `mibici/features.py`, `mibici/modeling/train_p1.py` y `mibici/registro_p1.py` |
| Entrada | 11 variables por estación-día (sección 1.5) |
| Salida | `demanda_salidas`: salidas esperadas de la estación en el día D, un número real mayor o igual a cero. La API la devuelve en el campo `prediction` |

Fuente: `scripts/p1/campeon_api_p1.json` y las etiquetas y parámetros del run en el almacén publicado.

**Sobre la numeración de versiones.** Esta ficha documenta la versión 1, que es la que trae el almacén publicado en el Release `almacen-entrega2-r1` y la que `registry-init` siembra al levantar el sistema. La versión 2 (`challenger`) es el mismo modelo, con los mismos hiperparámetros, reentrenado con objetivos hasta el 29 de septiembre de 2026. Existe solo en el Registry local donde se corrió el A/B de octubre, que parte de ese mismo almacén, y no en el que siembra el Compose.

## 1.2 Uso previsto

Cada noche, a las 23:00 del día D−1, pronosticar cuántos viajes saldrán de cada estación activa durante el día D.

El uso previsto es **de planeación operativa**: el pronóstico apoya el rebalanceo del día siguiente, es decir, cuántas bicicletas y turnos preparar y qué estaciones requieren más atención. Agregado por semanas o meses orienta decisiones de flota, anclajes y mantenimiento. La salida del modelo es un número de salidas por estación y día.

Usuarios previstos: el equipo de operación y rebalanceo del sistema y quienes hacen planeación. El modelo informa una decisión que toma una persona; no la sustituye.

## 1.3 Fuera de alcance

- **No pronostica por hora ni dentro del mismo día**, ni a más de un día de anticipación. El riesgo horario de que una estación se quede sin bicicletas o sin anclajes es la pregunta de P2.
- **No pronostica estaciones recién abiertas.** Antes de cumplir la regla de elegibilidad (sección 1.4) una estación no recibe pronóstico.
- **No estima la demanda no atendida.** El modelo aprende de viajes realizados, no de intentos fallidos (sección 1.8).
- **No se traslada a otro sistema ni a otra ciudad** sin reentrenar y volver a evaluar.
- **No sirve para decisiones automáticas sin revisión**, como retirar estaciones o recortar servicio.
- **No dice nada sobre personas.** La unidad es la estación-día. No hay atributos de usuario en el modelo.

## 1.4 Datos de entrenamiento y evaluación

| Conjunto | Fechas objetivo | Filas estación-día | Uso |
|---|---|---|---|
| Ajuste para la selección | hasta el 27 feb 2026 | 144,676 | Entrenar las candidatas que se comparan en validación |
| Validación | 1 mar 2026 – 29 abr 2026 | 22,133 | Elegir hiperparámetros y candidata |
| Entrenamiento final | 30 ene 2025 – 29 abr 2026 | 167,177 | Ajustar el modelo que se registra |
| Prueba | 1 may 2026 – 30 sep 2026 | 62,426 (6,225 de estaciones nuevas) | Diagnóstico del modelo ya elegido |
| Mes posterior | octubre de 2026 | 14,508, de 468 estaciones (369 existentes y 99 nuevas) | Monitoreo y A/B fuera de línea |

La partición es temporal, por fecha objetivo, y la evaluación es con origen de pronóstico móvil: los rezagos de cada día se van conociendo conforme pasan los días. La prueba empieza en mayo de 2026, el mismo mes en que empiezan las aperturas de la expansión, así que es el primer periodo con estaciones nuevas. **Todo lo que se elige —hiperparámetros y campeón— usa solo la validación.**

**La prueba no está intacta.** La política de dominio de la sección 1.6 se agregó después de haber visto resultados de prueba, así que el MAE de prueba se reporta como referencia histórica y no como una medición sobre datos nunca vistos. Octubre de 2026 sí quedó fuera del entrenamiento de las dos versiones que se comparan en la sección 1.9.

**El panel.** La unidad es la estación-día: 245,185 filas de 472 estaciones. P1 lee del dataset combinado solo `Inicio_del_viaje` y `Origen_Id` (`COLUMNAS_VIAJES` en `mibici/features.py`). Dos supuestos condicionan la lectura de los resultados:

- Los días sin viajes cuentan como cero desde la primera salida de cada estación, bajo el supuesto de que operó todos los días. No hay inventario ni historial de cierres.
- Se cuentan todos los registros publicados con fecha y estación interpretables (7,894,097), incluidos los 245,300 viajes de menos de un minuto y los 758 duplicados exactos de septiembre de 2026 que la limpieza descrita en la parte común descarta.

**Elegibilidad:** una fila entra si la estación tiene al menos 29 días de historia y alguna salida en los últimos 28 días disponibles.

Versión de los datos: `66dd3d190bba`, registrada como `data_version` en cada run de P1 y en la versión del Registry. Cubre de enero de 2025 a septiembre de 2026; octubre de 2026 se descargó aparte para el monitoreo y no forma parte de esa versión.

## 1.5 Variables

**Entrada del modelo (11 variables)**, todas derivadas del conteo diario de salidas de la propia estación y del calendario:

| Variable | Qué es |
|---|---|
| `dia_semana` | Día de la semana de D (0 = lunes, 6 = domingo) |
| `festivo`, `festivo_lag7` | 1 si D es festivo; 1 si D−7 lo fue |
| `lag_2` | Salidas en D−2, el último día completo al emitir el pronóstico |
| `lag_7`, `lag_14`, `lag_21`, `lag_28` | Salidas el mismo día de la semana, de una a cuatro semanas atrás |
| `media_7d`, `media_28d` | Media diaria de D−8 a D−2 y de D−29 a D−2 |
| `media_misma_dsem_4` | Media de los cuatro rezagos semanales |

**Regla contra la fuga de información.** A las 23:00 de D−1 ese día todavía no cierra, así que ninguna variable usa datos de D−1 ni de D; solo el calendario del día objetivo, que se conoce de antemano.

El modelo no recibe el identificador de la estación, su ubicación ni su capacidad. Tampoco clima ni eventos.

## 1.6 Métricas y regla de decisión

La métrica principal es el MAE, en viajes por estación-día, y es el criterio de selección entre las candidatas elegibles. Se acompañan RMSE (penaliza los errores grandes de las estaciones de alto volumen), WAPE (suma de errores absolutos entre suma de salidas reales) y sesgo (pronóstico menos valor real; positivo significa que sobrestima).

| Modelo | MAE val. | RMSE val. | WAPE val. | MAE prueba | RMSE prueba | WAPE prueba |
|---|---|---|---|---|---|---|
| Referencia: semana previa (D−7) | 8.4393 | 16.0660 | 26.14 % | 7.9550 | 15.3896 | 25.92 % |
| Referencia: persistencia (D−2) | 11.7063 | 19.9769 | 36.26 % | 11.2166 | 19.0482 | 36.54 % |
| Poisson regularizada | 5.8805 | 11.0907 | 18.22 % | 6.3535 | 12.1162 | 20.70 % |
| **HGB Poisson (champion)** | **5.6099** | **10.5407** | **17.38 %** | **6.2847** | **12.4086** | **20.48 %** |
| HGB absoluto (inelegible) | 5.6326 | 10.6633 | 17.45 % | 6.2989 | 12.3999 | 20.52 % |

Frente a la referencia de semana previa, el campeón baja el MAE 33.5 % en validación y 21.0 % en prueba. Su sesgo de prueba es −0.78 viajes por estación-día: en promedio subestima.

Prueba por mes:

| Mes de 2026 | Mayo | Junio | Julio | Agosto | Septiembre |
|---|---|---|---|---|---|
| MAE del campeón | 5.2258 | 7.6944 | 6.0421 | 5.1558 | 7.3546 |

Junio y septiembre son los meses con más error. La tabla describe la variación y no identifica su causa.

### La regla de selección

La selección tiene dos pasos: antes de comparar el MAE hay una condición de elegibilidad.

- **Política de dominio.** Una candidata solo es elegible si todas sus predicciones de validación son numéricas, finitas y mayores o iguales a cero. La que no cumple conserva su run y sus métricas, pero no puede ser campeona, y su salida no se recorta para hacerla pasar.
- **Selección.** Gana el menor MAE de validación entre la referencia de semana previa y las candidatas elegibles. La prueba no interviene.
- **Promoción.** El alias `champion` solo se mueve después de cargar la versión por su URI inmutable y obtener una predicción válida.

En esta corrida la política de dominio no cambió al ganador, porque HGB Poisson ya tenía el menor MAE de validación (5.6099 contra 5.6326 de HGB absoluto). Sí dejó fuera a HGB absoluto, que dio 22 predicciones negativas en validación: un conteo de salidas no puede ser negativo.

Fuente: métricas de los runs de la corrida de cierre en el almacén publicado.

## 1.7 Desempeño desagregado (estaciones nuevas contra existentes)

En P1 una estación es nueva si no tiene ningún día elegible en el entrenamiento, que termina el 29 de abril de 2026. En la prueba son 101 estaciones (parámetro `estaciones_nuevas_test` del run), todas con su primera salida entre mayo y agosto de 2026: las 99 de la expansión que describe la parte común y dos más que no figuran en la nomenclatura. Aparecen en la prueba conforme juntan sus 29 días de historia y son el 10.0 % de las filas (6,225 de 62,426). En octubre el grupo son exactamente las 99.

Periodo de prueba, mayo a septiembre de 2026:

| Grupo | Filas | MAE del campeón | WAPE del campeón | MAE de semana previa | Cambio en MAE |
|---|---|---|---|---|---|
| Existentes | 56,201 | 6.6119 | 20.17 % | 8.3809 | −21.1 % |
| Nuevas | 6,225 | 3.3310 | **28.01 %** | 4.1099 | −19.0 % |

Las existentes, separadas por tercil de volumen (los terciles se calculan solo con el entrenamiento):

| Segmento | Filas | MAE del campeón | WAPE del campeón | MAE de semana previa | Cambio en MAE |
|---|---|---|---|---|---|
| Volumen bajo | 18,684 | 2.8611 | 28.16 % | 3.6812 | −22.3 % |
| Volumen medio | 18,749 | 5.2900 | 20.90 % | 6.8620 | −22.9 % |
| Volumen alto | 18,768 | 11.6665 | 18.59 % | 14.5768 | −20.0 % |

**El MAE de las nuevas es la mitad del de las existentes, pero su error relativo es mayor:** 28.0 % contra 20.2 %. El MAE bajo refleja su menor volumen, no un mejor pronóstico: de las métricas se deduce que una estación nueva mueve alrededor de 12 salidas al día y una existente alrededor de 33. Como las nuevas son el 10 % de las filas y cerca del 4 % de las salidas, el WAPE global (20.48 %) casi no se separa del de las existentes (20.17 %) y no deja ver su 28 %.

Tres cosas que sí dicen los datos:

1. **El error relativo de las nuevas es el de las existentes de volumen bajo:** 28.0 % contra 28.2 % del tercil bajo, que tiene un volumen parecido (unas 10 salidas al día). La diferencia frente al conjunto de las existentes es coherente con el volumen; los datos disponibles no permiten separar cuánto se debe a la poca historia.
2. **El modelo sigue ganándole a la referencia en las nuevas**, aunque por menos: −19.0 % contra −21.1 % en las existentes.
3. **En el A/B de octubre de 2026 (sección 1.9), reentrenar no mostró mejora en las nuevas.** La versión reentrenada con datos hasta septiembre mejora el MAE en las 369 existentes (−0.168, IC del 95 % [−0.362, −0.038]); en las 99 nuevas la diferencia no es concluyente (+0.012, IC [−0.016, +0.040]).

Lo que no está medido: el error según las semanas transcurridas desde la apertura. Las estaciones con menos de 29 días de historia no reciben pronóstico, así que no tienen error que medir. La consecuencia operativa está declarada en la sección 1.3: durante su primer mes una estación nueva no tiene pronóstico, y después conviene leer el suyo con el margen de una estación de volumen bajo.

Fuente: métricas por segmento (`test_n_*`, `test_mae_*` y `test_wape_*`) del run del campeón y del run de la referencia de semana previa (`3e78d72b8ed94f59b8e4d6dc3b0edfbd`). Las filas de existentes son el agregado de los tres terciles, calculado a partir de esas métricas.

## 1.8 Limitaciones y riesgos

- **Demanda censurada.** Las salidas observadas son demanda atendida. Una estación vacía no registra viajes, así que el historial subestima la demanda justo donde el servicio falla. Si el pronóstico se usara como único criterio para repartir bicicletas, mandaría menos a las estaciones que más se vacían y reforzaría el problema. Por eso se propone leerlo junto con el riesgo de P2 y no usarlo para recortar servicio.
- **Subestimación.** El sesgo es negativo (−0.78 en prueba) y la dispersión del error crece con el nivel del pronóstico: en estaciones grandes un error de decenas de viajes es normal.
- **Sin clima ni eventos.** Días de lluvia fuerte, eventos masivos o cierres viales quedan fuera de lo que el modelo puede anticipar. La prueba cubre meses de lluvias y los partidos del Mundial 2026 en Guadalajara, que el modelo no conoce.
- **Error relativo mayor en estaciones nuevas y de volumen bajo**, documentado en 1.7.
- **Equidad territorial sin evaluar.** El error no se midió por zona ni por municipio, ni por día de la semana o en festivos. No se sabe si el modelo se equivoca más en unas partes de la ciudad que en otras; es el análisis pendiente con más peso antes de un uso real.
- **La prueba no está intacta**, como se explica en 1.4.
- **Horizonte único.** El modelo está validado para un día de anticipación con datos hasta D−2. Si el conteo del día llega con más retraso, las variables ya no corresponden.
- **Quién arma las variables.** La API no calcula las variables ni comprueba la elegibilidad: quien la consume debe armarlas con las reglas de `mibici.features.construir_features`, respetar la regla D−2 y descartar las filas que esa función marca como no elegibles.
- **Evaluación fuera de línea.** No hay tráfico real. El Registry del que la API carga `champion` es la copia local que siembra `registry-init` a partir del almacén publicado, no un Registry compartido.
- **Los viajes cortos y los duplicados se cuentan.** El panel de P1 no aplica la limpieza que describe la parte común (descarte de viajes de menos de un minuto y de duplicados exactos), así que un anclaje fallido o un registro repetido suman una salida.

## 1.9 Monitoreo

**Deriva de datos.** Con Evidently, comparando las 11 variables de octubre de 2026 contra las del entrenamiento. Con el umbral por omisión de Evidently (0.1), 9 de las 11 variables marcan deriva. Con el umbral calibrado por variable, definido como la mayor distancia de un mes de la referencia contra el resto, **quedan cinco** en el total de estaciones: `lag_2` (0.310 contra 0.117), `media_7d` (0.303 contra 0.139), `lag_7` (0.271 contra 0.165), `lag_14` (0.229 contra 0.208) y `media_28d` (0.218 contra 0.197); con las existentes o con las nuevas por separado pasan su umbral las 8 numéricas. Frente a octubre de 2025, la distancia de `lag_2` en las estaciones existentes es 0.382: la diferencia persiste al comparar el mismo mes de dos años seguidos.

**Calidad de datos.** 84 pruebas aprobadas, 0 fallidas y 15 advertencias, todas explicadas en el resumen de la corrida. No hay faltantes.

**A/B testing.** `p1_ab` compara el campeón (A) contra el mismo modelo reentrenado con objetivos hasta el 29 de septiembre de 2026 (B), sobre las mismas 14,508 filas de octubre, que ninguna de las dos versiones vio al entrenar. La diferencia de MAE lleva un intervalo del 95 % por bootstrap pareado por estación (1,000 réplicas, semilla 29).

La regla es la misma que se aplicó al ensayo de septiembre antes de descargar octubre (consta en los registros locales de las corridas, no en un commit previo): B cumple si el intervalo de todas las estaciones queda completo bajo cero, en ningún grupo queda completo arriba de cero y sus predicciones son finitas y no negativas. Cumplir la regla permite proponer B; no la promueve.

| Grupo | Estaciones | B − A (MAE) | IC del 95 % |
|---|---|---|---|
| Todas | 468 | −0.130 | [−0.275, −0.024] |
| Existentes | 369 | −0.168 | [−0.362, −0.038] |
| Nuevas | 99 | +0.012 | [−0.016, +0.040] |

Resultado de octubre de 2026: el MAE es 7.886 con A y 7.756 con B. **B cumple la regla, pero la mejora es pequeña** (1.6 %) y cinco estaciones explican el 83 %. Con bootstrap por día, el intervalo toca el cero con 1,000 réplicas ([−0.226, +0.0002]) y queda bajo cero con 10,000. Se mantiene el campeón: B quedó registrada como `challenger` y el alias `champion` no se movió. La versión 2 se promueve solo si en noviembre de 2026, sin reentrenar, vuelve a cumplir la regla contra A.

El MAE del campeón sube de 6.28 en la prueba (mayo a septiembre) a 7.89 en octubre.

**Qué dispararía una revisión.** Cada mes conviene revisar la deriva calibrada, la calidad de datos y el MAE contra la referencia de semana previa; esta última comparación todavía no forma parte de los módulos de monitoreo. Conviene reentrenar o promover al retador cuando el MAE mensual del campeón deja de ser menor que el de esa referencia o cuando una versión nueva cumple la regla del A/B dos meses seguidos.

Salidas del monitoreo: `reports/monitoring/2026_10/p1/` (fuera de control de versiones).

---

# Ficha 2 · P2 · Riesgo horario de ocupación

`<POR RELLENAR>`

## 2.1 Identificación
`<POR RELLENAR>`

## 2.2 Uso previsto
`<POR RELLENAR>`

## 2.3 Fuera de alcance
`<POR RELLENAR>`

## 2.4 Datos de entrenamiento y evaluación
`<POR RELLENAR>`

## 2.5 Variables
`<POR RELLENAR>`

## 2.6 Métricas y regla de decisión
`<POR RELLENAR>`

## 2.7 Desempeño desagregado (estaciones nuevas contra existentes)
`<POR RELLENAR>`

## 2.8 Limitaciones y riesgos
`<POR RELLENAR>`

## 2.9 Monitoreo
`<POR RELLENAR>`

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
| Identificación de P1 | `scripts/p1/campeon_api_p1.json`; etiquetas y parámetros del run en `.model-store/mlflow.db` |
| Conjuntos de P1 | Parámetros `n_fit`, `n_val`, `n_train` y `n_test` del run `a8fea49c` en `.model-store/mlflow.db` |
| Métricas de P1 y desagregado | Métricas de los runs `a8fea49c` (campeón) y `3e78d72b` (semana previa) en `.model-store/mlflow.db` |
| Panel y variables de P1 | `mibici/features.py`; sección "P1 · demanda diaria por estación" del README en `main` |
| Monitoreo de P1 | Secciones "Resultados históricos de P1 (octubre de 2026)", "P1: A/B testing de versiones del Registry (octubre de 2026)" y "P1: calidad de datos y drift calibrado (octubre de 2026)" del README en `main` |
| Identificación de P3 | `scripts/p3/campeon_api_p3.json`; etiquetas y parámetros del run en `.model-store/mlflow.db` |
| Métricas de P3 | Métricas del run en `.model-store/mlflow.db` |
| Candidatas y criterios | `data/criterios_interpretabilidad.csv` y `data/grupos_interpretabilidad.csv` |
| Perfil de los grupos | `data/perfil_uso_por_grupo.csv` |
| Desagregado | `data/asignacion_estaciones.csv` |
| Monitoreo de P3 | Sección "P3: drift y A/B testing" del README en `main` |
