# ML Canvas P1: demanda diaria de salidas por estación

| Bloque | Contenido |
|---|---|
| 1. Decisiones | Cada noche, el equipo de operación decide cuántas bicicletas y cuántos turnos de rebalanceo prepara para el día siguiente y qué estaciones requieren más atención. A mediano plazo, la demanda esperada orienta ampliaciones de flota o de anclajes y la programación de mantenimiento. |
| 2. Tareas de predicción | Regresión de conteos por estación y día. Para el día D se predice el número de salidas (viajes cuyo origen es la estación, según la fecha local de `Inicio_del_viaje`). El pronóstico se emite a las 23:00 de D-1 (horizonte de un día). |
| 3. Propuesta de valor | Un pronóstico de demanda por estación con error conocido y comparado contra un baseline simple, para anticipar días de alta o baja demanda (fines de semana, festivos) sin depender de la intuición del turno. |
| 4. Recolección de datos | Viajes mensuales y nomenclatura descargados con `python -m mibici.dataset`; panel estación-día con `python -m mibici.features`. |
| 5. Fuentes de datos | Datos abiertos de MiBici, viajes 2025-01 a 2026-09. Solo se leen `Inicio_del_viaje` y `Origen_Id`; no se usan `Usuario_Id`, `Genero` ni `Año_de_nacimiento`. Se cuentan todos los registros publicados con fecha y estación interpretables (incluye viajes cortos y posibles duplicados). Los timestamps son hora local sin offset. |
| 6. Hacer predicciones | Cada día a las 23:00 se pronostica el total de salidas del día siguiente para cada estación activa (en esta entrega, evaluación fuera de línea con origen de pronóstico móvil: los rezagos de val y prueba se van conociendo conforme pasan los días). |
| 7. Monitoreo | MAE, WAPE y sesgo por mes, por tercil de volumen de la estación y en estaciones nuevas; comparación contra el baseline de semana previa. Hay que vigilar eventos que el modelo no conoce (por ejemplo, los partidos del Mundial 2026 en Guadalajara el 11, 18, 23 y 26 de junio) y cierres o reaperturas de estaciones. |
| 8. Simulación de impacto | Comparación contra los baselines (semana previa D-7 y persistencia D-2) en la prueba de 2026-05 a 2026-09; la selección del modelo se hace solo con la validación de 2026-03 a 2026-04. |
| 9. Features | Día de la semana y festivos del día objetivo; salidas de la estación en D-2 y hace 1, 2, 3 y 4 semanas; medias de los 7 y 28 días más recientes disponibles (hasta D-2) y media del mismo día de la semana. Ninguna variable usa D-1 ni D (ver `mibici/features.py`). |
| 10. Construcción de modelos | Baselines: semana previa (D-7) y persistencia (D-2, el último día completo al emitir). Candidatos: regresión de Poisson regularizada y gradient boosting con pérdida de Poisson. Partición temporal igual que P2: entrenamiento 2025-01 a 2026-04 (validación 2026-03 a 2026-04 para ajuste; con un día de embargo para no usar etiquetas que aún no existen), prueba 2026-05 a 2026-09. Métrica principal MAE en salidas por estación-día; también RMSE, sesgo y WAPE. |

## Supuestos y límites

- Las salidas observadas son demanda satisfecha: si una estación está vacía no registra viajes, así que el conteo subestima la demanda real (demanda censurada).
- Los ceros se crean desde la primera salida de cada estación, bajo el supuesto de que estuvo operando todos los días; no hay inventario ni historial de cierres.
- Un día entra a la evaluación solo si la estación tiene al menos 29 días de historia y alguna salida en los últimos 28 días disponibles; las estaciones nuevas se reportan por separado.
- Estación nueva es la que no tiene ningún día elegible en el entrenamiento (en la práctica, la que registró su primera salida después del 31 de marzo de 2026). P2 usa otra regla (primera salida dentro de la prueba), así que el número de estaciones nuevas no se compara entre preguntas.
- No se incluyen clima ni eventos especiales; la prueba cubre meses de lluvias y el Mundial 2026, que el modelo no conoce.
