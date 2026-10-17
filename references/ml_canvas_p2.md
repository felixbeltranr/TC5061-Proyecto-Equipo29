# ML Canvas P2: riesgo de estación sin bicicletas o sin anclajes

| Bloque | Contenido |
|---|---|
| 1. Decisiones | Cada hora, el equipo de rebalanceo elige a qué estaciones ir: retirar bicicletas de las que se van a llenar y llevar a las que se van a vaciar. A mediano plazo, la frecuencia de riesgo orienta ampliaciones de anclajes o de flota. |
| 2. Tareas de predicción | Clasificación multiclase por estación y hora. Para la hora t se predice la clase al cierre de t+1: sin_bicicletas (ocupación estimada < 10 % de la capacidad), sin_anclajes (> 90 %) o segura. |
| 3. Propuesta de valor | Lista priorizada de estaciones en riesgo para el equipo de rebalanceo; menos usuarios sin bicicleta o sin lugar para dejarla. |
| 4. Recolección de datos | Viajes mensuales y nomenclatura descargados con `python -m mibici.dataset`; manifiesto sha256 con `python -m mibici.manifiesto_p2`. |
| 5. Fuentes de datos | Datos abiertos de MiBici, viajes 2025-01 a 2026-09 y nomenclatura 2026-09 (capacidad `dpcapacity`). Sin inventario por estación. Solo se usan fechas y estaciones; no se usan Usuario_Id, Genero ni Año_de_nacimiento. |
| 6. Hacer predicciones | Al cierre de cada hora se actualiza el flujo neto y se predice la hora siguiente (en esta entrega, evaluación fuera de línea). |
| 7. Monitoreo | Macro F1 y recall de clases de riesgo por mes, proporción de clases predichas, desempeño en estaciones nuevas. |
| 8. Simulación de impacto | Comparación contra baselines (clase mayoritaria, semana previa, persistencia) en prueba 2026-05 a 2026-09. |
| 9. Features | Hora y día del objetivo, capacidad, ocupación estimada actual, flujo neto con rezagos 1/24/168 h, medias móviles, ocupación hace 24 y 168 h, estadísticas por estación calculadas solo con entrenamiento (ver `mibici/features_p2.py`). |
| 10. Construcción de modelos | Baselines: mayoritaria, semana previa, persistencia. Candidatos: HistGradientBoostingClassifier con y sin pesos de clase. Partición temporal: train 2025-01 a 2026-04 (validación 2026-03 a 2026-04 para ajuste), prueba 2026-05 a 2026-09. Métrica principal macro F1; recall, precisión y PR AUC por clase de riesgo. |
