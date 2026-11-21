| Usuarios | RPS sostenido | p50 (ms) | p95 (ms) | p99 (ms) | Errores | CPU API p95 (núcleos) | Memoria API máx. (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 7.8 | 23 | 170 | 480 | 0.00% | 0.48 | 265 |
| 15 | 11.4 | 26 | 310 | 740 | 0.00% | 0.77 | 265 |
| 20 | 15.0 | 29 | 360 | 970 | 0.00% | 0.90 | 263 |
| 30 | 17.9 | 320 | 1100 | 1400 | 0.00% | 1.02 | 269 |
| 50 | 18.6 | 1400 | 2400 | 2800 | 0.00% | 1.02 | 273 |

Primera inferencia de cada pregunta después de que /ready responde 200 (no incluye el arranque del contenedor ni la carga de los modelos): P1 40 ms, P2 19 ms, P3 43 ms.

Medido: capacidad sostenida de 14.2 RPS de inferencias (15.0 RPS en total con las sondas GET /live) con 20 usuarios, p95 de 360 ms, 0.53 núcleos en promedio (0.90 en p95) y 263 MiB de memoria máxima. Saturación observada: 18.6 RPS (17.8 de inferencias) con 50 usuarios y p95 de 2,400 ms. Tamaño recomendado: 1 vCPU y 2 GB, con esa capacidad medida de 14.2 RPS de inferencias. La demanda de MiBici ocuparía el 0.727% de esa capacidad.

| Escenario | Tamaño | USD al mes | Inferencias al mes | USD por 1,000 inferencias |
| --- | --- | ---: | ---: | ---: |
| Piso: instancia recomendada a su capacidad sostenida | 1 vCPU, 2 GB | 36.04 | 37,266,332 | 0.000967 |
| MiBici, siempre encendido (24 x 7) | 1 vCPU, 2 GB | 36.04 | 270,785 | 0.1331 |
| MiBici, encendido 19 h al día (5:00 a 24:00) | 1 vCPU, 2 GB | 28.53 | 270,785 | 0.1054 |
| MiBici, siempre encendido en ARM (Graviton) | 1 vCPU, 2 GB | 28.83 | 270,785 | 0.1065 |
| MiBici, Cloud Run por petición, una estación por petición | 1 vCPU, 2 GB | 1.68 | 270,785 | 0.006195 |
| MiBici, Cloud Run por petición, P2 y P3 en lote | 1 vCPU, 2 GB | 0.09 | 270,785 | 0.000332 |
| Servidor de MLflow encendido 24 x 7 (la API carga los modelos al arrancar) | 0.25 vCPU, 1 GB | 10.63 | n/d | n/d |
| Servidor de MLflow 24 x 7 como lo levanta compose.yaml, con su ejecutor de trabajos | 0.5 vCPU, 3 GB | 24.51 | n/d | n/d |

**Optimizaciones**

- La demanda de MiBici (270,785 inferencias al mes) ocupa el 0.727% de la capacidad sostenida medida de la instancia (14.2 RPS de inferencias): de los 36.04 USD al mes de tenerla siempre encendida, 35.78 pagan tiempo ocioso (99.3%). Conviene escalar a cero o facturar por petición.
- Encender la instancia sólo 19 h al día (5:00 a 24:00) baja el costo de 36.04 a 28.53 USD al mes, un ahorro del 20.8%.
- Con el mismo tamaño en ARM (Graviton) la instancia cuesta 28.83 USD al mes en vez de 36.04, un ahorro del 20.0%; requiere validar la construcción de la imagen y el rendimiento de la API en linux/arm64.
- Cloud Run con facturación por petición, una estación por petición (270,785 peticiones al mes, cada pregunta con su latencia medida), cuesta 1.68 USD al mes, un ahorro del 95.3% frente a 36.04 USD; si P2 y P3 se envían en lote (14,775 peticiones al mes en vez de 270,785), baja a 0.09 USD al mes, un ahorro del 99.8%. Ambos sin contar el nivel gratuito. Al escalar a cero, la primera petición paga además el arranque del contenedor y la carga de los modelos desde el Registry, que esta prueba no mide, y la facturación por petición también cobra el tiempo de arranque y apagado de la instancia.
- El servidor de MLflow (0.25 vCPU y 1 GB) encendido 24 x 7 cuesta 10.63 USD al mes. Hoy no se puede apagar: la API resuelve el alias y carga los modelos del Registry cada vez que arranca, así que lo necesita al escalar o reiniciarse. Para ahorrarlo habría que empaquetar el modelo campeón en la imagen al desplegar o servir los artefactos desde almacenamiento de objetos; esos 10.63 USD al mes son el ahorro máximo posible.
- Apagar el ejecutor de trabajos de MLflow, que el proyecto no usa (`MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`), baja su memoria máxima de 1,975 a 412 MiB y su tamaño de 0.5 vCPU y 3 GB a 0.25 vCPU y 1 GB: de 24.51 a 10.63 USD al mes, un ahorro del 56.6%.
- En conjunto, la API en Cloud Run por petición con P2 y P3 en lote (0.09 USD) más el servidor de MLflow (10.63 USD, a tarifa de Fargate) suman 10.72 USD al mes, frente a 46.67 USD con la API siempre encendida: un ahorro del 77.0%. El servidor de MLflow pasa a ser el 99.2% de ese total. La suma combina las tarifas de dos proveedores como referencia de precio; con todo en una sola nube, el servidor de MLflow se cobraría con la tarifa de esa nube.
