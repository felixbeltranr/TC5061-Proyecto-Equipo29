Mínimo y mediana de 50 peticiones por tamaño, sobre una conexión reutilizada y con el sistema en reposo salvo la propia sonda. El costo se calcula con el mínimo, la mejor estimación del tiempo de servicio. Tarifa: Google Cloud Run, facturación por petición (nivel 1), 1 vCPU y 2 GB. La demanda de P3 es de 468 inferencias al mes.

| Estaciones por petición | Mínimo (ms) | Mediana (ms) | ms por inferencia | Veces más barato | USD por 1,000 inferencias | Peticiones al mes | USD al mes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.03 | 5.19 | 4.0310 | 1.0 | 0.003300 | 468 | 0.0015 |
| 10 | 4.21 | 5.07 | 0.4214 | 9.6 | 0.000330 | 47 | 0.0002 |
| 100 | 5.51 | 6.45 | 0.0551 | 73.2 | 0.000033 | 5 | 0.0000 |
| 468 | 11.48 | 13.87 | 0.0245 | 164.3 | 0.000007 | 1 | 0.0000 |

El primer tamaño se midió dos veces, al principio y al final del barrido: la diferencia de -0.35 ms entre ambas es el costo de inicialización del camino de inferencia, que así queda separado del tiempo de servicio que se costea.
