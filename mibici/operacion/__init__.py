"""Operación del sistema: pruebas de carga, consumo de recursos y análisis FinOps.

- ``payloads``: peticiones válidas y variadas para P1, P2 y P3, que usa el escenario de Locust.
- ``recursos``: muestreo de ``docker stats`` (o ``podman stats``) durante la prueba.
- ``carga``: lectura de los CSV de Locust y resumen por etapa.
- ``finops``: costo por hora, costo por inferencia y dimensionamiento con tarifas de referencia.
- ``graficas``: figuras de desempeño y de recursos para el reporte.
"""
