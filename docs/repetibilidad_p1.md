# Repetibilidad acotada de candidatos P1

El test `tests/modeling/test_reproducibilidad_p1.py` ajusta dos veces cada candidato
(`poisson`, `hgb_poisson`, `hgb_absoluto`) con el mismo panel sintético, orden,
hiperparámetros pequeños fijos en la prueba (no la rejilla de producción) y la semilla
`CONFIG_P1.semilla` (29). Construye las variables con
`mibici.features` y conserva el embargo D-2 (separación de exactamente el rezago mínimo).
Comprueba forma, finitud, dominio no negativo, predicciones cercanas (`rtol=atol=1e-10`),
MAE consistente y que el entrenamiento no modifica los datos de entrada. En los dos HGB
comprueba además que la semilla de la configuración llega al modelo (`random_state`) y que
se completan todas las iteraciones. Son tres casos y seis ajustes de candidatos.

En este test los tres candidatos son deterministas: Poisson usa lbfgs y los HGB corren
sin paro temprano y con menos de 200,000 filas, así que tampoco muestrean para calcular
los cortes. Con otra semilla las predicciones del fixture no cambian; el test confirma
la repetibilidad y que la semilla se propaga, no que la semilla cambie el resultado.

No son dos ejecuciones completas de `train_p1.main`, búsquedas completas de
hiperparámetros, runs MLflow o promociones del registro. No demuestra igualdad
entre versiones de bibliotecas o sistemas operativos ni resultados de datos reales.

## Ejecución

Desde la raíz del repositorio y con el entorno principal Python 3.14 ya
preparado, usar su ejecutable explícito en lugar de `python` si hay varios entornos:

```bash
python -m pytest tests/modeling/test_reproducibilidad_p1.py -v -rs -p no:cacheprovider
```

Confirmar antes que `mibici.features.__file__` y
`mibici.modeling.train_p1.__file__` pertenecen a este checkout y no a otra copia
instalada como editable. No mezclar este test con el entorno DeepChecks Python 3.11.

## Verificación

Windows 11, entorno principal Python 3.14.8, desde la raíz del checkout:

- Sobre main `d2bec0502daaed12762da790ae6d29523124ad32` con sólo este test y esta guía:
  `test_reproducibilidad_p1.py` **3 passed**; suite completa
  `python -m pytest`: **223 passed**, sin omisiones.
- Con el PR #13 (`2438e1b`) y los auxiliares de datos P1 también presentes: este test
  **3 passed**; suite completa **223 passed, 3 skipped**. Las tres omisiones son los
  módulos de DeepChecks (`test_deepchecks_p3.py`, `test_drift_p1.py`,
  `test_integridad_p1.py`), que se saltan solos fuera de su entorno Python 3.11.

`mibici.features` y `mibici.modeling.train_p1` se importaron desde el checkout probado.
No se requieren cambios de producción, dependencias nuevas o el PR #13 para este test.

La ubicación `tests/modeling/` sigue la estructura de los tests nuevos del repositorio;
no reorganiza los tests P1 existentes.
