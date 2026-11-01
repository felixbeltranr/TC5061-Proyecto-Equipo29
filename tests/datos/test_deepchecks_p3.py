"""Validación de los perfiles de estaciones de P3 con DeepChecks: integridad y drift.

Revisa las tablas que deja ``python -m mibici.features_p3`` en ``data/interim`` (o
``mlflow run . -e train``), no datos fabricados. Corre en el entorno aparte de DeepChecks
(Python 3.11, ``requirements-deepchecks.txt``); en el entorno del proyecto se salta sola porque
deepchecks no está instalado. También se salta si todavía no existen los perfiles.

Los reportes HTML quedan en ``reports/deepchecks/`` (no se versionan).

No importa nada de ``mibici``: ese paquete necesita las librerías del entorno del proyecto.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("deepchecks", reason="requiere el entorno de DeepChecks (ver README)")

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import PercentOfNulls
from deepchecks.tabular.suites import data_integrity, train_test_validation

RAIZ = Path(__file__).resolve().parents[2]
INTERIM = RAIZ / "data" / "interim"
REPORTES = RAIZ / "reports" / "deepchecks"
PERIODOS = ("entrenamiento", "evaluacion")
# Las 12 variables candidatas con las que entrena P3; deben coincidir con
# CONFIG_P3.prioridad_variables en mibici/config.py. Son proporciones, promedios diarios o
# logaritmos, así que se pueden comparar entre periodos de distinta duración (los conteos
# absolutos como dias_activos o salidas crecen con la duración del periodo y quedan fuera).
VARIABLES = [
    "log_trips_day",
    "weekend_share",
    "net_am",
    "net_pm",
    "dep_am",
    "dep_pm",
    "dep_mid",
    "dep_night",
    "arr_am",
    "arr_pm",
    "arr_mid",
    "arr_night",
]


@pytest.fixture(scope="module")
def perfiles() -> dict[str, pd.DataFrame]:
    """Perfiles de las estaciones elegibles de cada periodo (una fila por estación)."""
    rutas = {p: INTERIM / f"p3_perfiles_{p}.parquet" for p in PERIODOS}
    faltantes = [r.name for r in rutas.values() if not r.exists()]
    if faltantes:
        pytest.skip(
            f"No existen {faltantes} en data/interim. Genera los perfiles con "
            "'mlflow run . -e train -P question=p3 --env-manager local' o "
            "'python -m mibici.features_p3' en el entorno del proyecto."
        )
    return {p: pd.read_parquet(r)[VARIABLES] for p, r in rutas.items()}


def _dataset(df: pd.DataFrame) -> Dataset:
    # P3 es agrupamiento: no hay variable objetivo (label) y todas las variables son numéricas
    return Dataset(df, cat_features=[])


def _suite_integridad():
    """Suite de integridad de DeepChecks más el porcentaje de faltantes por columna.

    La suite estándar solo revisa que los faltantes no vengan escritos de varias formas (Mixed
    Nulls); PercentOfNulls revisa además que no haya ninguno: los perfiles no deben tenerlos.
    """
    suite = data_integrity()
    suite.add(PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(0))
    return suite


def _guardar(resultado, nombre: str) -> Path:
    REPORTES.mkdir(parents=True, exist_ok=True)
    ruta = REPORTES / f"p3_{nombre}.html"
    ruta.unlink(missing_ok=True)  # deepchecks no reemplaza: crearía "p3_... (1).html"
    # HTML plano: el formato de widget de Jupyter (el predeterminado) queda en blanco en el
    # navegador porque deepchecks 0.19.1 espera una versión anterior de ipywidgets
    resultado.save_as_html(str(ruta), as_widget=False)
    return ruta


def _condiciones(resultado) -> dict[str, str]:
    """Condición -> PASS/FAIL/WARN de los chequeos que sí corrieron."""
    return {
        f"{r.get_header()}: {c.name}": c.category.name
        for r in resultado.results
        if hasattr(r, "conditions_results")
        for c in r.conditions_results
    }


# --- integridad: cada tabla por sí sola -----------------------------------------------------


@pytest.mark.parametrize("periodo", PERIODOS)
def test_integridad_de_los_perfiles(perfiles, periodo):
    """Sin faltantes, duplicados, tipos mezclados, columnas constantes ni variables redundantes.

    Los chequeos que necesitan variable objetivo no corren (P3 no tiene) y no cuentan como falla.
    """
    resultado = _suite_integridad().run(_dataset(perfiles[periodo]))
    _guardar(resultado, f"integridad_{periodo}")
    no_pasan = {k: v for k, v in _condiciones(resultado).items() if v != "PASS"}
    assert resultado.passed(fail_if_warning=True, fail_if_check_not_run=False), no_pasan


# --- drift: entrenamiento contra evaluación -------------------------------------------------


def test_drift_entre_entrenamiento_y_evaluacion(perfiles):
    """Mide el drift y lo deja en el reporte; no falla por encontrarlo.

    El drift entre periodos describe cómo cambió el uso de las estaciones (temporada, estaciones
    nuevas), no un error en los datos. La prueba falla solo si los chequeos de drift no corren o
    si la misma fila (perfil idéntico) aparece en los dos periodos.
    """
    resultado = train_test_validation().run(
        _dataset(perfiles["entrenamiento"]), _dataset(perfiles["evaluacion"])
    )
    _guardar(resultado, "drift")
    por_check = {r.get_header(): r for r in resultado.results if hasattr(r, "value")}
    assert {"Feature Drift", "Multivariate Drift"} <= por_check.keys(), "el drift no se calculó"

    por_variable = {
        v: round(float(d["Drift score"]), 4) for v, d in por_check["Feature Drift"].value.items()
    }
    multivariado = por_check["Multivariate Drift"].value
    resumen = {
        "estaciones": {p: len(perfiles[p]) for p in PERIODOS},
        "drift_por_variable_ks": dict(sorted(por_variable.items(), key=lambda kv: -kv[1])),
        "drift_multivariado": round(float(multivariado["domain_classifier_drift_score"]), 4),
        "auc_clasificador_de_dominio": round(float(multivariado["domain_classifier_auc"]), 4),
        "condiciones": _condiciones(resultado),
    }
    (REPORTES / "p3_drift_resumen.json").write_text(json.dumps(resumen, indent=2) + "\n")

    assert set(por_variable) == set(VARIABLES)
    assert all(0 <= s <= 1 for s in por_variable.values())
    assert 0 <= resumen["drift_multivariado"] <= 1
    mezcla = [k for k in resumen["condiciones"] if k.startswith("Train Test Samples Mix")]
    assert mezcla and all(resumen["condiciones"][k] == "PASS" for k in mezcla)
