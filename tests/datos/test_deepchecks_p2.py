"""Drift de P2 entre train y test con DeepChecks: por variable, de la etiqueta y multivariado.

Revisa las tablas que deja ``python -m mibici.modeling.exportar_p2`` en ``data/interim``.
Corre en el entorno aparte de DeepChecks (Python 3.11, ``requirements-deepchecks.txt``); en el
entorno del proyecto se salta sola porque deepchecks no esta instalado. Tambien se salta si
todavia no existen las tablas.

El reporte HTML y su resumen quedan en ``reports/deepchecks/`` (no se versionan).

No importa nada de ``mibici``: ese paquete necesita las librerias del entorno del proyecto.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("deepchecks", reason="requiere el entorno de DeepChecks (ver README)")

from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import FeatureDrift, LabelDrift, MultivariateDrift

RAIZ = Path(__file__).resolve().parents[2]
INTERIM = RAIZ / "data" / "interim"
REPORTES = RAIZ / "reports" / "deepchecks"
PARTICIONES = ("train", "test")
MUESTRA = 100_000  # filas por particion; el panel completo tiene millones
SEMILLA = 29
CATEGORICAS = ["hora", "dia_semana", "fin_semana"]
# Umbrales de alarma: el drift medido con los datos reales queda muy por debajo (ver README)
MAX_DRIFT_VARIABLE = 0.2
MAX_DRIFT_ETIQUETA = 0.2
MAX_AUC_DOMINIO = 0.75


@pytest.fixture(scope="module")
def particiones() -> dict[str, pd.DataFrame]:
    """Muestra de train y de test con las features de P2 y el objetivo."""
    rutas = {p: INTERIM / f"p2_{p}.parquet" for p in PARTICIONES}
    faltantes = [r.name for r in rutas.values() if not r.exists()]
    if faltantes:
        pytest.skip(
            f"No existen {faltantes} en data/interim. Generalos con "
            "'python -m mibici.modeling.exportar_p2' en el entorno del proyecto."
        )
    return {p: pd.read_parquet(r).sample(MUESTRA, random_state=SEMILLA) for p, r in rutas.items()}


def test_drift_entre_train_y_test(particiones):
    """Mide el drift, guarda el reporte y falla solo si supera los umbrales de alarma."""
    datasets = [
        Dataset(particiones[p], label="objetivo", cat_features=CATEGORICAS) for p in PARTICIONES
    ]
    suite = Suite("Drift P2", FeatureDrift(), LabelDrift(), MultivariateDrift())
    resultado = suite.run(*datasets)
    REPORTES.mkdir(parents=True, exist_ok=True)
    ruta = REPORTES / "p2_drift.html"
    ruta.unlink(missing_ok=True)  # deepchecks no reemplaza el archivo
    resultado.save_as_html(str(ruta), as_widget=False)

    por_check = {r.get_header(): r.value for r in resultado.results if hasattr(r, "value")}
    assert {"Feature Drift", "Label Drift", "Multivariate Drift"} <= por_check.keys()
    por_variable = {v: float(d["Drift score"]) for v, d in por_check["Feature Drift"].items()}
    etiqueta = float(por_check["Label Drift"]["Drift score"])
    multi = por_check["Multivariate Drift"]
    importancia = multi["domain_classifier_feature_importance"]
    resumen = {
        "filas_por_particion": MUESTRA,
        "drift_por_variable": {
            v: round(s, 4) for v, s in sorted(por_variable.items(), key=lambda kv: -kv[1])
        },
        "drift_etiqueta_cramers_v": round(etiqueta, 4),
        "auc_clasificador_de_dominio": round(float(multi["domain_classifier_auc"]), 4),
        "drift_multivariado": round(float(multi["domain_classifier_drift_score"]), 4),
        "importancia_dominio": {
            v: round(float(s), 4)
            for v, s in sorted(importancia.items(), key=lambda kv: -kv[1])[:5]
        },
    }
    (REPORTES / "p2_drift_resumen.json").write_text(json.dumps(resumen, indent=2) + "\n")

    assert max(por_variable.values()) < MAX_DRIFT_VARIABLE, resumen["drift_por_variable"]
    assert etiqueta < MAX_DRIFT_ETIQUETA
    assert resumen["auc_clasificador_de_dominio"] < MAX_AUC_DOMINIO
