"""Prueba de humo del analisis de horizontes de P2; todo lo que escribe va a tmp_path."""

import json

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from tests.modeling.test_train_p2 import CORTES_CHICOS, hacer_panel

from mibici.config import CONFIG_P2
from mibici.modeling import horizontes_p2 as hp
from mibici.modeling import train_p2 as t2


def test_main_crea_un_run_padre_con_hijos_por_horizonte_sin_registrar_modelos(
    tmp_path, monkeypatch
):
    for clave, valor in CORTES_CHICOS.items():
        monkeypatch.setitem(t2.CORTES, clave, valor)
    hacer_panel().to_parquet(tmp_path / "panel.parquet", index=False)
    params = {"learning_rate": 0.1, "max_leaf_nodes": 7, "max_iter": 10, "l2_regularization": 0.0}
    (tmp_path / "resultados.json").write_text(json.dumps({"params_hgb": params}))
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    uri_previa = mlflow.get_tracking_uri()
    try:
        hp.main(
            [
                *("--panel", str(tmp_path / "panel.parquet")),
                *("--horizontes", "1", "2"),
                *("--tracking-uri", uri),
                *("--artefactos", str(tmp_path / "mlartifacts")),
                *("--figuras", str(tmp_path / "figuras")),
                *("--resultados", str(tmp_path / "resultados.json")),
            ]
        )
        cliente = MlflowClient(uri)
        experimento = cliente.get_experiment_by_name(CONFIG_P2.experimento)
        runs = {r.info.run_name: r for r in cliente.search_runs([experimento.experiment_id])}
        modelos = cliente.search_registered_models()
    finally:
        mlflow.set_tracking_uri(uri_previa)
    assert set(runs) == {"p2_horizontes_ini50", "p2_h1_ini50", "p2_h2_ini50"}
    padre = runs["p2_horizontes_ini50"].info.run_id
    for h in (1, 2):
        hijo = runs[f"p2_h{h}_ini50"].data
        assert hijo.tags["mlflow.parentRunId"] == padre
        assert hijo.tags["horizonte_horas"] == str(h)
        assert {"hgb_test_macro_f1", "persistencia_test_macro_f1"} <= set(hijo.metrics)
    assert modelos == []
    tabla = pd.read_csv(tmp_path / "figuras" / "p2_horizontes.csv", index_col=0)
    assert list(tabla.index) == [1, 2]
    assert {"hgb", "hgb_sin_pesos", "persistencia", "semana_previa", "mayoritaria"} <= set(
        tabla.columns
    )
    assert (tmp_path / "figuras" / "p2_horizontes_macro_f1.png").exists()
