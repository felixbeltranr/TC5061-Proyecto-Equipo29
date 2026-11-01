import pandas as pd
from tests.modeling.test_train_p2 import CORTES_CHICOS, hacer_panel

from mibici import features_p2 as fp
from mibici.modeling import exportar_p2 as ex
from mibici.modeling import train_p2 as t2


def test_exporta_train_y_test_con_features_objetivo_y_stats_de_train(tmp_path, monkeypatch):
    for clave, valor in CORTES_CHICOS.items():
        monkeypatch.setitem(t2.CORTES, clave, valor)
    rutas = ex.exportar_particiones(hacer_panel(), 0.5, tmp_path)
    train, test = (pd.read_parquet(rutas[k]) for k in ("train", "test"))
    assert list(train.columns) == list(test.columns) == [*fp.FEATURES, "objetivo"]
    assert len(train) and len(test)
    # la estacion nueva solo aparece en test y no tiene estadisticas de train
    assert test["est_tasa_sin_bicis"].isna().any() and train["est_tasa_sin_bicis"].notna().all()
