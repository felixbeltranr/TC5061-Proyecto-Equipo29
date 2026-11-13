import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, f1_score, recall_score

from mibici.monitoring import p2_ab
from mibici.registro import EMPATE_RECALL


def test_la_regla_usa_el_mismo_empate_que_la_seleccion_del_campeon():
    assert EMPATE_RECALL == 0.005


def test_promueve_si_el_recall_sube_lo_suficiente_con_intervalo_positivo():
    r = p2_ab.decidir(0.006, (0.001, 0.011), -0.02, (-0.03, -0.01))
    assert r["decision"] == p2_ab.PROMOVER and r["criterio"] == "recall_medio_riesgo"


def test_mantiene_si_el_recall_sube_pero_el_intervalo_incluye_el_cero():
    r = p2_ab.decidir(0.008, (-0.001, 0.017), 0.05, (0.01, 0.09))
    assert r["decision"] == p2_ab.MANTENER and r["criterio"] == "recall_medio_riesgo"


def test_mantiene_si_el_recall_baja_aunque_mejore_el_pr_auc():
    r = p2_ab.decidir(-0.006, (-0.01, -0.002), 0.05, (0.01, 0.09))
    assert r["decision"] == p2_ab.MANTENER


@pytest.mark.parametrize(
    ("delta_pr_auc", "ic_pr_auc", "esperada"),
    [
        (0.01, (0.002, 0.018), p2_ab.PROMOVER),
        (0.01, (-0.002, 0.018), p2_ab.MANTENER),
        (-0.01, (-0.018, -0.002), p2_ab.MANTENER),
    ],
)
def test_con_empate_de_recall_decide_el_pr_auc(delta_pr_auc, ic_pr_auc, esperada):
    r = p2_ab.decidir(0.004, (-0.001, 0.009), delta_pr_auc, ic_pr_auc)
    assert r["criterio"] == "pr_auc_medio_riesgo" and r["decision"] == esperada


def test_el_limite_del_empate_cuenta_como_mejora_de_recall():
    r = p2_ab.decidir(EMPATE_RECALL, (0.0001, 0.01), -0.1, (-0.2, -0.05))
    assert r["criterio"] == "recall_medio_riesgo" and r["decision"] == p2_ab.PROMOVER


def _datos_sinteticos(n_estaciones: int = 12, por_estacion: int = 40, seed: int = 3):
    rng = np.random.default_rng(seed)
    estaciones = np.repeat(np.arange(100, 100 + n_estaciones), por_estacion)
    y = rng.choice(p2_ab.CLASES, size=len(estaciones), p=[0.7, 0.15, 0.15])
    proba = rng.dirichlet(np.ones(3), size=len(estaciones))
    pred = np.array(p2_ab.CLASES, dtype=object)[proba.argmax(1)]
    return pd.Series(estaciones), pd.Series(y, dtype=object), pred, proba


def test_ap_ponderado_igual_a_sklearn_con_pesos_por_fila():
    rng = np.random.default_rng(0)
    codigos = rng.integers(0, 5, 300)
    puntaje = np.round(rng.random(300), 2)  # con empates
    positivos = rng.random(300) < 0.3
    pesos = np.array([0.0, 1.0, 2.0, 3.0, 1.0])
    orden = p2_ab.preparar_ap(puntaje, positivos, codigos)
    esperado = average_precision_score(positivos, puntaje, sample_weight=pesos[codigos])
    assert p2_ab.ap_ponderado(orden, pesos) == pytest.approx(esperado)
    assert p2_ab.ap_ponderado(orden, np.ones(5)) == pytest.approx(
        average_precision_score(positivos, puntaje)
    )


def test_metricas_de_confusion_iguales_a_sklearn():
    estaciones, y, pred, _ = _datos_sinteticos()
    codigos, unicas = pd.factorize(estaciones, sort=True)
    cm = p2_ab.matrices_por_estacion(codigos, y.to_numpy(), pred, len(unicas)).sum(axis=0)
    m = p2_ab.metricas_de_confusion(cm)
    clases = list(p2_ab.CLASES)
    recall = recall_score(y, pred, labels=clases, average=None, zero_division=0)
    assert m["macro_f1"] == pytest.approx(
        f1_score(y, pred, labels=clases, average="macro", zero_division=0)
    )
    for i, clase in enumerate(clases):
        assert m[f"recall_{clase}"] == pytest.approx(recall[i])
    assert m["recall_medio_riesgo"] == pytest.approx(recall[1:].mean())


def test_brier_multiclase_va_de_0_a_2():
    perfecto = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert p2_ab.brier_multiclase(perfecto, np.array([0, 2])).tolist() == [0.0, 0.0]
    assert p2_ab.brier_multiclase(perfecto, np.array([1, 0])).tolist() == [2.0, 2.0]


def test_proba_en_orden_reacomoda_columnas():
    proba = np.array([[0.2, 0.5, 0.3]])
    reordenada = p2_ab.proba_en_orden(proba, ["sin_anclajes", "segura", "sin_bicicletas"])
    assert reordenada.tolist() == [[0.5, 0.3, 0.2]]


def test_comparar_versiones_identicas_da_diferencia_cero_y_es_determinista():
    estaciones, y, pred, proba = _datos_sinteticos()
    tabla = p2_ab.comparar(estaciones, y, pred, proba, pred, proba, n_bootstrap=50, seed=29)
    con_ic = tabla.dropna(subset=["ic95_bajo"])
    assert (tabla["b_menos_a"].abs() < 1e-12).all()
    assert (con_ic[["ic95_bajo", "ic95_alto"]].abs() < 1e-12).all().all()
    assert set(tabla["filas"]) == {len(y)} and set(tabla["estaciones"]) == {12}
    assert {"recall_medio_riesgo", "pr_auc_medio_riesgo", "brier_multiclase"} <= set(
        tabla["metrica"]
    )


def test_comparar_bootstrap_reproducible_con_la_semilla():
    estaciones, y, pred_a, proba_a = _datos_sinteticos(seed=3)
    _, _, pred_b, proba_b = _datos_sinteticos(seed=4)
    uno = p2_ab.comparar(estaciones, y, pred_a, proba_a, pred_b, proba_b, 40, 29)
    dos = p2_ab.comparar(estaciones, y, pred_a, proba_a, pred_b, proba_b, 40, 29)
    pd.testing.assert_frame_equal(uno, dos)
    fila = uno.set_index("metrica").loc["recall_medio_riesgo"]
    assert fila.ic95_bajo <= fila.b_menos_a <= fila.ic95_alto


def test_decidir_tabla_usa_recall_y_pr_auc_medios():
    tabla = pd.DataFrame(
        {
            "metrica": ["recall_medio_riesgo", "pr_auc_medio_riesgo"],
            "b_menos_a": [0.03, -0.01],
            "ic95_bajo": [0.01, -0.02],
            "ic95_alto": [0.05, 0.0],
        }
    )
    assert p2_ab.decidir_tabla(tabla)["decision"] == p2_ab.PROMOVER


def test_subgrupos_por_estaciones_de_cada_train():
    estaciones = pd.Series([1, 2, 3, 4])
    mascaras = p2_ab.subgrupos(estaciones, {1, 2}, {1, 2, 3})
    assert mascaras["vistas_por_a"].tolist() == [True, True, False, False]
    assert mascaras["nuevas_para_a"].tolist() == [False, False, True, False]
    assert mascaras["nuevas_para_ambos"].tolist() == [False, False, False, True]
    assert mascaras["todas"].all()


def test_cortes_retador_entrena_hasta_el_corte_de_config():
    cortes = p2_ab.cortes_retador("2026-11-01")
    assert cortes["train_inicio"] == "2025-01-01"
    assert cortes["test_inicio"] == "2026-10-01"
    assert cortes["test_fin"] == "2026-11-01"
