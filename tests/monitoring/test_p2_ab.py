import pytest

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
