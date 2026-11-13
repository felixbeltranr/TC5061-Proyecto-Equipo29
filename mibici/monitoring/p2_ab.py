"""A/B testing de P2: campeón (A) contra la versión reentrenada (B) en el mes actual.

Regla de decisión, fijada antes de correr la comparación:

- La métrica principal es el recall medio de las clases de riesgo (sin_bicicletas y
  sin_anclajes), la misma con que se eligió al campeón (``mibici.registro``).
- B reemplaza a A si su recall medio sube al menos ``EMPATE_RECALL`` y el límite inferior del
  intervalo de confianza del 95 % de la diferencia B - A es mayor que cero.
- Si la diferencia de recall medio está por debajo de ``EMPATE_RECALL`` en valor absoluto, hay
  empate y decide el PR AUC medio de las clases de riesgo: B reemplaza a A solo si su diferencia
  es positiva y el límite inferior de su intervalo es mayor que cero.
- En cualquier otro caso se mantiene A.

Los intervalos salen de un bootstrap pareado por estación: se remuestrean estaciones con
reemplazo y las dos versiones se califican con las mismas filas.
"""

from __future__ import annotations

from mibici.registro import EMPATE_RECALL

PROMOVER = "promover B"
MANTENER = "mantener A"


def decidir(
    delta_recall: float,
    ic_recall: tuple[float, float],
    delta_pr_auc: float,
    ic_pr_auc: tuple[float, float],
    empate: float = EMPATE_RECALL,
) -> dict:
    """Aplica la regla de decisión del módulo a las diferencias B - A y sus intervalos."""
    if abs(delta_recall) < empate:
        criterio = "pr_auc_medio_riesgo"
        promover = delta_pr_auc > 0 and ic_pr_auc[0] > 0
    else:
        criterio = "recall_medio_riesgo"
        promover = delta_recall >= empate and ic_recall[0] > 0
    return {
        "decision": PROMOVER if promover else MANTENER,
        "criterio": criterio,
        "empate_recall": empate,
        "delta_recall_medio_riesgo": delta_recall,
        "ic95_recall_medio_riesgo": list(ic_recall),
        "delta_pr_auc_medio_riesgo": delta_pr_auc,
        "ic95_pr_auc_medio_riesgo": list(ic_pr_auc),
    }
