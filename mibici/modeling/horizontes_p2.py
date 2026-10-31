"""Analisis de horizontes de P2: macro F1 de prueba a 1, 2, 3 y 4 horas.

Entrena la HGB con y sin pesos de clase con los hiperparametros ya elegidos a 1 h (sin
busqueda) y la compara con las baselines en filas pareadas: la misma hora objetivo para todos
los horizontes y origen desde el inicio del dia de servicio. Registra un run padre con un run
hijo por horizonte; no registra modelos ni toca el Model Registry.

    python -m mibici.modeling.horizontes_p2 --horizontes 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from mibici import features_p2 as fp
from mibici.config import CONFIG_P2
from mibici.modeling import train_p2 as t2

METRICAS = (
    "test_macro_f1",
    *(f"test_{m}_{c}" for c in t2.RIESGO for m in ("recall", "precision")),
)


def evaluar_horizonte(
    panel: pd.DataFrame,
    h: int,
    nivel: float,
    params: dict,
    semilla: int,
    horas_objetivo: tuple[int, ...],
) -> dict[str, dict[str, float]]:
    """Metricas de prueba de las baselines y de las dos HGB a horizonte h."""
    partes = t2.preparar(nivel, panel, h, horas_objetivo)
    por_est, por_hora = fp.stats_estacion(partes["train"])
    train = fp.unir_stats(partes["train"], por_est, por_hora)
    test = fp.unir_stats(partes["test"], por_est, por_hora)
    preds = t2.predicciones_baseline(test, train["objetivo"].mode()[0])
    for clave, balanceado in (("hgb", True), ("hgb_sin_pesos", False)):
        modelo = t2.ajustar(train, params, semilla, balanceado=balanceado)
        preds[clave] = t2.predecir(modelo, test)[2]
    salida = {}
    for clave, pred in preds.items():
        m = t2.metricas(test["objetivo"], pred, None, fp.CLASES, "test")
        salida[clave] = {k: m[k] for k in METRICAS}
        logger.info("h={} {} macro F1 {:.4f}", h, clave, m["test_macro_f1"])
    return salida


def tabla_horizontes(resultados: dict[int, dict[str, dict[str, float]]]) -> pd.DataFrame:
    """Macro F1 por horizonte y modelo, con la diferencia de cada HGB contra la persistencia."""
    tabla = pd.DataFrame(
        {h: {k: v["test_macro_f1"] for k, v in mods.items()} for h, mods in resultados.items()}
    ).T
    tabla.index.name = "horizonte_horas"
    for clave in ("hgb", "hgb_sin_pesos"):
        tabla[f"delta_{clave}_vs_persistencia"] = tabla[clave] - tabla["persistencia"]
    return tabla


def figura_horizontes(tabla: pd.DataFrame, ruta: Path) -> Path:
    """Lineas de macro F1 de prueba contra el horizonte, una por modelo."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for clave in ("hgb", "hgb_sin_pesos", "persistencia", "semana_previa", "mayoritaria"):
        ax.plot(tabla.index, tabla[clave], marker="o", label=clave)
    ax.set_xticks(tabla.index)
    ax.set_xlabel("Horizonte (horas)")
    ax.set_ylabel("Macro F1 en prueba")
    ax.set_title("P2: macro F1 contra horizonte (horas objetivo pareadas)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    return ruta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Analisis de horizontes de P2 en MLflow")
    ap.add_argument("--panel", default=str(CONFIG_P2.panel_path))
    ap.add_argument(
        "--horizontes", type=int, nargs="+", default=list(CONFIG_P2.horizontes_analisis)
    )
    ap.add_argument("--nivel", type=float, default=CONFIG_P2.nivel_inicial)
    ap.add_argument("--semilla", type=int, default=CONFIG_P2.seed)
    ap.add_argument("--tracking-uri", default=CONFIG_P2.tracking_uri)
    ap.add_argument("--artefactos", default=str(CONFIG_P2.artefactos_dir))
    ap.add_argument("--figuras", default=str(CONFIG_P2.figuras_dir))
    ap.add_argument(
        "--resultados",
        default=str(CONFIG_P2.figuras_dir / "p2_resultados.json"),
        help="p2_resultados.json de la corrida a 1 h; de ahi salen los hiperparametros de la HGB",
    )
    args = ap.parse_args(argv)
    params = json.loads(Path(args.resultados).read_text())["params_hgb"]
    horas = CONFIG_P2.horas_objetivo_horizontes
    figs = Path(args.figuras)
    figs.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(args.tracking_uri)
    if mlflow.get_experiment_by_name(CONFIG_P2.experimento) is None:
        mlflow.create_experiment(
            CONFIG_P2.experimento, artifact_location=Path(args.artefactos).resolve().as_uri()
        )
    mlflow.set_experiment(CONFIG_P2.experimento)
    panel = pd.read_parquet(args.panel)
    sha, sucio = t2.git_commit()
    sufijo = t2.sufijo_nivel(args.nivel)
    tags = {
        "question": CONFIG_P2.question,
        "team": CONFIG_P2.team,
        "author": CONFIG_P2.author,
        "seed": str(args.semilla),
        "analisis": "horizontes",
        "horas_objetivo": f"{min(horas)}-{max(horas)}",
        "mlflow.source.git.commit": sha,
        "git_commit": sha,
        "git_dirty": str(sucio),
    }
    comunes = {"nivel_inicial": args.nivel, "semilla": args.semilla} | params
    resultados = {}
    with mlflow.start_run(run_name=f"p2_horizontes_{sufijo}"):
        mlflow.set_tags(tags)
        mlflow.log_params(comunes | {"horizontes": " ".join(map(str, args.horizontes))})
        for h in args.horizontes:
            resultados[h] = evaluar_horizonte(panel, h, args.nivel, params, args.semilla, horas)
            t2.registrar(
                f"p2_h{h}_{sufijo}",
                tags | {"horizonte_horas": str(h)},
                comunes | {"horizonte_horas": h},
                {f"{k}_{m}": v for k, mets in resultados[h].items() for m, v in mets.items()},
                [],
                anidado=True,
            )
        tabla = tabla_horizontes(resultados)
        ruta_csv = figs / "p2_horizontes.csv"
        tabla.round(4).to_csv(ruta_csv)
        ruta_json = figs / "p2_horizontes.json"
        ruta_json.write_text(json.dumps({str(h): r for h, r in resultados.items()}, indent=2))
        ruta_png = figura_horizontes(tabla, figs / "p2_horizontes_macro_f1.png")
        for ruta, carpeta in ((ruta_csv, "data"), (ruta_json, "data"), (ruta_png, "plots")):
            mlflow.log_artifact(str(ruta), carpeta)
    logger.info("Horizontes P2:\n{}", tabla.round(4))


if __name__ == "__main__":
    main()
