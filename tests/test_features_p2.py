import numpy as np
import pytest

from mibici.features_p2 import acumular_ocupacion, etiquetar_ocupacion


def test_etiquetas_por_umbral():
    cap = np.array([20, 20, 20, 20, 20])
    occ = np.array([0, 1.9, 2, 18, 18.5])
    assert list(etiquetar_ocupacion(occ, cap)) == [
        "sin_bicicletas",
        "sin_bicicletas",
        "segura",
        "segura",
        "sin_anclajes",
    ]


def test_estacion_llena_es_sin_anclajes():
    assert etiquetar_ocupacion([15], [15])[0] == "sin_anclajes"


def test_rechaza_ocupacion_fuera_de_rango():
    with pytest.raises(ValueError):
        etiquetar_ocupacion([16], [15])
    with pytest.raises(ValueError):
        etiquetar_ocupacion([-1], [15])


def test_rechaza_capacidad_cero():
    with pytest.raises(ValueError):
        etiquetar_ocupacion([0], [0])


def test_acumulado_respeta_capacidad():
    neto = np.array([[5, 5, -20, 3]])
    occ = acumular_ocupacion(neto, np.array([10]), 0.5)
    assert occ.tolist() == [[10, 10, 0, 3]]
    assert (occ <= 10).all() and (occ >= 0).all()


def test_features_no_usan_la_hora_objetivo():
    import pandas as pd

    from mibici.features_p2 import FEATURES, agregar_ocupacion, construir_features

    horas = pd.date_range("2026-01-01 05:00", periods=24 * 10, freq="h")
    rng = np.random.default_rng(0)
    base = pd.DataFrame(
        {
            "ts": horas,
            "estacion": 1,
            "capacidad": np.float32(20),
            "salidas": rng.integers(0, 4, len(horas)).astype("float32"),
            "llegadas": rng.integers(0, 4, len(horas)).astype("float32"),
        }
    )
    base["neto"] = base["llegadas"] - base["salidas"]
    base["primer_dia"] = horas[0]
    t = 24 * 9 + 10
    alterado = base.copy()
    # cambiar solo lo que pasa en t+1 y despues
    alterado.loc[t + 1 :, "llegadas"] += 15
    alterado["neto"] = alterado["llegadas"] - alterado["salidas"]

    cols = [c for c in FEATURES if not c.startswith("est_")]
    a = construir_features(agregar_ocupacion(base, 0.5)).set_index("ts")
    b = construir_features(agregar_ocupacion(alterado, 0.5)).set_index("ts")
    ts_t = horas[t]
    pd.testing.assert_series_equal(a.loc[ts_t, cols], b.loc[ts_t, cols])


def test_media_misma_hora_usa_cuatro_semanas():
    import pandas as pd

    from mibici.features_p2 import agregar_ocupacion, construir_features

    # 36 dias de servicio por estacion; el neto de cada hora es distinto y conocido
    horas = pd.date_range("2026-01-01 05:00", periods=24 * 36, freq="h")
    piezas = []
    for est, desfase in [(1, 0.0), (2, 10_000.0)]:
        llegadas = np.arange(len(horas), dtype="float32") + desfase
        piezas.append(
            pd.DataFrame(
                {
                    "ts": horas,
                    "estacion": est,
                    "capacidad": np.float32(20),
                    "salidas": np.float32(0),
                    "llegadas": llegadas,
                }
            )
        )
    panel = pd.concat(piezas, ignore_index=True)
    panel["neto"] = panel["llegadas"] - panel["salidas"]
    panel["primer_dia"] = horas[0]
    neto = panel.set_index(["estacion", "ts"])["neto"]

    df = construir_features(agregar_ocupacion(panel, 0.5))
    completas = df[df["ts_objetivo"] - pd.Timedelta(weeks=4) >= horas[0]]
    assert len(completas) > 0
    for est, ts_obj, valor, lag168 in completas[
        ["estacion", "ts_objetivo", "neto_media168_misma_hora", "neto_lag168"]
    ].itertuples(index=False):
        # mismo instante del objetivo hace 1, 2, 3 y 4 semanas, dentro de la misma estacion
        esperado = np.mean([neto[(est, ts_obj - pd.Timedelta(weeks=k))] for k in range(1, 5)])
        assert valor == esperado
        assert lag168 == neto[(est, ts_obj - pd.Timedelta(weeks=1))]
