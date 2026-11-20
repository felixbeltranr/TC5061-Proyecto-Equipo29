"""Las figuras se generan sin errores con datos sintéticos (el contenido se revisa a ojo)."""

import pandas as pd

from mibici.operacion import graficas
from mibici.operacion.recursos import MIB, Muestra


def _historial(usuarios, fallas=0.0):
    filas = []
    for segundo in range(20):
        filas.append(
            {
                "Timestamp": 1000 + segundo,
                "User Count": usuarios,
                "Name": "Aggregated",
                "Requests/s": usuarios * 0.8 if segundo else 0.0,
                "Failures/s": fallas,
                "50%": 20.0 if segundo else 0.0,
                "95%": 60.0,
                "99%": 90.0,
                "Total Request Count": segundo * usuarios,
                "segundo": segundo,
            }
        )
    return pd.DataFrame(filas)


def _no_vacio(ruta):
    return ruta.exists() and ruta.stat().st_size > 1000


def _capturar_figura(monkeypatch):
    """Guarda los títulos y textos de cada panel antes de que la figura se cierre."""
    paneles = []
    original = graficas._guardar

    def guardar(figura, salida):
        paneles.clear()
        for eje in figura.axes:
            # Los títulos van a la izquierda (axes.titlelocation).
            paneles.append((eje.get_title(loc="left"), [t.get_text() for t in eje.texts]))
        return original(figura, salida)

    monkeypatch.setattr(graficas, "_guardar", guardar)
    return paneles


def test_desempeno_con_y_sin_fallas(tmp_path, monkeypatch):
    paneles = _capturar_figura(monkeypatch)
    historiales = [(10, _historial(10)), (50, _historial(50))]
    assert _no_vacio(graficas.grafica_desempeno(historiales, tmp_path / "a.png"))
    # El panel de errores aparece aunque no haya fallas, con la anotación del 0 %.
    titulos = [titulo for titulo, _ in paneles]
    assert titulos == ["Peticiones por segundo", "Latencia por percentil", "Tasa de error"]
    assert "0.00 % de errores en todas las etapas" in paneles[2][1]
    historiales.append((100, _historial(100, fallas=1.0)))
    assert _no_vacio(graficas.grafica_desempeno(historiales, tmp_path / "b.png"))
    assert len(paneles) == 3 and paneles[2][0] == "Tasa de error"
    assert "0.00 % de errores en todas las etapas" not in paneles[2][1]


def test_tasa_de_error_en_porcentaje():
    datos = pd.DataFrame({"Requests/s": [0.0, 50.0, 80.0], "Failures/s": [0.0, 1.0, None]})
    tasa = graficas.tasa_error(datos)
    assert pd.isna(tasa.iloc[0])
    assert tasa.iloc[1] == 2.0 and tasa.iloc[2] == 0.0


def test_serie_continua_quita_ceros_sin_peticiones():
    datos, cortes = graficas.serie_continua([(10, _historial(10)), (50, _historial(50))])
    assert [u for _, u in cortes] == [10, 50]
    primera = datos[datos["etapa"] == 10].iloc[0]
    assert pd.isna(primera["50%"])
    assert datos["t"].is_monotonic_increasing


def test_capacidad_y_endpoints(tmp_path):
    resumenes = [
        {"usuarios": u, "rps_sostenido": u * 0.8, "p95_ms": 50 + u}
        for u in (10, 50, 100, 200)
    ]
    assert _no_vacio(graficas.grafica_capacidad(resumenes, tmp_path / "c.png"))
    etapa = {
        "usuarios": 100,
        "endpoints": [
            {"endpoint": "P1 POST /api/v1/predict", "peticiones": 10, "p50_ms": 20, "p95_ms": 70},
            {"endpoint": "GET /live", "peticiones": 5, "p50_ms": 2, "p95_ms": 10},
            {"endpoint": "P3 POST /api/v1/p3/predict", "peticiones": 0, "p50_ms": 0, "p95_ms": 0},
        ],
    }
    assert _no_vacio(graficas.grafica_endpoints(etapa, tmp_path / "e.png"))


def test_recursos(tmp_path):
    muestras = [
        Muestra(t, nombre, cpu, mem * MIB, 1024 * MIB, 0, 0)
        for t in range(10)
        for nombre, cpu, mem in (("src-api-1", 40 + t, 300), ("src-mlflow-1", 0.4, 210))
    ]
    ruta = graficas.grafica_recursos(
        muestras,
        tmp_path / "r.png",
        cortes=[(0, 10), (5, 50)],
        nombres={"src-api-1": "api", "src-mlflow-1": "mlflow"},
    )
    assert _no_vacio(ruta)
