import math

import pytest

from mibici.operacion import finops

CAPACIDAD = {"usuarios": 100, "rps_sostenido": 60.0, "promedio_ms": 30.0, "p95_ms": 120.0}
API = {"cpu_nucleos_promedio": 0.6, "cpu_nucleos_p95": 0.7, "memoria_mib_max": 320.0}
MLFLOW = {"cpu_nucleos_promedio": 0.01, "cpu_nucleos_p95": 0.02, "memoria_mib_max": 210.0}
LOTES = {"estaciones": 468, "p2_ms": 180.0, "p3_ms": 60.0}


def test_usd_hora_fargate():
    tarifa = finops.TARIFAS["fargate_x86"]
    # 1 vCPU y 2 GB: (0.000011244 + 2 * 0.000001235) * 3600
    assert finops.usd_hora(tarifa, 1, 2) == pytest.approx(0.0493704)


def test_usd_por_peticion_redondea_a_100_ms():
    tarifa = finops.TARIFAS["cloud_run_peticion"]
    una = finops.usd_por_peticion(tarifa, 1, 1, 0.03)
    assert una == pytest.approx((0.000024 + 0.0000025) * 0.1 + 0.40 / 1e6)
    assert finops.usd_por_peticion(tarifa, 1, 1, 0.1) == pytest.approx(una)
    assert finops.usd_por_peticion(tarifa, 1, 1, 0.11) > una


@pytest.mark.parametrize(
    "vcpu,gb,esperado",
    [
        (0.1, 0.3, (0.25, 0.5)),
        (0.3, 0.5, (0.5, 1.0)),
        (0.7, 0.5, (1.0, 2.0)),
        (1.0, 2.5, (1.0, 3.0)),
        (1.2, 1.0, (2.0, 4.0)),
    ],
)
def test_tamano_fargate(vcpu, gb, esperado):
    assert finops.tamano_fargate(vcpu, gb) == esperado


def test_tamano_fargate_sin_opcion():
    with pytest.raises(ValueError):
        finops.tamano_fargate(16, 1)


def test_dimensionar_con_margen():
    # 0.7 núcleos y 320 MiB con 30 % de margen: 0.91 vCPU y 0.41 GB → 1 vCPU con 2 GB.
    assert finops.dimensionar(0.7, 320) == (1.0, 2.0)


def test_demanda_mibici():
    demanda = finops.Demanda(estaciones=468, dias_mes=30.4)
    inferencias = demanda.inferencias()
    assert inferencias["p1"] == pytest.approx(468 * 30.4)
    assert inferencias["p2"] == pytest.approx(468 * 18 * 30.4)
    assert inferencias["total"] == pytest.approx(inferencias["p1"] + inferencias["p2"] + 468)
    con_lote = demanda.peticiones(lote=1000)
    assert con_lote["p2"] == pytest.approx(18 * 30.4)
    assert demanda.peticiones(lote=1)["p2"] == pytest.approx(inferencias["p2"])


@pytest.fixture
def resultado():
    return finops.analizar(CAPACIDAD, API, lotes=LOTES, recursos_mlflow=MLFLOW)


def _fila(resultado, inicio):
    return next(f for f in resultado["escenarios"] if f["escenario"].startswith(inicio))


def test_analizar_dimension_y_capacidad(resultado):
    dimension = resultado["dimension_recomendada"]
    assert (dimension["vcpu"], dimension["gb"]) == (1.0, 2.0)
    # La capacidad es la medida (60 RPS con 1 vCPU), no el RPS por núcleo extrapolado.
    assert dimension["capacidad_rps"] == pytest.approx(60.0)
    assert resultado["medido"]["rps_por_nucleo"] == pytest.approx(100.0)
    assert "utilizacion_objetivo" not in resultado["supuestos"]
    assert 0 < resultado["utilizacion_mibici"] < 0.01


def test_relaciones_entre_escenarios(resultado):
    piso = _fila(resultado, "Piso")
    siempre = _fila(resultado, "MiBici, siempre encendido (24")
    horario = _fila(resultado, "MiBici, encendido 19")
    arm = _fila(resultado, "MiBici, siempre encendido en ARM")
    sin_lote = _fila(resultado, "MiBici, Cloud Run por petición, una")
    con_lote = _fila(resultado, "MiBici, Cloud Run por petición, P2")
    assert piso["usd_mes"] == pytest.approx(siempre["usd_mes"])
    assert piso["usd_por_1000"] < siempre["usd_por_1000"]
    assert horario["usd_mes"] == pytest.approx(siempre["usd_mes"] * 19 * 30.4 / 730)
    assert arm["usd_mes"] < siempre["usd_mes"]
    assert con_lote["usd_mes"] < sin_lote["usd_mes"] < siempre["usd_mes"]
    assert finops.ahorro(siempre, arm) == pytest.approx(1 - arm["usd_mes"] / siempre["usd_mes"])


def test_mlflow_aparte(resultado):
    fila = _fila(resultado, "Servidor de MLflow")
    assert fila["escenario"] == (
        "Servidor de MLflow encendido 24 x 7 (la API carga los modelos al arrancar)"
    )
    assert (fila["vcpu"], fila["gb"]) == (0.25, 0.5)
    assert math.isnan(fila["usd_por_1000"])


def test_sin_lotes_no_hay_escenario_en_lote():
    resultado = finops.analizar(CAPACIDAD, API)
    nombres = [f["escenario"] for f in resultado["escenarios"]]
    assert not any("en lote" in n for n in nombres)
    assert any("una estación por petición" in n for n in nombres)


def test_markdown(resultado):
    texto = finops.a_markdown(resultado)
    assert "| Escenario |" in texto
    assert "1 vCPU, 2 GB" in texto
    # Encabezado, separador y un renglón por escenario.
    assert texto.count("\n| ") == len(resultado["escenarios"]) + 2


def test_escenarios_con_clave_estable(resultado):
    claves = [f["clave"] for f in resultado["escenarios"]]
    assert claves == [
        "piso",
        "siempre_encendido",
        "horario",
        "arm",
        "cloud_run",
        "cloud_run_lote",
        "mlflow",
    ]


def _por_clave(resultado, clave):
    return next(f for f in resultado["escenarios"] if f["clave"] == clave)


def test_optimizaciones_con_cifras(resultado):
    frases = resultado["optimizaciones"]
    assert frases == finops.optimizaciones(resultado)
    assert len(frases) == 5
    uso, horario, arm, nube, mlflow = frases
    base = _por_clave(resultado, "siempre_encendido")
    usd_base = f"{base['usd_mes']:,.2f}"
    assert f"{resultado['utilizacion_mibici']:.3%}" in uso
    assert usd_base in uso and "escalar a cero" in uso
    fila = _por_clave(resultado, "horario")
    assert f"{fila['usd_mes']:,.2f} USD al mes" in horario
    assert f"{finops.ahorro(base, fila):.1%}" in horario
    assert f"{1 - 19 * 30.4 / 730:.1%}" in horario
    fila = _por_clave(resultado, "arm")
    assert f"{fila['usd_mes']:,.2f} USD" in arm and f"{finops.ahorro(base, fila):.1%}" in arm
    sin_lote = _por_clave(resultado, "cloud_run")
    con_lote = _por_clave(resultado, "cloud_run_lote")
    assert f"{sin_lote['usd_mes']:,.2f} USD al mes" in nube
    assert f"{con_lote['usd_mes']:,.2f} USD al mes" in nube
    assert f"{finops.ahorro(base, sin_lote):.1%}" in nube
    assert f"{finops.ahorro(base, con_lote):.1%}" in nube
    supuestos = resultado["supuestos"]
    assert f"{supuestos['peticiones_mes_sin_lote']['total']:,.0f} peticiones" in nube
    assert f"({supuestos['peticiones_mes_con_lote']['total']:,.0f} peticiones al mes" in nube
    assert "arranque del contenedor y la carga de los modelos desde el Registry" in nube
    assert "tiempo de arranque y apagado de la instancia" in nube
    assert "validar la construcción de la imagen y el rendimiento" in arm
    assert "linux/arm64" in arm and "sólo requiere" not in arm
    fila = _por_clave(resultado, "mlflow")
    assert f"{fila['usd_mes']:,.2f} USD al mes" in mlflow
    assert "(0.25 vCPU y 0.5 GB)" in mlflow
    assert "no se puede apagar" in mlflow and "carga los modelos del Registry" in mlflow
    assert "empaquetar el modelo campeón en la imagen" in mlflow
    assert "almacenamiento de objetos" in mlflow and "ahorro máximo posible" in mlflow
    assert "4 h" not in mlflow and "horas_mlflow_mes" not in supuestos


def test_optimizaciones_sin_lotes_ni_mlflow():
    frases = finops.optimizaciones(finops.analizar(CAPACIDAD, API))
    assert len(frases) == 4
    assert not any("en lote" in f for f in frases)
    assert not any("MLflow" in f for f in frases)
    assert any("una estación por petición" in f for f in frases)


def test_optimizaciones_omite_valores_nan():
    resultado = finops.analizar({**CAPACIDAD, "promedio_ms": math.nan}, API)
    frases = finops.optimizaciones(resultado)
    assert not any("Cloud Run" in f for f in frases)
    fila = _por_clave(resultado, "horario")
    fila["usd_mes"] = math.nan
    assert not any("h al día" in f for f in finops.optimizaciones(resultado))


def test_markdown_termina_con_optimizaciones(resultado):
    texto = finops.a_markdown(resultado)
    cola = texto.split("**Optimizaciones**\n\n", 1)[1]
    assert cola == "".join(f"- {frase}\n" for frase in resultado["optimizaciones"])


def test_dimension_acotada_a_los_nucleos_utiles():
    # 0.99 núcleos con 30 % de margen piden 1.29 vCPU, pero uvicorn con un proceso usa uno.
    assert finops.dimensionar(0.99, 610, vcpu_max=1.0) == (1.0, 2.0)
    assert finops.dimensionar(0.99, 610) == (2.0, 4.0)
    api = {"cpu_nucleos_promedio": 0.82, "cpu_nucleos_p95": 0.99, "memoria_mib_max": 610.0}
    uno = finops.analizar(CAPACIDAD, api)
    assert (uno["dimension_recomendada"]["vcpu"], uno["dimension_recomendada"]["gb"]) == (1, 2)
    assert uno["supuestos"]["vcpu_max"] == 1.0
    dos = finops.analizar(CAPACIDAD, api, vcpu_max=2.0)
    assert (dos["dimension_recomendada"]["vcpu"], dos["dimension_recomendada"]["gb"]) == (2, 4)
    # Medida con 1 núcleo: con 1 vCPU es la misma; con 2 vCPU útiles se escala linealmente.
    assert uno["dimension_recomendada"]["capacidad_rps"] == pytest.approx(60.0)
    assert dos["dimension_recomendada"]["capacidad_rps"] == pytest.approx(120.0)
    # Si la prueba ya se hizo con 2 núcleos, la capacidad de 2 vCPU es la medida, sin escalar.
    medida_con_dos = finops.analizar(CAPACIDAD, api, vcpu_max=2.0, vcpu_medidos=2.0)
    assert medida_con_dos["dimension_recomendada"]["capacidad_rps"] == pytest.approx(60.0)
    assert medida_con_dos["supuestos"]["vcpu_medidos"] == 2.0


def test_capacidad_no_cuenta_nucleos_no_medidos():
    api = {"cpu_nucleos_promedio": 0.82, "cpu_nucleos_p95": 1.6, "memoria_mib_max": 610.0}
    # Aunque el tamaño quede en 2 vCPU por un tope mayor, la capacidad usa a lo más vcpu_max.
    resultado = finops.analizar(CAPACIDAD, api, vcpu_max=1.5)
    assert resultado["dimension_recomendada"]["vcpu"] == 2.0
    capacidad = resultado["dimension_recomendada"]["capacidad_rps"]
    assert capacidad == pytest.approx(60.0 * 1.5 / 1.0)


def test_usa_rps_y_latencia_de_inferencias():
    capacidad = {**CAPACIDAD, "rps_inferencias": 54.0, "promedio_inferencia_ms": 150.0}
    resultado = finops.analizar(capacidad, API)
    medido = resultado["medido"]
    assert medido["rps_sostenido"] == 60.0 and medido["rps_inferencias"] == 54.0
    assert medido["latencia_promedio_ms"] == 30.0 and medido["latencia_inferencia_ms"] == 150.0
    assert medido["rps_por_nucleo"] == pytest.approx(54.0 / 0.6)
    nube = _por_clave(resultado, "cloud_run")
    vcpu, gb = nube["vcpu"], nube["gb"]
    tarifa = finops.TARIFAS["cloud_run_peticion"]
    unitario = finops.usd_por_peticion(tarifa, vcpu, gb, 0.15)
    sin_lote = resultado["supuestos"]["peticiones_mes_sin_lote"]["total"]
    assert nube["usd_mes"] == pytest.approx(sin_lote * unitario)
    texto = finops.a_markdown(resultado)
    assert "Medido: capacidad sostenida de 54.0 RPS de inferencias" in texto
    assert "60.0 RPS en total con las sondas GET /live" in texto
    # La capacidad de la instancia es la medida de inferencias, no la del agregado.
    assert resultado["dimension_recomendada"]["capacidad_rps"] == pytest.approx(54.0)


def test_sin_datos_de_inferencias_usa_el_agregado():
    medido = finops.analizar(CAPACIDAD, API)["medido"]
    assert medido["rps_inferencias"] == 60.0
    assert medido["latencia_inferencia_ms"] == 30.0
    assert medido["latencias_pregunta_ms"] == {"p1": 30.0, "p2": 30.0, "p3": 30.0}


def _endpoint(nombre, peticiones, promedio_ms):
    return {"endpoint": nombre, "peticiones": peticiones, "promedio_ms": promedio_ms}


def test_cloud_run_cobra_la_latencia_de_cada_pregunta():
    capacidad = {
        **CAPACIDAD,
        "promedio_inferencia_ms": 110.0,
        "endpoints": [
            _endpoint("GET /live", 50, 3.0),
            _endpoint("P1 POST /api/v1/predict", 30, 40.0),
            _endpoint("P2 POST /api/v1/p2/predict", 500, 124.0),
            _endpoint("P3 POST /api/v1/p3/predict", 30, 50.0),
        ],
    }
    resultado = finops.analizar(capacidad, API, lotes=LOTES)
    assert resultado["medido"]["latencias_pregunta_ms"] == {"p1": 40.0, "p2": 124.0, "p3": 50.0}
    nube = _por_clave(resultado, "cloud_run")
    tarifa = finops.TARIFAS["cloud_run_peticion"]
    vcpu, gb = nube["vcpu"], nube["gb"]

    def unitario(segundos):
        return finops.usd_por_peticion(tarifa, vcpu, gb, segundos)

    # P2 promedia 124 ms y se factura a 0.2 s; P1 (40 ms) y P3 (50 ms), a 0.1 s.
    assert unitario(0.124) == pytest.approx(unitario(0.2))
    assert unitario(0.124) > unitario(0.1)
    sin_lote = resultado["supuestos"]["peticiones_mes_sin_lote"]
    esperado = sin_lote["p1"] * unitario(0.1) + sin_lote["p2"] * unitario(0.2)
    esperado += sin_lote["p3"] * unitario(0.1)
    assert nube["usd_mes"] == pytest.approx(esperado)
    # Con lote, P1 conserva su latencia y P2 y P3 usan las de la sonda de lotes.
    con_lote = resultado["supuestos"]["peticiones_mes_con_lote"]
    lote = _por_clave(resultado, "cloud_run_lote")
    esperado_lote = con_lote["p1"] * unitario(0.1) + con_lote["p2"] * unitario(0.18)
    esperado_lote += con_lote["p3"] * unitario(0.06)
    assert lote["usd_mes"] == pytest.approx(esperado_lote)


def test_pregunta_sin_endpoint_usa_el_promedio_de_inferencias():
    capacidad = {
        **CAPACIDAD,
        "promedio_inferencia_ms": 90.0,
        "endpoints": [_endpoint("P2 POST /api/v1/p2/predict", 100, 124.0)],
    }
    latencias = finops.latencias_por_pregunta(capacidad)
    assert latencias == {"p1": 90.0, "p2": 124.0, "p3": 90.0}


SATURACION = {
    "usuarios": 60,
    "rps_sostenido": 22.6,
    "rps_inferencias": 21.5,
    "p95_ms": 2300.0,
}


def test_saturacion_en_medido_y_en_el_texto():
    resultado = finops.analizar(CAPACIDAD, API, saturacion=SATURACION)
    medido = resultado["medido"]
    assert medido["rps_saturacion"] == 22.6
    assert medido["rps_inferencias_saturacion"] == 21.5
    assert medido["usuarios_saturacion"] == 60
    assert medido["p95_saturacion_ms"] == 2300.0
    texto = finops.a_markdown(resultado)
    assert "Saturación observada: 22.6 RPS (21.5 de inferencias) con 60 usuarios" in texto
    assert "p95 de 2,300 ms" in texto
    assert "70 %" not in texto and "% de uso" not in texto
    assert "con esa capacidad medida de 60.0 RPS de inferencias." in texto
    sin = finops.a_markdown(finops.analizar(CAPACIDAD, API))
    assert "Saturación" not in sin


def test_texto_dice_cuando_la_capacidad_se_escala():
    resultado = finops.analizar(CAPACIDAD, API, vcpu_max=2.0, vcpu_medidos=0.5)
    assert resultado["dimension_recomendada"]["capacidad_rps"] == pytest.approx(120.0)
    assert "escalada linealmente de 0.5 a 1 núcleos" in finops.a_markdown(resultado)


def test_piso_usa_la_capacidad_medida(resultado):
    piso = _por_clave(resultado, "piso")
    capacidad = resultado["dimension_recomendada"]["capacidad_rps"]
    assert piso["inferencias_mes"] == pytest.approx(capacidad * 3600 * finops.HORAS_MES)
    total = resultado["supuestos"]["inferencias_mes"]["total"]
    assert resultado["utilizacion_mibici"] == pytest.approx(total / piso["inferencias_mes"])
