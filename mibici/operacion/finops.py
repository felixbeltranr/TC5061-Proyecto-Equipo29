"""Costo por inferencia de la API de MiBici con tarifas públicas de referencia.

Método, en la línea de *FinOps for AI* de la FinOps Foundation: el costo unitario de una
inferencia es el costo del cómputo que la sostiene dividido entre las inferencias que atiende.
Con lo medido en la prueba de carga (RPS sostenido, latencia, núcleos y memoria del contenedor de
la API) se calculan:

1. **El piso**: una instancia del tamaño recomendado trabajando a su capacidad sostenida, que es
   la medida (el RPS de inferencias de la etapa que cumple el objetivo de latencia), no una
   extrapolación del RPS por núcleo.
2. **La demanda real de MiBici**: P1 una vez al día por estación, P2 cada hora de 6 a 23 h por
   estación y P3 una vez al mes; con la instancia siempre encendida, con horario, en ARM y sin
   servidor (Cloud Run, que cobra el tiempo activo de cada petición con la latencia medida de
   cada pregunta).

La diferencia entre ambos es el costo de la capacidad ociosa, y de ahí salen las optimizaciones.
Todas las cifras están en dólares de EE. UU. Las tarifas son de lista, sin impuestos, sin
descuentos por compromiso y sin el nivel gratuito; se consultaron el 19 de noviembre de 2026.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

HORAS_MES = 730.0
DIAS_MES = 30.4
CONSULTA = "2026-11-19"
PREGUNTAS = ("p1", "p2", "p3")


@dataclass(frozen=True)
class Tarifa:
    clave: str
    nombre: str
    usd_vcpu_segundo: float
    usd_gb_segundo: float
    usd_millon_peticiones: float = 0.0
    granularidad_s: float = 1.0  # redondeo del tiempo facturado por petición
    fuente: str = ""
    consultada: str = CONSULTA


TARIFAS = {
    "fargate_x86": Tarifa(
        "fargate_x86",
        "AWS Fargate, Linux/x86, us-east-1",
        usd_vcpu_segundo=0.000011244,
        usd_gb_segundo=0.000001235,
        fuente="https://aws.amazon.com/fargate/pricing/",
    ),
    "fargate_arm": Tarifa(
        "fargate_arm",
        "AWS Fargate, Linux/ARM (Graviton), us-east-1",
        usd_vcpu_segundo=0.0000089944,
        usd_gb_segundo=0.0000009889,
        fuente="https://aws.amazon.com/fargate/pricing/",
    ),
    "cloud_run_instancia": Tarifa(
        "cloud_run_instancia",
        "Google Cloud Run, facturación por instancia (nivel 1)",
        usd_vcpu_segundo=0.000018,
        usd_gb_segundo=0.000002,
        fuente="https://cloud.google.com/run/pricing",
    ),
    "cloud_run_peticion": Tarifa(
        "cloud_run_peticion",
        "Google Cloud Run, facturación por petición (nivel 1)",
        usd_vcpu_segundo=0.000024,
        usd_gb_segundo=0.0000025,
        usd_millon_peticiones=0.40,
        granularidad_s=0.1,
        fuente="https://cloud.google.com/run/pricing",
    ),
}

# Combinaciones válidas de Fargate: vCPU → memorias en GB (1 vCPU exige al menos 2 GB).
TAMANOS_FARGATE = {
    0.25: (0.5, 1.0, 2.0),
    0.5: (1.0, 2.0, 3.0, 4.0),
    1.0: tuple(float(gb) for gb in range(2, 9)),
    2.0: tuple(float(gb) for gb in range(4, 17)),
    4.0: tuple(float(gb) for gb in range(8, 31)),
}


@dataclass(frozen=True)
class Demanda:
    """Inferencias que pediría la operación de MiBici en un mes."""

    estaciones: int = 468  # red actual según el ML Canvas del equipo (de 368 a 468 estaciones)
    horas_p2_por_dia: int = 18  # horas objetivo de 6 a 23 h (CONFIG_P2.horas_etiqueta)
    dias_mes: float = DIAS_MES
    recalculos_p3_mes: int = 1

    def inferencias(self) -> dict[str, float]:
        p1 = self.estaciones * self.dias_mes
        p2 = self.estaciones * self.horas_p2_por_dia * self.dias_mes
        p3 = self.estaciones * self.recalculos_p3_mes
        return {"p1": p1, "p2": p2, "p3": p3, "total": p1 + p2 + p3}

    def peticiones(self, lote: int) -> dict[str, float]:
        """P1 recibe una estación por petición; P2 y P3 aceptan listas de hasta 1000."""
        lotes = math.ceil(self.estaciones / max(1, lote))
        p1 = self.estaciones * self.dias_mes
        p2 = self.horas_p2_por_dia * self.dias_mes * lotes
        p3 = self.recalculos_p3_mes * lotes
        return {"p1": p1, "p2": p2, "p3": p3, "total": p1 + p2 + p3}


def usd_hora(tarifa: Tarifa, vcpu: float, gb: float) -> float:
    """Costo de una instancia encendida durante una hora."""
    return (vcpu * tarifa.usd_vcpu_segundo + gb * tarifa.usd_gb_segundo) * 3600


def usd_por_peticion(tarifa: Tarifa, vcpu: float, gb: float, segundos_activos: float) -> float:
    """Costo de una petición cuando sólo se cobra el tiempo activo (Cloud Run por petición)."""
    facturados = math.ceil(segundos_activos / tarifa.granularidad_s) * tarifa.granularidad_s
    computo = (vcpu * tarifa.usd_vcpu_segundo + gb * tarifa.usd_gb_segundo) * facturados
    return computo + tarifa.usd_millon_peticiones / 1e6


def tamano_fargate(vcpu_minimo: float, gb_minimo: float) -> tuple[float, float]:
    """El tamaño válido de Fargate más chico que cubre la CPU y la memoria pedidas."""
    for vcpu in sorted(TAMANOS_FARGATE):
        if vcpu + 1e-9 < vcpu_minimo:
            continue
        for gb in TAMANOS_FARGATE[vcpu]:
            if gb + 1e-9 >= gb_minimo:
                return vcpu, gb
    raise ValueError(f"Ningún tamaño de Fargate cubre {vcpu_minimo} vCPU y {gb_minimo} GB")


def dimensionar(
    cpu_nucleos_p95: float,
    memoria_mib_max: float,
    margen: float = 0.3,
    vcpu_max: float | None = None,
):
    """Tamaño recomendado: lo medido en la prueba más un margen (30 % por omisión).

    Con ``vcpu_max`` la CPU pedida no pasa de los núcleos que el servicio puede aprovechar.
    """
    vcpu_minimo = cpu_nucleos_p95 * (1 + margen)
    if vcpu_max is not None:
        vcpu_minimo = min(vcpu_minimo, vcpu_max)
    return tamano_fargate(vcpu_minimo, memoria_mib_max / 1024 * (1 + margen))


def _escenario(clave: str, nombre: str, tarifa: Tarifa, vcpu: float, gb: float, **valores) -> dict:
    fila = {"clave": clave, "escenario": nombre, "tarifa": tarifa.nombre, "vcpu": vcpu, "gb": gb}
    fila.update(valores)
    inferencias = fila.get("inferencias_mes", math.nan)
    fila["usd_por_1000"] = (
        fila["usd_mes"] / inferencias * 1000 if inferencias and inferencias > 0 else math.nan
    )
    return fila


def latencias_por_pregunta(capacidad: dict) -> dict[str, float]:
    """Latencia promedio en ms de P1, P2 y P3 en la etapa, de sus endpoints (``"P1 …"``).

    A la pregunta sin datos se le asigna la latencia promedio de todas las inferencias o, si
    también falta, la del agregado.
    """
    respaldo = _medida(capacidad.get("promedio_inferencia_ms"), capacidad.get("promedio_ms"))
    endpoints = capacidad.get("endpoints") or []
    latencias = {}
    for clave in PREGUNTAS:
        propias = [
            e
            for e in endpoints
            if str(e.get("endpoint", "")).startswith(f"{clave.upper()} ")
            and e.get("peticiones")
            and _es_numero(e.get("promedio_ms"))
        ]
        peso = sum(e["peticiones"] for e in propias)
        suma = sum(e["promedio_ms"] * e["peticiones"] for e in propias)
        latencias[clave] = suma / peso if peso else respaldo
    return latencias


def analizar(
    capacidad: dict,
    recursos_api: dict,
    lotes: dict | None = None,
    recursos_mlflow: dict | None = None,
    demanda: Demanda | None = None,
    margen: float = 0.3,
    horas_con_horario: float = 19.0,
    vcpu_max: float = 1.0,
    vcpu_medidos: float = 1.0,
    saturacion: dict | None = None,
) -> dict:
    """Escenarios de costo y optimizaciones a partir de la prueba de carga.

    ``capacidad`` es el resumen de la etapa con más RPS que cumplió el objetivo (de
    ``carga.capacidad_sostenida``), con ``recursos_api`` de esa misma etapa; ``saturacion``, el
    de la etapa con más RPS (``carga.saturacion``). ``lotes`` trae la latencia medida de
    peticiones con todas las estaciones (``p2_ms``, ``p3_ms``). ``vcpu_max`` son los núcleos que
    la API puede aprovechar (un proceso de uvicorn usa uno) y ``vcpu_medidos``, el límite de CPU
    del contenedor de la API durante la prueba. La capacidad de la instancia recomendada es la
    medida; sólo se escala linealmente si sus núcleos útiles difieren de los medidos.
    """
    demanda = demanda or Demanda()
    lotes = lotes or {}
    saturacion = saturacion or {}
    rps_total = float(capacidad["rps_sostenido"])
    # Las sondas GET /live no son inferencias: se usa sólo el tráfico de los modelos.
    rps = _medida(capacidad.get("rps_inferencias"), rps_total)
    latencia_ms = _medida(capacidad.get("promedio_inferencia_ms"), capacidad.get("promedio_ms"))
    latencias = latencias_por_pregunta(capacidad)
    cpu_promedio = float(recursos_api["cpu_nucleos_promedio"])
    rps_por_nucleo = rps / cpu_promedio if cpu_promedio > 0 else math.nan
    vcpu, gb = dimensionar(
        recursos_api["cpu_nucleos_p95"], recursos_api["memoria_mib_max"], margen, vcpu_max
    )
    nucleos_utiles = min(vcpu, vcpu_max)
    capacidad_rps = rps * nucleos_utiles / vcpu_medidos if vcpu_medidos > 0 else math.nan
    x86, arm = TARIFAS["fargate_x86"], TARIFAS["fargate_arm"]
    run = TARIFAS["cloud_run_peticion"]
    inferencias = demanda.inferencias()
    con_lote = demanda.peticiones(lote=1000)
    sin_lote = demanda.peticiones(lote=1)

    escenarios = [
        _escenario(
            "piso",
            "Piso: instancia recomendada a su capacidad sostenida",
            x86,
            vcpu,
            gb,
            usd_hora=usd_hora(x86, vcpu, gb),
            usd_mes=usd_hora(x86, vcpu, gb) * HORAS_MES,
            inferencias_mes=capacidad_rps * 3600 * HORAS_MES,
        ),
        _escenario(
            "siempre_encendido",
            "MiBici, siempre encendido (24 x 7)",
            x86,
            vcpu,
            gb,
            usd_hora=usd_hora(x86, vcpu, gb),
            usd_mes=usd_hora(x86, vcpu, gb) * HORAS_MES,
            inferencias_mes=inferencias["total"],
        ),
        _escenario(
            "horario",
            f"MiBici, encendido {horas_con_horario:g} h al día (5:00 a 24:00)",
            x86,
            vcpu,
            gb,
            usd_hora=usd_hora(x86, vcpu, gb),
            usd_mes=usd_hora(x86, vcpu, gb) * horas_con_horario * demanda.dias_mes,
            inferencias_mes=inferencias["total"],
        ),
        _escenario(
            "arm",
            "MiBici, siempre encendido en ARM (Graviton)",
            arm,
            vcpu,
            gb,
            usd_hora=usd_hora(arm, vcpu, gb),
            usd_mes=usd_hora(arm, vcpu, gb) * HORAS_MES,
            inferencias_mes=inferencias["total"],
        ),
    ]

    if _es_numero(*latencias.values()):
        # Cada pregunta paga su propia latencia, redondeada a la granularidad de la tarifa.
        costo = {
            clave: usd_por_peticion(run, vcpu, gb, latencias[clave] / 1000) for clave in PREGUNTAS
        }
        p2_lote_s = _medida(lotes.get("p2_ms"), math.nan) / 1000
        p3_lote_s = _medida(lotes.get("p3_ms"), math.nan) / 1000
        usd_sin_lote = sum(sin_lote[clave] * costo[clave] for clave in PREGUNTAS)
        escenarios.append(
            _escenario(
                "cloud_run",
                "MiBici, Cloud Run por petición, una estación por petición",
                run,
                vcpu,
                gb,
                usd_hora=math.nan,
                usd_mes=usd_sin_lote,
                inferencias_mes=inferencias["total"],
                peticiones_mes=sin_lote["total"],
            )
        )
        if not math.isnan(p2_lote_s):
            p3_s = p3_lote_s if not math.isnan(p3_lote_s) else p2_lote_s
            usd_con_lote = (
                con_lote["p1"] * costo["p1"]
                + con_lote["p2"] * usd_por_peticion(run, vcpu, gb, p2_lote_s)
                + con_lote["p3"] * usd_por_peticion(run, vcpu, gb, p3_s)
            )
            escenarios.append(
                _escenario(
                    "cloud_run_lote",
                    "MiBici, Cloud Run por petición, P2 y P3 en lote",
                    run,
                    vcpu,
                    gb,
                    usd_hora=math.nan,
                    usd_mes=usd_con_lote,
                    inferencias_mes=inferencias["total"],
                    peticiones_mes=con_lote["total"],
                )
            )

    if recursos_mlflow:
        # MLflow está ocioso durante la carga: su p95 de CPU sólo refleja el proceso de Python
        # que lanza el healthcheck cada 10 s, así que se dimensiona con el promedio.
        cpu_mlflow = recursos_mlflow.get(
            "cpu_nucleos_promedio", recursos_mlflow["cpu_nucleos_p95"]
        )
        vcpu_m, gb_m = dimensionar(cpu_mlflow, recursos_mlflow["memoria_mib_max"], margen)
        escenarios.append(
            _escenario(
                "mlflow",
                "Servidor de MLflow encendido 24 x 7 (la API carga los modelos al arrancar)",
                x86,
                vcpu_m,
                gb_m,
                usd_hora=usd_hora(x86, vcpu_m, gb_m),
                usd_mes=usd_hora(x86, vcpu_m, gb_m) * HORAS_MES,
                inferencias_mes=math.nan,
            )
        )

    resultado = {
        "supuestos": {
            "tarifas": {clave: asdict(t) for clave, t in TARIFAS.items()},
            "demanda": asdict(demanda),
            "inferencias_mes": inferencias,
            "peticiones_mes_con_lote": con_lote,
            "peticiones_mes_sin_lote": sin_lote,
            "margen": margen,
            "horas_mes": HORAS_MES,
            "horas_con_horario": horas_con_horario,
            "vcpu_max": vcpu_max,
            "vcpu_medidos": vcpu_medidos,
        },
        "medido": {
            "usuarios": capacidad.get("usuarios"),
            "rps_sostenido": rps_total,
            "rps_inferencias": rps,
            "latencia_promedio_ms": capacidad.get("promedio_ms"),
            "latencia_inferencia_ms": latencia_ms,
            "latencias_pregunta_ms": latencias,
            "p95_ms": capacidad.get("p95_ms"),
            "cpu_nucleos_promedio": cpu_promedio,
            "cpu_nucleos_p95": recursos_api["cpu_nucleos_p95"],
            "memoria_mib_max": recursos_api["memoria_mib_max"],
            "rps_por_nucleo": rps_por_nucleo,  # sólo como dato: la capacidad no se extrapola
            "rps_saturacion": saturacion.get("rps_sostenido"),
            "rps_inferencias_saturacion": saturacion.get("rps_inferencias"),
            "usuarios_saturacion": saturacion.get("usuarios"),
            "p95_saturacion_ms": saturacion.get("p95_ms"),
            "lotes": lotes,
        },
        "dimension_recomendada": {
            "vcpu": vcpu,
            "gb": gb,
            "capacidad_rps": capacidad_rps,
            "nucleos_utiles": nucleos_utiles,
        },
        "utilizacion_mibici": inferencias["total"] / (capacidad_rps * 3600 * HORAS_MES)
        if capacidad_rps > 0
        else math.nan,
        "escenarios": escenarios,
    }
    resultado["optimizaciones"] = optimizaciones(resultado)
    return resultado


def _medida(valor, respaldo) -> float:
    """``valor`` como número; si falta o no es un número, ``respaldo``."""
    numero = float(valor) if isinstance(valor, int | float) else math.nan
    if math.isnan(numero):
        return float(respaldo) if respaldo is not None else math.nan
    return numero


def ahorro(base: dict, alternativa: dict) -> float:
    """Ahorro relativo del costo mensual de ``alternativa`` frente a ``base``."""
    return 1 - alternativa["usd_mes"] / base["usd_mes"] if base["usd_mes"] else math.nan


def _es_numero(*valores) -> bool:
    return all(
        valor is not None and not (isinstance(valor, float) and math.isnan(valor))
        for valor in valores
    )


def optimizaciones(resultado: dict) -> list[str]:
    """Optimizaciones con cifras, frente a la instancia de MiBici siempre encendida.

    Cada frase cita los USD al mes y el ahorro; se omite si falta su escenario o su valor.
    """
    filas = {fila.get("clave"): fila for fila in resultado["escenarios"]}
    supuestos = resultado["supuestos"]
    base = filas.get("siempre_encendido")
    if base is None or not _es_numero(base["usd_mes"]):
        return []
    usd_base = base["usd_mes"]
    frases = []

    uso = resultado.get("utilizacion_mibici", math.nan)
    capacidad_rps = resultado.get("dimension_recomendada", {}).get("capacidad_rps", math.nan)
    if _es_numero(uso, capacidad_rps):
        frases.append(
            f"La demanda de MiBici ({_numero(supuestos['inferencias_mes']['total'])} inferencias "
            f"al mes) ocupa el {uso:.3%} de la capacidad sostenida medida de la instancia "
            f"({capacidad_rps:.1f} RPS de inferencias): de los "
            f"{_usd(usd_base)} USD al mes de tenerla siempre encendida, {_usd(usd_base * (1 - uso))}"
            f" pagan tiempo ocioso ({1 - uso:.1%}). Conviene escalar a cero o facturar por petición."
        )

    horario = filas.get("horario")
    if horario is not None and _es_numero(horario["usd_mes"], ahorro(base, horario)):
        horas = supuestos.get("horas_con_horario", math.nan)
        frases.append(
            f"Encender la instancia sólo {horas:g} h al día (5:00 a 24:00) baja el costo de "
            f"{_usd(usd_base)} a {_usd(horario['usd_mes'])} USD al mes, un ahorro del "
            f"{ahorro(base, horario):.1%}."
        )

    arm = filas.get("arm")
    if arm is not None and _es_numero(arm["usd_mes"], ahorro(base, arm)):
        frases.append(
            f"Con el mismo tamaño en ARM (Graviton) la instancia cuesta {_usd(arm['usd_mes'])} USD"
            f" al mes en vez de {_usd(usd_base)}, un ahorro del {ahorro(base, arm):.1%}; requiere "
            "validar la construcción de la imagen y el rendimiento de la API en linux/arm64."
        )

    run = filas.get("cloud_run")
    if run is not None and _es_numero(run["usd_mes"], ahorro(base, run)):
        sin_lote = supuestos["peticiones_mes_sin_lote"]["total"]
        frase = (
            f"Cloud Run con facturación por petición, una estación por petición "
            f"({_numero(sin_lote)} peticiones al mes, cada pregunta con su latencia medida), "
            f"cuesta {_usd(run['usd_mes'])} USD al mes, un ahorro del {ahorro(base, run):.1%} "
            f"frente a {_usd(usd_base)} USD"
        )
        lote = filas.get("cloud_run_lote")
        if lote is not None and _es_numero(lote["usd_mes"], ahorro(base, lote)):
            con_lote = supuestos["peticiones_mes_con_lote"]["total"]
            frase += (
                f"; si P2 y P3 se envían en lote ({_numero(con_lote)} peticiones al mes en vez de "
                f"{_numero(sin_lote)}), baja a {_usd(lote['usd_mes'])} USD al mes, un ahorro del "
                f"{ahorro(base, lote):.1%}"
            )
        frases.append(
            frase + ". Ambos sin contar el nivel gratuito. Al escalar a cero, la primera petición "
            "paga además el arranque del contenedor y la carga de los modelos desde el Registry, "
            "que esta prueba no mide, y la facturación por petición también cobra el tiempo de "
            "arranque y apagado de la instancia."
        )

    mlflow = filas.get("mlflow")
    if mlflow is not None and _es_numero(mlflow["usd_mes"]):
        frases.append(
            f"El servidor de MLflow ({mlflow['vcpu']:g} vCPU y {mlflow['gb']:g} GB) encendido "
            f"24 x 7 cuesta {_usd(mlflow['usd_mes'])} USD al mes. Hoy no se puede apagar: la API "
            "resuelve el alias y carga los modelos del Registry cada vez que arranca, así que lo "
            "necesita al escalar o reiniciarse. Para ahorrarlo habría que empaquetar el modelo "
            "campeón en la imagen al desplegar o servir los artefactos desde almacenamiento de "
            f"objetos; esos {_usd(mlflow['usd_mes'])} USD al mes son el ahorro máximo posible."
        )
    return frases


def _usd(valor: float, decimales: int = 2) -> str:
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return "n/d"
    if 0 < abs(valor) < 0.01:
        return f"{valor:.6f}"
    return f"{valor:,.{decimales}f}"


def _numero(valor: float) -> str:
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return "n/d"
    return f"{valor:,.0f}"


def a_markdown(resultado: dict) -> str:
    """Tabla de escenarios y optimizaciones, listas para el README o el reporte."""
    medido, dimension = resultado["medido"], resultado["dimension_recomendada"]
    rps_inferencias = medido.get("rps_inferencias", medido["rps_sostenido"])
    renglones = [
        "| Escenario | Tamaño | USD al mes | Inferencias al mes | USD por 1,000 inferencias |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for fila in resultado["escenarios"]:
        renglones.append(
            f"| {fila['escenario']} | {fila['vcpu']:g} vCPU, {fila['gb']:g} GB "
            f"| {_usd(fila['usd_mes'])} | {_numero(fila['inferencias_mes'])} "
            f"| {_usd(fila['usd_por_1000'], 4)} |"
        )
    encabezado = (
        f"Medido: capacidad sostenida de {rps_inferencias:.1f} RPS de inferencias "
        f"({medido['rps_sostenido']:.1f} RPS en total con las sondas GET /live) con "
        f"{medido['usuarios']} usuarios, p95 de {medido['p95_ms']:,.0f} ms, "
        f"{medido['cpu_nucleos_promedio']:.2f} núcleos en promedio "
        f"({medido['cpu_nucleos_p95']:.2f} en p95) y {medido['memoria_mib_max']:.0f} MiB de "
        "memoria máxima."
    )
    if _es_numero(medido.get("rps_saturacion"), medido.get("usuarios_saturacion")):
        inferencias_sat = medido.get("rps_inferencias_saturacion")
        detalle = f" ({inferencias_sat:.1f} de inferencias)" if _es_numero(inferencias_sat) else ""
        p95_sat = medido.get("p95_saturacion_ms")
        encabezado += (
            f" Saturación observada: {medido['rps_saturacion']:.1f} RPS{detalle} con "
            f"{medido['usuarios_saturacion']} usuarios"
            + (f" y p95 de {p95_sat:,.0f} ms." if _es_numero(p95_sat) else ".")
        )
    supuestos = resultado["supuestos"]
    nucleos_utiles = dimension.get("nucleos_utiles", dimension["vcpu"])
    vcpu_medidos = supuestos.get("vcpu_medidos", nucleos_utiles)
    escala = (
        f", escalada linealmente de {vcpu_medidos:g} a {nucleos_utiles:g} núcleos"
        if abs(nucleos_utiles - vcpu_medidos) > 1e-9
        else ""
    )
    encabezado += (
        f" Tamaño recomendado: {dimension['vcpu']:g} vCPU y {dimension['gb']:g} GB, con esa "
        f"capacidad medida de {dimension['capacidad_rps']:.1f} RPS de inferencias{escala}. "
        f"La demanda de MiBici ocuparía el {resultado['utilizacion_mibici']:.3%} de esa capacidad."
    )
    frases = resultado.get("optimizaciones")
    if frases is None:
        frases = optimizaciones(resultado)
    lista = [f"- {frase}" for frase in frases] or ["- Sin escenarios suficientes."]
    return (
        encabezado
        + "\n\n"
        + "\n".join(renglones)
        + "\n\n**Optimizaciones**\n\n"
        + "\n".join(lista)
        + "\n"
    )
