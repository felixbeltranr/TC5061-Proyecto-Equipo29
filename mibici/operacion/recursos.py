"""Consumo de CPU y memoria de los contenedores durante la prueba de carga.

``Muestreador`` corre ``docker stats --no-stream`` (o ``podman stats``) cada pocos segundos en
un hilo aparte mientras Locust genera la carga, y guarda una muestra por contenedor. ``resumir``
convierte las muestras en núcleos de CPU y MiB de memoria, que son las cantidades con las que
``finops`` dimensiona el servicio.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Self

_UNIDADES = {
    "b": 1,
    "kb": 1e3,
    "kib": 1024,
    "mb": 1e6,
    "mib": 1024**2,
    "gb": 1e9,
    "gib": 1024**3,
    "tb": 1e12,
    "tib": 1024**4,
}
_TAMANO = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)\s*$")
MIB = 1024**2


def bytes_de(texto: str) -> float:
    """Convierte ``'45.6MiB'``, ``'1.2kB'`` o ``'0B'`` a bytes; ``'--'`` cuenta como cero."""
    texto = str(texto).strip()
    if texto in ("", "--"):
        return 0.0
    coincidencia = _TAMANO.match(texto)
    if not coincidencia:
        raise ValueError(f"Tamaño no reconocido: {texto!r}")
    valor, unidad = float(coincidencia.group(1)), (coincidencia.group(2) or "B").lower()
    if unidad not in _UNIDADES:
        raise ValueError(f"Unidad no reconocida: {texto!r}")
    return valor * _UNIDADES[unidad]


def porcentaje(texto) -> float:
    """``'12.5%'`` → 12.5; ``'--'`` (contenedor detenido) → 0."""
    texto = str(texto).strip().rstrip("%")
    return 0.0 if texto in ("", "--") else float(texto)


@dataclass(frozen=True)
class Muestra:
    marca: float  # segundos desde la época Unix
    contenedor: str
    cpu_pct: float  # porcentaje de un núcleo: 100 equivale a un núcleo completo
    memoria_bytes: float
    limite_memoria_bytes: float
    red_entrada_bytes: float
    red_salida_bytes: float


def parsear_registro(registro: dict, marca: float) -> Muestra:
    """Un renglón de ``docker stats`` o de ``podman stats`` en formato JSON."""
    nombre = (
        registro.get("Name")
        or registro.get("name")
        or registro.get("Container")
        or registro.get("id", "")
    )
    cpu = registro.get("CPUPerc", registro.get("cpu_percent", "0%"))
    memoria = registro.get("MemUsage", registro.get("mem_usage", "0B / 0B"))
    red = registro.get("NetIO", registro.get("net_io", "0B / 0B"))
    uso, _, limite = str(memoria).partition("/")
    entrada, _, salida = str(red).partition("/")
    return Muestra(
        marca=marca,
        contenedor=str(nombre),
        cpu_pct=porcentaje(cpu),
        memoria_bytes=bytes_de(uso),
        limite_memoria_bytes=bytes_de(limite),
        red_entrada_bytes=bytes_de(entrada),
        red_salida_bytes=bytes_de(salida),
    )


def registros_de_salida(texto: str) -> list[dict]:
    """Docker imprime un objeto JSON por renglón; podman, una lista JSON."""
    texto = texto.strip()
    if not texto:
        return []
    if texto.startswith("["):
        return list(json.loads(texto))
    return [json.loads(renglon) for renglon in texto.splitlines() if renglon.strip()]


def comando_stats(motor: str, contenedores: list[str]) -> list[str]:
    formato = "json" if motor == "podman" else "{{json .}}"
    return [motor, "stats", "--no-stream", "--format", formato, *contenedores]


class Muestreador:
    """Toma muestras en segundo plano mientras dura el bloque ``with``."""

    def __init__(
        self,
        contenedores: list[str],
        motor: str = "docker",
        intervalo: float = 2.0,
        ejecutar=subprocess.run,
    ):
        self.contenedores = list(contenedores)
        self.motor = motor
        self.intervalo = intervalo
        self.ejecutar = ejecutar
        self.muestras: list[Muestra] = []
        self.errores: list[str] = []
        self._alto = threading.Event()
        self._hilo = threading.Thread(target=self._bucle, name="muestreador-stats", daemon=True)

    def tomar(self) -> list[Muestra]:
        resultado = self.ejecutar(
            comando_stats(self.motor, self.contenedores),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        if resultado.returncode != 0:
            self.errores.append((resultado.stderr or "").strip()[:300])
            return []
        marca = time.time()
        return [parsear_registro(r, marca) for r in registros_de_salida(resultado.stdout)]

    def _bucle(self) -> None:
        while not self._alto.is_set():
            inicio = time.monotonic()
            try:
                self.muestras.extend(self.tomar())
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                self.errores.append(f"{type(exc).__name__}: {exc}"[:300])
            self._alto.wait(max(0.0, self.intervalo - (time.monotonic() - inicio)))

    def __enter__(self) -> Self:
        self._hilo.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._alto.set()
        self._hilo.join(timeout=self.intervalo + 35)


def guardar_csv(muestras: list[Muestra], ruta: Path) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    columnas = [campo.name for campo in fields(Muestra)]
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        for muestra in muestras:
            escritor.writerow(asdict(muestra))


def leer_csv(ruta: Path) -> list[Muestra]:
    with Path(ruta).open(newline="", encoding="utf-8") as archivo:
        return [
            Muestra(
                marca=float(r["marca"]),
                contenedor=r["contenedor"],
                cpu_pct=float(r["cpu_pct"]),
                memoria_bytes=float(r["memoria_bytes"]),
                limite_memoria_bytes=float(r["limite_memoria_bytes"]),
                red_entrada_bytes=float(r["red_entrada_bytes"]),
                red_salida_bytes=float(r["red_salida_bytes"]),
            )
            for r in csv.DictReader(archivo)
        ]


def percentil(valores: list[float], q: float) -> float:
    """Percentil con interpolación lineal (como ``numpy.percentile``), sin dependencias."""
    if not valores:
        return math.nan
    ordenados = sorted(valores)
    posicion = (len(ordenados) - 1) * q / 100
    abajo, arriba = math.floor(posicion), math.ceil(posicion)
    peso = posicion - abajo
    return ordenados[abajo] * (1 - peso) + ordenados[arriba] * peso


def en_ventana(
    muestras: list[Muestra],
    marcas: list[float],
    descartar_s: float,
    inicio: float,
    fin: float,
) -> list[Muestra]:
    """Las muestras de la ventana estable, la misma con la que se calcula el RPS sostenido.

    ``marcas`` son las del historial de Locust (segundos de época): la ventana va de la primera
    más ``descartar_s`` a la última. Sin marcas utilizables se usa ``inicio + descartar_s`` y
    ``fin``. Si ninguna muestra cae en la ventana, se devuelven todas.
    """
    validas = [float(m) for m in marcas if m is not None and math.isfinite(float(m))]
    if validas:
        desde, hasta = min(validas) + descartar_s, max(validas)
    else:
        desde, hasta = inicio + descartar_s, fin
    dentro = [m for m in muestras if desde <= m.marca <= hasta]
    return dentro or list(muestras)


def resumir(muestras: list[Muestra]) -> dict[str, dict]:
    """Por contenedor: núcleos de CPU (promedio, p95 y máximo) y MiB de memoria."""
    por_contenedor: dict[str, list[Muestra]] = {}
    for muestra in muestras:
        por_contenedor.setdefault(muestra.contenedor, []).append(muestra)
    resumen = {}
    for contenedor, lista in sorted(por_contenedor.items()):
        nucleos = [m.cpu_pct / 100 for m in lista]
        memoria = [m.memoria_bytes / MIB for m in lista]
        resumen[contenedor] = {
            "muestras": len(lista),
            "cpu_nucleos_promedio": sum(nucleos) / len(nucleos),
            "cpu_nucleos_p95": percentil(nucleos, 95),
            "cpu_nucleos_max": max(nucleos),
            "memoria_mib_promedio": sum(memoria) / len(memoria),
            "memoria_mib_max": max(memoria),
            "limite_memoria_mib": max(m.limite_memoria_bytes for m in lista) / MIB,
        }
    return resumen


def contenedor_de_servicio(resumen: dict[str, dict], servicio: str) -> str | None:
    """Encuentra el contenedor de un servicio de Compose (``<proyecto>-api-1`` para ``api``)."""
    candidatos = [
        nombre
        for nombre in resumen
        if nombre == servicio or nombre.endswith((f"-{servicio}-1", f"_{servicio}_1"))
    ]
    return candidatos[0] if candidatos else None
