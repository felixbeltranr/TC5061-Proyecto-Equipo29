"""Prueba de carga por etapas, consumo de recursos y análisis FinOps de la API de MiBici.

Con el sistema levantado con el tamaño de la medición, desde la raíz del repositorio:

    docker compose -f compose.yaml -f carga/compose.medicion.yaml up -d --wait
    python scripts/carga/correr_carga.py

``carga/compose.medicion.yaml`` fija los límites de CPU y memoria con los que se mide y se
costea; el orquestador los lee de los contenedores y los deja en el resumen. Por omisión corre
cinco etapas de 90 segundos (10, 15, 20, 30 y 50 usuarios) y mientras tanto
toma ``docker stats`` de los contenedores ``api`` y ``mlflow`` cada 2 segundos. Al final mide la
latencia de una petición con todas las estaciones (P2 y P3 aceptan listas), y escribe en
``reports/carga/<fecha>/``:

- ``resumen.json`` y ``resumen.md``: cada etapa (RPS sostenido, p50/p95/p99, errores y
  recursos), la capacidad sostenida que cumple el objetivo de latencia y la saturación
  observada (la etapa con más RPS);
- ``finops.json`` y ``finops.md``: escenarios de costo y costo por 1,000 inferencias;
- ``desempeno.png``, ``capacidad.png``, ``endpoints.png`` y ``recursos.png``;
- los CSV y el HTML de Locust de cada etapa, y los CSV de ``docker stats``.

Con ``--actualizar-readme`` reemplaza el bloque de resultados del README. Termina con código 0
si al menos una etapa dejó datos; 2 si la API no está lista o ninguna etapa los dejó.

Con ``--reposo SEGUNDOS`` no corre la carga: sólo toma ``docker stats`` ese tiempo y escribe
``reposo.json`` y ``reposo_recursos.csv``. Sirve para medir MLflow tal como lo levanta
``compose.yaml`` solo; ese ``reposo.json`` se pasa después con ``--mlflow-sin-ajuste`` para
costear la diferencia.

Con ``--recalcular CARPETA`` tampoco corre la carga: vuelve a calcular ``finops.json``,
``finops.md`` y ``resumen.md`` desde el ``resumen.json`` de esa corrida (por ejemplo, si
cambian las tarifas) y, con ``--actualizar-readme``, reescribe el bloque del README. Usa los
parámetros que la corrida guardó; ``--estaciones`` y ``--vcpu-max`` sólo cuentan para una
corrida que no los guardó, y las demás opciones de la carga no aplican.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

from mibici.operacion import carga, finops, graficas, recursos
from mibici.operacion.payloads import cuerpo_p1, cuerpo_p2, cuerpo_p3

RAIZ = Path(__file__).resolve().parents[2]
LOCUSTFILE = RAIZ / "carga" / "locustfile.py"
README = RAIZ / "README.md"
INICIO_BLOQUE = "<!-- resultados-carga:inicio -->"
FIN_BLOQUE = "<!-- resultados-carga:fin -->"
PREGUNTAS = ("p1", "p2", "p3")


def argumentos(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--host",
        default="http://127.0.0.1:8000",
        help="con localhost, en Windows cada conexión nueva de Python tarda unos 2 s (IPv6)",
    )
    p.add_argument("--etapas", default="10,15,20,30,50", help="usuarios de cada etapa")
    p.add_argument("--duracion", type=int, default=90, help="segundos por etapa")
    p.add_argument("--tasa", type=float, default=20.0, help="usuarios nuevos por segundo")
    p.add_argument("--salida", type=Path, default=None)
    p.add_argument("--motor", choices=("docker", "podman"), default="docker")
    p.add_argument("--contenedores", default="", help="nombres separados por coma")
    p.add_argument("--intervalo", type=float, default=2.0, help="segundos entre muestras")
    p.add_argument("--sin-recursos", action="store_true", help="no toma docker stats")
    p.add_argument("--p95-objetivo", type=float, default=500.0, help="milisegundos")
    p.add_argument("--error-maximo", type=float, default=0.01)
    p.add_argument("--estaciones", type=int, default=finops.Demanda().estaciones)
    p.add_argument("--lote-sonda", type=int, default=finops.Demanda().estaciones)
    p.add_argument("--repeticiones-sonda", type=int, default=20)
    p.add_argument(
        "--vcpu-max",
        type=float,
        default=1.0,
        help="núcleos que la API puede aprovechar (un proceso de uvicorn usa uno)",
    )
    p.add_argument(
        "--vcpu-medidos",
        type=float,
        default=None,
        help="límite de CPU de la API durante la prueba; por omisión se lee del contenedor",
    )
    p.add_argument(
        "--reposo",
        type=int,
        default=0,
        help="sólo toma docker stats durante estos segundos, sin carga, y termina",
    )
    p.add_argument(
        "--mlflow-sin-ajuste",
        type=Path,
        default=None,
        help="reposo.json tomado con compose.yaml solo, para costear MLflow sin el ajuste",
    )
    p.add_argument(
        "--recalcular",
        type=Path,
        default=None,
        help=(
            "carpeta de una corrida ya hecha: recalcula sus costos sin repetir la carga, con "
            "los parámetros que guardó (--estaciones y --vcpu-max sólo si no los guardó)"
        ),
    )
    p.add_argument("--actualizar-readme", action="store_true")
    return p.parse_args(argv)


def _get(url: str, timeout: float = 10.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as respuesta:
            return respuesta.status, respuesta.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace")


def preguntas_sin_modelo(detalle: str) -> set[str]:
    """``'Models not loaded: p2 (X), p3 (Y)'`` → ``{'p2', 'p3'}``."""
    if "No model is loaded" in detalle:
        return set(PREGUNTAS)
    return set(re.findall(r"\b(p[123])\s*\(", detalle))


def endpoints_disponibles(host: str) -> list[str]:
    estado, cuerpo = _get(f"{host}/live")
    if estado != 200:
        raise RuntimeError(f"GET /live respondió {estado}: ¿está levantada la API en {host}?")
    estado, cuerpo = _get(f"{host}/ready")
    if estado == 200:
        return list(PREGUNTAS)
    try:
        detalle = json.loads(cuerpo).get("detail", cuerpo)
    except ValueError:
        detalle = cuerpo
    faltan = preguntas_sin_modelo(str(detalle))
    return [p for p in PREGUNTAS if p not in faltan]


def nombres_compose(texto: str) -> dict[str, str]:
    """Salida de ``docker compose ps --format json`` (una lista o un objeto por renglón)."""
    texto = texto.strip()
    if not texto:
        return {}
    registros = json.loads(texto) if texto.startswith("[") else [
        json.loads(renglon) for renglon in texto.splitlines() if renglon.strip()
    ]
    return {r.get("Service", ""): r.get("Name", "") for r in registros if r.get("Name")}


def servicio_de(nombre: str) -> str:
    """``'proyecto-api-1'`` → ``'api'``; un nombre que no es de esos servicios se queda igual.

    Decide por el final del nombre, como los nombra Compose (``<proyecto>-<servicio>-<n>``):
    un proyecto que se llame ``mibici-api`` no convierte a su MLflow en la API.
    """
    encontrado = re.fullmatch(r"(?:.*[-_])?(api|mlflow)(?:[-_]\d+)?", nombre)
    return encontrado.group(1) if encontrado else nombre


def contenedores_del_sistema(args: argparse.Namespace) -> dict[str, str]:
    """``{'api': '<contenedor>', 'mlflow': '<contenedor>'}`` del proyecto de Compose actual."""
    if args.contenedores:
        nombres = [n.strip() for n in args.contenedores.split(",") if n.strip()]
        return {servicio_de(n): n for n in nombres}
    comando = [args.motor, "compose", "ps", "--format", "json"]
    resultado = subprocess.run(
        comando,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=RAIZ,
        check=False,
    )
    if resultado.returncode != 0:
        raise RuntimeError(f"{' '.join(comando)} falló: {resultado.stderr.strip()[:300]}")
    servicios = nombres_compose(resultado.stdout)
    return {s: n for s, n in servicios.items() if s in ("api", "mlflow")}


def leer_limites(texto: str) -> dict[str, float | None]:
    """``'1000000000 1073741824'`` (NanoCpus y Memory de ``inspect``) → núcleos y MiB.

    Un cero, o un valor que no se pudo leer, significa que el contenedor no tiene ese límite.
    """
    partes = texto.split()
    nano = int(partes[0]) if len(partes) > 0 and partes[0].isdigit() else 0
    memoria = int(partes[1]) if len(partes) > 1 and partes[1].isdigit() else 0
    return {
        "cpus": nano / 1e9 if nano else None,
        "memoria_mib": memoria / recursos.MIB if memoria else None,
    }


def limites_de(contenedores: dict[str, str], motor: str) -> dict[str, dict]:
    """Límites de CPU y memoria de cada servicio, leídos del contenedor que está corriendo."""
    formato = "{{.HostConfig.NanoCpus}} {{.HostConfig.Memory}}"
    limites = {}
    for servicio, nombre in contenedores.items():
        try:
            resultado = subprocess.run(
                [motor, "inspect", "--format", formato, nombre],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            leido = resultado.returncode == 0
            salida = resultado.stdout if leido else ""
        except OSError:
            leido, salida = False, ""
        # ``leido`` distingue «sin límite» (inspect respondió cero) de «no se pudo leer».
        limites[servicio] = dict(leer_limites(salida), leido=leido)
    return limites


def texto_limites(limites: dict[str, dict]) -> str:
    """``'Límites durante la corrida: api con 1 vCPU y 1,024 MiB; mlflow con CPU sin límite…'``."""
    partes = []
    for servicio, limite in limites.items():
        if limite.get("leido") is False:
            partes.append(f"{servicio} con límites desconocidos (no se pudieron leer)")
            continue
        cpus, memoria = limite.get("cpus"), limite.get("memoria_mib")
        cpu = f"{cpus:g} vCPU" if cpus else "CPU sin límite"
        ram = f"{memoria:,.0f} MiB" if memoria else "memoria sin límite"
        partes.append(f"{servicio} con {cpu} y {ram}")
    return "Límites durante la corrida: " + "; ".join(partes) + "." if partes else ""


def medir_reposo(args: argparse.Namespace, salida: Path, fecha: str) -> int:
    """Toma ``docker stats`` sin carga y escribe ``reposo.json`` y ``reposo_recursos.csv``."""
    contenedores = contenedores_del_sistema(args)
    if not contenedores:
        print("No encontré los contenedores del sistema; uso --contenedores.")
        return 2
    print(f"Tomando {args.motor} stats en reposo durante {args.reposo} s…")
    with recursos.Muestreador(
        list(contenedores.values()), args.motor, args.intervalo
    ) as muestreador:
        time.sleep(args.reposo)
    por_contenedor = recursos.resumir(muestreador.muestras)
    sin_muestras = [s for s, nombre in contenedores.items() if not por_contenedor.get(nombre)]
    if sin_muestras:
        detalle = f" ({muestreador.errores[-1]})" if muestreador.errores else ""
        print(
            f"Sin muestras de {', '.join(sin_muestras)} en reposo{detalle}: no se escribe "
            "reposo.json."
        )
        return 2
    recursos.guardar_csv(muestreador.muestras, salida / "reposo_recursos.csv")
    reposo = {
        "fecha": fecha,
        "segundos": args.reposo,
        "contenedores": contenedores,
        "limites": limites_de(contenedores, args.motor),
        "recursos": {s: por_contenedor.get(n) for s, n in contenedores.items()},
    }
    _escribir_json(salida / "reposo.json", reposo)
    for servicio, medido in reposo["recursos"].items():
        if medido:
            print(f"  {servicio}: {medido['memoria_mib_max']:,.0f} MiB de memoria máxima")
    return 0


def mlflow_sin_ajuste(ruta: Path | None) -> dict | None:
    """Consumo de MLflow guardado por ``--reposo`` con ``compose.yaml`` solo, o ``None``."""
    if ruta is None:
        return None
    reposo = json.loads(ruta.read_text(encoding="utf-8"))
    medido = (reposo.get("recursos") or {}).get("mlflow")
    if not isinstance(medido, dict) or "memoria_mib_max" not in medido:
        raise ValueError(f"{ruta} no trae la medición de MLflow en reposo")
    return medido


def tabla_preguntas(etapa: dict) -> str:
    """Peticiones y latencia de cada endpoint en una etapa; cadena vacía si no hay endpoints."""
    renglones = [
        "| Endpoint | Peticiones | Promedio (ms) | p95 (ms) | p99 (ms) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for e in etapa.get("endpoints") or []:
        renglones.append(
            f"| {e['endpoint']} | {e['peticiones']:,} | {e['promedio_ms']:.0f} "
            f"| {e['p95_ms']:.0f} | {e['p99_ms']:.0f} |"
        )
    return "\n".join(renglones) if len(renglones) > 2 else ""


def etapas_ocupadas(salida: Path, etapas: list[int]) -> list[int]:
    """Etapas que ya tienen archivos en la carpeta (``etapa_015_stats.csv``, ``etapa_015.html``)."""
    if not salida.is_dir():
        return []
    nombres = [ruta.name for ruta in salida.iterdir()]
    return [
        usuarios
        for usuarios in etapas
        if any(n.startswith((f"etapa_{usuarios:03d}_", f"etapa_{usuarios:03d}.")) for n in nombres)
    ]


def correr_etapa(args, usuarios: int, salida: Path, endpoints: list[str]) -> tuple[float, float]:
    prefijo = salida / f"etapa_{usuarios:03d}"
    entorno = dict(os.environ, MIBICI_CARGA_ENDPOINTS=",".join(endpoints))
    comando = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(LOCUSTFILE),
        "--headless",
        "--host",
        args.host,
        "-u",
        str(usuarios),
        "-r",
        str(args.tasa),
        "-t",
        f"{args.duracion}s",
        "--csv",
        str(prefijo),
        "--csv-full-history",
        "--html",
        f"{prefijo}.html",
        "--only-summary",
        "--stop-timeout",
        "5",
        "--loglevel",
        "WARNING",
    ]
    inicio = time.time()
    # Locust devuelve 1 si hubo fallas; se registran en el CSV y no detienen las demás etapas.
    subprocess.run(comando, cwd=RAIZ, env=entorno, check=False)
    return inicio, time.time()


def primera_peticion(host: str, endpoints: list[str]) -> dict:
    """Primera inferencia de cada pregunta después de que /ready responde 200, antes de la carga.

    No incluye el arranque del contenedor ni la carga de los modelos, que ya ocurrieron: mide la
    primera llamada a cada modelo ya cargado. Además deja esas rutas calientes, para que su
    primera llamada no caiga en los percentiles de la primera etapa.
    """
    rng = random.Random(29)
    resultado: dict[str, float | str] = {}
    for clave, ruta, generar in (
        ("p1", "/api/v1/predict", cuerpo_p1),
        ("p2", "/api/v1/p2/predict", cuerpo_p2),
        ("p3", "/api/v1/p3/predict", cuerpo_p3),
    ):
        if clave not in endpoints:
            continue
        datos = json.dumps(generar(rng)).encode("utf-8")
        peticion = urllib.request.Request(
            f"{host}{ruta}", data=datos, headers={"Content-Type": "application/json"}
        )
        inicio = time.perf_counter()
        try:
            with urllib.request.urlopen(peticion, timeout=60) as respuesta:
                respuesta.read()
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            resultado[f"{clave}_error"] = f"{type(error).__name__}: {error}"[:300]
            continue
        resultado[f"{clave}_ms"] = (time.perf_counter() - inicio) * 1000
    return resultado


def texto_primera_peticion(primera: dict) -> str:
    """``'Primera inferencia …: P1 120 ms, P2 2,600 ms.'`` o cadena vacía si no hay medidas."""
    medidas = [
        f"{clave.upper()} {primera[f'{clave}_ms']:,.0f} ms"
        for clave in PREGUNTAS
        if isinstance(primera.get(f"{clave}_ms"), (int, float))
    ]
    if not medidas:
        return ""
    return (
        "Primera inferencia de cada pregunta después de que /ready responde 200 (no incluye el "
        "arranque del contenedor ni la carga de los modelos): " + ", ".join(medidas) + "."
    )


def sonda_de_lotes(host: str, estaciones: int, repeticiones: int, endpoints: list[str]) -> dict:
    """Latencia mediana de una petición con ``estaciones`` estaciones para P2 y P3.

    Si un endpoint falla, se guarda el error en ``<clave>_error`` y se sigue con el siguiente.
    """
    rng = random.Random(29)
    estaciones = max(1, min(estaciones, 1000))
    resultado: dict[str, float | str] = {"estaciones": estaciones}
    for clave, ruta, generar in (
        ("p2", "/api/v1/p2/predict", cuerpo_p2),
        ("p3", "/api/v1/p3/predict", cuerpo_p3),
    ):
        if clave not in endpoints:
            continue
        tiempos = []
        try:
            for _ in range(repeticiones):
                datos = json.dumps(generar(rng, estaciones)).encode("utf-8")
                peticion = urllib.request.Request(
                    f"{host}{ruta}", data=datos, headers={"Content-Type": "application/json"}
                )
                inicio = time.perf_counter()
                with urllib.request.urlopen(peticion, timeout=60) as respuesta:
                    respuesta.read()
                tiempos.append((time.perf_counter() - inicio) * 1000)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            resultado[f"{clave}_error"] = f"{type(error).__name__}: {error}"[:300]
            print(f"  la sonda de lotes de {clave.upper()} falló: {error}")
            continue
        if tiempos:
            resultado[f"{clave}_ms"] = statistics.median(tiempos)
    return resultado


def _sin_nan(valor):
    if isinstance(valor, float) and math.isnan(valor):
        return None
    if isinstance(valor, dict):
        return {k: _sin_nan(v) for k, v in valor.items()}
    if isinstance(valor, list):
        return [_sin_nan(v) for v in valor]
    return valor


def _escribir_json(ruta: Path, datos) -> None:
    ruta.write_text(json.dumps(_sin_nan(datos), ensure_ascii=False, indent=2), encoding="utf-8")


def _con_nan(valor, clave: str = ""):
    """Inverso de ``_sin_nan`` para las medidas de un ``resumen.json`` ya escrito.

    Los ``null`` de ``recursos`` y de sus servicios no eran medidas: significan que no hubo
    muestras de ese contenedor, y se quedan como ``None``.
    """
    if valor is None:
        return None if clave in ("recursos", "api", "mlflow") else math.nan
    if isinstance(valor, dict):
        return {k: _con_nan(v, k) for k, v in valor.items()}
    if isinstance(valor, list):
        return [_con_nan(v, clave) for v in valor]
    return valor


def _o_nd(valor, formato: str) -> str:
    """El número con ese formato, o ``n/d`` si no hay medida."""
    if not isinstance(valor, (int, float)) or math.isnan(valor):
        return "n/d"
    return format(valor, formato)


def tabla_etapas(resumenes: list[dict]) -> str:
    renglones = [
        (
            "| Usuarios | RPS sostenido | p50 (ms) | p95 (ms) | p99 (ms) | Errores | CPU API p95 "
            "(núcleos) | Memoria API máx. (MiB) |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in resumenes:
        api = (r.get("recursos") or {}).get("api") or {}
        renglones.append(
            f"| {r['usuarios']} | {r['rps_sostenido']:.1f} | {r['p50_ms']:.0f} | {r['p95_ms']:.0f}"
            f" | {r['p99_ms']:.0f} | {r['tasa_error']:.2%} | "
            f"{_o_nd(api.get('cpu_nucleos_p95'), '.2f')} | "
            f"{_o_nd(api.get('memoria_mib_max'), '.0f')} |"
        )
    return "\n".join(renglones)


def _texto_capacidad(capacidad: dict, saturacion: dict) -> str:
    """Capacidad sostenida (RPS de inferencias, usuarios y p95) y saturación observada."""
    if capacidad:
        rps = capacidad.get("rps_inferencias")
        if not isinstance(rps, (int, float)) or math.isnan(rps):
            rps = capacidad["rps_sostenido"]
        texto = (
            f"{rps:.1f} RPS de inferencias ({capacidad['rps_sostenido']:.1f} en total con "
            f"GET /live) con {capacidad['usuarios']} usuarios y p95 de "
            f"{capacidad['p95_ms']:,.0f} ms."
        )
    else:
        texto = "ninguna etapa cumplió el objetivo."
    if saturacion:
        texto += (
            f" Saturación observada: {saturacion['rps_sostenido']:.1f} RPS con "
            f"{saturacion['usuarios']} usuarios y p95 de {saturacion['p95_ms']:,.0f} ms."
        )
    return texto


def bloque_readme(resumen: dict, finops_md: str, carpeta_relativa: str) -> str:
    capacidad = resumen.get("capacidad") or {}
    saturacion = resumen.get("saturacion") or {}
    primera = texto_primera_peticion(resumen.get("primera_peticion") or {})
    limites = texto_limites(resumen.get("limites") or {})
    if limites and resumen.get("vcpu_medidos_supuesto"):
        limites += (
            f" El costo supone {resumen.get('vcpu_medidos', 1.0):g} núcleo para la API, porque "
            "no se leyó un límite de CPU."
        )
    avisos = resumen.get("avisos") or []
    if avisos:
        limites = (limites + " " if limites else "") + (
            "Avisos de la corrida: " + "; ".join(avisos) + "."
        )
    preguntas = tabla_preguntas(capacidad)
    if preguntas:
        preguntas = (
            f"Por endpoint en la etapa de capacidad ({capacidad['usuarios']} usuarios); con pocas "
            "peticiones, el p95 y el p99 de un endpoint son poco estables:\n\n" + preguntas
        )
    return "\n".join(
        [
            INICIO_BLOQUE,
            f"Corrida del {resumen['fecha']} (commit `{resumen.get('commit', 'n/d')}`), "
            f"{len(resumen['etapas'])} etapas de {resumen['duracion_s']} s; endpoints "
            f"probados: {', '.join(resumen['endpoints'])}. Capacidad sostenida con p95 de "
            f"{resumen['p95_objetivo_ms']:.0f} ms o menos: "
            + _texto_capacidad(capacidad, saturacion),
            "",
            *([limites, ""] if limites else []),
            tabla_etapas(resumen["etapas"]),
            "",
            *([preguntas, ""] if preguntas else []),
            *([primera, ""] if primera else []),
            finops_md.strip(),
            "",
            f"![Desempeño por etapa]({carpeta_relativa}/desempeno.png)",
            f"![Capacidad]({carpeta_relativa}/capacidad.png)",
            f"![Latencia por endpoint]({carpeta_relativa}/endpoints.png)",
            f"![Recursos por contenedor]({carpeta_relativa}/recursos.png)",
            "",
            (
                f"Detalle completo en `{carpeta_relativa}/` (CSV de Locust, muestras de "
                "`docker stats`, `resumen.json` y `finops.json`); los HTML de Locust se generan "
                "en la misma carpeta y no se versionan."
            ),
            FIN_BLOQUE,
        ]
    )


def reemplazar_bloque(texto: str, bloque: str) -> str:
    patron = re.compile(re.escape(INICIO_BLOQUE) + r".*?" + re.escape(FIN_BLOQUE), re.DOTALL)
    if not patron.search(texto):
        raise ValueError("El README no tiene el bloque de resultados de la prueba de carga")
    return patron.sub(lambda _: bloque, texto, count=1)


def _commit() -> str:
    resultado = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=RAIZ,
    )
    return resultado.stdout.strip() or "n/d"


SIN_CAPACIDAD = "Sin etapa que cumpla el objetivo: no se calcula el costo a capacidad."
SIN_RECURSOS = (
    "Sin muestras de recursos de la API (--sin-recursos o docker stats sin datos): no se "
    "calcula el costo."
)


def analisis_de_costo(resumen: dict, salida: Path, args: argparse.Namespace) -> str:
    """Escribe ``finops.json``, ``finops.md`` y ``resumen.md`` de una corrida; devuelve el texto.

    Todo sale del resumen de la corrida, así que sirve igual al terminar la carga que al
    recalcular una corrida ya hecha.
    """
    capacidad = resumen.get("capacidad") or {}
    medidos = capacidad.get("recursos") or {}
    if not capacidad or not isinstance(medidos.get("api"), dict):
        finops_md = SIN_RECURSOS if capacidad else SIN_CAPACIDAD
        # Un finops.json de un cálculo anterior contradiría al finops.md que se escribe ahora.
        (salida / "finops.json").unlink(missing_ok=True)
    else:
        mlflow = medidos.get("mlflow")
        resultado = finops.analizar(
            capacidad,
            medidos["api"],
            lotes=resumen.get("lotes"),
            recursos_mlflow=mlflow if isinstance(mlflow, dict) else None,
            demanda=finops.Demanda(estaciones=resumen.get("estaciones") or args.estaciones),
            vcpu_max=resumen.get("vcpu_max") or args.vcpu_max,
            vcpu_medidos=resumen.get("vcpu_medidos") or 1.0,
            saturacion=resumen.get("saturacion"),
            recursos_mlflow_sin_ajuste=resumen.get("mlflow_sin_ajuste"),
        )
        _escribir_json(salida / "finops.json", resultado)
        finops_md = finops.a_markdown(resultado)
    (salida / "finops.md").write_text(finops_md, encoding="utf-8")
    partes = [
        tabla_etapas(resumen["etapas"]),
        texto_primera_peticion(resumen.get("primera_peticion") or {}),
        finops_md,
    ]
    (salida / "resumen.md").write_text(
        "\n\n".join(parte for parte in partes if parte), encoding="utf-8"
    )
    return finops_md


def actualizar_readme(resumen: dict, finops_md: str, salida: Path) -> None:
    relativa = salida.resolve().relative_to(RAIZ).as_posix()
    README.write_text(
        reemplazar_bloque(
            README.read_text(encoding="utf-8"), bloque_readme(resumen, finops_md, relativa)
        ),
        encoding="utf-8",
    )
    print("README actualizado con los resultados.")


def recalcular(args: argparse.Namespace, carpeta: Path) -> int:
    """Vuelve a calcular los costos de una corrida ya hecha, sin repetir la carga."""
    ruta = carpeta / "resumen.json"
    if not ruta.is_file():
        print(f"No encontré {ruta}: --recalcular necesita la carpeta de una corrida.")
        return 2
    resumen = json.loads(ruta.read_text(encoding="utf-8"))
    for clave in ("etapas", "capacidad", "saturacion", "lotes"):
        if resumen.get(clave):
            resumen[clave] = _con_nan(resumen[clave])
    finops_md = analisis_de_costo(resumen, carpeta, args)
    if args.actualizar_readme:
        actualizar_readme(resumen, finops_md, carpeta)
    print(f"Listo. Costos recalculados en {carpeta}")
    print(finops_md)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = argumentos(argv)
    etapas = [int(u) for u in args.etapas.split(",") if u.strip()]
    fecha = datetime.now().strftime("%Y-%m-%d_%H%M")
    salida = args.recalcular or args.salida or RAIZ / "reports" / "carga" / fecha
    if args.actualizar_readme and not salida.resolve().is_relative_to(RAIZ):
        # Las figuras del README se enlazan con rutas relativas al repositorio.
        print(f"--actualizar-readme necesita que la salida esté dentro de {RAIZ}.")
        return 2
    if args.recalcular:
        return recalcular(args, args.recalcular)
    salida.mkdir(parents=True, exist_ok=True)

    if args.reposo:
        return medir_reposo(args, salida, fecha)

    ocupadas = etapas_ocupadas(salida, etapas)
    if ocupadas:
        # Si Locust no llegara a escribir, los CSV de otra corrida pasarían por los de ésta.
        cuales = "la etapa" if len(ocupadas) == 1 else "las etapas"
        print(
            f"{salida} ya tiene resultados de {cuales} de {', '.join(map(str, ocupadas))} "
            "usuarios. Cada corrida necesita una carpeta propia: uso otra --salida."
        )
        return 2
    try:
        sin_ajuste = mlflow_sin_ajuste(args.mlflow_sin_ajuste)
    except (OSError, ValueError) as error:
        print(f"--mlflow-sin-ajuste: {error}")
        return 2

    try:
        endpoints = endpoints_disponibles(args.host)
    except (RuntimeError, urllib.error.URLError) as error:
        print(f"La API no está lista: {error}")
        return 2
    if not endpoints:
        print("Ningún modelo está cargado (/ready). Siembra el Registry antes de la prueba.")
        return 2
    print(f"Endpoints con modelo: {', '.join(endpoints)}")

    contenedores = {} if args.sin_recursos else contenedores_del_sistema(args)
    if not args.sin_recursos and "api" not in contenedores:
        print("No encontré el contenedor de la API; uso --contenedores o --sin-recursos.")
        return 2

    limites = limites_de(contenedores, args.motor) if contenedores else {}
    vcpu_medidos = args.vcpu_medidos or (limites.get("api") or {}).get("cpus")
    if texto_limites(limites):
        print(texto_limites(limites))
    supuesto = not vcpu_medidos
    if supuesto:
        if contenedores and (limites.get("api") or {}).get("leido") is False:
            print(
                "No se pudieron leer los límites de la API y no se indicó --vcpu-medidos. Se "
                "toma 1 núcleo, el que aprovecha un proceso de uvicorn, y queda anotado como "
                "supuesto."
            )
        elif contenedores:
            print(
                "La API no tiene límite de CPU y no se indicó --vcpu-medidos: la capacidad "
                "depende de esta computadora. Para costear, levanta el sistema con "
                "carga/compose.medicion.yaml. Se toma 1 núcleo, el que aprovecha un proceso de "
                "uvicorn."
            )
        vcpu_medidos = 1.0

    primera = primera_peticion(args.host, endpoints)
    if texto_primera_peticion(primera):
        print(texto_primera_peticion(primera))

    resumenes, historiales, muestras, cortes, avisos = [], [], [], [], []
    for usuarios in etapas:
        print(f"Etapa de {usuarios} usuarios durante {args.duracion} s…")
        if contenedores:
            with recursos.Muestreador(
                list(contenedores.values()), args.motor, args.intervalo
            ) as muestreador:
                inicio, fin = correr_etapa(args, usuarios, salida, endpoints)
            propias = muestreador.muestras
            error_stats = muestreador.errores[-1] if muestreador.errores else ""
            if error_stats:
                print(f"  avisos de {args.motor} stats: {error_stats}")
        else:
            inicio, fin = correr_etapa(args, usuarios, salida, endpoints)
            propias, error_stats = [], ""
        prefijo = salida / f"etapa_{usuarios:03d}"
        ruta_stats = Path(f"{prefijo}_stats.csv")
        ruta_historial = Path(f"{prefijo}_stats_history.csv")
        if not (ruta_stats.exists() and ruta_historial.exists()):
            print(f"  la etapa de {usuarios} usuarios no dejó los CSV de Locust; la salto.")
            continue
        stats = carga.leer_stats(ruta_stats)
        historial = carga.leer_historial(ruta_historial)
        descartar = max(10.0, usuarios / args.tasa + 5)
        resumen = carga.resumen_etapa(stats, historial, usuarios, args.duracion, descartar)
        # CPU y RPS con la misma ventana: sin el arranque de Locust, la rampa ni el cierre.
        marcas = historial["Timestamp"].tolist() if "Timestamp" in historial else []
        estables = recursos.en_ventana(propias, marcas, descartar, inicio, fin)
        por_contenedor = recursos.resumir(estables)
        resumen["recursos"] = {
            servicio: por_contenedor.get(nombre) for servicio, nombre in contenedores.items()
        }
        faltan = [servicio for servicio, medido in resumen["recursos"].items() if not medido]
        if faltan:
            # Sin esas muestras el servicio no entra al costo: que no pase inadvertido.
            aviso = f"etapa de {usuarios} usuarios sin muestras de {', '.join(faltan)}"
            aviso += f" ({error_stats})" if error_stats else ""
            avisos.append(aviso)
            print(f"  aviso: {aviso}")
        recursos.guardar_csv(propias, Path(f"{prefijo}_recursos.csv"))
        resumenes.append(resumen)
        historiales.append((usuarios, historial))
        muestras.extend(propias)
        cortes.append((inicio, usuarios))
        print(
            f"  {resumen['rps_sostenido']:.1f} RPS, p95 {resumen['p95_ms']:.0f} ms, "
            f"errores {resumen['tasa_error']:.2%}"
        )

    if not resumenes:
        print("Ninguna etapa dejó resultados de Locust; revisa que Locust esté instalado y corra.")
        return 2

    lotes = sonda_de_lotes(args.host, args.lote_sonda, args.repeticiones_sonda, endpoints)
    capacidad = carga.capacidad_sostenida(resumenes, args.p95_objetivo, args.error_maximo)
    saturacion = carga.saturacion(resumenes)
    resumen_total = {
        "fecha": fecha,
        "commit": _commit(),
        "host": args.host,
        "endpoints": endpoints,
        "duracion_s": args.duracion,
        "tasa_usuarios_s": args.tasa,
        "p95_objetivo_ms": args.p95_objetivo,
        "contenedores": contenedores,
        "limites": limites,
        "vcpu_medidos": vcpu_medidos,
        "vcpu_medidos_supuesto": supuesto,
        "avisos": avisos,
        "vcpu_max": args.vcpu_max,
        "estaciones": args.estaciones,
        "mlflow_sin_ajuste": sin_ajuste,
        "etapas": resumenes,
        "capacidad": capacidad,
        "saturacion": saturacion,
        "lotes": lotes,
        "primera_peticion": primera,
    }
    _escribir_json(salida / "resumen.json", resumen_total)

    graficas.grafica_desempeno(historiales, salida / "desempeno.png")
    graficas.grafica_capacidad(resumenes, salida / "capacidad.png", args.p95_objetivo)
    graficas.grafica_endpoints(capacidad or resumenes[0], salida / "endpoints.png")
    if muestras:
        nombres = {nombre: servicio for servicio, nombre in contenedores.items()}
        graficas.grafica_recursos(muestras, salida / "recursos.png", cortes, nombres)

    finops_md = analisis_de_costo(resumen_total, salida, args)
    if args.actualizar_readme:
        actualizar_readme(resumen_total, finops_md, salida)

    print(f"Listo. Resultados en {salida}")
    print(finops_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
