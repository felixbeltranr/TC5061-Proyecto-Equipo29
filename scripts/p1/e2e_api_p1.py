"""Recorrido de extremo a extremo de la API de P1 en Compose, con comprobaciones que fallan.

Uso, desde cualquier carpeta (el script ubica la raíz del repositorio):

    python scripts/p1/e2e_api_p1.py [--build] [--salida reporte.json]
    python scripts/p1/e2e_api_p1.py --sin-levantar   # revisa un Compose ya iniciado

Levanta ``mlflow`` y ``api`` con ``docker compose up -d``, espera ``/ready`` y comprueba contra
``campeon_api_p1.json``:

- que el puerto publicado de ``api`` sea el de ``--url`` (que responde este proyecto y no otro);
- ``/live`` y ``/ready``;
- ``/api/v1/model``: nombre, versión, alias, run y URI inmutable del campeón;
- ``POST /api/v1/predict`` con la petición real: HTTP 200, la misma versión y la predicción de
  referencia (tolerancia relativa ``--tolerancia``);
- que las entradas fuera de dominio (lag negativo, festivo=7, NaN, campo extra, falta un campo)
  respondan 422;
- dentro del contenedor ``api``, que las versiones instaladas sean las del requirements.txt del
  modelo servido y que Python tenga la misma versión mayor.menor.

Si la API se cierra al arrancar (por ejemplo, con el Registry vacío) falla en cuanto lo detecta y
muestra el final de sus logs. Termina con código 0 sólo si todo pasa. No detiene ni borra
contenedores o volúmenes (para detenerlos: ``docker compose stop``). Sólo usa la biblioteca
estándar.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
from urllib.parse import urlsplit
import urllib.request

RAIZ = Path(__file__).resolve().parents[2]
CAMPEON = Path(__file__).resolve().with_name("campeon_api_p1.json")

# Corre dentro del contenedor api: compara lo instalado con el requirements.txt del modelo.
VERSIONES_EN_CONTENEDOR = r"""
import importlib.metadata as md, json, os, platform, sys, tempfile
import mlflow
from mlflow.artifacts import download_artifacts

uri = sys.argv[1]
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_registry_uri(os.environ["MLFLOW_TRACKING_URI"])
with tempfile.TemporaryDirectory() as tmp:
    carpeta = download_artifacts(artifact_uri=uri, dst_path=tmp)
    with open(os.path.join(carpeta, "requirements.txt"), encoding="utf-8") as f:
        requisitos = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    python_modelo = None
    with open(os.path.join(carpeta, "python_env.yaml"), encoding="utf-8") as f:
        for l in f:
            if l.startswith("python:"):
                python_modelo = l.split(":", 1)[1].strip()
instaladas = {}
for linea in requisitos:
    nombre = linea.split("==")[0].split("[")[0].strip()
    try:
        instaladas[nombre] = md.version(nombre)
    except md.PackageNotFoundError:
        instaladas[nombre] = None
otras = {}
for nombre in ("fastapi", "starlette", "pydantic", "uvicorn"):
    try:
        otras[nombre] = md.version(nombre)
    except md.PackageNotFoundError:
        otras[nombre] = None
print(json.dumps({
    "python": platform.python_version(),
    "python_modelo": python_modelo,
    "requisitos_modelo": requisitos,
    "instaladas": instaladas,
    "otras": otras,
}))
"""


@dataclass
class Comprobacion:
    nombre: str
    ok: bool
    detalle: str


def cargar_campeon(ruta: Path = CAMPEON) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


def compose(*args: str, mostrar: bool = False) -> subprocess.CompletedProcess:
    """``docker compose`` desde la raíz del repositorio."""
    return subprocess.run(
        ["docker", "compose", *args],
        check=False,
        cwd=RAIZ,
        stdout=None if mostrar else subprocess.PIPE,
        stderr=None if mostrar else subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def comando(*args: str) -> str | None:
    try:
        salida = subprocess.run(
            list(args), cwd=RAIZ, capture_output=True, text=True, encoding="utf-8", check=False
        )
    except FileNotFoundError:
        return None
    return salida.stdout.strip() if salida.returncode == 0 else None


def config_compose() -> dict:
    return json_o_none(compose("config", "--format", "json").stdout or "") or {}


def parsear_ps(texto: str) -> list[dict]:
    """``docker compose ps --format json`` da un arreglo o una línea JSON por servicio."""
    texto = texto.strip()
    if not texto:
        return []
    if texto.startswith("["):
        return json.loads(texto)
    return [json.loads(linea) for linea in texto.splitlines() if linea.strip()]


def estado_servicio(servicio: str) -> dict | None:
    salida = compose("ps", "-a", "--format", "json", servicio)
    if salida.returncode != 0:
        return None
    filas = [f for f in parsear_ps(salida.stdout) if f.get("Service") == servicio]
    return filas[0] if filas else None


def http(metodo: str, url: str, cuerpo: str | None = None) -> tuple[int, str]:
    datos = None if cuerpo is None else cuerpo.encode("utf-8")
    peticion = urllib.request.Request(
        url, data=datos, method=metodo, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(peticion, timeout=30) as respuesta:
            return respuesta.status, respuesta.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def json_o_none(texto: str):
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        return None


def puerto_coincide(publicado: str, url: str) -> bool:
    """``docker compose port api 8000`` (``127.0.0.1:8000``) contra el host y puerto de la URL."""
    partes = urlsplit(url)
    esperado_puerto = partes.port or (443 if partes.scheme == "https" else 80)
    host, _, puerto = publicado.strip().rpartition(":")
    hosts_locales = {"127.0.0.1", "localhost", "0.0.0.0", "::", "[::]"}
    mismo_host = host == partes.hostname or (
        host in hosts_locales and partes.hostname in hosts_locales
    )
    return mismo_host and puerto == str(esperado_puerto)


def predicciones_iguales(obtenida: float, referencia: float, tolerancia: float) -> bool:
    return math.isfinite(obtenida) and math.isclose(
        obtenida, referencia, rel_tol=tolerancia, abs_tol=0.0
    )


def cuerpos_fuera_de_dominio(features: dict) -> dict[str, str]:
    """Peticiones que la API debe rechazar con 422; NaN va como token literal de JSON."""
    sin_lag_28 = {k: v for k, v in features.items() if k != "lag_28"}
    return {
        "lag_2 = -1": json.dumps({"features": {**features, "lag_2": -1.0}}),
        "festivo = 7": json.dumps({"features": {**features, "festivo": 7}}),
        "lag_2 = NaN": json.dumps({"features": {**features, "lag_2": float("nan")}}),
        "campo extra en features": json.dumps({"features": {**features, "estacion": 51}}),
        "campo extra fuera de features": json.dumps({"features": features, "modelo": "p2"}),
        "falta lag_28": json.dumps({"features": sin_lag_28}),
    }


def comparar_versiones(datos: dict) -> list[Comprobacion]:
    """Requisitos fijados del modelo contra lo instalado en el contenedor."""
    resultado = []
    for linea in datos["requisitos_modelo"]:
        nombre, separador, version = linea.partition("==")
        nombre = nombre.split("[")[0].strip()
        instalada = datos["instaladas"].get(nombre)
        if not separador:
            resultado.append(
                Comprobacion(f"dependencia {nombre}", False, f"el modelo no fija versión: {linea}")
            )
            continue
        resultado.append(
            Comprobacion(
                f"dependencia {nombre}",
                instalada == version.strip(),
                f"modelo {version.strip()}, contenedor {instalada}",
            )
        )
    python, modelo = datos["python"], datos.get("python_modelo") or ""
    resultado.append(
        Comprobacion(
            "python mayor.menor",
            python.split(".")[:2] == modelo.split(".")[:2],
            f"modelo {modelo}, contenedor {python}",
        )
    )
    return resultado


def esperar_ready(url: str, limite: float) -> tuple[bool, str]:
    fin = time.monotonic() + limite
    ultimo = "sin respuesta"
    while time.monotonic() < fin:
        estado = estado_servicio("api")
        if estado and estado.get("State") == "exited":
            return False, f"el contenedor api se cerró con código {estado.get('ExitCode')}"
        try:
            codigo, texto = http("GET", f"{url}/ready")
            if codigo == 200 and json_o_none(texto) == {"status": "ready"}:
                return True, texto
            ultimo = f"HTTP {codigo} {texto[:200]}"
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            ultimo = f"sin respuesta: {exc}"
        time.sleep(2)
    return False, f"/ready no respondió ready en {limite:.0f} s ({ultimo})"


def recorrido(args: argparse.Namespace) -> tuple[list[Comprobacion], dict]:
    campeon = cargar_campeon()
    modelo, real = campeon["modelo"], campeon["peticion_real"]
    url = args.url.rstrip("/")
    checks: list[Comprobacion] = []
    datos: dict = {}

    if not args.sin_levantar:
        up = ["up", "-d", "--build" if args.build else "--no-build", "mlflow", "api"]
        imagen = config_compose().get("services", {}).get("api", {}).get("image")
        if not args.build and not (imagen and comando("docker", "image", "inspect", imagen)):
            up.remove("--no-build")  # sin imagen local, Compose la construye
        salida = compose(*up, mostrar=True)
        checks.append(
            Comprobacion(
                "docker compose up -d", salida.returncode == 0, f"código {salida.returncode}"
            )
        )
        if salida.returncode != 0:
            return checks, datos

    listo, detalle = esperar_ready(url, args.espera)
    checks.append(Comprobacion("api lista (/ready)", listo, detalle))
    if not listo:
        logs = compose("logs", "--no-color", "--tail", "40", "api")
        datos["logs_api"] = logs.stdout.splitlines()
        datos["pista"] = (
            "Si el Registry está vacío o no tiene el campeón, siembra volúmenes nuevos con "
            "scripts/p1/sembrar_registry_p1.py antes de iniciar MLflow."
        )
        return checks, datos

    publicado = compose("port", "api", "8000").stdout.strip()
    checks.append(
        Comprobacion("puerto de api", puerto_coincide(publicado, url), f"{publicado} vs {url}")
    )
    contenedor = compose("ps", "-q", "api").stdout.strip()
    datos["imagen_api"] = comando("docker", "inspect", "-f", "{{.Image}}", contenedor)

    codigo, texto = http("GET", f"{url}/live")
    checks.append(
        Comprobacion("/live", codigo == 200 and json_o_none(texto) == {"status": "alive"}, texto)
    )

    codigo, texto = http("GET", f"{url}/api/v1/model")
    info = json_o_none(texto) or {}
    datos["modelo_cargado"] = info
    esperado = {
        "name": modelo["nombre"],
        "version": modelo["version"],
        "alias": modelo["alias"],
        "run_id": modelo["run_id"],
        "model_uri": f"models:/{modelo['nombre']}/{modelo['version']}",
    }
    checks.append(Comprobacion("/api/v1/model", codigo == 200 and info == esperado, texto))

    cuerpo = json.dumps(real["cuerpo"])
    codigo, texto = http("POST", f"{url}/api/v1/predict", cuerpo)
    respuesta = json_o_none(texto) or {}
    datos["peticion"] = real["cuerpo"]
    datos["respuesta"] = respuesta
    checks.append(Comprobacion("predict HTTP 200", codigo == 200, f"HTTP {codigo}"))
    checks.append(
        Comprobacion(
            "predict reporta la versión cargada",
            respuesta.get("model_name") == modelo["nombre"]
            and respuesta.get("model_version") == modelo["version"],
            f"{respuesta.get('model_name')} v{respuesta.get('model_version')}",
        )
    )
    prediccion = respuesta.get("prediction")
    referencia = real["prediccion_referencia"]
    iguales = isinstance(prediccion, (int, float)) and predicciones_iguales(
        float(prediccion), referencia, args.tolerancia
    )
    diferencia = (
        abs(float(prediccion) - referencia) if isinstance(prediccion, (int, float)) else None
    )
    checks.append(
        Comprobacion(
            "predicción = referencia",
            iguales,
            f"{prediccion} vs {referencia} (diferencia {diferencia}, tolerancia {args.tolerancia})",
        )
    )

    for nombre, cuerpo_malo in cuerpos_fuera_de_dominio(real["cuerpo"]["features"]).items():
        codigo, texto = http("POST", f"{url}/api/v1/predict", cuerpo_malo)
        checks.append(Comprobacion(f"422 con {nombre}", codigo == 422, f"HTTP {codigo}"))

    salida = compose(
        "exec", "-T", "api", "python", "-c", VERSIONES_EN_CONTENEDOR, esperado["model_uri"]
    )
    lineas = (salida.stdout or "").strip().splitlines()
    versiones = json_o_none(lineas[-1]) if lineas else None
    if salida.returncode != 0 or not versiones:
        checks.append(
            Comprobacion("versiones del contenedor", False, (salida.stderr or "")[-500:])
        )
    else:
        datos["versiones_contenedor"] = versiones
        checks.extend(comparar_versiones(versiones))
    return checks, datos


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", action="store_true", help="reconstruye la imagen al levantar")
    parser.add_argument("--sin-levantar", action="store_true", help="no llama docker compose up")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="URL base de la API")
    parser.add_argument("--espera", type=float, default=180, help="segundos para /ready")
    parser.add_argument("--tolerancia", type=float, default=1e-9, help="tolerancia relativa")
    parser.add_argument("--salida", type=Path, help="escribe el reporte JSON en esta ruta")
    args = parser.parse_args(argv)

    inicio = datetime.now().astimezone().isoformat(timespec="seconds")
    try:
        checks, datos = recorrido(args)
    except FileNotFoundError as exc:
        print(f"ERROR: no se encontró {exc.filename} (¿Docker instalado?)", file=sys.stderr)
        return 1
    estado_git = comando("git", "status", "--porcelain")
    reporte = {
        "inicio": inicio,
        "fin": datetime.now().astimezone().isoformat(timespec="seconds"),
        "commit": comando("git", "rev-parse", "HEAD"),
        "cambios_sin_commit": None if estado_git is None else estado_git.splitlines(),
        "proyecto_compose": config_compose().get("name"),
        "docker": comando("docker", "version", "--format", "{{.Server.Version}}"),
        "compose": comando("docker", "compose", "version", "--short"),
        "url": args.url,
        "comprobaciones": [asdict(c) for c in checks],
        **datos,
    }
    for c in checks:
        print(f"{'OK   ' if c.ok else 'FALLA'} {c.nombre}: {c.detalle}")
    for linea in datos.get("logs_api", []):
        print(f"  | {linea}")
    if "pista" in datos:
        print(datos["pista"])
    if args.salida:
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        args.salida.write_text(
            json.dumps(reporte, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Reporte: {args.salida.resolve()}")
    fallas = [c for c in checks if not c.ok]
    print(f"{len(checks) - len(fallas)} de {len(checks)} comprobaciones pasaron")
    return 0 if checks and not fallas else 1


if __name__ == "__main__":
    sys.exit(main())
