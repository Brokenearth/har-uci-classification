"""Arranca Streamlit y guarda log si el proceso termina inesperadamente."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "streamlit.log"
LOG.parent.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable,
    "-u",
    "-m",
    "streamlit",
    "run",
    str(ROOT / "app" / "streamlit_app.py"),
    "--server.headless",
    "true",
    "--server.port",
    "8501",
    "--logger.level",
    "info",
]

print("Iniciando Streamlit…")
print("  URL:  http://localhost:8501")
print(f"  Log:  {LOG}")
print("  Detener con Ctrl+C\n")

with open(LOG, "w", encoding="utf-8") as log_file:
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    print(f"PID del servidor: {proc.pid}")
    code = proc.wait()

print(f"\nStreamlit terminó (código {code}). Revisa el log: {LOG}")
if code != 0:
    tail = LOG.read_text(encoding="utf-8", errors="replace")[-2000:]
    print("\n--- Últimas líneas del log ---")
    print(tail)
input("\nPulsa Enter para cerrar…")
