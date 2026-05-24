#!/usr/bin/env python3
"""Copia artefactos de results/ a evidencias/ para entrega."""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCIAS = ROOT / "evidencias"
FIGURES = EVIDENCIAS / "figures"

FILES = {
    ROOT / "results" / "metrics" / "metrics_final.json": EVIDENCIAS / "metrics_final.json",
    ROOT / "results" / "metrics" / "comparacion_modelos.csv": EVIDENCIAS / "comparacion_modelos.csv",
    ROOT / "results" / "figures" / "confusion_matrix.png": EVIDENCIAS / "confusion_matrix.png",
    ROOT / "results" / "figures" / "loss_curves.png": EVIDENCIAS / "loss_curves.png",
    ROOT / "results" / "models" / "final_model.pt": EVIDENCIAS / "final_model.pt",
}

FIGURE_ASSETS = [
  "confusion_matrix.png",
  "loss_curves.png",
  "aciertos_1.png",
  "errores_1.png",
]


def main() -> None:
    EVIDENCIAS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for src, dst in FILES.items():
        if not src.exists():
            print(f"Falta: {src}")
            sys.exit(1)
        shutil.copy2(src, dst)
        print(f"  {dst.name}")

    for name in FIGURE_ASSETS:
        src = ROOT / "results" / "figures" / name
        if src.exists():
            shutil.copy2(src, FIGURES / name)
            print(f"  figures/{name}")

    print(f"\nEvidencias actualizadas en {EVIDENCIAS}")


if __name__ == "__main__":
    main()
