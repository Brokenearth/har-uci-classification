#!/usr/bin/env python3
"""Copia artefactos de results/ a evidencias/ para entrega."""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCIAS = ROOT / "evidencias"

FILES = {
    ROOT / "results" / "metrics" / "metrics_final.json": EVIDENCIAS / "metrics_final.json",
    ROOT / "results" / "metrics" / "comparacion_modelos.csv": EVIDENCIAS / "comparacion_modelos.csv",
    ROOT / "results" / "figures" / "confusion_matrix.png": EVIDENCIAS / "confusion_matrix.png",
    ROOT / "results" / "figures" / "loss_curves.png": EVIDENCIAS / "loss_curves.png",
    ROOT / "results" / "models" / "final_model.pt": EVIDENCIAS / "final_model.pt",
}


def main() -> None:
    EVIDENCIAS.mkdir(parents=True, exist_ok=True)
    for src, dst in FILES.items():
        if not src.exists():
            print(f"Falta: {src}")
            sys.exit(1)
        shutil.copy2(src, dst)
        print(f"  {dst.name}")
    print(f"\nEvidencias actualizadas en {EVIDENCIAS}")


if __name__ == "__main__":
    main()
