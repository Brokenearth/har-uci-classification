#!/usr/bin/env python3
"""Genera loss_curves.png desde training_history.json (sin reentrenar)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from har_classifier.config import FIGURES_DIR, METRICS_DIR
from har_classifier.evaluation.visualization import plot_loss_curves


def main():
    path = METRICS_DIR / "training_history.json"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecute scripts/04_final_training.py primero.")
    with open(path, encoding="utf-8") as f:
        history = json.load(f)
    out = FIGURES_DIR / "loss_curves.png"
    plot_loss_curves(history, out)
    print(f"Guardado: {out}")


if __name__ == "__main__":
    main()
