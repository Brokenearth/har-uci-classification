#!/usr/bin/env python3
"""Evalúa el modelo final: test (hold-out) y val (referencia; incluido en entrenamiento final)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from har_classifier.config import BATCH_SIZE, FIGURES_DIR, METRICS_DIR, MODELS_DIR
from har_classifier.evaluation.metrics import evaluate_model, load_model_from_checkpoint
from har_classifier.evaluation.visualization import plot_confusion_matrix, plot_loss_curves
from har_classifier.preprocessing import load_processed
from har_classifier.torch_dataset import HARDataset
from har_classifier.utils.device import get_device
from har_classifier.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (MODELS_DIR / "final_model.pt")
    data = load_processed()

    loaders = {
        "test": DataLoader(
            HARDataset(data["X_test"], data["y_test"]),
            batch_size=args.batch_size,
            shuffle=False,
        ),
        "val_referencia": DataLoader(
            HARDataset(data["X_val"], data["y_val"]),
            batch_size=args.batch_size,
            shuffle=False,
        ),
    }

    model = load_model_from_checkpoint(ckpt_path, device)
    results = evaluate_model(model, loaders, device)

    history_path = METRICS_DIR / "training_history.json"
    if history_path.exists():
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)
        plot_loss_curves(history, FIGURES_DIR / "loss_curves.png")
        print(f"Curvas de pérdida: {FIGURES_DIR / 'loss_curves.png'}")
    else:
        print("Sin training_history.json; ejecute 04_final_training.py para loss_curves.png")

    metrics_path = METRICS_DIR / "metrics_final.json"
    with open(metrics_path, encoding="utf-8") as f:
        saved = json.load(f)
    saved["_nota_val"] = (
        "val_referencia: sujetos usados en entrenamiento final (train+val). "
        "Métrica principal: test."
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2)

    for split, metrics in results.items():
        print(f"\n=== {split.upper()} ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 macro: {metrics['f1_macro']:.4f}")
        print(metrics["classification_report"])

    cm = np.array(results["test"]["confusion_matrix"])
    plot_confusion_matrix(cm, FIGURES_DIR / "confusion_matrix.png")
    print(f"\nMatriz de confusión (test): {FIGURES_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
