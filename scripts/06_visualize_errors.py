#!/usr/bin/env python3
"""Visualiza 5 aciertos y 5 errores + reporte Markdown."""

import argparse
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from har_classifier.config import BATCH_SIZE, MODELS_DIR
from har_classifier.evaluation.metrics import load_model_from_checkpoint, predict
from har_classifier.evaluation.visualization import (
    visualize_correct_and_errors,
    write_error_analysis_report,
)
from har_classifier.preprocessing import load_processed
from har_classifier.torch_dataset import HARDataset
from har_classifier.utils.device import get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    device = get_device()
    data = load_processed()
    loader = DataLoader(
        HARDataset(data["X_test"], data["y_test"]),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    ckpt_path = args.checkpoint or (MODELS_DIR / "final_model.pt")
    model = load_model_from_checkpoint(ckpt_path, device)
    y_true, y_pred = predict(model, loader, device)

    visualize_correct_and_errors(data["X_test"], y_true, y_pred, n_each=args.n)
    write_error_analysis_report(y_true, y_pred)
    print("Figuras y reporte generados en results/figures y results/reports.")


if __name__ == "__main__":
    main()
