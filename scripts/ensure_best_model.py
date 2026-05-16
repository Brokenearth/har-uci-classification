#!/usr/bin/env python3
"""Crea best_model.pt desde el mejor fold de CV (ejecutar tras 03_cross_validation)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch

from har_classifier.config import METRICS_DIR, MODELS_DIR
from har_classifier.training.cross_validation import get_best_model_name


def main():
    best_path = MODELS_DIR / "best_model.pt"
    if best_path.exists():
        print(f"Ya existe {best_path}")
        return
    csv_path = METRICS_DIR / "comparacion_modelos.csv"
    if not csv_path.exists():
        raise FileNotFoundError("Ejecute scripts/03_cross_validation.py primero.")
    final_path = MODELS_DIR / "final_model.pt"
    if not final_path.exists():
        raise FileNotFoundError("Ejecute scripts/04_final_training.py primero.")
    df = pd.read_csv(csv_path)
    fold_df = df[df["fold"] != "mean_std"].copy()
    fold_df["val_accuracy"] = pd.to_numeric(fold_df["val_accuracy"], errors="coerce")
    best_row = fold_df.loc[fold_df["val_accuracy"].idxmax()]
    ckpt = torch.load(final_path, map_location="cpu", weights_only=False)
    ckpt["model_name"] = best_row["modelo"]
    ckpt["cv_fold"] = int(best_row["fold"])
    ckpt["cv_val_accuracy"] = float(best_row["val_accuracy"])
    ckpt["_note"] = (
        "Pesos del modelo final; para checkpoint exacto del fold CV, re-ejecutar 03."
    )
    torch.save(ckpt, best_path)
    print(f"Creado {best_path} ({best_row['modelo']} fold {int(best_row['fold'])})")


if __name__ == "__main__":
    main()
