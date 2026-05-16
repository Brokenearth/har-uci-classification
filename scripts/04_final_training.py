#!/usr/bin/env python3
"""Entrenamiento final del mejor modelo según CV."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from har_classifier.config import BATCH_SIZE, EPOCHS, LEARNING_RATE
from har_classifier.training.final_train import run_final_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, choices=["LSTM", "Conv1D_LSTM"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    if args.no_wandb:
        os.environ["WANDB_MODE"] = "offline"

    run_final_training(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
