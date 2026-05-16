"""Validación cruzada 2-fold por sujeto: LSTM vs Conv1D+LSTM."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from har_classifier.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    METRICS_DIR,
    MODELS_DIR,
    N_CV_FOLDS,
    RANDOM_SEED,
    WANDB_PROJECT,
)
from har_classifier.models import Conv1DLSTMClassifier, LSTMClassifier
from har_classifier.preprocessing import apply_normalization, fit_channel_stats, load_raw_for_cv
from har_classifier.splits import get_cv_folds, print_fold_info
from har_classifier.torch_dataset import HARDataset
from har_classifier.training.trainer import Trainer
from har_classifier.utils.device import get_device
from har_classifier.utils.seed import set_seed


MODEL_BUILDERS = {
    "LSTM": lambda: LSTMClassifier(),
    "Conv1D_LSTM": lambda: Conv1DLSTMClassifier(),
}


def run_cross_validation(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    use_wandb: bool = True,
) -> pd.DataFrame:
    set_seed(RANDOM_SEED)
    device = get_device()
    data = load_raw_for_cv()
    folds = get_cv_folds(data["y_train"], data["subjects_train"], n_splits=N_CV_FOLDS)
    print_fold_info(data["subjects_train"], folds)

    rows: list[dict] = []
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_cv: dict = {"accuracy": -1.0}

    for model_name, builder in MODEL_BUILDERS.items():
        for fold_i, (train_idx, val_idx) in enumerate(folds):
            X_tr = data["X_train"][train_idx]
            y_tr = data["y_train"][train_idx]
            X_val = data["X_train"][val_idx]
            y_val = data["y_train"][val_idx]

            mean, std = fit_channel_stats(X_tr)
            X_tr = apply_normalization(X_tr, mean, std)
            X_val = apply_normalization(X_val, mean, std)

            train_ds = HARDataset(X_tr, y_tr)
            val_ds = HARDataset(X_val, y_val)
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available()
            )
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available()
            )

            model = builder()
            trainer = Trainer(
                device,
                use_wandb=use_wandb,
                wandb_config={
                    "project": WANDB_PROJECT,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                },
            )
            history = trainer.fit(
                model,
                train_loader,
                val_loader,
                epochs=epochs,
                lr=lr,
                model_name=model_name,
                fold=fold_i,
            )

            best_acc = max(history["val_accuracy"])
            best_loss = min(history["val_loss"])
            rows.append(
                {
                    "modelo": model_name,
                    "fold": fold_i,
                    "val_accuracy": best_acc,
                    "val_loss": best_loss,
                }
            )
            if best_acc > best_cv["accuracy"]:
                best_cv = {
                    "accuracy": best_acc,
                    "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    "model_name": model_name,
                    "fold": fold_i,
                    "mean": mean,
                    "std": std,
                }

    if best_cv.get("state_dict") is not None:
        torch.save(
            {
                "state_dict": best_cv["state_dict"],
                "model_name": best_cv["model_name"],
                "fold": best_cv["fold"],
                "cv_val_accuracy": best_cv["accuracy"],
                "mean": best_cv["mean"],
                "std": best_cv["std"],
            },
            MODELS_DIR / "best_model.pt",
        )
        print(f"Mejor modelo CV guardado en {MODELS_DIR / 'best_model.pt'}")

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("modelo")
        .agg(val_accuracy_mean=("val_accuracy", "mean"), val_accuracy_std=("val_accuracy", "std"))
        .reset_index()
    )
    for _, row in summary.iterrows():
        rows.append(
            {
                "modelo": row["modelo"],
                "fold": "mean_std",
                "val_accuracy": f"{row['val_accuracy_mean']:.4f} ± {row['val_accuracy_std']:.4f}",
                "val_loss": "",
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = METRICS_DIR / "comparacion_modelos.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nResultados guardados en {out_path}")
    print(summary)
    return out_df


def get_best_model_name(csv_path: Path | None = None) -> str:
    path = csv_path or METRICS_DIR / "comparacion_modelos.csv"
    if not path.exists():
        return "Conv1D_LSTM"
    df = pd.read_csv(path)
    fold_df = df[df["fold"] != "mean_std"].copy()
    if fold_df.empty:
        return "Conv1D_LSTM"
    fold_df["val_accuracy"] = pd.to_numeric(fold_df["val_accuracy"], errors="coerce")
    means = fold_df.groupby("modelo")["val_accuracy"].mean()
    return str(means.idxmax())
