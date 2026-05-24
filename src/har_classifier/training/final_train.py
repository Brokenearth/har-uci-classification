"""Entrenamiento final del mejor modelo sobre train; val para early stopping (literal c)."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from har_classifier.config import (
    BATCH_SIZE,
    EPOCHS,
    FIGURES_DIR,
    LEARNING_RATE,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    WANDB_PROJECT,
)
from har_classifier.evaluation.visualization import plot_loss_curves
from har_classifier.models import Conv1DLSTMClassifier, LSTMClassifier
from har_classifier.preprocessing import load_processed
from har_classifier.torch_dataset import HARDataset
from har_classifier.training.cross_validation import get_best_model_name
from har_classifier.training.trainer import Trainer
from har_classifier.utils.device import get_device
from har_classifier.utils.seed import set_seed


def build_model(name: str) -> torch.nn.Module:
    if name == "LSTM":
        return LSTMClassifier()
    return Conv1DLSTMClassifier()


def run_final_training(
    model_name: str | None = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    use_wandb: bool = True,
) -> Path:
    """
    Entrena solo con X_train / y_train (Proyecto 13, literal c).
    X_val se usa durante el entrenamiento para early stopping; las métricas
    oficiales del modelo final se reportan en val con el script 05.
    """
    set_seed(RANDOM_SEED)
    device = get_device()
    if model_name is None:
        model_name = get_best_model_name()

    data = load_processed()
    train_ds = HARDataset(data["X_train"], data["y_train"])
    val_ds = HARDataset(data["X_val"], data["y_val"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_model(model_name)
    trainer = Trainer(
        device,
        use_wandb=use_wandb,
        wandb_config={
            "project": WANDB_PROJECT,
            "epochs": epochs,
            "batch_size": batch_size,
            "model": model_name,
            "phase": "final_train",
        },
    )
    history = trainer.fit(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr,
        model_name=model_name,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    out_path = MODELS_DIR / "final_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": model_name,
            "mean": data.get("mean"),
            "std": data.get("std"),
        },
        out_path,
    )

    history_path = METRICS_DIR / "training_history.json"
    serializable = {k: v for k, v in history.items() if isinstance(v, list)}
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    plot_loss_curves(serializable, FIGURES_DIR / "loss_curves.png")

    stats = {
        "model_name": model_name,
        "epochs": epochs,
        "best_val_accuracy": max(history["val_accuracy"]) if history["val_accuracy"] else None,
        "n_train_samples": int(len(data["X_train"])),
        "n_val_samples": int(len(data["X_val"])),
        "note": "Entrenado solo con train; evaluar métricas finales en val (script 05, literal c).",
    }
    with open(METRICS_DIR / "final_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Modelo final guardado en {out_path} ({model_name})")
    print(f"Curvas de pérdida: {FIGURES_DIR / 'loss_curves.png'}")
    return out_path
