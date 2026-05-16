"""Métricas de clasificación."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from har_classifier.config import ACTIVITY_NAMES, METRICS_DIR
from har_classifier.models import Conv1DLSTMClassifier
from har_classifier.training.trainer import Trainer


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = Trainer._forward(model, x)
            preds = logits.argmax(1).cpu().numpy()
            y_pred.append(preds)
            y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_per_class": precision_score(
            y_true, y_pred, average=None, zero_division=0
        ).tolist(),
        "recall_per_class": recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "f1_per_class": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=ACTIVITY_NAMES, zero_division=0
        ),
    }


def evaluate_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    save_path: Path | None = None,
) -> dict:
    results = {}
    for split_name, loader in loaders.items():
        y_true, y_pred = predict(model, loader, device)
        results[split_name] = compute_metrics(y_true, y_pred)
        results[split_name]["y_true"] = y_true.tolist()
        results[split_name]["y_pred"] = y_pred.tolist()

    if save_path is None:
        save_path = METRICS_DIR / "metrics_final.json"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    serializable = {
        k: {sk: sv for sk, sv in v.items() if sk not in ("y_true", "y_pred")}
        for k, v in results.items()
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    return results


def load_model_from_checkpoint(path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    name = ckpt.get("model_name", "Conv1D_LSTM")
    if name == "LSTM":
        from har_classifier.models import LSTMClassifier

        model = LSTMClassifier()
    else:
        model = Conv1DLSTMClassifier()
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device)
