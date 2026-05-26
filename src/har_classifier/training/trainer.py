"""Bucle de entrenamiento y validación con logging W&B."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from har_classifier.config import (
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    WANDB_ENTITY,
    WANDB_PROJECT,
)


class Trainer:
    def __init__(
        self,
        device: torch.device,
        use_wandb: bool = True,
        wandb_config: dict | None = None,
    ):
        self.device = device
        self.use_wandb = use_wandb
        self.wandb_config = wandb_config or {}
        self._run = None

    def _init_wandb(self, model_name: str, fold: int | None = None) -> None:
        if not self.use_wandb:
            return
        import wandb

        config = dict(self.wandb_config)
        config["model"] = model_name
        if fold is not None:
            config["fold"] = fold

        project = self.wandb_config.get("project") or WANDB_PROJECT
        phase = config.get("phase", "train")
        if fold is not None:
            run_name = f"{model_name}-fold{fold}"
            group = f"cv-{model_name}"
            job_type = "cross-validation"
        elif phase == "final_train":
            run_name = f"{model_name}-final"
            group = "final-train"
            job_type = "final"
        else:
            run_name = model_name
            group = None
            job_type = None

        init_kwargs: dict[str, Any] = {
            "project": project,
            "config": config,
            "name": run_name,
            "tags": [model_name, phase],
        }
        if WANDB_ENTITY:
            init_kwargs["entity"] = WANDB_ENTITY
        if group:
            init_kwargs["group"] = group
        if job_type:
            init_kwargs["job_type"] = job_type

        self._run = wandb.init(**init_kwargs)

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float = LEARNING_RATE,
        model_name: str = "model",
        fold: int | None = None,
        early_stopping: bool = True,
    ) -> dict[str, list[float]]:
        self._init_wandb(model_name, fold)
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }
        best_val_loss = float("inf")
        patience_counter = 0
        best_state: dict[str, Any] | None = None

        for epoch in range(epochs):
            train_loss, train_acc = self._epoch(model, train_loader, criterion, optimizer, train=True)
            val_loss, val_acc = self._epoch(model, val_loader, criterion, optimizer, train=False)
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)

            if self.use_wandb and self._run is not None:
                import wandb

                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_accuracy": train_acc,
                        "val_accuracy": val_acc,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} acc={val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            elif early_stopping:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping en época {epoch + 1}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        if self.use_wandb and self._run is not None:
            import wandb

            wandb.finish()

        history["best_val_loss"] = [best_val_loss]
        history["best_val_accuracy"] = [max(history["val_accuracy"]) if history["val_accuracy"] else 0.0]
        return history

    def _epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train: bool,
    ) -> tuple[float, float]:
        model.train(train)
        total_loss = 0.0
        correct = 0
        total = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x, y in tqdm(loader, leave=False, desc="train" if train else "val"):
                x, y = x.to(self.device), y.to(self.device)
                if train:
                    optimizer.zero_grad()
                logits = self._forward(model, x)
                loss = criterion(logits, y)
                if train:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                total += len(y)

        return total_loss / max(total, 1), correct / max(total, 1)

    @staticmethod
    def _forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        from har_classifier.models import Conv1DLSTMClassifier

        if isinstance(model, Conv1DLSTMClassifier):
            return model(x)
        return model(x.transpose(1, 2))
