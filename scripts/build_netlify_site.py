#!/usr/bin/env python3
"""Genera site/data/ para despliegue estático en Netlify (sin backend)."""

import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from har_classifier.config import ACTIVITY_NAMES, INERTIAL_CHANNELS, MODELS_DIR
from har_classifier.evaluation.metrics import load_model_from_checkpoint
from har_classifier.preprocessing import load_processed
from har_classifier.training.trainer import Trainer
from har_classifier.utils.device import get_device

SITE = ROOT / "site"
EVIDENCIAS = ROOT / "evidencias"

METRICS_CANDIDATES = (
    EVIDENCIAS / "metrics_final.json",
    ROOT / "results" / "metrics" / "metrics_final.json",
)


def load_metrics_accuracies() -> tuple[float, float]:
    """Carga exactitud en val y test desde metrics_final.json."""
    metrics_path = next((p for p in METRICS_CANDIDATES if p.exists()), None)
    if metrics_path is None:
        searched = ", ".join(str(p) for p in METRICS_CANDIDATES)
        raise FileNotFoundError(
            f"No se encontró metrics_final.json. Rutas probadas: {searched}. "
            "Ejecute scripts/05_evaluate.py y scripts/sync_evidencias.py."
        )

    with open(metrics_path, encoding="utf-8") as f:
        metrics_file = json.load(f)

    accuracies: dict[str, float] = {}
    for split in ("val", "test"):
        block = metrics_file.get(split)
        if not isinstance(block, dict) or "accuracy" not in block:
            raise KeyError(
                f"{metrics_path}: falta la clave '{split}.accuracy'. "
                "Regenere métricas con 05_evaluate.py."
            )
        accuracies[split] = float(block["accuracy"])

    return accuracies["val"], accuracies["test"]


def main() -> None:
    import torch

    print("Cargando modelo y datos de test...")
    device = get_device(verbose=False)
    ckpt = MODELS_DIR / "final_model.pt"
    if not ckpt.exists():
        ckpt = EVIDENCIAS / "final_model.pt"
    model = load_model_from_checkpoint(ckpt, device)
    data = load_processed()
    X_test = data["X_test"]
    y_test = data["y_test"]

    model.eval()
    samples = []
    n = len(y_test)
    for i in range(n):
        x = X_test[i]
        x_t = torch.from_numpy(x.transpose(1, 0).astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            logits = Trainer._forward(model, x_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        y_pred = int(probs.argmax())
        samples.append(
            {
                "idx": i,
                "y_true": ACTIVITY_NAMES[int(y_test[i])],
                "y_pred": ACTIVITY_NAMES[y_pred],
                "match": bool(y_test[i] == y_pred),
                "probs": {ACTIVITY_NAMES[j]: float(probs[j]) for j in range(6)},
                "signal": x.tolist(),
            }
        )
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}")

    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "data").mkdir(exist_ok=True)
    assets = SITE / "assets"
    assets.mkdir(exist_ok=True)

    val_acc, test_acc = load_metrics_accuracies()

    meta = {
        "activity_names": ACTIVITY_NAMES,
        "channels": INERTIAL_CHANNELS,
        "n_samples": n,
        "metrics": {
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "note": (
                "Explorador: predicciones precomputadas en test. "
                "Figuras: matriz y curvas según evidencias/ (val, literal c)."
            ),
        },
    }
    with open(SITE / "data" / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    raw = json.dumps(samples, separators=(",", ":")).encode("utf-8")
    gz_path = SITE / "data" / "samples.json.gz"
    with gzip.open(gz_path, "wb", compresslevel=6) as f:
        f.write(raw)
    print(f"Guardado {gz_path} ({gz_path.stat().st_size / 1e6:.1f} MB)")

    for name in ("confusion_matrix.png", "loss_curves.png", "comparacion_modelos.csv"):
        src = EVIDENCIAS / name
        if src.exists():
            shutil.copy2(src, assets / name)

    print(f"Sitio listo en {SITE}")


if __name__ == "__main__":
    main()
