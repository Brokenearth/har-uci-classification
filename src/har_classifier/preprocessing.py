"""Preprocesamiento, normalización y persistencia de datos."""

import json
from pathlib import Path

import numpy as np

from har_classifier.config import (
    ACTIVITY_NAMES,
    DATA_PROCESSED,
    INERTIAL_CHANNELS,
    RANDOM_SEED,
    VAL_SUBJECT_RATIO,
)
from har_classifier.loader import load_all_raw
from har_classifier.splits import get_final_train_val_split, _verify_no_subject_leak


def fit_channel_stats(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcula mean/std por canal sobre (n, T, C)."""
    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1))
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_normalization(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def normalize_by_channel(
    X: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normaliza por canal; calcula stats si no se proporcionan."""
    if mean is None or std is None:
        mean, std = fit_channel_stats(X)
    return apply_normalization(X, mean, std), mean, std


def prepare_and_save(val_ratio: float = VAL_SUBJECT_RATIO) -> dict:
    """
    Carga datos, divide train/val por sujeto, normaliza con stats de train,
    guarda har_processed.npz y stats.json.
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    data = load_all_raw()
    np.savez_compressed(
        DATA_PROCESSED / "har_raw.npz",
        X_train=data["X_train"],
        y_train=data["y_train"],
        subjects_train=data["subjects_train"],
        X_test=data["X_test"],
        y_test=data["y_test"],
    )

    train_idx, val_idx = get_final_train_val_split(
        data["y_train"], data["subjects_train"], val_ratio=val_ratio
    )
    _verify_no_subject_leak(data["subjects_train"], train_idx, val_idx)

    X_tr = data["X_train"][train_idx]
    y_tr = data["y_train"][train_idx]
    X_val = data["X_train"][val_idx]
    y_val = data["y_train"][val_idx]

    mean, std = fit_channel_stats(X_tr)
    X_tr = apply_normalization(X_tr, mean, std)
    X_val = apply_normalization(X_val, mean, std)
    X_test = apply_normalization(data["X_test"], mean, std)

    np.savez_compressed(
        DATA_PROCESSED / "har_processed.npz",
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=data["y_test"],
        train_idx=train_idx,
        val_idx=val_idx,
        mean=mean,
        std=std,
    )

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "random_seed": RANDOM_SEED,
        "val_ratio": val_ratio,
        "channels": INERTIAL_CHANNELS,
        "activity_names": ACTIVITY_NAMES,
    }
    with open(DATA_PROCESSED / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    meta = {
        "shapes": {
            "X_train": list(X_tr.shape),
            "X_val": list(X_val.shape),
            "X_test": list(X_test.shape),
        },
        "n_train_subjects": int(len(np.unique(data["subjects_train"][train_idx]))),
        "n_val_subjects": int(len(np.unique(data["subjects_train"][val_idx]))),
    }
    with open(DATA_PROCESSED / "preprocess_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Guardado en {DATA_PROCESSED / 'har_processed.npz'}")
    print(f"  X_train: {X_tr.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    return {
        "X_train": X_tr,
        "y_train": y_tr,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": data["y_test"],
        "subjects_train": data["subjects_train"],
        "mean": mean,
        "std": std,
    }


def load_processed() -> dict:
    """Carga har_processed.npz; genera si no existe."""
    path = DATA_PROCESSED / "har_processed.npz"
    if not path.exists():
        return prepare_and_save()
    loaded = np.load(path)
    result = {k: loaded[k] for k in loaded.files}
    raw_path = DATA_PROCESSED / "har_raw.npz"
    if "subjects_train" not in result and raw_path.exists():
        raw = np.load(raw_path)
        result["subjects_train"] = raw["subjects_train"]
    elif "subjects_train" not in result:
        data = load_all_raw()
        result["subjects_train"] = data["subjects_train"]
    return result


def load_raw_for_cv() -> dict:
    """Carga datos sin normalizar para CV por fold."""
    path = DATA_PROCESSED / "har_raw.npz"
    if path.exists():
        loaded = np.load(path)
        return {k: loaded[k] for k in loaded.files}
    data = load_all_raw()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X_train=data["X_train"],
        y_train=data["y_train"],
        subjects_train=data["subjects_train"],
        X_test=data["X_test"],
        y_test=data["y_test"],
    )
    return data
