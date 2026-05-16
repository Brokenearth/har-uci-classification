"""Lectura del UCI HAR Dataset desde archivos oficiales."""

from pathlib import Path

import numpy as np

from har_classifier.config import INERTIAL_CHANNELS, NUM_CHANNELS, WINDOW_SIZE
from har_classifier.download import get_dataset_root


def _load_signal_file(path: Path) -> np.ndarray:
    """Carga un archivo de señal (una ventana por línea)."""
    return np.loadtxt(path, dtype=np.float32)


def _load_labels(path: Path) -> np.ndarray:
    """Carga etiquetas 1-based y convierte a 0-based."""
    labels = np.loadtxt(path, dtype=np.int64)
    if labels.ndim == 0:
        labels = np.array([labels])
    return labels - 1


def _load_subjects(path: Path) -> np.ndarray:
    """Carga IDs de sujetos."""
    subjects = np.loadtxt(path, dtype=np.int64)
    if subjects.ndim == 0:
        subjects = np.array([subjects])
    return subjects


def load_split(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Carga X, y y opcionalmente subjects para 'train' o 'test'.

    Returns:
        X: (n_samples, window_size, n_channels)
        y: (n_samples,) enteros 0-5
        subjects: (n_samples,) solo para train, None para test
    """
    if split not in ("train", "test"):
        raise ValueError("split debe ser 'train' o 'test'")

    root = get_dataset_root()
    split_dir = root / split
    inertial_dir = split_dir / "Inertial Signals"

    channels = []
    for ch in INERTIAL_CHANNELS:
        fname = f"{ch}_{split}.txt"
        data = _load_signal_file(inertial_dir / fname)
        channels.append(data)

    X = np.stack(channels, axis=-1).astype(np.float32)
    y = _load_labels(split_dir / f"y_{split}.txt")

    subjects = None
    if split == "train":
        subjects = _load_subjects(split_dir / "subject_train.txt")

    assert X.shape[1] == WINDOW_SIZE, f"Esperado T={WINDOW_SIZE}, got {X.shape[1]}"
    assert X.shape[2] == NUM_CHANNELS
    assert len(y) == X.shape[0]

    return X, y, subjects


def load_all_raw() -> dict:
    """Carga train y test completos."""
    X_train, y_train, subjects_train = load_split("train")
    X_test, y_test, _ = load_split("test")
    return {
        "X_train": X_train,
        "y_train": y_train,
        "subjects_train": subjects_train,
        "X_test": X_test,
        "y_test": y_test,
    }
