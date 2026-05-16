"""Particiones por sujeto sin fuga de datos."""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from har_classifier.config import N_CV_FOLDS, RANDOM_SEED, VAL_SUBJECT_RATIO


def get_cv_folds(
    y: np.ndarray, groups: np.ndarray, n_splits: int = N_CV_FOLDS
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Genera índices de validación cruzada agrupados por sujeto.

    Returns:
        Lista de (train_idx, val_idx) por fold.
    """
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED
    )
    folds = []
    for train_idx, val_idx in sgkf.split(np.zeros(len(y)), y, groups):
        folds.append((train_idx, val_idx))
        _verify_no_subject_leak(groups, train_idx, val_idx)
    return folds


def get_final_train_val_split(
    y: np.ndarray, groups: np.ndarray, val_ratio: float = VAL_SUBJECT_RATIO
) -> tuple[np.ndarray, np.ndarray]:
    """
    Divide train en train/val por sujetos (hold-out de sujetos).
    """
    unique_subjects = np.unique(groups)
    try:
        subject_labels = np.array(
            [
                np.bincount(y[groups == s], minlength=6).argmax()
                for s in unique_subjects
            ]
        )
        train_subjects, val_subjects = train_test_split(
            unique_subjects,
            test_size=val_ratio,
            random_state=RANDOM_SEED,
            stratify=subject_labels,
        )
    except ValueError:
        train_subjects, val_subjects = train_test_split(
            unique_subjects,
            test_size=val_ratio,
            random_state=RANDOM_SEED,
        )

    train_idx = np.where(np.isin(groups, train_subjects))[0]
    val_idx = np.where(np.isin(groups, val_subjects))[0]
    _verify_no_subject_leak(groups, train_idx, val_idx)
    return train_idx, val_idx


def _verify_no_subject_leak(
    groups: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray
) -> None:
    """Verifica que ningún sujeto aparezca en train y val simultáneamente."""
    train_subjects = set(groups[train_idx])
    val_subjects = set(groups[val_idx])
    overlap = train_subjects & val_subjects
    if overlap:
        raise ValueError(f"Fuga de datos: sujetos en ambos conjuntos: {overlap}")


def print_fold_info(groups: np.ndarray, folds: list) -> None:
    """Imprime información de depuración de folds."""
    for i, (train_idx, val_idx) in enumerate(folds):
        n_train_subj = len(np.unique(groups[train_idx]))
        n_val_subj = len(np.unique(groups[val_idx]))
        print(
            f"Fold {i}: train={len(train_idx)} muestras ({n_train_subj} sujetos), "
            f"val={len(val_idx)} muestras ({n_val_subj} sujetos)"
        )
