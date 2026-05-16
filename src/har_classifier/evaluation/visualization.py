"""Visualizaciones: matriz de confusión, señales, aciertos/errores."""

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from har_classifier.config import ACTIVITY_NAMES, FIGURES_DIR, INERTIAL_CHANNELS, REPORTS_DIR


def plot_signals_base64(X: np.ndarray, title: str = "", theme: str = "light") -> str:
    """Genera PNG en base64 de las 9 señales (T, C)."""
    colors = [
        "#1d4ed8", "#047857", "#7c3aed", "#b45309",
        "#be185d", "#0e7490", "#c2410c", "#475569", "#15803d",
    ]
    if theme == "light":
        fig_bg, ax_bg, title_c, tick_c, spine_c = "#f1f5f9", "#ffffff", "#0f172a", "#475569", "#cbd5e1"
    else:
        fig_bg, ax_bg, title_c, tick_c, spine_c = "#141820", "#1c2230", "#e8ecf4", "#6b7280", "#2a3344"

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, facecolor=fig_bg)
    fig.patch.set_facecolor(fig_bg)
    axes = axes.flatten()
    for i, ch in enumerate(INERTIAL_CHANNELS):
        axes[i].set_facecolor(ax_bg)
        axes[i].plot(X[:, i], linewidth=2.0, color=colors[i % len(colors)], alpha=0.95)
        axes[i].set_title(ch, fontsize=10, fontweight="600", color=title_c, pad=6)
        axes[i].tick_params(colors=tick_c, labelsize=8)
        axes[i].grid(True, linestyle="--", linewidth=0.5, alpha=0.45, color=spine_c)
        for spine in axes[i].spines.values():
            spine.set_color(spine_c)
    if title:
        fig.suptitle(title, fontsize=14, fontweight="600", color=title_c, y=1.01)
    fig.tight_layout(pad=2.0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor=fig_bg)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_confusion_matrix(cm: np.ndarray, save_path: Path) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=ACTIVITY_NAMES,
        yticklabels=ACTIVITY_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_loss_curves(history: dict, save_path: Path) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.get("train_loss", []), label="train")
    axes[0].plot(history.get("val_loss", []), label="val")
    axes[0].set_title("Pérdida")
    axes[0].legend()
    axes[1].plot(history.get("train_accuracy", []), label="train")
    axes[1].plot(history.get("val_accuracy", []), label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_signal_window(
    X: np.ndarray,
    y_true: int,
    y_pred: int,
    save_path: Path,
    title: str = "",
) -> None:
    """Grafica 9 canales de una ventana (T, C)."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    for i, ch in enumerate(INERTIAL_CHANNELS):
        axes[i].plot(X[:, i], linewidth=0.8)
        axes[i].set_title(ch, fontsize=8)
    true_name = ACTIVITY_NAMES[y_true]
    pred_name = ACTIVITY_NAMES[y_pred]
    fig.suptitle(f"{title}\nReal: {true_name} | Pred: {pred_name}", fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def visualize_correct_and_errors(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_each: int = 5,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    correct_idx = np.where(y_true == y_pred)[0]
    error_idx = np.where(y_true != y_pred)[0]

    np.random.seed(42)
    for i, idx in enumerate(np.random.choice(correct_idx, min(n_each, len(correct_idx)), replace=False)):
        plot_signal_window(
            X[idx],
            int(y_true[idx]),
            int(y_pred[idx]),
            FIGURES_DIR / f"aciertos_{i + 1}.png",
            title=f"Acierto #{i + 1}",
        )

    for i, idx in enumerate(np.random.choice(error_idx, min(n_each, len(error_idx)), replace=False)):
        plot_signal_window(
            X[idx],
            int(y_true[idx]),
            int(y_pred[idx]),
            FIGURES_DIR / f"errores_{i + 1}.png",
            title=f"Error #{i + 1}",
        )


def write_error_analysis_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path | None = None,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if save_path is None:
        save_path = REPORTS_DIR / "error_analysis.md"

    error_mask = y_true != y_pred
    lines = [
        "# Análisis de errores — UCI HAR\n",
        f"Total de muestras: {len(y_true)}\n",
        f"Errores: {error_mask.sum()} ({100 * error_mask.mean():.2f}%)\n\n",
        "## Confusiones más frecuentes\n",
    ]

    pairs: dict[tuple[str, str], int] = {}
    for t, p in zip(y_true[error_mask], y_pred[error_mask]):
        key = (ACTIVITY_NAMES[t], ACTIVITY_NAMES[p])
        pairs[key] = pairs.get(key, 0) + 1

    for (true_l, pred_l), count in sorted(pairs.items(), key=lambda x: -x[1])[:10]:
        lines.append(f"- **{true_l}** → **{pred_l}**: {count} muestras\n")

    lines.extend(
        [
            "\n## Observaciones\n",
            "- `SITTING` vs `STANDING` suelen confundirse por señales estáticas similares.\n",
            "- Las actividades de marcha (`WALKING*`) comparten patrones periódicos.\n",
            "- `LAYING` suele separarse bien del resto por orientación del acelerómetro.\n",
        ]
    )

    save_path.write_text("".join(lines), encoding="utf-8")
    print(f"Reporte guardado en {save_path}")
