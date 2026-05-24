#!/usr/bin/env python3
"""Gráfico de barras: comparación CV LSTM vs Conv1D+LSTM (presentación Beamer)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_CANDIDATES = [
    ROOT / "evidencias" / "comparacion_modelos.csv",
    ROOT / "results" / "metrics" / "comparacion_modelos.csv",
]
OUT = ROOT / "evidencias" / "figures" / "comparacion_cv.png"


def load_csv() -> pd.DataFrame:
    for path in CSV_CANDIDATES:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError("No se encontró comparacion_modelos.csv")


def parse_mean_std(row_val: str) -> tuple[float, float]:
    parts = str(row_val).replace("%", "").split("±")
    mean = float(parts[0].strip())
    std = float(parts[1].strip()) if len(parts) > 1 else 0.0
    return mean, std


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = load_csv()

    folds = df[df["fold"].isin([0, 1])].copy()
    folds["val_accuracy_pct"] = folds["val_accuracy"] * 100

    summary = df[df["fold"] == "mean_std"].copy()
    labels_display = {"LSTM": "LSTM", "Conv1D_LSTM": "Conv1D+LSTM"}
    models_order = ["LSTM", "Conv1D_LSTM"]

    means, stds = [], []
    for m in models_order:
        row = summary[summary["modelo"] == m]
        if len(row):
            mean, std = parse_mean_std(row.iloc[0]["val_accuracy"])
            means.append(mean * 100 if mean < 1 else mean)
            stds.append(std * 100 if std < 1 else std)
        else:
            sub = folds[folds["modelo"] == m]["val_accuracy_pct"]
            means.append(sub.mean())
            stds.append(sub.std())

    colors = ["#94a3b8", "#1d4ed8"]
    x = np.arange(len(models_order))
    width = 0.55

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=160)
    fig.patch.set_facecolor("white")

    bars = ax.bar(
        x,
        means,
        width,
        yerr=stds,
        capsize=6,
        color=colors,
        edgecolor="#0f172a",
        linewidth=0.8,
        error_kw={"elinewidth": 1.2, "ecolor": "#0f172a"},
        zorder=2,
    )

    for i, m in enumerate(models_order):
        sub = folds[folds["modelo"] == m]
        jitter = (np.random.default_rng(42).random(len(sub)) - 0.5) * 0.08
        ax.scatter(
            np.full(len(sub), x[i]) + jitter,
            sub["val_accuracy_pct"],
            color="#0f172a",
            s=28,
            zorder=3,
            alpha=0.75,
        )

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.35,
            f"{mean:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#0f172a",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([labels_display[m] for m in models_order], fontsize=11)
    ax.set_ylabel("Exactitud en validación (%)", fontsize=10)
    ax.set_title("Validación cruzada (2 folds, agrupado por sujeto)", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylim(88, 95)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.5,
        0.02,
        "Barras: media ± desv. est.   ·   Puntos: cada fold",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
        color="#64748b",
    )

    plt.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Guardado: {OUT}")


if __name__ == "__main__":
    main()
