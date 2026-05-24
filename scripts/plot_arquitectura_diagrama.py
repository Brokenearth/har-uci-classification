#!/usr/bin/env python3
"""Genera diagrama de bloques Conv1D+LSTM para la presentación Beamer."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidencias" / "figures" / "diagrama_arquitectura.png"


def add_box(ax, x, y, w, h, text, fc, ec, fs=8.5):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.06",
        linewidth=1.3,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#0f172a", linespacing=1.3)
    return x + w, y + h / 2


def add_arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.1,
            color="#475569",
        )
    )


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.5, 3.8), dpi=160)
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 3.8)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    y = 1.35
    h = 1.15
    cy = y + h / 2

    ax.text(6.75, 3.45, "Arquitectura Conv1D + LSTM (Conv1DLSTMClassifier)", ha="center", fontsize=11, fontweight="bold", color="#0f172a")

    blocks = [
        (0.15, 1.35, "Entrada\n(B, 128, 9)", "#f1f5f9", "#64748b"),
        (1.55, 1.35, "Conv1d\n9 → 32, k=5", "#dbeafe", "#1d4ed8"),
        (2.85, 1.35, "BN + ReLU\nMaxPool ÷2", "#dbeafe", "#1d4ed8"),
        (4.15, 1.35, "Conv1d\n32 → 64, k=5", "#dbeafe", "#1d4ed8"),
        (5.45, 1.35, "BN + ReLU\nMaxPool ÷2", "#dbeafe", "#1d4ed8"),
        (6.75, 1.35, "LSTM\n2 capas, h=128\nsec. T=32", "#d1fae5", "#047857"),
        (8.35, 1.35, "Dropout\np = 0.3", "#fef3c7", "#b45309"),
        (9.65, 1.35, "Linear\n128 → 6", "#fce7f3", "#be185d"),
        (11.05, 1.35, "Softmax\n6 clases", "#f1f5f9", "#64748b"),
    ]

    widths = [1.25, 1.15, 1.15, 1.15, 1.15, 1.45, 1.15, 1.15, 1.25]
    x = 0.15
    rights = []
    for (bx, by, txt, fc, ec), w in zip(blocks, widths):
        rights.append(add_box(ax, bx if bx > 0.1 else x, by, w, h, txt, fc, ec))
        x += w + 0.18

    # redraw with consistent x positions
    ax.clear()
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 3.8)
    ax.axis("off")

    ax.text(6.75, 3.45, "Arquitectura Conv1D + LSTM (Conv1DLSTMClassifier)", ha="center", fontsize=11, fontweight="bold", color="#0f172a")

    x = 0.15
    prev_right = None
    for (_, _, txt, fc, ec), w in zip(blocks, widths):
        right, mid_y = add_box(ax, x, y, w, h, txt, fc, ec)
        if prev_right is not None:
            add_arrow(ax, prev_right + 0.04, cy, x - 0.04, cy)
        prev_right = right
        x += w + 0.18

    ax.text(6.75, 0.55, "LSTM (alternativa):  (B, 128, 9)  →  LSTM×2 (128)  →  Dropout  →  Linear (6)", ha="center", fontsize=8.5, color="#64748b", style="italic")
    ax.text(0.15, 2.75, "Tras conv: (B, 64, 32)  →  transpose  →  (B, 32, 64) para LSTM", fontsize=7.5, color="#64748b")
    ax.text(8.0, 2.75, "Salida: último estado oculto h_n[-1]", fontsize=7.5, color="#047857")

    plt.tight_layout(pad=0.2)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Guardado: {OUT}")


if __name__ == "__main__":
    main()
