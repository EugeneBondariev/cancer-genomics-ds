"""
Confusion matrix heatmap for PAM50 subtype classification results.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


def plot_confusion_matrix(cm_df: pd.DataFrame, *, ax: Axes | None = None) -> Axes:
    """
    Heatmap of a confusion matrix DataFrame.

    Each cell shows the count and percentage of true-label samples.
    Rows = true labels, columns = predicted labels.
    """
    if ax is None:
        n = len(cm_df)
        _, ax = plt.subplots(figsize=(max(5, n), max(4, n - 1)))

    cm = cm_df.values.astype(float)
    row_totals = cm.sum(axis=1, keepdims=True)
    pct = np.where(row_totals > 0, cm / row_totals * 100, 0.0)

    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax, label="% of true label")

    labels = cm_df.index.tolist()
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            text_color = "white" if pct[i, j] > 60 else "black"
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}\n({pct[i, j]:.0f}%)",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix", fontsize=11, fontweight="bold")
    return ax
