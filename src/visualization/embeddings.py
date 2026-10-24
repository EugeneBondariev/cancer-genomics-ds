"""
Scatter and scree plots for PCA / UMAP embeddings of TCGA BRCA data.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

_SCATTER_KW: dict = dict(s=10, alpha=0.7, linewidths=0, rasterized=True)


def plot_embedding(
    emb: pd.DataFrame,
    labels: pd.Series,
    *,
    title: str,
    ax: Axes | None = None,
) -> Axes:
    """
    2-D scatter of emb (first two columns) coloured by categorical labels.

    emb must have at least two columns; the first two are used as x and y.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    categories = sorted(labels.unique())
    cmap = plt.get_cmap("tab10", len(categories))
    color_map: dict[str, tuple] = {cat: cmap(i) for i, cat in enumerate(categories)}

    x_col, y_col = emb.columns[0], emb.columns[1]
    for cat in categories:
        mask = labels == cat
        ax.scatter(
            emb.loc[mask, x_col],
            emb.loc[mask, y_col],
            color=color_map[cat],
            label=cat,
            **_SCATTER_KW,
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(markerscale=2, fontsize=8, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    return ax


def plot_scree(
    pca: PCA,
    *,
    n_components: int = 20,
    ax: Axes | None = None,
) -> Axes:
    """
    Scree plot: per-component explained variance as bars, cumulative as a line.

    n_components limits how many components are displayed (not how many were fitted).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    n = min(n_components, len(pca.explained_variance_ratio_))
    var = pca.explained_variance_ratio_[:n] * 100
    cumvar = np.cumsum(var)
    indices = np.arange(1, n + 1)

    ax.bar(indices, var, color="#4878cf", alpha=0.8, label="Per component")

    ax2 = ax.twinx()
    ax2.plot(indices, cumvar, color="#e84646", lw=2, marker="o", ms=4, label="Cumulative")
    ax2.set_ylabel("Cumulative explained variance (%)", color="#e84646")
    ax2.tick_params(axis="y", labelcolor="#e84646")
    ax2.set_ylim(0, 105)

    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("PCA scree plot", fontsize=11, fontweight="bold")
    ax.set_xticks(indices)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, frameon=False)

    return ax
