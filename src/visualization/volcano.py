"""
Volcano plots for one-vs-rest differential expression results.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

_UP_COLOR = "#d62728"    # red   — higher in subtype
_DOWN_COLOR = "#1f77b4"  # blue  — lower in subtype
_NS_COLOR = "#cccccc"    # grey  — not significant
_SCATTER_KW: dict = dict(s=8, alpha=0.6, linewidths=0, rasterized=True)


def _safe_neg_log10(s: pd.Series) -> pd.Series:
    return -np.log10(s.clip(lower=1e-300))


def plot_volcano(
    results: pd.DataFrame,
    subtype: str,
    *,
    fdr_threshold: float = 0.05,
    log2fc_threshold: float = 1.0,
    n_label: int = 10,
    ax: Axes | None = None,
) -> Axes:
    """
    Volcano plot for one subtype.

    x = log2FC (positive → higher in subtype vs all others)
    y = −log10(BH-adjusted p-value)
    Dashed lines mark the FDR and log2FC thresholds.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    df = results.loc[results["subtype"] == subtype].copy()
    if df.empty:
        raise ValueError(f"No results for subtype '{subtype}'")

    df["neg_log10_padj"] = _safe_neg_log10(df["padj"])

    sig_up = df["significant"] & (df["log2fc"] > 0)
    sig_down = df["significant"] & (df["log2fc"] < 0)
    ns = ~df["significant"]

    ax.scatter(df.loc[ns, "log2fc"], df.loc[ns, "neg_log10_padj"],
               c=_NS_COLOR, **_SCATTER_KW, label="n.s.")
    ax.scatter(df.loc[sig_down, "log2fc"], df.loc[sig_down, "neg_log10_padj"],
               c=_DOWN_COLOR, **_SCATTER_KW, label=f"down ({int(sig_down.sum())})")
    ax.scatter(df.loc[sig_up, "log2fc"], df.loc[sig_up, "neg_log10_padj"],
               c=_UP_COLOR, **_SCATTER_KW, label=f"up ({int(sig_up.sum())})")

    ax.axhline(-np.log10(fdr_threshold), color="black", lw=0.8, ls="--", alpha=0.7)
    ax.axvline(log2fc_threshold, color="black", lw=0.8, ls="--", alpha=0.7)
    ax.axvline(-log2fc_threshold, color="black", lw=0.8, ls="--", alpha=0.7)

    to_label = df.loc[df["significant"]].nsmallest(n_label, "padj")
    for _, row in to_label.iterrows():
        ax.annotate(
            row["gene"],
            xy=(row["log2fc"], row["neg_log10_padj"]),
            fontsize=5.5,
            ha="left" if row["log2fc"] > 0 else "right",
        )

    ax.set_title(subtype, fontsize=11, fontweight="bold")
    ax.set_xlabel("log₂ fold change (subtype vs rest)")
    ax.set_ylabel("−log₁₀(FDR-adjusted p-value)")
    ax.legend(markerscale=2, fontsize=8, frameon=False)
    return ax


def plot_all_volcanos(
    results: pd.DataFrame,
    *,
    fdr_threshold: float = 0.05,
    log2fc_threshold: float = 1.0,
    n_label: int = 8,
    ncols: int = 3,
) -> plt.Figure:
    """Grid of volcano plots, one panel per subtype."""
    subtypes = sorted(results["subtype"].unique())
    nrows = (len(subtypes) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 6, nrows * 5), squeeze=False
    )
    axes_flat: np.ndarray = axes.flatten()

    for i, subtype in enumerate(subtypes):
        plot_volcano(
            results,
            subtype,
            fdr_threshold=fdr_threshold,
            log2fc_threshold=log2fc_threshold,
            n_label=n_label,
            ax=axes_flat[i],
        )
    for j in range(len(subtypes), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout()
    return fig
