"""
Kaplan-Meier and Cox forest plot visualizations for BRCA survival analysis.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from matplotlib.axes import Axes

from src.models.survival import fit_km

logger = logging.getLogger(__name__)


def plot_km_by_subtype(df: pd.DataFrame, *, ax: Axes | None = None) -> Axes:
    """
    KM survival curves for all subtypes on a single axes.

    df must have columns: subtype, OS_time, OS_event.
    Uses lifelines add_at_risk_counts.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    subtypes = sorted(df["subtype"].unique())
    fitters: list[KaplanMeierFitter] = []
    for subtype in subtypes:
        kmf = fit_km(df=df, subtype=subtype)
        kmf.plot_survival_function(ax=ax, ci_show=True)
        fitters.append(kmf)

    from lifelines.plotting import add_at_risk_counts
    add_at_risk_counts(*fitters, ax=ax)

    ax.set_title("Overall survival by PAM50 subtype", fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival probability")
    ax.legend(fontsize=8, frameon=False)
    return ax


def plot_cox_forest(cox_fitter: CoxPHFitter, *, ax: Axes | None = None) -> Axes:
    """
    Forest plot of hazard ratios from a fitted CoxPHFitter.

    Each covariate gets a point estimate (HR) and 95% CI. HR < 1 is protective,
    HR > 1 is a risk factor. Vertical dashed line at HR = 1 (no effect).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(4, len(cox_fitter.params_) * 0.5 + 2)))

    summary = cox_fitter.summary.copy()
    summary = summary.sort_values("exp(coef)")

    covariates = summary.index.tolist()
    y_pos = np.arange(len(covariates))

    hr = summary["exp(coef)"].values
    ci_lo = summary["exp(coef) lower 95%"].values
    ci_hi = summary["exp(coef) upper 95%"].values

    ax.scatter(hr, y_pos, color="#d62728", zorder=3, s=40)
    for i, (lo, hi) in enumerate(zip(ci_lo, ci_hi)):
        ax.hlines(y_pos[i], lo, hi, color="#d62728", lw=2)

    ax.axvline(1.0, color="black", lw=0.8, ls="--", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(covariates)
    ax.set_xlabel("Hazard ratio (95% CI)")
    ax.set_title("Cox model — hazard ratios", fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    return ax
