"""
One-vs-rest differential gene expression for PAM50 breast cancer subtypes.

Applies log2(counts+1) normalization, runs Welch's t-test per gene per subtype
vs all remaining samples, and corrects for multiple testing via Benjamini-Hochberg FDR.

Run: python -m src.features.differential_expression
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.constants import InterimFile
from src.paths import interim_file

logger = logging.getLogger(__name__)

_PSEUDOCOUNT: float = 1.0


def log2_normalize(X: pd.DataFrame) -> pd.DataFrame:
    """Log2(counts + 1) transform. Pseudocount prevents log(0) on zero-count genes."""
    return pd.DataFrame(np.log2(X + _PSEUDOCOUNT), index=X.index, columns=X.columns)


def _test_one_subtype(
    X_log: pd.DataFrame,
    labels: pd.Series,
    subtype: str,
) -> pd.DataFrame:
    mask = labels == subtype
    a = X_log.loc[mask].values    # (n_subtype, n_genes)
    b = X_log.loc[~mask].values   # (n_rest,    n_genes)

    _, pvals = stats.ttest_ind(a, b, equal_var=False, axis=0)
    _, padj, _, _ = multipletests(pvals, method="fdr_bh")
    log2fc = X_log.loc[mask].mean() - X_log.loc[~mask].mean()

    return pd.DataFrame(
        {
            "gene": X_log.columns.tolist(),
            "subtype": subtype,
            "log2fc": log2fc.values,
            "pval": pvals,
            "padj": padj,
            "mean_subtype": X_log.loc[mask].mean().values,
            "mean_rest": X_log.loc[~mask].mean().values,
            "n_subtype": int(mask.sum()),
            "n_rest": int((~mask).sum()),
        }
    )


def run_one_vs_rest(
    X: pd.DataFrame,
    labels: pd.Series,
    *,
    fdr_threshold: float = 0.05,
    log2fc_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Run one-vs-rest Welch's t-test DGE for every unique subtype in `labels`.

    X: samples × genes DataFrame of raw counts (log2 normalization applied internally).
    labels: sample-aligned Series of subtype strings.
    Returns long-format DataFrame with one row per (gene, subtype) pair.
    """
    if not X.index.equals(labels.index):
        raise ValueError("X and labels must share the same index in the same order")

    X_log = log2_normalize(X)
    subtypes = sorted(labels.unique())

    logger.info(
        "One-vs-rest DGE: %d subtypes × %d genes = %d tests",
        len(subtypes),
        X.shape[1],
        len(subtypes) * X.shape[1],
    )

    parts: list[pd.DataFrame] = []
    for subtype in subtypes:
        n = int((labels == subtype).sum())
        logger.info("  %-8s  n=%d", subtype, n)
        parts.append(_test_one_subtype(X_log=X_log, labels=labels, subtype=subtype))

    result = pd.concat(parts, ignore_index=True)
    result["significant"] = (result["padj"] < fdr_threshold) & (
        result["log2fc"].abs() >= log2fc_threshold
    )

    n_sig = int(result["significant"].sum())
    logger.info(
        "Significant pairs (FDR<%.2f, |log2FC|≥%.1f): %d / %d",
        fdr_threshold,
        log2fc_threshold,
        n_sig,
        len(result),
    )
    return result


def top_de_genes(
    results: pd.DataFrame,
    *,
    n: int = 20,
    direction: Literal["up", "down", "both"] = "both",
) -> pd.DataFrame:
    """
    Top n significant DE genes per subtype, ranked by adjusted p-value.

    direction:
      'up'   — higher in subtype vs rest (log2fc > 0)
      'down' — lower in subtype vs rest  (log2fc < 0)
      'both' — no direction filter
    """
    sig = results.loc[results["significant"]].copy()
    if direction == "up":
        sig = sig.loc[sig["log2fc"] > 0]
    elif direction == "down":
        sig = sig.loc[sig["log2fc"] < 0]
    return (
        sig.sort_values("padj")
        .groupby("subtype", sort=False)
        .head(n)
        .reset_index(drop=True)
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    X = pd.read_csv(interim_file(InterimFile.X), index_col=0)
    _raw = pd.read_csv(interim_file(InterimFile.LABELS), index_col=0).squeeze()
    labels: pd.Series = _raw  # type: ignore[assignment]  # squeeze() return type is overly broad
    results = run_one_vs_rest(X=X, labels=labels)
    out = interim_file(InterimFile.DE_RESULTS)
    results.to_csv(out, index=False)
    logger.info("Saved %s  (%d rows)", out.name, len(results))


if __name__ == "__main__":
    main()
