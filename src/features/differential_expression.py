"""
One-vs-rest and pairwise differential gene expression for PAM50 breast cancer subtypes.

Runs Welch's t-test per gene per subtype vs all remaining samples and corrects
for multiple testing via Benjamini-Hochberg FDR.

Expects log2(count+1)-normalized input — the UCSC Xena STAR counts file is already
stored in that form, so no further normalization is applied inside this module.

Run: python -m src.features.differential_expression
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Literal

import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.constants import InterimFile
from src.paths import interim_file

logger = logging.getLogger(__name__)


def _test_one_subtype(
    X: pd.DataFrame,
    labels: pd.Series,
    subtype: str,
) -> pd.DataFrame:
    mask = labels == subtype
    a = X.loc[mask].values    # (n_subtype, n_genes)
    b = X.loc[~mask].values   # (n_rest,    n_genes)

    _, pvals = stats.ttest_ind(a, b, equal_var=False, axis=0)
    _, padj, _, _ = multipletests(pvals, method="fdr_bh")
    mean_subtype = X.loc[mask].mean()
    mean_rest = X.loc[~mask].mean()

    return pd.DataFrame(
        {
            "gene": X.columns.tolist(),
            "subtype": subtype,
            "log2fc": (mean_subtype - mean_rest).values,
            "pval": pvals,
            "padj": padj,
            "mean_subtype": mean_subtype.values,
            "mean_rest": mean_rest.values,
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

    X: samples × genes DataFrame, log2(count+1)-normalized.
    labels: sample-aligned Series of subtype strings.
    Returns long-format DataFrame with one row per (gene, subtype) pair.
    """
    if not X.index.equals(labels.index):
        raise ValueError("X and labels must share the same index in the same order")

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
        parts.append(_test_one_subtype(X=X, labels=labels, subtype=subtype))

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


def run_pairwise(
    X: pd.DataFrame,
    labels: pd.Series,
    *,
    fdr_threshold: float = 0.05,
    log2fc_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Run all pairwise Welch's t-test DGE comparisons between subtypes.

    X: samples × genes DataFrame, log2(count+1)-normalized.
    labels: sample-aligned Series of subtype strings.
    Returns long-format DataFrame with one row per (gene, subtype_a, subtype_b) pair.
    log2fc = mean(subtype_a) - mean(subtype_b); positive means higher in subtype_a.
    Subtypes are sorted alphabetically so each pair appears exactly once.
    """
    if not X.index.equals(labels.index):
        raise ValueError("X and labels must share the same index in the same order")

    subtypes = sorted(labels.unique())
    # C(5,2) = 5! / ((5-2)! * 2!) = 10
    # AB, AC, AD, AE, BC, BD, BE, CD, CE, DE
    pairs = list(combinations(subtypes, 2))

    logger.info(
        "Pairwise DGE: %d pairs × %d genes = %d tests",
        len(pairs),
        X.shape[1],
        len(pairs) * X.shape[1],
    )

    parts: list[pd.DataFrame] = []
    for subtype_a, subtype_b in pairs:
        mask_a = labels == subtype_a
        mask_b = labels == subtype_b
        n_a = int(mask_a.sum())
        n_b = int(mask_b.sum())
        logger.info("  %-8s vs %-8s  n=(%d, %d)", subtype_a, subtype_b, n_a, n_b)

        a = X.loc[mask_a].values
        b = X.loc[mask_b].values

        _, pvals = stats.ttest_ind(a, b, equal_var=False, axis=0)
        _, padj, _, _ = multipletests(pvals, method="fdr_bh")
        mean_a = X.loc[mask_a].mean()
        mean_b = X.loc[mask_b].mean()

        parts.append(pd.DataFrame({
            "gene": X.columns.tolist(),
            "subtype_a": subtype_a,
            "subtype_b": subtype_b,
            "log2fc": (mean_a - mean_b).values,
            "pval": pvals,
            "padj": padj,
            "mean_a": mean_a.values,
            "mean_b": mean_b.values,
            "n_a": n_a,
            "n_b": n_b,
        }))

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
        # realistically, log2fc -> [1, 21]
        sig = sig.loc[sig["log2fc"] > 0]
    elif direction == "down":
        # realistically, log2fc -> [-21, -1]
        sig = sig.loc[sig["log2fc"] < 0]
    return (
        sig.groupby("subtype", sort=True)
        .apply(lambda g: g.nsmallest(n, "padj"), include_groups=False)
        .reset_index(level=0)
        .reset_index(drop=True)
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    X = pd.read_csv(interim_file(InterimFile.X), index_col=0)
    _raw = pd.read_csv(interim_file(InterimFile.LABELS), index_col=0).squeeze()
    labels: pd.Series = _raw  # type: ignore[assignment]  # squeeze() return type is overly broad

    for df, key in [
        (run_one_vs_rest(X=X, labels=labels),  InterimFile.DE_RESULTS),
        (run_pairwise(X=X, labels=labels),     InterimFile.PAIRWISE_RESULTS),
    ]:
        out = interim_file(key)
        df.to_csv(out, index=False)
        logger.info("Saved %s  (%d rows)", out.name, len(df))


if __name__ == "__main__":
    main()
