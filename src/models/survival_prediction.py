"""
Prognostic gene signature via univariate Cox selection + regularized Cox model.

Selects the top N DE genes with independent prognostic value (univariate Cox),
fits a regularized Cox model, and computes per-sample risk scores.

Run: python -m src.models.survival_prediction
"""

from __future__ import annotations

import logging
import pickle

import pandas as pd
from lifelines import CoxPHFitter

from src.constants import DataFile, InterimFile
from src.features.differential_expression import log2_normalize
from src.models.survival import merge_survival
from src.paths import PROCESSED_DIR, interim_file, raw_file

logger = logging.getLogger(__name__)


def select_prognostic_genes(
    de_results: pd.DataFrame,
    survival_df: pd.DataFrame,
    X_log: pd.DataFrame,
    *,
    fdr_threshold: float = 0.05,
    n_genes: int = 50,
) -> list[str]:
    """
    Univariate Cox regression per significant DE gene; return the top n_genes by p-value.

    Only genes that are significant in at least one subtype comparison (padj < fdr_threshold)
    are screened. For each candidate gene we fit a CoxPHFitter with just that gene's
    log-expression as the covariate.
    """
    sig_genes = (
        de_results.loc[de_results["padj"] < fdr_threshold, "gene"]
        .unique()
        .tolist()
    )
    available = [g for g in sig_genes if g in X_log.columns]
    logger.info("Screening %d DE genes for univariate Cox significance ...", len(available))

    shared_samples = X_log.index.intersection(survival_df.index)
    surv_sub = survival_df.loc[shared_samples, ["OS_time", "OS_event"]].copy()

    pvals: list[tuple[str, float]] = []
    for gene in available:
        expr = X_log.loc[shared_samples, gene]
        cox_df = surv_sub.copy()
        cox_df["expr"] = expr.values
        cox_df = cox_df.dropna()
        if cox_df["expr"].std() == 0 or len(cox_df) < 10:
            continue
        try:
            cph = CoxPHFitter()
            cph.fit(cox_df, duration_col="OS_time", event_col="OS_event", show_progress=False)
            p = float(cph.summary.loc["expr", "p"])
            pvals.append((gene, p))
        except Exception:
            continue

    pvals.sort(key=lambda x: x[1])
    selected = [g for g, _ in pvals[:n_genes]]
    logger.info("Selected %d prognostic genes (top by univariate Cox p-value)", len(selected))
    return selected


def fit_cox_signature(
    X_log: pd.DataFrame,
    survival_df: pd.DataFrame,
    gene_ids: list[str],
    *,
    penalizer: float = 0.1,
) -> CoxPHFitter:
    """Fit a regularized Cox model on the selected gene_ids."""
    shared_samples = X_log.index.intersection(survival_df.index)
    available_genes = [g for g in gene_ids if g in X_log.columns]

    cox_df = X_log.loc[shared_samples, available_genes].copy()
    cox_df["OS_time"] = survival_df.loc[shared_samples, "OS_time"].values
    cox_df["OS_event"] = survival_df.loc[shared_samples, "OS_event"].values
    cox_df = cox_df.dropna()

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(cox_df, duration_col="OS_time", event_col="OS_event", show_progress=False)
    logger.info(
        "Cox signature: %d genes, C-index = %.3f",
        len(available_genes),
        cph.concordance_index_,
    )
    return cph


def compute_risk_score(
    X_log: pd.DataFrame,
    cox_fitter: CoxPHFitter,
    gene_ids: list[str],
) -> pd.Series:
    """
    Compute the log partial hazard (risk score) for each sample.

    Higher score = higher predicted hazard (worse prognosis).
    """
    available_genes = [g for g in gene_ids if g in X_log.columns]
    expr = X_log[available_genes].copy()

    log_hazard = expr.dot(cox_fitter.params_.loc[available_genes])
    log_hazard.name = "risk_score"
    logger.info(
        "Risk scores: %d samples, range [%.3f, %.3f]",
        len(log_hazard),
        log_hazard.min(),
        log_hazard.max(),
    )
    return log_hazard


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    X_raw = pd.read_csv(interim_file(InterimFile.X), index_col=0)
    X_log = log2_normalize(X_raw)

    _raw = pd.read_csv(interim_file(InterimFile.LABELS), index_col=0).squeeze()
    labels: pd.Series = _raw  # type: ignore[assignment]

    surv_raw = pd.read_csv(raw_file(DataFile.SURVIVAL), sep="\t", index_col=0)
    survival_df = merge_survival(labels=labels, raw_survival_df=surv_raw)

    de = pd.read_csv(interim_file(InterimFile.DE_RESULTS))

    genes = select_prognostic_genes(
        de_results=de,
        survival_df=survival_df,
        X_log=X_log,
        fdr_threshold=0.05,
        n_genes=50,
    )

    genes_series = pd.Series(genes, name="gene")
    out_genes = interim_file(InterimFile.PROGNOSTIC_GENES)
    genes_series.to_csv(out_genes, index=False)
    logger.info("Saved %s  (%d genes)", out_genes.name, len(genes))

    cph = fit_cox_signature(
        X_log=X_log,
        survival_df=survival_df,
        gene_ids=genes,
        penalizer=0.1,
    )

    risk_scores = compute_risk_score(X_log=X_log, cox_fitter=cph, gene_ids=genes)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_model = PROCESSED_DIR / "cox_signature.pkl"
    with open(out_model, "wb") as f:
        pickle.dump({"cph": cph, "genes": genes}, f)
    logger.info("Saved %s", out_model.name)

    out_scores = PROCESSED_DIR / "risk_scores.csv"
    risk_scores.to_csv(out_scores, header=True)
    logger.info("Saved %s", out_scores.name)


if __name__ == "__main__":
    main()
