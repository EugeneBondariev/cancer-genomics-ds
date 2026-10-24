"""
Unit tests for src.models.survival_prediction.
All tests use synthetic data; no real data files required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.survival_prediction import (
    compute_risk_score,
    fit_cox_signature,
    select_prognostic_genes,
)


@pytest.fixture
def synthetic_prognostic_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    150 samples × 60 genes log-expression; matching survival DataFrame;
    DE results with 30 significant genes (padj < 0.05).
    """
    rng = np.random.default_rng(11)
    n_samples, n_genes = 150, 60
    sample_ids = [f"s{i:03d}" for i in range(n_samples)]
    genes = [f"G{i:03d}" for i in range(n_genes)]

    X_log = pd.DataFrame(
        rng.standard_normal((n_samples, n_genes)),
        index=sample_ids,
        columns=genes,
    )

    os_time = rng.exponential(scale=800, size=n_samples)
    os_event = rng.integers(0, 2, size=n_samples).astype(float)
    survival_df = pd.DataFrame(
        {"OS_time": os_time, "OS_event": os_event},
        index=sample_ids,
    )

    sig_genes = genes[:30]
    non_sig_genes = genes[30:]
    rows = []
    for gene in sig_genes:
        rows.append(
            {
                "gene": gene,
                "subtype": "Basal",
                "log2fc": rng.uniform(1, 3),
                "pval": rng.uniform(0.001, 0.01),
                "padj": rng.uniform(0.001, 0.04),
            }
        )
    for gene in non_sig_genes:
        rows.append(
            {
                "gene": gene,
                "subtype": "Basal",
                "log2fc": rng.uniform(-0.1, 0.1),
                "pval": rng.uniform(0.3, 1.0),
                "padj": rng.uniform(0.3, 1.0),
            }
        )
    de_results = pd.DataFrame(rows)

    return X_log, survival_df, de_results


class TestSelectPrognosticGenes:
    def test_returns_list(
        self,
        synthetic_prognostic_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        X_log, surv, de = synthetic_prognostic_data
        genes = select_prognostic_genes(
            de_results=de,
            survival_df=surv,
            X_log=X_log,
            n_genes=20,
        )
        assert isinstance(genes, list)

    def test_returns_le_n_genes(
        self,
        synthetic_prognostic_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        X_log, surv, de = synthetic_prognostic_data
        for n in [5, 10, 20]:
            genes = select_prognostic_genes(
                de_results=de,
                survival_df=surv,
                X_log=X_log,
                n_genes=n,
            )
            assert len(genes) <= n

    def test_genes_are_subset_of_X_columns(
        self,
        synthetic_prognostic_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        X_log, surv, de = synthetic_prognostic_data
        genes = select_prognostic_genes(
            de_results=de,
            survival_df=surv,
            X_log=X_log,
            n_genes=15,
        )
        assert all(g in X_log.columns for g in genes)


class TestComputeRiskScore:
    def test_same_index_as_X(
        self,
        synthetic_prognostic_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        X_log, surv, de = synthetic_prognostic_data
        genes = select_prognostic_genes(
            de_results=de,
            survival_df=surv,
            X_log=X_log,
            n_genes=10,
        )
        cph = fit_cox_signature(
            X_log=X_log,
            survival_df=surv,
            gene_ids=genes,
            penalizer=0.1,
        )
        risk = compute_risk_score(X_log=X_log, cox_fitter=cph, gene_ids=genes)
        assert list(risk.index) == list(X_log.index)

    def test_risk_scores_are_finite(
        self,
        synthetic_prognostic_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        X_log, surv, de = synthetic_prognostic_data
        genes = select_prognostic_genes(
            de_results=de,
            survival_df=surv,
            X_log=X_log,
            n_genes=10,
        )
        cph = fit_cox_signature(
            X_log=X_log,
            survival_df=surv,
            gene_ids=genes,
            penalizer=0.1,
        )
        risk = compute_risk_score(X_log=X_log, cox_fitter=cph, gene_ids=genes)
        assert np.isfinite(risk.values).all()

    def test_length_matches_samples(
        self,
        synthetic_prognostic_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ) -> None:
        X_log, surv, de = synthetic_prognostic_data
        genes = select_prognostic_genes(
            de_results=de,
            survival_df=surv,
            X_log=X_log,
            n_genes=10,
        )
        cph = fit_cox_signature(
            X_log=X_log,
            survival_df=surv,
            gene_ids=genes,
            penalizer=0.1,
        )
        risk = compute_risk_score(X_log=X_log, cox_fitter=cph, gene_ids=genes)
        assert len(risk) == len(X_log)
