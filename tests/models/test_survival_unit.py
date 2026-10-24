"""
Unit tests for src.models.survival.
Uses synthetic survival data; no real data files required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.survival import fit_km, logrank_pairwise, merge_survival


@pytest.fixture
def synthetic_survival() -> tuple[pd.Series, pd.DataFrame]:
    """
    100 patients in 2 groups (A=50, B=50) with different survival distributions.
    Labels use short IDs (TCGA-XX-0001); survival uses full IDs (TCGA-XX-0001-01A).
    """
    rng = np.random.default_rng(99)
    n = 100
    short_ids = [f"TCGA-XX-{i:04d}" for i in range(n)]
    full_ids = [f"TCGA-XX-{i:04d}-01A" for i in range(n)]

    groups = ["A"] * 50 + ["B"] * 50
    labels = pd.Series(groups, index=pd.Index(short_ids), name="subtype")

    os_time = np.concatenate(
        [
            rng.exponential(scale=1000, size=50),
            rng.exponential(scale=500, size=50),
        ]
    )
    os_event = rng.integers(0, 2, size=n).astype(float)

    surv_df = pd.DataFrame(
        {"OS.time": os_time, "OS": os_event},
        index=pd.Index(full_ids),
    )
    return labels, surv_df


class TestMergeSurvival:
    def test_output_columns(
        self, synthetic_survival: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        labels, surv_df = synthetic_survival
        merged = merge_survival(labels=labels, raw_survival_df=surv_df)
        assert {"subtype", "OS_time", "OS_event"}.issubset(merged.columns)

    def test_drops_missing_os_time(
        self, synthetic_survival: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        labels, surv_df = synthetic_survival
        surv_df_with_nan = surv_df.copy()
        surv_df_with_nan.iloc[:5, surv_df_with_nan.columns.get_loc("OS.time")] = np.nan
        merged = merge_survival(labels=labels, raw_survival_df=surv_df_with_nan)
        assert merged["OS_time"].isna().sum() == 0

    def test_fewer_rows_when_nans_present(
        self, synthetic_survival: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        labels, surv_df = synthetic_survival
        full_merged = merge_survival(labels=labels, raw_survival_df=surv_df)
        surv_df_with_nan = surv_df.copy()
        surv_df_with_nan.iloc[:5, surv_df_with_nan.columns.get_loc("OS.time")] = np.nan
        partial_merged = merge_survival(labels=labels, raw_survival_df=surv_df_with_nan)
        assert len(partial_merged) < len(full_merged)

    def test_all_samples_have_matching_subtype(
        self, synthetic_survival: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        labels, surv_df = synthetic_survival
        merged = merge_survival(labels=labels, raw_survival_df=surv_df)
        assert merged["subtype"].isin(["A", "B"]).all()

    def test_joins_on_trimmed_index(
        self, synthetic_survival: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        labels, surv_df = synthetic_survival
        merged = merge_survival(labels=labels, raw_survival_df=surv_df)
        assert len(merged) == len(labels)


class TestFitKm:
    def test_returns_kmf(
        self, synthetic_survival: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        from lifelines import KaplanMeierFitter

        labels, surv_df = synthetic_survival
        merged = merge_survival(labels=labels, raw_survival_df=surv_df)
        kmf = fit_km(df=merged, subtype="A")
        assert isinstance(kmf, KaplanMeierFitter)

    def test_label_attribute(
        self, synthetic_survival: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        labels, surv_df = synthetic_survival
        merged = merge_survival(labels=labels, raw_survival_df=surv_df)
        kmf = fit_km(df=merged, subtype="A")
        assert kmf._label == "A"


class TestLogrankPairwise:
    def test_returns_symmetric_matrix(
        self, synthetic_survival: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        labels, surv_df = synthetic_survival
        merged = merge_survival(labels=labels, raw_survival_df=surv_df)
        p_mat = logrank_pairwise(merged)
        assert p_mat.shape == (2, 2)

    def test_diagonal_is_nan(
        self, synthetic_survival: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        labels, surv_df = synthetic_survival
        merged = merge_survival(labels=labels, raw_survival_df=surv_df)
        p_mat = logrank_pairwise(merged)
        for lbl in p_mat.index:
            assert np.isnan(p_mat.loc[lbl, lbl])

    def test_off_diagonal_in_unit_interval(
        self, synthetic_survival: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        labels, surv_df = synthetic_survival
        merged = merge_survival(labels=labels, raw_survival_df=surv_df)
        p_mat = logrank_pairwise(merged)
        off_diag = p_mat.values[~np.eye(len(p_mat), dtype=bool)]
        valid = off_diag[~np.isnan(off_diag)]
        assert ((valid >= 0) & (valid <= 1)).all()

    def test_symmetric(
        self, synthetic_survival: tuple[pd.Series, pd.DataFrame]
    ) -> None:
        labels, surv_df = synthetic_survival
        merged = merge_survival(labels=labels, raw_survival_df=surv_df)
        p_mat = logrank_pairwise(merged)
        assert p_mat.loc["A", "B"] == pytest.approx(p_mat.loc["B", "A"], nan_ok=True)
