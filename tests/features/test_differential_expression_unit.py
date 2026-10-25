from __future__ import annotations

from itertools import combinations

import pandas as pd
import pytest

from src.features.differential_expression import (
    run_one_vs_rest,
    run_pairwise,
    top_de_genes,
)


class TestRunOneVsRest:
    def test_row_count(self, synthetic_expression: tuple[pd.DataFrame, pd.Series]) -> None:
        X, labels = synthetic_expression
        result = run_one_vs_rest(X=X, labels=labels)
        assert len(result) == X.shape[1] * labels.nunique()

    def test_required_columns(self, synthetic_expression: tuple[pd.DataFrame, pd.Series]) -> None:
        X, labels = synthetic_expression
        result = run_one_vs_rest(X=X, labels=labels)
        required = {
            "gene", "subtype", "log2fc", "pval", "padj",
            "mean_subtype", "mean_rest", "n_subtype", "n_rest", "significant",
        }
        assert required.issubset(result.columns)

    def test_padj_in_unit_interval(self, synthetic_expression: tuple[pd.DataFrame, pd.Series]) -> None:
        X, labels = synthetic_expression
        result = run_one_vs_rest(X=X, labels=labels)
        assert (result["padj"] >= 0).all() and (result["padj"] <= 1).all()

    def test_significant_flag_consistent(
        self, synthetic_expression: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, labels = synthetic_expression
        result = run_one_vs_rest(X=X, labels=labels, fdr_threshold=0.05, log2fc_threshold=1.0)
        expected = (result["padj"] < 0.05) & (result["log2fc"].abs() >= 1.0)
        pd.testing.assert_series_equal(result["significant"], expected, check_names=False)

    def test_detects_marker_genes(
        self, synthetic_expression: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, labels = synthetic_expression
        result = run_one_vs_rest(X=X, labels=labels, fdr_threshold=0.05, log2fc_threshold=0.5)
        markers = {f"GENE_{i:03d}" for i in range(10)}
        detected = set(result.loc[(result["subtype"] == "A") & result["significant"], "gene"])
        assert len(detected & markers) >= 8, (
            f"Expected ≥8 of 10 markers; got {detected & markers}"
        )

    def test_raises_on_misaligned_index(
        self, synthetic_expression: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, labels = synthetic_expression
        bad_labels = labels.set_axis([f"x{i}" for i in range(len(labels))])
        with pytest.raises(ValueError, match="same index"):
            run_one_vs_rest(X=X, labels=bad_labels)

    def test_raises_on_shuffled_index(
        self, synthetic_expression: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, labels = synthetic_expression
        shuffled_labels = labels.sample(frac=1, random_state=0)
        with pytest.raises(ValueError, match="same index"):
            run_one_vs_rest(X=X, labels=shuffled_labels)

    def test_raises_on_subset_index(
        self, synthetic_expression: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, labels = synthetic_expression
        subset_labels = labels.iloc[:-10]
        with pytest.raises(ValueError, match="same index"):
            run_one_vs_rest(X=X, labels=subset_labels)

    def test_n_subtype_correct(
        self, synthetic_expression: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, labels = synthetic_expression
        result = run_one_vs_rest(X=X, labels=labels)
        for subtype in labels.unique():
            expected_n = int((labels == subtype).sum())
            rows = result.loc[result["subtype"] == subtype, "n_subtype"]
            assert (rows == expected_n).all()

    def test_all_subtypes_present(
        self, synthetic_expression: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, labels = synthetic_expression
        result = run_one_vs_rest(X=X, labels=labels)
        assert set(result["subtype"].unique()) == set(labels.unique())

    def test_all_genes_present_per_subtype(
        self, synthetic_expression: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, labels = synthetic_expression
        result = run_one_vs_rest(X=X, labels=labels)
        for subtype in labels.unique():
            genes_in_subtype = set(result.loc[result["subtype"] == subtype, "gene"])
            assert genes_in_subtype == set(X.columns)


class TestTopDeGenes:
    def test_respects_n(self, synthetic_expression: tuple[pd.DataFrame, pd.Series]) -> None:
        X, labels = synthetic_expression
        result = run_one_vs_rest(X=X, labels=labels, fdr_threshold=0.05, log2fc_threshold=0.5)
        top = top_de_genes(result, n=3)
        assert top.groupby("subtype").size().max() <= 3

    def test_direction_up_only_positive_fc(
        self, synthetic_expression: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, labels = synthetic_expression
        result = run_one_vs_rest(X=X, labels=labels, fdr_threshold=0.05, log2fc_threshold=0.5)
        top = top_de_genes(result, n=20, direction="up")
        assert (top["log2fc"] > 0).all()

    def test_direction_down_only_negative_fc(
        self, synthetic_expression: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, labels = synthetic_expression
        result = run_one_vs_rest(X=X, labels=labels, fdr_threshold=0.05, log2fc_threshold=0.5)
        top = top_de_genes(result, n=20, direction="down")
        assert (top["log2fc"] < 0).all()

    def test_only_significant_genes_returned(
        self, synthetic_expression: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, labels = synthetic_expression
        result = run_one_vs_rest(X=X, labels=labels, fdr_threshold=0.05, log2fc_threshold=1.0)
        top = top_de_genes(result, n=50)
        assert top["significant"].all()

    def test_empty_when_nothing_significant(
        self, synthetic_expression: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, labels = synthetic_expression
        result = run_one_vs_rest(X=X, labels=labels, fdr_threshold=0.05, log2fc_threshold=1.0)
        # force everything non-significant
        result["significant"] = False
        top = top_de_genes(result, n=10)
        assert len(top) == 0


class TestRunPairwise:
    def test_row_count(self, synthetic_expression: tuple[pd.DataFrame, pd.Series]) -> None:
        X, labels = synthetic_expression
        result = run_pairwise(X=X, labels=labels)
        n_pairs = len(list(combinations(labels.unique(), 2)))
        assert len(result) == n_pairs * X.shape[1]

    def test_required_columns(self, synthetic_expression: tuple[pd.DataFrame, pd.Series]) -> None:
        X, labels = synthetic_expression
        result = run_pairwise(X=X, labels=labels)
        required = {"gene", "subtype_a", "subtype_b", "log2fc", "pval", "padj",
                    "mean_a", "mean_b", "n_a", "n_b", "significant"}
        assert required.issubset(result.columns)

    def test_padj_in_unit_interval(self, synthetic_expression: tuple[pd.DataFrame, pd.Series]) -> None:
        X, labels = synthetic_expression
        result = run_pairwise(X=X, labels=labels)
        assert (result["padj"] >= 0).all() and (result["padj"] <= 1).all()

    def test_significant_flag_consistent(
        self, synthetic_expression: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, labels = synthetic_expression
        result = run_pairwise(X=X, labels=labels, fdr_threshold=0.05, log2fc_threshold=1.0)
        expected = (result["padj"] < 0.05) & (result["log2fc"].abs() >= 1.0)
        pd.testing.assert_series_equal(result["significant"], expected, check_names=False)

    def test_all_pairs_present(self, synthetic_expression: tuple[pd.DataFrame, pd.Series]) -> None:
        X, labels = synthetic_expression
        result = run_pairwise(X=X, labels=labels)
        expected_pairs = set(combinations(sorted(labels.unique()), 2))
        actual_pairs = set(zip(result["subtype_a"], result["subtype_b"]))
        assert expected_pairs == actual_pairs

    def test_n_a_n_b_correct(self, synthetic_expression: tuple[pd.DataFrame, pd.Series]) -> None:
        X, labels = synthetic_expression
        result = run_pairwise(X=X, labels=labels)
        for subtype_a, subtype_b in combinations(sorted(labels.unique()), 2):
            rows = result.loc[(result["subtype_a"] == subtype_a) & (result["subtype_b"] == subtype_b)]
            assert (rows["n_a"] == int((labels == subtype_a).sum())).all()
            assert (rows["n_b"] == int((labels == subtype_b).sum())).all()

    def test_detects_markers_in_pairwise(
        self, synthetic_expression: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, labels = synthetic_expression
        result = run_pairwise(X=X, labels=labels, fdr_threshold=0.05, log2fc_threshold=0.5)
        markers = {f"GENE_{i:03d}" for i in range(10)}
        detected = set(result.loc[
            (result["subtype_a"] == "A") & (result["subtype_b"] == "B")
            & result["significant"] & (result["log2fc"] > 0),
            "gene",
        ])
        assert len(detected & markers) >= 8, f"Expected ≥8 of 10 markers; got {detected & markers}"

    def test_raises_on_misaligned_index(
        self, synthetic_expression: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, labels = synthetic_expression
        bad_labels = labels.set_axis([f"x{i}" for i in range(len(labels))])
        with pytest.raises(ValueError, match="same index"):
            run_pairwise(X=X, labels=bad_labels)
