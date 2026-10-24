from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.differential_expression import (
    log2_normalize,
    run_one_vs_rest,
    top_de_genes,
)


class TestLog2Normalize:
    def test_zero_maps_to_zero(self) -> None:
        X = pd.DataFrame({"g": [0.0]})
        assert log2_normalize(X).iloc[0, 0] == pytest.approx(0.0)

    def test_one_maps_to_one(self) -> None:
        # log2(1 + 1) = log2(2) = 1.0
        X = pd.DataFrame({"g": [1.0]})
        assert log2_normalize(X).iloc[0, 0] == pytest.approx(1.0)

    def test_shape_preserved(self) -> None:
        X = pd.DataFrame(np.ones((5, 8)))
        assert log2_normalize(X).shape == (5, 8)

    def test_monotone_in_counts(self) -> None:
        counts = np.array([[0.0, 1.0, 10.0, 100.0]])
        result = log2_normalize(pd.DataFrame(counts))
        assert (result.diff(axis=1).iloc[0, 1:] > 0).all()

    def test_index_and_columns_preserved(self) -> None:
        X = pd.DataFrame(
            np.ones((3, 4)),
            index=["a", "b", "c"],
            columns=["g1", "g2", "g3", "g4"],
        )
        out = log2_normalize(X)
        assert list(out.index) == list(X.index)
        assert list(out.columns) == list(X.columns)


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
