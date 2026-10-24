"""
Unit tests for src.features.multiomics.
All tests use synthetic data; no real data files required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.features.multiomics import cnv_rna_correlation, joint_embedding


@pytest.fixture
def synthetic_cnv_rna() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    200 samples × 80 genes CNV matrix and 200 samples × 100 genes RNA matrix.
    The first 80 genes overlap between the two omics.
    """
    rng = np.random.default_rng(55)
    sample_ids = [f"s{i:03d}" for i in range(200)]
    cnv_genes = [f"G{i:03d}" for i in range(80)]
    rna_genes = [f"G{i:03d}" for i in range(100)]

    cnv_data = rng.integers(-2, 3, size=(200, 80)).astype(float)
    rna_data = rng.standard_normal((200, 100))
    rna_data[:, :80] += cnv_data * 0.5

    cnv = pd.DataFrame(cnv_data, index=sample_ids, columns=cnv_genes)
    rna = pd.DataFrame(rna_data, index=sample_ids, columns=rna_genes)
    return cnv, rna


class TestCnvRnaCorrelation:
    def test_returns_series(
        self, synthetic_cnv_rna: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        cnv, rna = synthetic_cnv_rna
        result = cnv_rna_correlation(cnv=cnv, rna_log=rna)
        assert isinstance(result, pd.Series)

    def test_values_in_minus1_plus1(
        self, synthetic_cnv_rna: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        cnv, rna = synthetic_cnv_rna
        result = cnv_rna_correlation(cnv=cnv, rna_log=rna)
        valid = result.dropna()
        assert ((valid >= -1.0 - 1e-9) & (valid <= 1.0 + 1e-9)).all()

    def test_indexed_by_shared_genes(
        self, synthetic_cnv_rna: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        cnv, rna = synthetic_cnv_rna
        result = cnv_rna_correlation(cnv=cnv, rna_log=rna)
        shared = cnv.columns.intersection(rna.columns)
        assert set(result.index) == set(shared)

    def test_positive_correlation_when_correlated(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.standard_normal(100)
        cnv = pd.DataFrame({"G": x})
        rna = pd.DataFrame({"G": x + rng.standard_normal(100) * 0.1})
        result = cnv_rna_correlation(cnv=cnv, rna_log=rna)
        assert result["G"] > 0.9


class TestJointEmbedding:
    def test_output_shape(
        self, synthetic_cnv_rna: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        cnv, rna = synthetic_cnv_rna
        emb = joint_embedding(cnv=cnv, rna_log=rna)
        assert emb.shape == (200, 2)

    def test_index_is_shared_samples(
        self, synthetic_cnv_rna: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        cnv, rna = synthetic_cnv_rna
        emb = joint_embedding(cnv=cnv, rna_log=rna)
        shared = cnv.index.intersection(rna.index)
        assert set(emb.index) == set(shared)

    def test_column_names(
        self, synthetic_cnv_rna: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        cnv, rna = synthetic_cnv_rna
        emb = joint_embedding(cnv=cnv, rna_log=rna)
        assert list(emb.columns) == ["UMAP1", "UMAP2"]

    def test_output_is_finite(
        self, synthetic_cnv_rna: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        cnv, rna = synthetic_cnv_rna
        emb = joint_embedding(cnv=cnv, rna_log=rna)
        assert np.isfinite(emb.values).all()

    def test_subset_of_samples_in_cnv(
        self, synthetic_cnv_rna: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        cnv, rna = synthetic_cnv_rna
        cnv_small = cnv.iloc[:100]
        emb = joint_embedding(cnv=cnv_small, rna_log=rna)
        assert len(emb) == 100


class TestLoadAlignCnv:
    def test_filters_to_requested_index(self, tmp_path: Path) -> None:
        """load_align_cnv should keep only samples present in X_index."""
        rng = np.random.default_rng(2)
        sample_ids = [f"s{i:03d}" for i in range(50)]
        genes = [f"G{j:03d}" for j in range(20)]

        cnv_data = rng.integers(-2, 3, size=(len(genes), len(sample_ids))).astype(float)
        cnv_raw = pd.DataFrame(cnv_data, index=genes, columns=sample_ids)

        raw_path = tmp_path / "TCGA-BRCA.gistic2thresholded.tsv.gz"
        cnv_raw.to_csv(raw_path, sep="\t")

        X_index = pd.Index(sample_ids[:20])

        from src.features.multiomics import load_align_cnv

        with patch("src.features.multiomics.raw_file", return_value=raw_path):
            cnv_aligned = load_align_cnv(X_index)

        assert set(cnv_aligned.index) == set(X_index)
        assert cnv_aligned.shape[0] == 20
