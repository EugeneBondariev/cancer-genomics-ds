"""
Unit tests for src.features.dimensionality_reduction.
All tests use synthetic data; no real data files required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.dimensionality_reduction import run_pca, run_umap


@pytest.fixture
def synthetic_X() -> pd.DataFrame:
    """200 samples × 150 genes with injected structure."""
    rng = np.random.default_rng(0)
    index = pd.Index([f"s{i:03d}" for i in range(200)])
    cols = [f"G{i:03d}" for i in range(150)]
    data = rng.standard_normal(size=(200, 150))
    return pd.DataFrame(data, index=index, columns=cols)


class TestRunPca:
    def test_output_shape(self, synthetic_X: pd.DataFrame) -> None:
        emb, _ = run_pca(synthetic_X, n_components=10)
        assert emb.shape == (200, 10)

    def test_index_preserved(self, synthetic_X: pd.DataFrame) -> None:
        emb, _ = run_pca(synthetic_X, n_components=5)
        assert list(emb.index) == list(synthetic_X.index)

    def test_column_names(self, synthetic_X: pd.DataFrame) -> None:
        n = 7
        emb, _ = run_pca(synthetic_X, n_components=n)
        assert list(emb.columns) == [f"PC{i}" for i in range(1, n + 1)]

    def test_explained_variance_ratio_sums_le_one(self, synthetic_X: pd.DataFrame) -> None:
        _, pca = run_pca(synthetic_X, n_components=20)
        assert pca.explained_variance_ratio_.sum() <= 1.0 + 1e-9

    def test_explained_variance_ratio_all_positive(self, synthetic_X: pd.DataFrame) -> None:
        _, pca = run_pca(synthetic_X, n_components=20)
        assert (pca.explained_variance_ratio_ >= 0).all()

    def test_n_components_capped_to_data_dims(self) -> None:
        rng = np.random.default_rng(1)
        small = pd.DataFrame(rng.standard_normal((10, 8)))
        emb, _ = run_pca(small, n_components=50)
        assert emb.shape[1] <= min(10, 8)

    def test_returns_pca_object(self, synthetic_X: pd.DataFrame) -> None:
        from sklearn.decomposition import PCA

        _, pca = run_pca(synthetic_X, n_components=5)
        assert isinstance(pca, PCA)


class TestRunUmap:
    def test_output_shape(self, synthetic_X: pd.DataFrame) -> None:
        emb = run_umap(synthetic_X, n_components=2)
        assert emb.shape == (200, 2)

    def test_index_preserved(self, synthetic_X: pd.DataFrame) -> None:
        emb = run_umap(synthetic_X, n_components=2)
        assert list(emb.index) == list(synthetic_X.index)

    def test_column_names(self, synthetic_X: pd.DataFrame) -> None:
        emb = run_umap(synthetic_X, n_components=2)
        assert list(emb.columns) == ["UMAP1", "UMAP2"]

    def test_output_is_finite(self, synthetic_X: pd.DataFrame) -> None:
        emb = run_umap(synthetic_X, n_components=2)
        assert np.isfinite(emb.values).all()
