"""
PCA and UMAP dimensionality reduction on filtered TCGA BRCA gene expression data.

Inputs:  InterimFile.X (samples × genes log-normalized counts)
Outputs: InterimFile.PCA_EMBEDDINGS, InterimFile.UMAP_EMBEDDINGS

Run: python -m src.features.dimensionality_reduction
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from src.constants import InterimFile
from src.paths import interim_file

logger = logging.getLogger(__name__)


def run_pca(X: pd.DataFrame, *, n_components: int = 50) -> tuple[pd.DataFrame, PCA]:
    """
    Standardize X and compute PCA embeddings.

    Returns (embeddings DataFrame with shape samples × n_components, fitted PCA object).
    Column names are PC1, PC2, …, PC{n_components}.
    """
    n_components = min(n_components, X.shape[0], X.shape[1])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X_scaled)

    cols = [f"PC{i + 1}" for i in range(n_components)]
    emb = pd.DataFrame(coords, index=X.index, columns=cols)

    logger.info(
        "PCA: %d components, top-10 explain %.1f%% variance",
        n_components,
        100 * pca.explained_variance_ratio_[:10].sum(),
    )
    return emb, pca


def run_umap(
    X: pd.DataFrame,
    *,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run UMAP on X (raw or PCA-reduced).

    Returns embeddings DataFrame with shape samples × n_components.
    Column names are UMAP1, UMAP2, … .
    """
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    coords = reducer.fit_transform(X.values)

    cols = [f"UMAP{i + 1}" for i in range(n_components)]
    emb = pd.DataFrame(coords, index=X.index, columns=cols)

    logger.info("UMAP: %d components on %d × %d input", n_components, *X.shape)
    return emb


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    X = pd.read_csv(interim_file(InterimFile.X), index_col=0)
    logger.info("Loaded X: %d samples × %d genes", *X.shape)

    pca_emb, _ = run_pca(X, n_components=50)
    out_pca = interim_file(InterimFile.PCA_EMBEDDINGS)
    pca_emb.to_csv(out_pca)
    logger.info("Saved %s", out_pca.name)

    umap_emb = run_umap(pca_emb, n_components=2)
    out_umap = interim_file(InterimFile.UMAP_EMBEDDINGS)
    umap_emb.to_csv(out_umap)
    logger.info("Saved %s", out_umap.name)


if __name__ == "__main__":
    main()
