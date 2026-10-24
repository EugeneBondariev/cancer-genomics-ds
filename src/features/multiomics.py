"""
Multi-omics integration: CNV + RNA expression for TCGA BRCA samples.

Aligns copy-number variation (GISTIC2) data to the RNA-seq sample index,
computes CNV–RNA correlations per gene, and builds a joint UMAP embedding.

Inputs:  DataFile.CNV (raw), InterimFile.X
Outputs: InterimFile.CNV_FILTERED

Run: python -m src.features.multiomics
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from src.constants import DataFile, InterimFile
from src.paths import interim_file, raw_file

logger = logging.getLogger(__name__)


def load_align_cnv(X_index: pd.Index) -> pd.DataFrame:
    """
    Load GISTIC2-thresholded CNV matrix and keep only samples present in X_index.

    The raw CNV file has genes as rows and samples as columns; we transpose to
    samples × genes to match the RNA convention.
    """
    cnv_raw = pd.read_csv(raw_file(DataFile.CNV), sep="\t", index_col=0)
    cnv = cnv_raw.T

    shared = X_index.intersection(cnv.index)
    if len(shared) == 0:
        raise ValueError("No shared samples between X_index and CNV file.")

    cnv_aligned = cnv.loc[shared]
    logger.info(
        "CNV: %d × %d after aligning to %d RNA samples (%d shared)",
        *cnv_aligned.shape,
        len(X_index),
        len(shared),
    )
    return cnv_aligned


def cnv_rna_correlation(cnv: pd.DataFrame, rna_log: pd.DataFrame) -> pd.Series:
    """
    Per-gene Pearson correlation between CNV and log-normalized RNA for shared genes.

    Both DataFrames must be indexed by sample. Returns a Series indexed by gene,
    values in [-1, 1].
    """
    shared_genes = cnv.columns.intersection(rna_log.columns)
    shared_samples = cnv.index.intersection(rna_log.index)

    if len(shared_genes) == 0 or len(shared_samples) == 0:
        raise ValueError("No shared genes or samples between CNV and RNA data.")

    cnv_sub = cnv.loc[shared_samples, shared_genes].values.astype(float)
    rna_sub = rna_log.loc[shared_samples, shared_genes].values.astype(float)

    correlations: list[float] = []
    for j in range(cnv_sub.shape[1]):
        c = cnv_sub[:, j]
        r = rna_sub[:, j]
        std_c, std_r = c.std(), r.std()
        if std_c == 0 or std_r == 0:
            correlations.append(float("nan"))
        else:
            correlations.append(float(np.corrcoef(c, r)[0, 1]))

    corr = pd.Series(correlations, index=shared_genes, name="cnv_rna_correlation")
    logger.info(
        "CNV-RNA correlation: %d genes, median r = %.3f",
        len(corr.dropna()),
        corr.median(),
    )
    return corr


def joint_embedding(cnv: pd.DataFrame, rna_log: pd.DataFrame) -> pd.DataFrame:
    """
    Build a joint UMAP embedding from CNV + RNA data.

    Each omic is scaled to zero mean / unit variance independently before
    concatenation so neither dominates the embedding due to scale differences.
    Returns a DataFrame with columns UMAP1, UMAP2 indexed by sample.
    """
    shared_samples = cnv.index.intersection(rna_log.index)
    cnv_sub = cnv.loc[shared_samples]
    rna_sub = rna_log.loc[shared_samples]

    scaler = StandardScaler()
    cnv_scaled = scaler.fit_transform(cnv_sub.values.astype(float))
    rna_scaled = scaler.fit_transform(rna_sub.values.astype(float))

    joint = np.hstack([cnv_scaled, rna_scaled])
    logger.info(
        "Joint embedding: %d samples, %d features (%d CNV + %d RNA)",
        joint.shape[0],
        joint.shape[1],
        cnv_sub.shape[1],
        rna_sub.shape[1],
    )

    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    coords = reducer.fit_transform(joint)

    return pd.DataFrame(coords, index=shared_samples, columns=["UMAP1", "UMAP2"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    X_raw = pd.read_csv(interim_file(InterimFile.X), index_col=0)
    cnv = load_align_cnv(X_raw.index)

    out = interim_file(InterimFile.CNV_FILTERED)
    cnv.to_csv(out)
    logger.info("Saved %s  (%d × %d)", out.name, *cnv.shape)


if __name__ == "__main__":
    main()
