from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def synthetic_expression() -> tuple[pd.DataFrame, pd.Series]:
    """
    150 samples × 100 genes; 3 subtypes (A/B/C, 50 each).

    Genes 0–9 are strongly upregulated in A (injected ~100 extra NegBin counts
    on top of background ~10), giving a log2FC of ~3–4. All other genes are pure
    background. Used by all DGE unit tests.
    """
    rng = np.random.default_rng(42)
    n_per_group, n_genes = 50, 100

    index = pd.Index([f"s{i:03d}" for i in range(n_per_group * 3)])
    subtypes = ["A"] * n_per_group + ["B"] * n_per_group + ["C"] * n_per_group
    labels = pd.Series(subtypes, index=index, name="subtype")

    counts = rng.negative_binomial(n=10, p=0.5, size=(len(index), n_genes)).astype(float)
    counts[:n_per_group, :10] += rng.negative_binomial(
        n=100, p=0.5, size=(n_per_group, 10)
    )

    X = pd.DataFrame(
        counts,
        index=index,
        columns=[f"GENE_{i:03d}" for i in range(n_genes)],
    )
    return X, labels
