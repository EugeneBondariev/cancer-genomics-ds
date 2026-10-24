"""
Unit tests for src.features.pathway_enrichment.
All tests use synthetic data; no real data files required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.pathway_enrichment import build_ranked_list


@pytest.fixture
def synthetic_de_and_annotation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Minimal synthetic DE results (3 subtypes × 50 genes) and gene annotation mapping.
    """
    rng = np.random.default_rng(33)
    genes = [f"ENSG{i:011d}.1" for i in range(50)]
    subtypes = ["Basal", "LumA", "LumB"]

    rows = []
    for subtype in subtypes:
        for gene in genes:
            rows.append(
                {
                    "gene": gene,
                    "subtype": subtype,
                    "log2fc": rng.uniform(-3, 3),
                    "pval": rng.uniform(0, 0.1),
                    "padj": rng.uniform(0, 0.2),
                }
            )
    de = pd.DataFrame(rows)

    annotation = pd.DataFrame(
        {
            "ensembl_id": [f"ENSG{i:011d}" for i in range(50)],
            "symbol": [f"GENE{i}" for i in range(40)] + [None] * 10,
            "name": [f"Gene {i} name" for i in range(40)] + [None] * 10,
        }
    )
    return de, annotation


class TestBuildRankedList:
    def test_returns_series(
        self,
        synthetic_de_and_annotation: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        de, ann = synthetic_de_and_annotation
        ranked = build_ranked_list(de_results=de, subtype="Basal", gene_symbols=ann)
        assert isinstance(ranked, pd.Series)

    def test_sorted_descending(
        self,
        synthetic_de_and_annotation: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        de, ann = synthetic_de_and_annotation
        ranked = build_ranked_list(de_results=de, subtype="Basal", gene_symbols=ann)
        assert (ranked.diff().dropna() <= 0).all()

    def test_unmapped_genes_dropped(
        self,
        synthetic_de_and_annotation: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        de, ann = synthetic_de_and_annotation
        ranked = build_ranked_list(de_results=de, subtype="Basal", gene_symbols=ann)
        assert len(ranked) == 40

    def test_score_direction_positive_log2fc(
        self,
        synthetic_de_and_annotation: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        de, ann = synthetic_de_and_annotation
        de_pos = de.copy()
        de_pos["log2fc"] = 2.0
        de_pos["padj"] = 0.01
        ranked = build_ranked_list(de_results=de_pos, subtype="Basal", gene_symbols=ann)
        assert (ranked > 0).all()

    def test_score_direction_negative_log2fc(
        self,
        synthetic_de_and_annotation: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        de, ann = synthetic_de_and_annotation
        de_neg = de.copy()
        de_neg["log2fc"] = -2.0
        de_neg["padj"] = 0.01
        ranked = build_ranked_list(de_results=de_neg, subtype="Basal", gene_symbols=ann)
        assert (ranked < 0).all()

    def test_raises_on_unknown_subtype(
        self,
        synthetic_de_and_annotation: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        de, ann = synthetic_de_and_annotation
        with pytest.raises(ValueError, match="No DE results"):
            build_ranked_list(de_results=de, subtype="UNKNOWN", gene_symbols=ann)

    def test_indexed_by_symbol(
        self,
        synthetic_de_and_annotation: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        de, ann = synthetic_de_and_annotation
        ranked = build_ranked_list(de_results=de, subtype="Basal", gene_symbols=ann)
        expected_symbols = {f"GENE{i}" for i in range(40)}
        assert set(ranked.index) == expected_symbols
