"""
Gene set enrichment analysis (GSEA pre-ranked) using gseapy.

Maps Ensembl IDs from DE results to HGNC symbols, builds a ranking score
(-log10(padj) * sign(log2fc)), and runs GSEApy prerank against MSigDB Hallmark.

Run: python -m src.features.pathway_enrichment
"""

from __future__ import annotations

import logging
import tempfile

import numpy as np
import pandas as pd

from src.constants import InterimFile
from src.paths import interim_file

logger = logging.getLogger(__name__)


def build_ranked_list(
    de_results: pd.DataFrame,
    subtype: str,
    gene_symbols: pd.DataFrame,
) -> pd.Series:
    """
    Build a pre-ranked gene list for GSEA.

    Filters DE results to the given subtype, maps ensembl_id → symbol,
    drops unmapped genes (no symbol), and computes:
        score = -log10(padj) * sign(log2fc)

    padj values of 0 are clipped to 1e-300 before log10.
    Returns a Series indexed by gene symbol, sorted descending.
    """
    sub = de_results.loc[de_results["subtype"] == subtype].copy()
    if sub.empty:
        raise ValueError(f"No DE results for subtype '{subtype}'")

    sub["ensembl_base"] = sub["gene"].str.split(".").str[0]

    sym_map = (
        gene_symbols[["ensembl_id", "symbol"]]
        .dropna(subset=["symbol"])
        .set_index("ensembl_id")["symbol"]
        .to_dict()
    )
    sub["symbol"] = sub["ensembl_base"].map(sym_map)
    sub = sub.dropna(subset=["symbol"])

    padj_clipped = sub["padj"].clip(lower=1e-300)
    sub = sub.copy()
    sub["score"] = -np.log10(padj_clipped) * np.sign(sub["log2fc"])

    ranked = (
        sub[["symbol", "score"]]
        .groupby("symbol")["score"]
        .mean()
        .sort_values(ascending=False)
    )
    logger.info("Ranked list for %s: %d genes", subtype, len(ranked))
    return ranked


def run_gsea_preranked(
    ranked: pd.Series,
    *,
    gene_sets: str = "MSigDB_Hallmark_2020",
    outdir: str = "tmp",
) -> pd.DataFrame:
    """
    Run GSEApy prerank analysis on a pre-ranked gene list.

    Returns a DataFrame with columns: Term, NES, pval, fdr.
    Returns an empty DataFrame if gseapy is unavailable or the API call fails.
    """
    try:
        import gseapy as gp
    except ImportError:
        logger.warning("gseapy not installed — returning empty GSEA results")
        return pd.DataFrame(columns=["Term", "NES", "pval", "fdr"])

    with tempfile.TemporaryDirectory() as tmpdir:
        actual_outdir = outdir if outdir != "tmp" else tmpdir
        try:
            res = gp.prerank(
                rnk=ranked,
                gene_sets=gene_sets,
                outdir=actual_outdir,
                min_size=15,
                max_size=500,
                permutation_num=100,
                seed=42,
                verbose=False,
            )
            df = res.res2d.rename(
                columns={"Term": "Term", "NES": "NES", "NOM p-val": "pval", "FDR q-val": "fdr"}
            )
            keep = ["Term", "NES", "pval", "fdr"]
            available = [c for c in keep if c in df.columns]
            result = df[available].reset_index(drop=True)
            logger.info("GSEA: %d gene sets tested", len(result))
            return result
        except Exception as exc:
            logger.warning("GSEApy prerank failed: %s — returning empty results", exc)
            return pd.DataFrame(columns=["Term", "NES", "pval", "fdr"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    de = pd.read_csv(interim_file(InterimFile.DE_RESULTS))
    annotation = pd.read_csv(interim_file(InterimFile.GENE_ANNOTATION))

    subtypes = sorted(de["subtype"].unique())
    all_results: list[pd.DataFrame] = []

    for subtype in subtypes:
        ranked = build_ranked_list(
            de_results=de, subtype=subtype, gene_symbols=annotation
        )
        gsea_df = run_gsea_preranked(ranked=ranked, gene_sets="MSigDB_Hallmark_2020")
        if not gsea_df.empty:
            gsea_df["subtype"] = subtype
            all_results.append(gsea_df)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        out = interim_file(InterimFile.DE_RESULTS).parent / "gsea_results.csv.gz"
        combined.to_csv(out, index=False)
        logger.info("Saved %s  (%d rows)", out.name, len(combined))
    else:
        logger.warning("No GSEA results produced.")


if __name__ == "__main__":
    main()
