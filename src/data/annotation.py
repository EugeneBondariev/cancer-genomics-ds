"""
Fetch HGNC gene symbols for Ensembl IDs via the MyGene.info REST API.

Queries in batches of 1000 using POST to https://mygene.info/v3/query.
Results are cached locally as a CSV to avoid re-fetching on every run.

Run: python -m src.data.annotation
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from src.constants import InterimFile
from src.paths import interim_file

logger = logging.getLogger(__name__)

_MYGENE_URL = "https://mygene.info/v3/query"
_BATCH_SIZE = 500
_RETRY_DELAY = 2.0


def fetch_gene_symbols(ensembl_ids: list[str], *, cache_path: Path) -> pd.DataFrame:
    """
    Map Ensembl base IDs to HGNC symbols and gene names via MyGene.info.

    If cache_path exists, load from disk instead of querying the API.
    Returns DataFrame with columns: ensembl_id, symbol, name.
    Unmapped IDs are included with NaN for symbol and name.
    """
    if cache_path.exists():
        logger.info("Loading gene annotation from cache: %s", cache_path.name)
        return pd.read_csv(cache_path)

    logger.info("Querying MyGene.info for %d Ensembl IDs ...", len(ensembl_ids))
    rows: list[dict] = []

    for start in range(0, len(ensembl_ids), _BATCH_SIZE):
        batch = ensembl_ids[start : start + _BATCH_SIZE]
        try:
            resp = requests.post(
                _MYGENE_URL,
                data={
                    "q": "\n".join(batch),
                    "scopes": "ensembl.gene",
                    "fields": "symbol,name",
                    "species": "human",
                    "size": 1,
                },
                timeout=60,
            )
            resp.raise_for_status()
            hits = resp.json()
        except requests.RequestException as exc:
            logger.warning("MyGene.info batch %d failed: %s — skipping batch", start, exc)
            for eid in batch:
                rows.append({"ensembl_id": eid, "symbol": None, "name": None})
            time.sleep(_RETRY_DELAY)
            continue

        found: set[str] = set()
        for hit in hits:
            query_id = hit.get("query", "")
            found.add(query_id)
            rows.append(
                {
                    "ensembl_id": query_id,
                    "symbol": hit.get("symbol"),
                    "name": hit.get("name"),
                }
            )
        for eid in batch:
            if eid not in found:
                rows.append({"ensembl_id": eid, "symbol": None, "name": None})

        logger.info("  batch %d–%d: %d mapped", start, start + len(batch), len(found))

    df = pd.DataFrame(rows, columns=["ensembl_id", "symbol", "name"])
    n_unmapped = int(df["symbol"].isna().sum())
    logger.info(
        "Mapped %d / %d Ensembl IDs to HGNC symbols (%.0f%% unmapped)",
        len(df) - n_unmapped,
        len(df),
        100 * n_unmapped / len(df) if df.shape[0] else 0,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("Saved annotation to %s  (%d rows)", cache_path.name, len(df))
    return df


def annotate_de_results(
    results: pd.DataFrame,
    annotation: pd.DataFrame,
    *,
    gene_col: str = "gene",
) -> pd.DataFrame:
    """
    Add a 'symbol' column to a DE results DataFrame by joining with annotation.

    annotation: DataFrame with columns 'ensembl_id' and 'symbol' (from fetch_gene_symbols).
    Returns a copy; rows without a mapping get NaN in 'symbol'.
    """
    out = results.copy()
    sym_map = annotation.dropna(subset=["symbol"]).set_index("ensembl_id")["symbol"]
    base_ids = out[gene_col].str.split(".").str[0]
    out["symbol"] = base_ids.map(sym_map)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    X = pd.read_csv(interim_file(InterimFile.X), index_col=0)
    raw_ids: list[str] = X.columns.tolist()
    ensembl_ids = [eid.split(".")[0] for eid in raw_ids]
    ensembl_ids = list(dict.fromkeys(ensembl_ids))

    cache = interim_file(InterimFile.GENE_ANNOTATION)
    fetch_gene_symbols(ensembl_ids=ensembl_ids, cache_path=cache)


if __name__ == "__main__":
    main()
