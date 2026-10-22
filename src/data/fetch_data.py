from __future__ import annotations
"""
Downloads TCGA BRCA datasets from the UCSC Xena GDC hub.
Source: https://xenabrowser.net/datapages/?cohort=GDC%20TCGA%20Breast%20Cancer%20(BRCA)
Run: python -m src.data.fetch_data
"""

import logging

import requests

from src.constants import DataFile
from src.paths import RAW_DIR, raw_file

logger = logging.getLogger(__name__)

_GDC_BASE = "https://gdc.xenahubs.net/download"


def _fetch(dataset: DataFile) -> None:
    dest = raw_file(dataset)
    if dest.exists():
        logger.info("Already present — skipping %s", dest.name)
        return
    logger.info("Downloading %s ...", dataset.name)
    with requests.get(f"{_GDC_BASE}/{dest.name}", stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
    mb = dest.stat().st_size / 1_048_576
    logger.info("Saved %s (%.1f MB)", dest.name, mb)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in DataFile:
        _fetch(dataset)


if __name__ == "__main__":
    main()
