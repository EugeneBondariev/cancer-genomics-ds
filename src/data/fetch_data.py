from __future__ import annotations
"""
Downloads TCGA BRCA datasets from UCSC Xena hubs.
GDC hub: https://xenabrowser.net/datapages/?cohort=GDC%20TCGA%20Breast%20Cancer%20(BRCA)
TCGA legacy hub: https://tcga.xenahubs.net (PAM50 clinical matrix)
Run: python -m src.data.fetch_data
"""

import gzip
import logging

import requests

from src.constants import DataFile
from src.paths import RAW_DIR, raw_file

logger = logging.getLogger(__name__)

_GDC_BASE = "https://gdc.xenahubs.net/download"

# PAM50 subtypes are in the legacy TCGA hub, not GDC — different URL pattern.
_TCGA_LEGACY = {
    DataFile.PAM50_CLINICAL: "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix",
}


def _fetch(dataset: DataFile) -> None:
    dest = raw_file(dataset)
    if dest.exists():
        logger.info("Already present — skipping %s", dest.name)
        return
    url = _TCGA_LEGACY.get(dataset) or f"{_GDC_BASE}/{dest.name}"
    logger.info("Downloading %s ...", dataset.name)
    needs_gzip = dataset in _TCGA_LEGACY
    opener = gzip.open if needs_gzip else open
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with opener(dest, "wb") as f:
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
