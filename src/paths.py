from __future__ import annotations

from pathlib import Path

from src.constants import DataFile

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"
EXTERNAL = DATA / "external"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"


def raw_file(dataset: DataFile) -> Path:
    return RAW_DIR / f"{dataset}.tsv.gz"
