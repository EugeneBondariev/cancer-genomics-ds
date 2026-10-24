from __future__ import annotations

from pathlib import Path

from src.constants import DataFile, InterimFile

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"
EXTERNAL_DIR = DATA / "external"
INTERIM_DIR = DATA / "interim"
PROCESSED_DIR = DATA / "processed"


def raw_file(dataset: DataFile) -> Path:
    return RAW_DIR / f"{dataset}.tsv.gz"

def interim_file(dataset: InterimFile) -> Path:
    return INTERIM_DIR / f"{dataset}.csv.gz"
