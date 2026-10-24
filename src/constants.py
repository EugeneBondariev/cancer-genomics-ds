from __future__ import annotations

from enum import StrEnum


class DataFile(StrEnum):
    STAR_COUNTS = "TCGA-BRCA.star_counts"
    SURVIVAL = "TCGA-BRCA.survival"
    PAM50_CLINICAL = "TCGA-BRCA.pam50-clinical"


class InterimFile(StrEnum):
    X = "star_filtered"
    LABELS = "labels"