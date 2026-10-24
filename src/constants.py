from __future__ import annotations

from enum import StrEnum


class DataFile(StrEnum):
    STAR_COUNTS = "TCGA-BRCA.star_counts"
    SURVIVAL = "TCGA-BRCA.survival"
    PAM50_CLINICAL = "TCGA-BRCA.pam50-clinical"


class InterimFile(StrEnum):
    X = "star_filtered"
    LABELS = "labels"
    DE_RESULTS = "de_results"


# Ensembl base IDs (without version suffix) for canonical PAM50 subtype markers.
# Source: Parker et al. 2009 (PAM50 paper) and TCGA BRCA marker validation studies.
KNOWN_MARKERS: dict[str, list[str]] = {
    "Basal": ["ENSG00000012048", "ENSG00000186081", "ENSG00000143228", "ENSG00000166033"],  # BRCA1, KRT5, NUF2, CENPF
    "HER2":  ["ENSG00000141736", "ENSG00000140443", "ENSG00000132965"],                      # ERBB2, IGF1R, ALOX5AP
    "LumA":  ["ENSG00000091831", "ENSG00000129514", "ENSG00000107485", "ENSG00000082175"],   # ESR1, FOXA1, GATA3, PGR
    "LumB":  ["ENSG00000148773", "ENSG00000100526", "ENSG00000117724"],                       # MKI67, CDKN3, CENPF
    "Normal": ["ENSG00000117625", "ENSG00000169031"],                                          # RCBTB2, COL4A6
}
