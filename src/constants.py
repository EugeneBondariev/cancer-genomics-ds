from enum import StrEnum

class DataFile(StrEnum):
  CLINICAL =    "TCGA-BRCA.clinical"
  STAR_COUNTS = "TCGA-BRCA.star_counts"
  SURVIVAL =    "TCGA-BRCA.survival"