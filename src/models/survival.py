"""
Kaplan-Meier and Cox proportional hazard survival analysis for PAM50 BRCA subtypes.

Inputs:  InterimFile.LABELS, DataFile.SURVIVAL (raw TSV)
Outputs: InterimFile.SURVIVAL_MERGED

Run: python -m src.models.survival
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import pairwise_logrank_test

from src.constants import DataFile, InterimFile
from src.paths import interim_file, raw_file

logger = logging.getLogger(__name__)


def merge_survival(labels: pd.Series, raw_survival_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join PAM50 labels with OS.time / OS columns from the raw survival DataFrame.

    The survival index uses the full TCGA sample barcode (e.g. TCGA-XX-XXXX-01A),
    while labels use a shorter form. We strip everything from the third hyphen-separated
    token onward to align the two indices.

    Drops rows where OS.time is missing. Returns DataFrame with columns:
    subtype, OS_time, OS_event.
    """
    surv = raw_survival_df[["OS.time", "OS"]].copy()

    def _trim(s: str) -> str:
        return s[:-1] if s and s[-1].isalpha() else s

    surv.index = pd.Index([_trim(idx) for idx in surv.index], name=surv.index.name)
    surv = surv[~surv.index.duplicated(keep="first")]

    merged = labels.rename("subtype").to_frame().join(surv, how="inner")
    before = len(merged)
    merged = merged.dropna(subset=["OS.time"])
    merged = merged.rename(columns={"OS.time": "OS_time", "OS": "OS_event"})

    logger.info(
        "merge_survival: %d samples joined, %d kept after dropping missing OS.time",
        before,
        len(merged),
    )
    return merged


def fit_km(df: pd.DataFrame, subtype: str) -> KaplanMeierFitter:
    """Fit KaplanMeierFitter on the subset of df matching subtype."""
    sub = df.loc[df["subtype"] == subtype]
    kmf = KaplanMeierFitter(label=subtype)
    kmf.fit(durations=sub["OS_time"], event_observed=sub["OS_event"])
    return kmf


def logrank_pairwise(df: pd.DataFrame) -> pd.DataFrame:
    """
    All pairwise log-rank p-values across subtypes.

    Returns a symmetric DataFrame of p-values (diagonal = NaN).
    pairwise_logrank_test returns one row per ordered pair (s1, s2) with s1 < s2
    in a MultiIndex; we fill both (s1, s2) and (s2, s1) in the output matrix.
    """
    result = pairwise_logrank_test(
        event_durations=df["OS_time"],
        groups=df["subtype"],
        event_observed=df["OS_event"],
    )

    subtypes = sorted(df["subtype"].unique())
    sym = pd.DataFrame(np.nan, index=subtypes, columns=subtypes)

    for (s1, s2), row in result.summary.iterrows():
        p = float(row["p"])
        if s1 in sym.index and s2 in sym.columns:
            sym.loc[s1, s2] = p
            sym.loc[s2, s1] = p

    return sym


def fit_cox(df: pd.DataFrame, *, covariates: list[str]) -> CoxPHFitter:
    """Fit a Cox proportional hazards model on the specified covariates."""
    cox_df = df[["OS_time", "OS_event"] + covariates].copy()
    cox_df = pd.get_dummies(cox_df, columns=[c for c in covariates if cox_df[c].dtype == object], drop_first=True)
    cox_df = cox_df.astype(float)

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="OS_time", event_col="OS_event")
    logger.info("Cox model fitted; concordance index = %.3f", cph.concordance_index_)
    return cph


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _raw = pd.read_csv(interim_file(InterimFile.LABELS), index_col=0).squeeze()
    labels: pd.Series = _raw  # type: ignore[assignment]

    surv_raw = pd.read_csv(raw_file(DataFile.SURVIVAL), sep="\t", index_col=0)
    df = merge_survival(labels=labels, raw_survival_df=surv_raw)

    out = interim_file(InterimFile.SURVIVAL_MERGED)
    df.to_csv(out)
    logger.info("Saved %s  (%d rows)", out.name, len(df))


if __name__ == "__main__":
    main()
