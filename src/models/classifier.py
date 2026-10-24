"""
PAM50 subtype classifier: multinomial logistic regression with cross-validation.

Inputs:  InterimFile.X (samples × genes), InterimFile.LABELS
Outputs: data/processed/classifier.pkl (fitted pipeline)

Run: python -m src.models.classifier
"""

from __future__ import annotations

import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constants import InterimFile
from src.features.differential_expression import log2_normalize
from src.paths import PROCESSED_DIR, interim_file

logger = logging.getLogger(__name__)


def build_pipeline() -> Pipeline:
    """StandardScaler + LogisticRegression (lbfgs, multinomial by default in sklearn ≥1.5)."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=1000,
                    C=0.1,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    pipeline: Pipeline,
    cv: int = 5,
) -> dict[str, np.ndarray]:
    """
    StratifiedKFold cross-validation.

    Returns dict with keys 'accuracy', 'balanced_accuracy', 'f1_macro',
    each an array of length cv.
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = ["accuracy", "balanced_accuracy", "f1_macro"]

    raw = cross_validate(
        pipeline,
        X.values,
        y.values,
        cv=cv_splitter,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    results = {
        "accuracy": raw["test_accuracy"],
        "balanced_accuracy": raw["test_balanced_accuracy"],
        "f1_macro": raw["test_f1_macro"],
    }

    for metric, vals in results.items():
        logger.info("  %-22s %.3f ± %.3f", metric, vals.mean(), vals.std())

    return results


def train(X: pd.DataFrame, y: pd.Series, pipeline: Pipeline) -> Pipeline:
    """Fit pipeline on the full dataset and return the fitted pipeline."""
    pipeline.fit(X.values, y.values)
    logger.info("Trained on %d samples, %d features", *X.shape)
    return pipeline


def confusion_matrix_df(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    """Confusion matrix as a labeled DataFrame (rows = true, cols = predicted)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    X_raw = pd.read_csv(interim_file(InterimFile.X), index_col=0)
    _raw = pd.read_csv(interim_file(InterimFile.LABELS), index_col=0).squeeze()
    labels: pd.Series = _raw  # type: ignore[assignment]

    X = log2_normalize(X_raw)

    logger.info("Building pipeline and running 5-fold CV ...")
    pipeline = build_pipeline()
    cross_validate_model(X=X, y=labels, pipeline=pipeline, cv=5)

    logger.info("Training on full dataset ...")
    fitted = train(X=X, y=labels, pipeline=build_pipeline())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "classifier.pkl"
    with open(out, "wb") as f:
        pickle.dump(fitted, f)
    logger.info("Saved %s", out.name)


if __name__ == "__main__":
    main()
