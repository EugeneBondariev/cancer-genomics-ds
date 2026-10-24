"""
Unit tests for src.models.classifier.
Uses synthetic 5-class expression data; no real data files required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.classifier import (
    build_pipeline,
    confusion_matrix_df,
    cross_validate_model,
    train,
)


@pytest.fixture
def synthetic_5class() -> tuple[pd.DataFrame, pd.Series]:
    """
    5 classes × 50 samples each = 250 samples; 100 features.
    Each class has 10 genes with injected signal (+5 mean shift).
    """
    rng = np.random.default_rng(7)
    n_per_class, n_classes, n_genes = 50, 5, 100
    classes = [f"C{i}" for i in range(n_classes)]

    index = pd.Index([f"s{i:04d}" for i in range(n_per_class * n_classes)])
    subtypes = [c for c in classes for _ in range(n_per_class)]
    labels = pd.Series(subtypes, index=index, name="subtype")

    data = rng.standard_normal((n_per_class * n_classes, n_genes))
    for i, _ in enumerate(classes):
        start, end = i * n_per_class, (i + 1) * n_per_class
        gene_start, gene_end = i * 10, (i + 1) * 10
        data[start:end, gene_start:gene_end] += 5.0

    X = pd.DataFrame(
        data,
        index=index,
        columns=[f"G{j:03d}" for j in range(n_genes)],
    )
    return X, labels


class TestBuildPipeline:
    def test_returns_pipeline(self) -> None:
        pipe = build_pipeline()
        assert isinstance(pipe, Pipeline)

    def test_has_scaler_step(self) -> None:
        pipe = build_pipeline()
        assert isinstance(pipe.named_steps["scaler"], StandardScaler)

    def test_has_lr_step(self) -> None:
        pipe = build_pipeline()
        assert isinstance(pipe.named_steps["lr"], LogisticRegression)

    def test_lr_max_iter(self) -> None:
        pipe = build_pipeline()
        assert pipe.named_steps["lr"].max_iter == 1000

    def test_lr_c(self) -> None:
        pipe = build_pipeline()
        assert pipe.named_steps["lr"].C == pytest.approx(0.1)


class TestCrossValidateModel:
    def test_returns_required_keys(
        self, synthetic_5class: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = synthetic_5class
        results = cross_validate_model(X=X, y=y, pipeline=build_pipeline(), cv=3)
        assert set(results.keys()) == {"accuracy", "balanced_accuracy", "f1_macro"}

    def test_array_length_equals_cv(
        self, synthetic_5class: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = synthetic_5class
        cv = 3
        results = cross_validate_model(X=X, y=y, pipeline=build_pipeline(), cv=cv)
        for arr in results.values():
            assert len(arr) == cv

    def test_accuracy_in_unit_interval(
        self, synthetic_5class: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = synthetic_5class
        results = cross_validate_model(X=X, y=y, pipeline=build_pipeline(), cv=3)
        for arr in results.values():
            assert (arr >= 0).all() and (arr <= 1).all()

    def test_high_accuracy_on_separable_data(
        self, synthetic_5class: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = synthetic_5class
        results = cross_validate_model(X=X, y=y, pipeline=build_pipeline(), cv=3)
        assert results["accuracy"].mean() > 0.8


class TestTrain:
    def test_returns_fitted_pipeline(
        self, synthetic_5class: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = synthetic_5class
        fitted = train(X=X, y=y, pipeline=build_pipeline())
        assert isinstance(fitted, Pipeline)

    def test_fitted_pipeline_predicts(
        self, synthetic_5class: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = synthetic_5class
        fitted = train(X=X, y=y, pipeline=build_pipeline())
        preds = fitted.predict(X.values)
        assert len(preds) == len(y)


class TestConfusionMatrixDf:
    def test_shape(self, synthetic_5class: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = synthetic_5class
        fitted = train(X=X, y=y, pipeline=build_pipeline())
        preds = fitted.predict(X.values)
        labels_list = sorted(y.unique())
        cm = confusion_matrix_df(y_true=y, y_pred=preds, labels=labels_list)
        assert cm.shape == (5, 5)

    def test_index_equals_columns(
        self, synthetic_5class: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = synthetic_5class
        fitted = train(X=X, y=y, pipeline=build_pipeline())
        preds = fitted.predict(X.values)
        labels_list = sorted(y.unique())
        cm = confusion_matrix_df(y_true=y, y_pred=preds, labels=labels_list)
        assert list(cm.index) == list(cm.columns)

    def test_row_sums_equal_class_counts(
        self, synthetic_5class: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = synthetic_5class
        fitted = train(X=X, y=y, pipeline=build_pipeline())
        preds = fitted.predict(X.values)
        labels_list = sorted(y.unique())
        cm = confusion_matrix_df(y_true=y, y_pred=preds, labels=labels_list)
        for label in labels_list:
            assert cm.loc[label].sum() == (y == label).sum()
