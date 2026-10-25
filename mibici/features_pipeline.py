"""Shared scikit-learn pipeline helpers for task-specific feature transformations.

Feature definitions and transformation choices remain with each project question. This module
provides common pipeline construction and train-only fitting helpers.
"""

from collections.abc import Sequence
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def make_tabular_pipeline(
    estimator: BaseEstimator,
    *,
    numerical_features: Sequence[str] = (),
    categorical_features: Sequence[str] = (),
    passthrough_features: Sequence[str] = (),
    numerical_transformer: BaseEstimator | str = "passthrough",
    categorical_transformer: BaseEstimator | str = "passthrough",
    remainder: str = "drop",
) -> Pipeline:
    """Build a named preprocessing-and-model pipeline for selected DataFrame columns."""
    groups = {
        "numerical": tuple(numerical_features),
        "categorical": tuple(categorical_features),
        "passthrough": tuple(passthrough_features),
    }
    assigned: set[str] = set()
    for group_name, columns in groups.items():
        if any(not column for column in columns):
            raise ValueError(f"{group_name} feature names must be non-empty")
        duplicates = assigned.intersection(columns)
        if duplicates:
            raise ValueError(f"Features assigned to multiple groups: {sorted(duplicates)}")
        if len(set(columns)) != len(columns):
            raise ValueError(f"Duplicate {group_name} feature names")
        assigned.update(columns)

    transformers = []
    if groups["numerical"]:
        transformers.append(
            ("numerical", numerical_transformer, list(groups["numerical"]))
        )
    if groups["categorical"]:
        transformers.append(
            ("categorical", categorical_transformer, list(groups["categorical"]))
        )
    if groups["passthrough"]:
        transformers.append(("passthrough", "passthrough", list(groups["passthrough"])))

    preprocessing: BaseEstimator | str = (
        ColumnTransformer(transformers, remainder=remainder) if transformers else "passthrough"
    )
    return Pipeline([("preprocessing", preprocessing), ("model", estimator)])


def fit_transform_train_and_holdouts(
    transformer: BaseEstimator, train: Any, *holdouts: Any
) -> tuple[Any, ...]:
    """Fit a transformer on training data only, then transform train and holdout data."""
    transformer.fit(train)
    return (transformer.transform(train), *(transformer.transform(data) for data in holdouts))
