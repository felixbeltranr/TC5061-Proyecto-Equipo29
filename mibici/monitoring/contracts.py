from __future__ import annotations

import numpy as np
import pandas as pd


def validate_feature_frame(
    frame: pd.DataFrame,
    name: str,
    expected_features: tuple[str, ...] | list[str],
    numerical_features: tuple[str, ...] | list[str],
    categorical_features: tuple[str, ...] | list[str],
) -> dict:
    """Validate an input frame against a problem-specific feature contract."""
    expected = list(expected_features)
    if frame.empty:
        raise ValueError(f"{name} has no eligible rows.")
    if len(set(expected)) != len(expected):
        raise ValueError("Expected feature names must be unique.")
    if frame.columns.tolist() != expected:
        raise ValueError(
            f"{name} must contain exactly the ordered contract features {expected}; "
            f"received {frame.columns.tolist()}."
        )
    if set(numerical_features) & set(categorical_features):
        raise ValueError("A feature cannot be both numerical and categorical.")
    if set(numerical_features) | set(categorical_features) != set(expected):
        raise ValueError("Every expected feature must have exactly one declared data type.")

    non_numeric = [
        column for column in numerical_features if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise TypeError(f"{name} has non-numeric features: {non_numeric}")

    infinite = [
        column
        for column in numerical_features
        if np.isinf(frame[column].to_numpy(dtype=float, na_value=np.nan)).any()
    ]
    if infinite:
        raise ValueError(f"{name} contains infinite values in: {infinite}")

    return {
        "rows": len(frame),
        "features": expected,
        "nulls_by_feature": {column: int(frame[column].isna().sum()) for column in expected},
    }
