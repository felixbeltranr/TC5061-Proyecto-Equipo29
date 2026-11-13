import numpy as np
import pandas as pd
import pytest

from mibici.features import FEATURES
from mibici.monitoring.contracts import validate_feature_frame

P1_CATEGORICAL_FEATURES = ("dia_semana", "festivo", "festivo_lag7")
P1_NUMERICAL_FEATURES = tuple(
    feature for feature in FEATURES if feature not in P1_CATEGORICAL_FEATURES
)


def valid_features(rows: int = 2) -> pd.DataFrame:
    return pd.DataFrame({feature: np.zeros(rows) for feature in FEATURES})


def test_contract_accepts_model_features_and_reports_null_counts():
    frame = valid_features()
    frame.loc[0, "lag_7"] = np.nan

    result = validate_feature_frame(
        frame,
        "sample",
        FEATURES,
        P1_NUMERICAL_FEATURES,
        P1_CATEGORICAL_FEATURES,
    )

    assert result["rows"] == 2
    assert result["nulls_by_feature"]["lag_7"] == 1


def test_contract_rejects_ids_labels_and_unexpected_columns():
    frame = valid_features()
    frame["objetivo"] = 1.0

    with pytest.raises(ValueError, match="exactly the ordered contract features"):
        validate_feature_frame(
            frame,
            "sample",
            FEATURES,
            P1_NUMERICAL_FEATURES,
            P1_CATEGORICAL_FEATURES,
        )


def test_contract_rejects_infinite_values():
    frame = valid_features()
    frame.loc[0, "lag_7"] = np.inf

    with pytest.raises(ValueError, match="infinite values"):
        validate_feature_frame(
            frame,
            "sample",
            FEATURES,
            P1_NUMERICAL_FEATURES,
            P1_CATEGORICAL_FEATURES,
        )


def test_common_contract_allows_string_categories_for_other_problems():
    frame = pd.DataFrame({"score": [0.2, 0.8], "label": ["safe", "risk"]})

    result = validate_feature_frame(
        frame,
        "P2 sample",
        ("score", "label"),
        ("score",),
        ("label",),
    )

    assert result["rows"] == 2
    assert result["nulls_by_feature"] == {"score": 0, "label": 0}
