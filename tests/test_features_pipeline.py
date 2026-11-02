import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mibici.features_pipeline import fit_transform_train_and_holdouts, make_tabular_pipeline


def test_make_tabular_pipeline_rechaza_features_duplicadas():
    with pytest.raises(ValueError, match="multiple groups"):
        make_tabular_pipeline(
            RandomForestRegressor(),
            numerical_features=["age"],
            categorical_features=["age"],
        )


def test_fit_transform_train_and_holdouts_fit_solo_con_train():
    train = pd.DataFrame({"value": [0.0, 2.0]})
    holdout = pd.DataFrame({"value": [100.0]})

    train_scaled, holdout_scaled = fit_transform_train_and_holdouts(
        StandardScaler(), train, holdout
    )

    np.testing.assert_allclose(train_scaled[:, 0], [-1.0, 1.0])
    np.testing.assert_allclose(holdout_scaled[:, 0], [99.0])


def test_modelo_compartido_reproduce_predicciones_con_la_misma_semilla():
    features = pd.DataFrame(
        {
            "distancia": [0.2, 0.4, 0.8, 1.0, 1.3, 1.7, 2.0, 2.4],
            "tipo": ["corta", "corta", "media", "media", "larga", "larga", "corta", "media"],
        }
    )
    target = np.array([3.0, 4.0, 8.0, 9.0, 13.0, 17.0, 5.0, 10.0])
    holdout = pd.DataFrame({"distancia": [0.6, 1.5], "tipo": ["media", "larga"]})

    def fit_and_predict():
        pipeline = make_tabular_pipeline(
            RandomForestRegressor(n_estimators=20, random_state=29, n_jobs=1),
            numerical_features=["distancia"],
            categorical_features=["tipo"],
            categorical_transformer=OneHotEncoder(handle_unknown="ignore"),
        )
        pipeline.fit(features, target)
        return pipeline.predict(holdout)

    first = fit_and_predict()
    second = fit_and_predict()

    np.testing.assert_array_equal(first, second)
