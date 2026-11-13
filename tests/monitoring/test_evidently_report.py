import pandas as pd

from mibici.features import FEATURES
from mibici.monitoring.evidently_report import generate_report

P1_CATEGORICAL_FEATURES = ("dia_semana", "festivo", "festivo_lag7")
P1_NUMERICAL_FEATURES = tuple(
    feature for feature in FEATURES if feature not in P1_CATEGORICAL_FEATURES
)


def test_generates_evidently_html_and_json(tmp_path):
    values = list(range(50))
    reference = pd.DataFrame(
        {
            feature: [value % 7 for value in values]
            if feature == "dia_semana"
            else [value % 2 for value in values]
            if feature in ("festivo", "festivo_lag7")
            else values
            for feature in FEATURES
        }
    )
    current = reference.copy()
    for column in FEATURES:
        if column == "dia_semana":
            current[column] = (current[column] + 1) % 7
        elif column in ("festivo", "festivo_lag7"):
            current[column] = 1 - current[column]
        else:
            current[column] = current[column] + 1
    html_path = tmp_path / "nested" / "report.html"
    json_path = tmp_path / "nested" / "report.json"

    generate_report(
        reference,
        current,
        html_path,
        json_path,
        P1_NUMERICAL_FEATURES,
        P1_CATEGORICAL_FEATURES,
    )

    assert html_path.is_file() and html_path.stat().st_size > 1000
    assert json_path.is_file() and json_path.stat().st_size > 100
