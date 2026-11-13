from __future__ import annotations

from pathlib import Path

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
import pandas as pd


def generate_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    html_path: Path,
    json_path: Path,
    numerical_features: tuple[str, ...] | list[str],
    categorical_features: tuple[str, ...] | list[str],
) -> None:
    """Write an Evidently feature-drift and data-summary report."""
    if reference.columns.tolist() != current.columns.tolist():
        raise ValueError("Reference and current columns must have the same order.")
    declared = set(numerical_features) | set(categorical_features)
    if set(reference.columns) != declared:
        raise ValueError("Declared feature types must cover exactly the input columns.")
    if set(numerical_features) & set(categorical_features):
        raise ValueError("A feature cannot be both numerical and categorical.")

    reference = reference.copy()
    current = current.copy()
    for column in categorical_features:
        reference[column] = reference[column].map(
            lambda value: str(value) if pd.notna(value) else None
        )
        current[column] = current[column].map(
            lambda value: str(value) if pd.notna(value) else None
        )

    definition = DataDefinition(
        numerical_columns=list(numerical_features),
        categorical_columns=list(categorical_features),
    )
    reference_data = Dataset.from_pandas(reference, data_definition=definition)
    current_data = Dataset.from_pandas(current, data_definition=definition)
    report = Report([DataDriftPreset(), DataSummaryPreset()])
    snapshot = report.run(current_data=current_data, reference_data=reference_data)

    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))
    json_path.write_text(snapshot.json(), encoding="utf-8")
