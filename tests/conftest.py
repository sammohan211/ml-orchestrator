"""
Shared fixtures for ML Orchestrator tests.
"""
import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_df():
    """Small mixed-type DataFrame representative of a tabular dataset."""
    return pd.DataFrame({
        "age":        [25, 32, 47, 51, 23, 38, 45, 29, 60, 34],
        "income":     [50000, 80000, None, 120000, 45000, 90000, 75000, 60000, 110000, 85000],
        "gender":     ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
        "city":       ["NY", "LA", "NY", "SF", "LA", "NY", "SF", "LA", "NY", "SF"],
        "subscribed": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    })


@pytest.fixture
def sample_df_with_issues():
    """DataFrame with nulls, duplicates, and a constant column — for cleaning tests."""
    return pd.DataFrame({
        "age":      [25, 32, None, 51, 25, 32, 45, None, 60, 34],
        "income":   [50000, 80000, 60000, None, 50000, 80000, 75000, 60000, None, 85000],
        "gender":   ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
        "constant": ["A", "A", "A", "A", "A", "A", "A", "A", "A", "A"],
        "label":    [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    })


@pytest.fixture
def project_dir(tmp_path):
    """Temporary project directory with artifacts subfolder."""
    proj = tmp_path / "test_project"
    (proj / "artifacts").mkdir(parents=True)
    return proj


@pytest.fixture
def minimal_state(project_dir, sample_df):
    """Minimal valid state dict representing a project after ingestion."""
    csv_path = project_dir / "source.csv"
    sample_df.to_csv(csv_path, index=False)

    return {
        "project": {
            "name": "test_project",
            "project_dir": str(project_dir),
            "created_at": "2026-04-05T00:00:00",
            "last_updated": "2026-04-05T00:00:00",
            "schema_version": "1.0",
        },
        "settings": {
            "test_size": 0.2,
            "random_seed": 42,
            "stratified": True,
            "feature_selection_threshold": 0.01,
            "cv_folds": 5,
            "tuning_n_iter": 20,
            "high_cardinality_threshold": 50,
            "class_imbalance_threshold": 0.75,
        },
        "stages": {
            "ingestion": "completed",
            "profiling": "pending",
            "cleaning": "pending",
            "feature_preparation": "pending",
            "feature_selection": "pending",
            "modeling": "pending",
            "tuning": "pending",
            "evaluation": "pending",
            "export": "pending",
        },
        "ingestion": {
            "source_path": str(csv_path),
            "format": "CSV",
            "row_count": len(sample_df),
            "column_count": len(sample_df.columns),
            "encoding": "utf-8",
            "delimiter": ",",
            "schema": {col: {"dtype": str(sample_df[col].dtype), "nullable": bool(sample_df[col].isnull().any())}
                       for col in sample_df.columns},
        },
        "data_dictionary": {"columns": {}},
        "artifacts": {},
        "dependencies": [],
    }
