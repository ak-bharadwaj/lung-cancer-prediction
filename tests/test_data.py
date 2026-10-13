"""
Unit and Integration Tests for Data Validation and Preprocessing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from ml.data_validation import (
    EXPECTED_SYMPTOM_COLUMNS,
    standardize_column_names,
    validate_cancer_patient_dataset,
    validate_dataset,
)
from ml.eda import run_eda
from ml.preprocessing import (
    FEATURE_ORDER,
    load_and_preprocess_raw,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "raw" / "survey_lung_cancer.csv"


def test_raw_csv_exists():
    """Verify raw dataset exists and is accessible."""
    assert CSV_PATH.exists(), f"Raw dataset not found at {CSV_PATH}"


def test_standardize_column_names():
    """Verify whitespace stripping and uppercase snake_case formatting."""
    df = pd.DataFrame(columns=[" GENDER ", "CHRONIC DISEASE", "FATIGUE "])
    cleaned = standardize_column_names(df)
    assert cleaned.columns.tolist() == ["GENDER", "CHRONIC_DISEASE", "FATIGUE"]


def test_validate_dataset_schema():
    """Verify validate_dataset returns correct schema metrics and zero missing values."""
    report = validate_dataset(CSV_PATH)
    assert report["schema_valid"] is True
    assert report["total_rows"] == 309
    assert report["total_columns"] == 16
    assert report["total_missing_values"] == 0
    assert len(report["missing_columns"]) == 0
    assert report["class_imbalance"]["positive_count"] == 270
    assert report["class_imbalance"]["negative_count"] == 39
    assert report["class_imbalance"]["positive_percentage"] == 87.38


def test_data_audit_json_output():
    """Verify reports/data_audit.json is generated and structurally valid."""
    audit_file = BASE_DIR / "reports" / "data_audit.json"
    assert audit_file.exists()
    with open(audit_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "class_imbalance" in data
    assert "demographics" in data
    assert data["demographics"]["age_statistics"]["min"] >= 1
    assert data["demographics"]["age_statistics"]["max"] <= 120


def test_eda_summary_json_output():
    """Verify reports/eda_summary.json is populated with relative risk and correlations."""
    eda_summary = run_eda(CSV_PATH)
    assert "target_distribution" in eda_summary
    assert "symptom_analysis" in eda_summary
    assert "correlation_with_target" in eda_summary
    assert "top_correlated_feature_pairs" in eda_summary

    # Ensure all symptoms are analyzed
    for symptom in EXPECTED_SYMPTOM_COLUMNS:
        assert symptom in eda_summary["symptom_analysis"]
        assert "relative_risk" in eda_summary["symptom_analysis"][symptom]


def test_load_and_preprocess_raw():
    """Verify load_and_preprocess_raw outputs X and y with matched rows."""
    X, y = load_and_preprocess_raw(str(CSV_PATH))
    assert len(X) == 309
    assert len(y) == 309
    assert list(X.columns) == FEATURE_ORDER
    assert set(y.unique()) == {0, 1}
    assert (y == 1).sum() == 270


def test_validate_expanded_1309_cohort():
    """Verify 1,309-record benchmark cohort validation."""
    cohort_path = BASE_DIR / "data" / "lung_cancer_cohort_1309.csv"
    assert cohort_path.exists(), "data/lung_cancer_cohort_1309.csv must exist."
    report = validate_dataset(cohort_path)
    assert report["schema_valid"] is True
    assert report["total_rows"] == 1309
    assert report["total_columns"] == 16
    assert report["total_missing_values"] == 0
    assert report["class_imbalance"]["positive_count"] == 1144
    assert report["class_imbalance"]["negative_count"] == 165


def test_validate_kaggle_1000_patient_dataset():
    """Verify Kaggle 1,000-patient cancer dataset validation."""
    cancer_path = BASE_DIR / "data" / "raw" / "cancer_patient_data_sets.csv"
    assert cancer_path.exists(), "data/raw/cancer_patient_data_sets.csv must exist."
    report = validate_cancer_patient_dataset(cancer_path)
    assert report["schema_valid"] is True
    assert report["total_rows"] == 1000
    assert report["total_columns"] == 26
    assert report["total_missing_values"] == 0
    assert "High" in report["level_counts"]
    assert "Medium" in report["level_counts"]
    assert "Low" in report["level_counts"]


def test_load_and_preprocess_1309_cohort():
    """Verify loading and preprocessing the 1,309 cohort."""
    cohort_path = BASE_DIR / "data" / "lung_cancer_cohort_1309.csv"
    X, y = load_and_preprocess_raw(str(cohort_path))
    assert len(X) == 1309
    assert len(y) == 1309
    assert list(X.columns) == FEATURE_ORDER
    assert (y == 1).sum() == 1144
    assert (y == 0).sum() == 165


def test_load_and_preprocess_cancer_patient_dataset():
    """Verify loading and transformer for Kaggle 1,000-patient dataset."""
    from ml.cancer_patient_pipeline import (
        CANCER_PATIENT_FEATURE_COLUMNS,
        CancerPatientFeatureTransformer,
        load_cancer_patient_data,
    )
    cancer_path = BASE_DIR / "data" / "raw" / "cancer_patient_data_sets.csv"
    assert cancer_path.exists()
    X, y = load_cancer_patient_data(cancer_path, binary_elevated_risk=True)
    assert len(X) == 1000
    assert len(y) == 1000
    assert list(X.columns) == CANCER_PATIENT_FEATURE_COLUMNS
    assert (y == 1).sum() == 697
    assert (y == 0).sum() == 303

    transformer = CancerPatientFeatureTransformer(scale_features=True)
    X_trans = transformer.fit_transform(X)
    assert X_trans.shape == (1000, 23)


