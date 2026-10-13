"""
Data Validation and Audit Module for Survey Lung Cancer Dataset.
Performs schema verification, range validation, missing value audit,
duplicate detection, and class imbalance analysis.
Exports findings to reports/data_audit.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_SYMPTOM_COLUMNS = [
    "SMOKING",
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC_DISEASE",
    "FATIGUE",
    "ALLERGY",
    "WHEEZING",
    "ALCOHOL_CONSUMING",
    "COUGHING",
    "SHORTNESS_OF_BREATH",
    "SWALLOWING_DIFFICULTY",
    "CHEST_PAIN",
]

ALL_FEATURE_COLUMNS = ["GENDER", "AGE"] + EXPECTED_SYMPTOM_COLUMNS
TARGET_COLUMN = "LUNG_CANCER"
ALL_COLUMNS = ALL_FEATURE_COLUMNS + [TARGET_COLUMN]


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and replace interior spaces with underscores."""
    df_clean = df.copy()
    df_clean.columns = (
        df_clean.columns.str.strip()
        .str.replace(" ", "_")
        .str.upper()
    )
    return df_clean


def validate_dataset(
    csv_path: str | Path | None = None,
    output_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Validate the survey lung cancer dataset and export an audit report.

    Args:
        csv_path: Path to the raw CSV file.
        output_report_path: Path where data_audit.json will be saved.

    Returns:
        Audit report dictionary.
    """
    base_dir = Path(__file__).resolve().parent.parent
    if csv_path is None:
        primary_cohort = base_dir / "data" / "lung_cancer_cohort_1309.csv"
        if primary_cohort.exists():
            csv_path = primary_cohort
        else:
            csv_path = base_dir / "data" / "raw" / "survey_lung_cancer.csv"
    else:
        csv_path = Path(csv_path)

    if output_report_path is None:
        output_report_path = base_dir / "reports" / "data_audit.json"
    else:
        output_report_path = Path(output_report_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    raw_df = pd.read_csv(csv_path)
    total_raw_rows, total_raw_cols = raw_df.shape

    df = standardize_column_names(raw_df)

    # 1. Column presence check
    missing_cols = [c for c in ALL_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in ALL_COLUMNS]

    # 2. Missing value audit
    null_counts = {col: int(df[col].isnull().sum()) for col in df.columns}
    total_nulls = sum(null_counts.values())

    # 3. Duplicate row audit
    duplicate_count = int(df.duplicated().sum())

    # 4. Value validation
    validation_errors: list[str] = []

    # Check GENDER
    invalid_gender = df[~df["GENDER"].isin(["M", "F"])]
    if not invalid_gender.empty:
        validation_errors.append(f"Found {len(invalid_gender)} invalid GENDER values.")

    # Check AGE
    invalid_age = df[(df["AGE"] < 1) | (df["AGE"] > 120)]
    if not invalid_age.empty:
        validation_errors.append(f"Found {len(invalid_age)} invalid AGE values.")

    # Check Symptom columns (must be 1 or 2)
    for col in EXPECTED_SYMPTOM_COLUMNS:
        if col in df.columns:
            invalid_vals = df[~df[col].isin([1, 2])]
            if not invalid_vals.empty:
                validation_errors.append(
                    f"Column '{col}' has {len(invalid_vals)} values outside [1, 2]."
                )

    # Check Target
    if TARGET_COLUMN in df.columns:
        invalid_target = df[~df[TARGET_COLUMN].isin(["YES", "NO"])]
        if not invalid_target.empty:
            validation_errors.append(
                f"Column '{TARGET_COLUMN}' has {len(invalid_target)} values outside ['YES', 'NO']."
            )

    # 5. Class imbalance analysis
    target_counts = df[TARGET_COLUMN].value_counts().to_dict()
    total_samples = len(df)
    yes_count = int(target_counts.get("YES", 0))
    no_count = int(target_counts.get("NO", 0))
    yes_pct = round((yes_count / total_samples) * 100, 2) if total_samples > 0 else 0.0
    no_pct = round((no_count / total_samples) * 100, 2) if total_samples > 0 else 0.0
    imbalance_ratio = round(yes_count / no_count, 2) if no_count > 0 else float("inf")

    # Demographic stats
    gender_counts = df["GENDER"].value_counts().to_dict()
    age_stats = {
        "min": int(df["AGE"].min()),
        "max": int(df["AGE"].max()),
        "mean": round(float(df["AGE"].mean()), 2),
        "median": round(float(df["AGE"].median()), 2),
        "std": round(float(df["AGE"].std()), 2),
    }

    dataset_name = (
        "Lung Cancer Cohort (1309 Records)"
        if "1309" in str(csv_path)
        else ("Survey Lung Cancer (309 Records)" if "survey" in str(csv_path).lower() else "Lung Cancer Dataset")
    )
    severity_msg = (
        f"Minority class NO comprises {no_pct}% of records ({no_count}/{total_samples})"
        if no_pct > 0
        else "Single-class distribution"
    )

    audit_report = {
        "dataset_name": dataset_name,
        "raw_file": str(csv_path),
        "total_rows": total_raw_rows,
        "total_columns": total_raw_cols,
        "standardized_columns": df.columns.tolist(),
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
        "total_missing_values": total_nulls,
        "missing_values_by_column": null_counts,
        "duplicate_rows_count": duplicate_count,
        "duplicate_notes": (
            "Duplicate entries in anonymous survey questionnaires are natural "
            "when categorical response combinations overlap across participants."
        ),
        "class_imbalance": {
            "positive_label": "YES",
            "positive_count": yes_count,
            "positive_percentage": yes_pct,
            "negative_label": "NO",
            "negative_count": no_count,
            "negative_percentage": no_pct,
            "imbalance_ratio": imbalance_ratio,
            "severity": severity_msg,
        },
        "demographics": {
            "gender_counts": gender_counts,
            "age_statistics": age_stats,
        },
        "schema_valid": len(missing_cols) == 0 and len(validation_errors) == 0,
        "validation_errors": validation_errors,
    }

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    return audit_report


def validate_cancer_patient_dataset(
    csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate Kaggle 1,000-Patient Cancer Dataset (23 clinical features, Level target)."""
    base_dir = Path(__file__).resolve().parent.parent
    if csv_path is None:
        csv_path = base_dir / "data" / "raw" / "cancer_patient_data_sets.csv"
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Cancer patient dataset not found at {csv_path}")

    raw_df = pd.read_csv(csv_path)
    total_rows, total_cols = raw_df.shape
    null_count = int(raw_df.isnull().sum().sum())

    feature_cols = [c for c in raw_df.columns if c not in ["index", "Patient Id"]]
    dedup_rows = len(raw_df[feature_cols].drop_duplicates())
    level_counts = raw_df["Level"].value_counts().to_dict() if "Level" in raw_df.columns else {}

    return {
        "dataset_name": "Kaggle 1,000-Patient Cancer Dataset",
        "raw_file": str(csv_path),
        "total_rows": total_rows,
        "total_columns": total_cols,
        "unique_profiles": dedup_rows,
        "total_missing_values": null_count,
        "level_counts": level_counts,
        "schema_valid": total_rows >= 1000 and null_count == 0,
    }


if __name__ == "__main__":
    report = validate_dataset()
    print("Dataset audit completed successfully.")
    print(f"Total Rows: {report['total_rows']}, Missing: {report['total_missing_values']}")
    print(f"Class Balance: YES={report['class_imbalance']['positive_count']} ({report['class_imbalance']['positive_percentage']}%), NO={report['class_imbalance']['negative_count']} ({report['class_imbalance']['negative_percentage']}%)")
    print(f"Schema Valid: {report['schema_valid']}")
