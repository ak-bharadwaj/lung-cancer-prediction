"""
Exploratory Data Analysis (EDA) Module for Survey Lung Cancer Dataset.
Calculates demographic distributions, symptom prevalence by target class,
relative risk ratios, and feature correlation matrix.
Exports detailed summary to reports/eda_summary.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any

import pandas as pd

from ml.preprocessing import SYMPTOM_FEATURES, clean_feature_names


def run_eda(
    csv_path: str | Path | None = None,
    output_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Compute statistical distributions and relationships across survey features.
    
    Args:
        csv_path: Path to survey CSV.
        output_report_path: Path to output JSON summary.
        
    Returns:
        Dictionary containing EDA insights.
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
        output_report_path = base_dir / "reports" / "eda_summary.json"
    else:
        output_report_path = Path(output_report_path)

    raw_df = pd.read_csv(csv_path)
    df = clean_feature_names(raw_df)

    total_records = len(df)
    target_series = df["LUNG_CANCER"].astype(str).str.strip().str.upper()
    df["TARGET_NUM"] = (target_series == "YES").astype(int)

    # 1. Target distribution
    cancer_yes = int((df["TARGET_NUM"] == 1).sum())
    cancer_no = int((df["TARGET_NUM"] == 0).sum())
    target_dist = {
        "positive": {
            "count": cancer_yes,
            "percentage": round((cancer_yes / total_records) * 100, 2),
        },
        "negative": {
            "count": cancer_no,
            "percentage": round((cancer_no / total_records) * 100, 2),
        },
        "imbalance_ratio": round(cancer_yes / cancer_no, 2) if cancer_no > 0 else 0.0,
    }

    # 2. Age distribution
    def get_stats(series: pd.Series) -> dict[str, float]:
        return {
            "count": int(series.count()),
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
            "min": round(float(series.min()), 2),
            "p25": round(float(series.quantile(0.25)), 2),
            "median": round(float(series.median()), 2),
            "p75": round(float(series.quantile(0.75)), 2),
            "max": round(float(series.max()), 2),
        }

    age_stats = {
        "overall": get_stats(df["AGE"]),
        "cancer_yes": get_stats(df[df["TARGET_NUM"] == 1]["AGE"]),
        "cancer_no": get_stats(df[df["TARGET_NUM"] == 0]["AGE"]),
    }

    # 3. Gender breakdown
    gender_breakdown = {}
    for g in ["M", "F"]:
        sub = df[df["GENDER"] == g]
        total_g = len(sub)
        pos_g = int((sub["TARGET_NUM"] == 1).sum())
        neg_g = int((sub["TARGET_NUM"] == 0).sum())
        gender_breakdown[g] = {
            "total": total_g,
            "cancer_yes": pos_g,
            "cancer_no": neg_g,
            "cancer_prevalence_pct": round((pos_g / total_g) * 100, 2) if total_g > 0 else 0.0,
        }

    # 4. Symptom prevalence and relative risk
    symptom_analysis = {}
    for symptom in SYMPTOM_FEATURES:
        # 2 = Presence of symptom, 1 = Absence
        has_symptom = df[symptom] == 2
        no_symptom = df[symptom] == 1

        p_cancer_given_symptom = (
            float((df[has_symptom]["TARGET_NUM"] == 1).mean()) if has_symptom.sum() > 0 else 0.0
        )
        p_cancer_given_no_symptom = (
            float((df[no_symptom]["TARGET_NUM"] == 1).mean()) if no_symptom.sum() > 0 else 0.0
        )
        relative_risk = (
            round(p_cancer_given_symptom / p_cancer_given_no_symptom, 3)
            if p_cancer_given_no_symptom > 0
            else 1.0
        )

        pos_group = df[df["TARGET_NUM"] == 1]
        neg_group = df[df["TARGET_NUM"] == 0]

        prev_in_cancer = (
            round(float((pos_group[symptom] == 2).mean()) * 100, 2)
            if len(pos_group) > 0
            else 0.0
        )
        prev_in_non_cancer = (
            round(float((neg_group[symptom] == 2).mean()) * 100, 2)
            if len(neg_group) > 0
            else 0.0
        )

        symptom_analysis[symptom] = {
            "prevalence_in_cancer_positive_pct": prev_in_cancer,
            "prevalence_in_cancer_negative_pct": prev_in_non_cancer,
            "relative_risk": relative_risk,
            "risk_elevation": (
                f"{round((relative_risk - 1.0) * 100, 1)}% increase in likelihood"
                if relative_risk >= 1.0
                else f"{round((1.0 - relative_risk) * 100, 1)}% decrease in likelihood"
            ),
        }

    # 5. Correlation with target
    encoded_df = df.copy()
    encoded_df["GENDER_NUM"] = (encoded_df["GENDER"] == "M").astype(int)
    corr_features = ["GENDER_NUM", "AGE"] + SYMPTOM_FEATURES
    corr_matrix = encoded_df[corr_features + ["TARGET_NUM"]].corr().round(4)

    target_correlations = (
        corr_matrix["TARGET_NUM"]
        .drop("TARGET_NUM")
        .sort_values(ascending=False)
        .to_dict()
    )

    # Top co-occurring pairs
    feature_corr = corr_matrix.drop(index="TARGET_NUM", columns="TARGET_NUM")
    pairs = []
    cols = feature_corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append({
                "feature_a": cols[i],
                "feature_b": cols[j],
                "correlation": float(feature_corr.iloc[i, j]),
            })
    top_pairs = sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)[:10]

    dataset_name = (
        "Lung Cancer Cohort (1309 Records)"
        if "1309" in str(csv_path)
        else ("Survey Lung Cancer (309 Records)" if "survey" in str(csv_path).lower() else "Lung Cancer Dataset")
    )

    eda_summary = {
        "dataset_name": dataset_name,
        "total_records": total_records,
        "target_distribution": target_dist,
        "age_statistics": age_stats,
        "gender_breakdown": gender_breakdown,
        "symptom_analysis": symptom_analysis,
        "correlation_with_target": target_correlations,
        "top_correlated_feature_pairs": top_pairs,
    }

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2)

    return eda_summary


if __name__ == "__main__":
    summary = run_eda()
    print("EDA analysis completed.")
    print("Top correlated features with target:")
    for feat, corr in list(summary["correlation_with_target"].items())[:5]:
        print(f"  {feat}: {corr}")
