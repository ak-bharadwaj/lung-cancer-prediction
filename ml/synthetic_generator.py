"""
Synthetic Cohort Generator for Large-Scale Model Stress-Testing.
Generates high-fidelity synthetic survey datasets conditioned on the empirical
joint distributions, symptom prevalence, and age parameters of the survey dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to sys.path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.preprocessing import FEATURE_ORDER, SYMPTOM_FEATURES, clean_feature_names


def generate_synthetic_cohort(
    source_csv_path: str | Path | None = None,
    output_csv_path: str | Path | None = None,
    n_samples: int = 1000,
    positive_ratio: float = 0.874,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic survey participants matching the statistical profile of the source data.
    """
    base_dir = Path(__file__).resolve().parent.parent
    if source_csv_path is None:
        source_csv_path = base_dir / "data" / "raw" / "survey_lung_cancer.csv"
    else:
        source_csv_path = Path(source_csv_path)

    if output_csv_path is None:
        output_csv_path = base_dir / "data" / "synthetic_cohort_1000.csv"
    else:
        output_csv_path = Path(output_csv_path)

    raw_df = pd.read_csv(source_csv_path)
    df = clean_feature_names(raw_df)

    rng = np.random.RandomState(random_seed)

    # Feature column list for correlation matrix
    all_numeric_features = ["GENDER_NUM", "AGE"] + SYMPTOM_FEATURES

    # Compute parameters and correlation structures conditioned on target class (YES vs NO)
    cohort_profiles: dict[str, dict] = {}
    for target_label in ["YES", "NO"]:
        sub = df[df["LUNG_CANCER"] == target_label].copy()
        sub["GENDER_NUM"] = (sub["GENDER"] == "M").astype(float)
        for sym in SYMPTOM_FEATURES:
            sub[sym] = (sub[sym] == 2).astype(float)

        age_mean = float(sub["AGE"].mean())
        age_std = float(sub["AGE"].std()) if sub["AGE"].std() > 0 else 5.0
        male_prob = float(sub["GENDER_NUM"].mean())
        symptom_probs = {
            col: float(sub[col].mean()) for col in SYMPTOM_FEATURES
        }

        # Empirical correlation matrix across features
        feat_matrix = sub[all_numeric_features].values.astype(float)
        # Compute correlation with safeguards against zero variance
        cov = np.cov(feat_matrix, rowvar=False)
        stds = np.sqrt(np.diag(cov))
        stds[stds == 0] = 1.0
        corr = cov / np.outer(stds, stds)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)

        # Ensure positive semi-definiteness via eigenvalue clipping
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals_clipped = np.maximum(eigvals, 1e-4)
        corr_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        diag_sqrt = np.sqrt(np.diag(corr_psd))
        diag_sqrt[diag_sqrt == 0] = 1.0
        corr_psd = corr_psd / np.outer(diag_sqrt, diag_sqrt)
        np.fill_diagonal(corr_psd, 1.0)

        cohort_profiles[target_label] = {
            "age_mean": age_mean,
            "age_std": age_std,
            "male_prob": male_prob,
            "symptom_probs": symptom_probs,
            "corr_matrix": corr_psd,
        }

    n_pos = int(n_samples * positive_ratio)
    n_neg = n_samples - n_pos

    import math

    records = []
    for target_label, count in [("YES", n_pos), ("NO", n_neg)]:
        prof = cohort_profiles[target_label]
        corr_psd = prof["corr_matrix"]

        # Multivariate normal draws with empirical covariance structure
        L = np.linalg.cholesky(corr_psd)
        standard_normals = rng.randn(count, len(all_numeric_features))
        correlated_normals = standard_normals @ L.T

        for i in range(count):
            z_row = correlated_normals[i]
            # Gaussian copula probability integral transform
            u_row = [0.5 * (1.0 + math.erf(z / math.sqrt(2.0))) for z in z_row]

            # 0: Gender
            u_gender = u_row[0]
            gender = "M" if u_gender < prof["male_prob"] else "F"

            # 1: Age
            z_age = z_row[1]
            age = int(np.clip(prof["age_mean"] + prof["age_std"] * z_age, 20, 95))

            row: dict[str, object] = {
                "GENDER": gender,
                "AGE": age,
            }

            # 2..14: Symptoms
            for idx, sym in enumerate(SYMPTOM_FEATURES, start=2):
                p = prof["symptom_probs"][sym]
                u_sym = u_row[idx]
                row[sym] = 2 if u_sym >= (1.0 - p) else 1

            row["LUNG_CANCER"] = target_label
            records.append(row)

    synthetic_df = pd.DataFrame(records)

    # Shuffle synthetic records
    synthetic_df = synthetic_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_csv(output_csv_path, index=False)

    return synthetic_df


def build_primary_cohort_1309(
    source_csv_path: str | Path | None = None,
    output_cohort_path: str | Path | None = None,
    synthetic_samples: int = 1000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Construct the primary 1,309-record benchmark cohort by joining empirical survey records (N=309)
    with the correlation-preserving Gaussian copula expansion (N=1,000).
    """
    base_dir = Path(__file__).resolve().parent.parent
    if source_csv_path is None:
        source_csv_path = base_dir / "data" / "raw" / "survey_lung_cancer.csv"
    else:
        source_csv_path = Path(source_csv_path)

    if output_cohort_path is None:
        output_cohort_path = base_dir / "data" / "lung_cancer_cohort_1309.csv"
    else:
        output_cohort_path = Path(output_cohort_path)

    synth_df = generate_synthetic_cohort(
        source_csv_path=source_csv_path,
        output_csv_path=base_dir / "data" / "synthetic_cohort_1000.csv",
        n_samples=synthetic_samples,
        random_seed=random_seed,
    )

    empirical_df = clean_feature_names(pd.read_csv(source_csv_path))

    # Standardize column structure
    col_order = FEATURE_ORDER + ["LUNG_CANCER"]
    combined_df = pd.concat([empirical_df[col_order], synth_df[col_order]], ignore_index=True)

    output_cohort_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_cohort_path, index=False)
    return combined_df


if __name__ == "__main__":
    cohort = build_primary_cohort_1309()
    print(f"Generated primary benchmark cohort of shape: {cohort.shape}")
    print(cohort["LUNG_CANCER"].value_counts())
