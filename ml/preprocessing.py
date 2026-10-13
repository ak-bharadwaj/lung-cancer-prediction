"""
Preprocessing Module for Survey Lung Cancer Dataset.
Provides scikit-learn compatible transformation pipeline to ensure
zero data leakage across training, validation, test, and inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

SYMPTOM_FEATURES = [
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

FEATURE_ORDER = ["GENDER", "AGE"] + SYMPTOM_FEATURES


def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names by stripping whitespace, replacing spaces with underscores, and capitalizing."""
    df_clean = df.copy()
    df_clean.columns = [
        str(col).strip().replace(" ", "_").upper() for col in df_clean.columns
    ]
    return df_clean


class SurveyFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms raw survey inputs into normalized numerical features.
    
    1. GENDER: 'M'/1 -> 1.0, 'F'/0 -> 0.0
    2. AGE: standard scaled using mean and std fitted exclusively on training data.
    3. Symptoms (13 features): converts survey scale [1 (No/Low), 2 (Yes/High)] to binary [0.0, 1.0].
       Also safely handles pre-converted [0, 1] values.
    """

    def __init__(self, scale_age: bool = True):
        self.scale_age = scale_age
        self.age_scaler = StandardScaler() if scale_age else None
        self.feature_names_out_: list[str] = FEATURE_ORDER.copy()
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame | np.ndarray | dict, y=None):
        df = self._ensure_dataframe(X)
        if self.scale_age:
            age_values = df[["AGE"]].values.astype(float)
            self.age_scaler.fit(age_values)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame | np.ndarray | dict) -> np.ndarray:
        df = self._ensure_dataframe(X)
        output = np.zeros((len(df), len(FEATURE_ORDER)), dtype=np.float32)

        # 1. Gender encoding (M=1, F=0)
        gender_col = df["GENDER"].astype(str).str.strip().str.upper()
        output[:, 0] = np.where(gender_col.isin(["M", "MALE", "1"]), 1.0, 0.0)

        # 2. Age standard scaling
        age_vals = df[["AGE"]].values.astype(float)
        if self.scale_age and self.age_scaler is not None:
            output[:, 1] = self.age_scaler.transform(age_vals).flatten()
        else:
            output[:, 1] = age_vals.flatten()

        # 3. Symptoms encoding: 1 -> 0.0, 2 -> 1.0 (if 0 or 1 already, keep as is)
        for idx, col in enumerate(SYMPTOM_FEATURES, start=2):
            val = df[col].astype(float).values
            # If values are in [1, 2], map 1->0, 2->1. If already in [0, 1], retain.
            normalized = np.where(val == 2.0, 1.0, np.where(val == 1.0, 0.0, val))
            output[:, idx] = normalized

        return output

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)

    def _ensure_dataframe(self, X: Any) -> pd.DataFrame:
        if isinstance(X, dict):
            # Single or multi-record dictionary
            if not any(isinstance(v, list) for v in X.values()):
                df = pd.DataFrame([X])
            else:
                df = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        elif isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=FEATURE_ORDER)
        else:
            df = pd.DataFrame(X)

        df = clean_feature_names(df)

        # Verify required columns
        for col in FEATURE_ORDER:
            if col not in df.columns:
                raise ValueError(f"Required feature '{col}' missing from input dataset.")

        return df[FEATURE_ORDER]


def load_and_preprocess_raw(
    csv_path: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load raw CSV, clean column names, separate X features and y target.
    
    Target: 'YES' -> 1, 'NO' -> 0.
    """
    raw_df = pd.read_csv(csv_path)
    df = clean_feature_names(raw_df)

    if "LUNG_CANCER" not in df.columns:
        raise ValueError("Target column 'LUNG_CANCER' not found in dataset.")

    y = (df["LUNG_CANCER"].astype(str).str.strip().str.upper() == "YES").astype(int)
    X = df[FEATURE_ORDER].copy()
    return X, y
