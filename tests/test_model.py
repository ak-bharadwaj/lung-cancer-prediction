"""
Unit and Integration Tests for ML Modeling and CatGBM Pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from ml.predict import CatGBMPredictor
from ml.preprocessing import SurveyFeatureTransformer
from ml.train import find_optimal_threshold

BASE_DIR = Path(__file__).resolve().parent.parent
PIPELINE_PATH = BASE_DIR / "models" / "pipeline.joblib"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"
TEST_EVAL_PATH = BASE_DIR / "reports" / "test_evaluation.json"


def test_survey_feature_transformer():
    """Verify transformer encodes gender, scales age, and maps symptom scales."""
    sample_df = pd.DataFrame([{
        "GENDER": "M",
        "AGE": 60,
        "SMOKING": 2,
        "YELLOW_FINGERS": 1,
        "ANXIETY": 2,
        "PEER_PRESSURE": 1,
        "CHRONIC_DISEASE": 2,
        "FATIGUE": 2,
        "ALLERGY": 1,
        "WHEEZING": 2,
        "ALCOHOL_CONSUMING": 1,
        "COUGHING": 2,
        "SHORTNESS_OF_BREATH": 2,
        "SWALLOWING_DIFFICULTY": 1,
        "CHEST_PAIN": 2,
    }])

    transformer = SurveyFeatureTransformer(scale_age=True)
    transformed = transformer.fit_transform(sample_df)

    assert transformed.shape == (1, 15)
    # GENDER is index 0 -> 'M' becomes 1.0
    assert transformed[0, 0] == 1.0
    # SMOKING is index 2 -> 2 becomes 1.0
    assert transformed[0, 2] == 1.0
    # YELLOW_FINGERS is index 3 -> 1 becomes 0.0
    assert transformed[0, 3] == 0.0


def test_find_optimal_threshold():
    """Verify threshold search selects threshold with high macro-f1."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
    best_th, best_f1 = find_optimal_threshold(y_true, probs)
    assert 0.25 <= best_th <= 0.75
    assert best_f1 > 0.80


def test_pipeline_artifact_and_metadata():
    """Verify saved pipeline exists, loads properly, and metadata is complete."""
    assert PIPELINE_PATH.exists(), "Model pipeline.joblib must exist."
    assert METADATA_PATH.exists(), "Model metadata.json must exist."

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["model_name"] == "CatGBM Ensemble"
    assert "optimal_decision_threshold" in meta
    assert 0.1 <= float(meta["optimal_decision_threshold"]) <= 0.9
    assert len(meta["feature_importances"]) == 15


def test_catgbm_predictor():
    """Verify predictor generates valid probability scores and contributing factors."""
    predictor = CatGBMPredictor()

    sample = {
        "GENDER": "M",
        "AGE": 65,
        "SMOKING": 2,
        "YELLOW_FINGERS": 2,
        "ANXIETY": 2,
        "PEER_PRESSURE": 2,
        "CHRONIC_DISEASE": 2,
        "FATIGUE": 2,
        "ALLERGY": 2,
        "WHEEZING": 2,
        "ALCOHOL_CONSUMING": 2,
        "COUGHING": 2,
        "SHORTNESS_OF_BREATH": 2,
        "SWALLOWING_DIFFICULTY": 2,
        "CHEST_PAIN": 2,
    }

    result = predictor.predict_sample(sample)
    assert result["predicted_class"] in ["YES", "NO"]
    assert 0.0 <= result["probability_score"] <= 1.0
    assert 0.0 <= result["probability_percentage"] <= 100.0
    assert len(result["contributing_factors"]) == 15
    assert "disclaimer" in result


def test_test_evaluation_report():
    """Verify test_evaluation.json contains comprehensive test metrics."""
    assert TEST_EVAL_PATH.exists(), "reports/test_evaluation.json must exist."
    with open(TEST_EVAL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data["metrics"]
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "roc_auc" in metrics
    assert "minority_class_no" in metrics
    assert metrics["minority_class_no"]["recall"] >= 0.50
    assert "confusion_matrix" in data


def test_cancer_patient_evaluation_report():
    """Verify cancer_patient_test_evaluation.json exists and contains 23 features."""
    cp_test_path = BASE_DIR / "reports" / "cancer_patient_test_evaluation.json"
    cp_comp_path = BASE_DIR / "reports" / "cancer_patient_model_comparison.json"
    assert cp_test_path.exists(), "reports/cancer_patient_test_evaluation.json must exist."
    assert cp_comp_path.exists(), "reports/cancer_patient_model_comparison.json must exist."

    with open(cp_test_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_samples"] == 1000
    assert "accuracy" in data["metrics"]
    assert "roc_auc" in data["metrics"]
    assert len(data["feature_importances"]) == 23
    assert "deduplication_audit" in data

