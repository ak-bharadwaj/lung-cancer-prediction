"""
API Route Handlers for Prediction, Model Metadata, Metrics, and EDA.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.app.config import settings
from backend.app.schemas import (
    ExplanationResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionInput,
    PredictionOutput,
)
from backend.app.services.explanation import get_explanation_for_prediction
from backend.app.services.model_loader import get_metadata, get_pipeline
from backend.app.services.prediction import predict_from_input
from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Check service health and verify CatGBM model artifact readiness."""
    try:
        pipeline = get_pipeline()
        is_ready = pipeline is not None
    except Exception:
        is_ready = False

    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_loaded=is_ready,
        version=settings.api_version,
    )


@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_information() -> ModelInfoResponse:
    """Return model architecture, ensemble weights, features, and optimal threshold."""
    meta = get_metadata()
    if not meta:
        raise HTTPException(status_code=500, detail="Model metadata is unavailable.")

    return ModelInfoResponse(
        model_name=meta.get("model_name", "CatGBM Ensemble"),
        version=meta.get("version", "1.0.0"),
        description=meta.get("description", "High-Performance CatGBM Ensemble"),
        estimators=meta.get("estimators", ["CatBoost", "LightGBM", "XGBoost"]),
        ensemble_weights=meta.get("ensemble_weights", {}),
        optimal_decision_threshold=float(meta.get("optimal_decision_threshold", 0.5)),
        features=meta.get("features", []),
        feature_importances=meta.get("feature_importances", {}),
        split_summary=meta.get("split_summary", {}),
    )


@router.get("/metrics")
def get_evaluation_metrics() -> dict[str, Any]:
    """Retrieve candidate model comparison and locked test split evaluation metrics."""
    reports_path = Path(settings.reports_dir)
    comp_file = reports_path / "model_comparison.json"
    test_file = reports_path / "test_evaluation.json"
    cp_comp_file = reports_path / "cancer_patient_model_comparison.json"
    cp_test_file = reports_path / "cancer_patient_test_evaluation.json"

    comparison_data = []
    test_data = {}
    cp_comparison = []
    cp_test = {}

    if comp_file.exists():
        with open(comp_file, "r", encoding="utf-8") as f:
            comparison_data = json.load(f)

    if test_file.exists():
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)

    if cp_comp_file.exists():
        with open(cp_comp_file, "r", encoding="utf-8") as f:
            cp_comparison = json.load(f)

    if cp_test_file.exists():
        with open(cp_test_file, "r", encoding="utf-8") as f:
            cp_test = json.load(f)

    return {
        "model_comparison": comparison_data,
        "test_evaluation": test_data,
        "cancer_patient_comparison": cp_comparison,
        "cancer_patient_test_evaluation": cp_test,
    }


@router.get("/metrics/cancer-patient")
def get_cancer_patient_metrics() -> dict[str, Any]:
    """Retrieve Kaggle 1,000-Patient benchmark comparisons and test evaluation metrics."""
    reports_path = Path(settings.reports_dir)
    cp_comp_file = reports_path / "cancer_patient_model_comparison.json"
    cp_test_file = reports_path / "cancer_patient_test_evaluation.json"

    cp_comparison = []
    cp_test = {}

    if cp_comp_file.exists():
        with open(cp_comp_file, "r", encoding="utf-8") as f:
            cp_comparison = json.load(f)

    if cp_test_file.exists():
        with open(cp_test_file, "r", encoding="utf-8") as f:
            cp_test = json.load(f)

    return {
        "cancer_patient_comparison": cp_comparison,
        "cancer_patient_test_evaluation": cp_test,
    }


@router.get("/eda")
def get_eda_data() -> dict[str, Any]:
    """Retrieve precomputed statistical distributions and data audit reports."""
    reports_path = Path(settings.reports_dir)
    eda_file = reports_path / "eda_summary.json"
    audit_file = reports_path / "data_audit.json"

    eda_data = {}
    audit_data = {}

    if eda_file.exists():
        with open(eda_file, "r", encoding="utf-8") as f:
            eda_data = json.load(f)

    if audit_file.exists():
        with open(audit_file, "r", encoding="utf-8") as f:
            audit_data = json.load(f)

    return {
        "eda_summary": eda_data,
        "data_audit": audit_data,
    }


@router.post("/predict", response_model=PredictionOutput)
def predict_endpoint(payload: PredictionInput) -> PredictionOutput:
    """Predict lung cancer survey risk score for 15 input features."""
    try:
        return predict_from_input(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc!s}")


@router.post("/explain", response_model=ExplanationResponse)
def explain_endpoint(payload: PredictionInput) -> ExplanationResponse:
    """Generate prediction alongside contextual educational explanation (LLM or fallback)."""
    try:
        prediction = predict_from_input(payload)
        return get_explanation_for_prediction(prediction)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explanation generation error: {exc!s}")
