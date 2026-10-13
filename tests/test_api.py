"""
FastAPI Endpoints Integration Tests using TestClient.
"""

from __future__ import annotations

from backend.app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

VALID_PAYLOAD = {
    "gender": "M",
    "age": 65,
    "smoking": 2,
    "yellow_fingers": 2,
    "anxiety": 2,
    "peer_pressure": 1,
    "chronic_disease": 2,
    "fatigue": 2,
    "allergy": 1,
    "wheezing": 2,
    "alcohol_consuming": 2,
    "coughing": 2,
    "shortness_of_breath": 2,
    "swallowing_difficulty": 2,
    "chest_pain": 2,
}


def test_health_endpoint():
    """Verify health endpoint returns status 200 and model_loaded=True."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

    # Also check /api/health prefix
    res_api = client.get("/api/health")
    assert res_api.status_code == 200
    assert res_api.json()["status"] == "healthy"


def test_model_info_endpoint():
    """Verify model-info endpoint returns ensemble configuration and features."""
    res = client.get("/model-info")
    assert res.status_code == 200
    data = res.json()
    assert data["model_name"] == "CatGBM Ensemble"
    assert "CatBoost" in data["estimators"]
    assert "LightGBM" in data["estimators"]
    assert "XGBoost" in data["estimators"]
    assert len(data["features"]) == 15
    assert len(data["feature_importances"]) == 15


def test_metrics_endpoint():
    """Verify metrics endpoint returns candidate model comparison and test metrics."""
    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "model_comparison" in data
    assert "test_evaluation" in data
    assert len(data["model_comparison"]) >= 5
    assert "cancer_patient_comparison" in data
    assert "cancer_patient_test_evaluation" in data


def test_cancer_patient_metrics_endpoint():
    """Verify dedicated /metrics/cancer-patient endpoint."""
    res = client.get("/metrics/cancer-patient")
    assert res.status_code == 200
    data = res.json()
    assert "cancer_patient_comparison" in data
    assert "cancer_patient_test_evaluation" in data
    assert len(data["cancer_patient_comparison"]) >= 5



def test_eda_endpoint():
    """Verify eda endpoint returns distributions and demographic data."""
    res = client.get("/eda")
    assert res.status_code == 200
    data = res.json()
    assert "eda_summary" in data
    assert "data_audit" in data


def test_predict_success():
    """Verify prediction endpoint with valid payload."""
    res = client.post("/predict", json=VALID_PAYLOAD)
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_class"] in ["YES", "NO"]
    assert 0.0 <= data["probability_score"] <= 1.0
    assert len(data["contributing_factors"]) == 15
    assert "disclaimer" in data


def test_predict_boundary_ages():
    """Verify predict accepts minimum (1) and maximum (120) valid boundary ages."""
    for age in [1, 120]:
        payload = VALID_PAYLOAD.copy()
        payload["age"] = age
        res = client.post("/predict", json=payload)
        assert res.status_code == 200


def test_predict_invalid_inputs():
    """Verify schema validation rejects invalid inputs with HTTP 422."""
    # Age below 1
    invalid_age_low = VALID_PAYLOAD.copy()
    invalid_age_low["age"] = 0
    assert client.post("/predict", json=invalid_age_low).status_code == 422

    # Age above 120
    invalid_age_high = VALID_PAYLOAD.copy()
    invalid_age_high["age"] = 121
    assert client.post("/predict", json=invalid_age_high).status_code == 422

    # Invalid symptom scale value (3 instead of 1 or 2)
    invalid_symptom = VALID_PAYLOAD.copy()
    invalid_symptom["smoking"] = 3
    assert client.post("/predict", json=invalid_symptom).status_code == 422

    # Invalid gender
    invalid_gender = VALID_PAYLOAD.copy()
    invalid_gender["gender"] = "Other"
    assert client.post("/predict", json=invalid_gender).status_code == 422


def test_explain_endpoint():
    """Verify explain endpoint returns prediction alongside explanation."""
    res = client.post("/explain", json=VALID_PAYLOAD)
    assert res.status_code == 200
    data = res.json()
    assert "prediction" in data
    assert "explanation" in data
    assert data["source"] in ["llm", "deterministic_fallback"]
    assert len(data["explanation"]) > 50
