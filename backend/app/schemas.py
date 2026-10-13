"""
Pydantic Schemas for API Request and Response Validation.
Enforces strict boundaries, typing, and self-documenting data structures.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    """
    Survey participant questionnaire responses.
    Symptom scales: 1 = No / Low, 2 = Yes / High.
    """
    gender: Literal["M", "F"] = Field(
        ..., description="Gender identifier ('M' for Male, 'F' for Female)."
    )
    age: int = Field(
        ..., ge=1, le=120, description="Participant age in years (1 to 120)."
    )
    smoking: Literal[1, 2] = Field(
        ..., description="Smoking status: 1 = Non-smoker, 2 = Active / Former smoker."
    )
    yellow_fingers: Literal[1, 2] = Field(
        ..., description="Yellow finger staining: 1 = Absent, 2 = Present."
    )
    anxiety: Literal[1, 2] = Field(
        ..., description="Anxiety symptoms: 1 = Absent/Low, 2 = Present/High."
    )
    peer_pressure: Literal[1, 2] = Field(
        ..., description="Peer / environmental exposure to smoke: 1 = Low, 2 = High."
    )
    chronic_disease: Literal[1, 2] = Field(
        ..., description="Pre-existing chronic lung/respiratory condition: 1 = No, 2 = Yes."
    )
    fatigue: Literal[1, 2] = Field(
        ..., description="Fatigue and lethargy: 1 = Normal, 2 = Frequent / Persistent."
    )
    allergy: Literal[1, 2] = Field(
        ..., description="Allergies / respiratory sensitivity: 1 = None, 2 = Present."
    )
    wheezing: Literal[1, 2] = Field(
        ..., description="Wheezing or stridor breathing sound: 1 = None, 2 = Frequent."
    )
    alcohol_consuming: Literal[1, 2] = Field(
        ..., description="Alcohol consumption: 1 = Low / None, 2 = Regular / High."
    )
    coughing: Literal[1, 2] = Field(
        ..., description="Persistent cough: 1 = Infrequent / None, 2 = Frequent / Chronic."
    )
    shortness_of_breath: Literal[1, 2] = Field(
        ..., description="Dyspnea / Shortness of breath: 1 = None, 2 = Present."
    )
    swallowing_difficulty: Literal[1, 2] = Field(
        ..., description="Dysphagia / Difficulty swallowing: 1 = None, 2 = Present."
    )
    chest_pain: Literal[1, 2] = Field(
        ..., description="Chest pain / pressure: 1 = None, 2 = Present."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "M",
                "age": 69,
                "smoking": 2,
                "yellow_fingers": 2,
                "anxiety": 2,
                "peer_pressure": 1,
                "chronic_disease": 1,
                "fatigue": 2,
                "allergy": 1,
                "wheezing": 2,
                "alcohol_consuming": 2,
                "coughing": 2,
                "shortness_of_breath": 2,
                "swallowing_difficulty": 2,
                "chest_pain": 2,
            }
        }
    }


class ContributingFactor(BaseModel):
    feature: str
    label: str
    value: str
    importance_score: float
    impact: Literal["high", "moderate", "low"]
    description: str


class PredictionOutput(BaseModel):
    model_name: str = "CatGBM Ensemble (CatBoost + LightGBM + XGBoost)"
    model_version: str = "1.0.0"
    predicted_class: Literal["YES", "NO"]
    probability_score: float
    probability_percentage: float
    decision_threshold: float
    contributing_factors: list[ContributingFactor]
    disclaimer: str


class ExplanationResponse(BaseModel):
    prediction: PredictionOutput
    explanation: str
    source: Literal["llm", "deterministic_fallback"]
    disclaimer: str


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    description: str
    estimators: list[str]
    ensemble_weights: dict[str, float]
    optimal_decision_threshold: float
    features: list[str]
    feature_importances: dict[str, float]
    split_summary: dict[str, int]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
