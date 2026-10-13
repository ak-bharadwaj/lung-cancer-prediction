"""
Inference and Prediction Service for CatGBM Ensemble.
Provides single-sample and batch prediction functions with feature contribution
attribution and calibrated decision thresholding.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Any

import joblib
import pandas as pd

from ml.preprocessing import FEATURE_ORDER

HUMAN_READABLE_SYMPTOMS = {
    "SMOKING": "Smoking History",
    "YELLOW_FINGERS": "Yellow Finger Staining",
    "ANXIETY": "Anxiety Symptoms",
    "PEER_PRESSURE": "Peer Exposure to Smoking",
    "CHRONIC_DISEASE": "Chronic Respiratory / Lung Condition",
    "FATIGUE": "Fatigue and Low Energy",
    "ALLERGY": "Respiratory Allergies",
    "WHEEZING": "Wheezing / Stridor",
    "ALCOHOL_CONSUMING": "Alcohol Consumption",
    "COUGHING": "Persistent Cough",
    "SHORTNESS_OF_BREATH": "Shortness of Breath (Dyspnea)",
    "SWALLOWING_DIFFICULTY": "Difficulty Swallowing (Dysphagia)",
    "CHEST_PAIN": "Chest Pain / Tightness",
    "AGE": "Participant Age",
    "GENDER": "Gender Demographic",
}

DISCLAIMER_TEXT = (
    "EDUCATIONAL DISCLAIMER: This machine learning prediction and analysis is strictly "
    "for educational and informational purposes. It is NOT a clinical screening tool, medical "
    "diagnosis, or substitute for professional medical consultation. Always consult a qualified "
    "healthcare provider for medical advice."
)


class CatGBMPredictor:
    """Inference predictor for the trained CatGBM Ensemble pipeline."""

    def __init__(
        self,
        pipeline_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ):
        base_dir = Path(__file__).resolve().parent.parent
        self.pipeline_path = (
            Path(pipeline_path) if pipeline_path else base_dir / "models" / "pipeline.joblib"
        )
        self.metadata_path = (
            Path(metadata_path) if metadata_path else base_dir / "models" / "metadata.json"
        )

        if not self.pipeline_path.exists():
            raise FileNotFoundError(
                f"Model pipeline not found at {self.pipeline_path}. Run ml/train.py first."
            )

        self.pipeline = joblib.load(self.pipeline_path)
        self.metadata = self._load_metadata()
        self.threshold = float(self.metadata.get("optimal_decision_threshold", 0.5))
        self.feature_importances: dict[str, float] = self.metadata.get(
            "feature_importances", {}
        )

    def _load_metadata(self) -> dict[str, Any]:
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "model_name": "CatGBM Ensemble",
            "version": "1.0.0",
            "optimal_decision_threshold": 0.5,
        }

    def predict_sample(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """
        Predict lung cancer survey class for a single participant record.
        """
        # Build DataFrame with standardized uppercase column names
        cleaned_dict = {
            k.strip().replace(" ", "_").upper(): v for k, v in input_data.items()
        }
        df_input = pd.DataFrame([cleaned_dict])

        # Predict probability
        prob_positive = float(self.pipeline.predict_proba(df_input)[0, 1])
        predicted_class = "YES" if prob_positive >= self.threshold else "NO"

        # Calculate contributing factors
        contributing_factors = self._compute_contributing_factors(cleaned_dict)

        return {
            "model_name": self.metadata.get("model_name", "CatGBM Ensemble"),
            "model_version": self.metadata.get("version", "1.0.0"),
            "predicted_class": predicted_class,
            "probability_score": round(prob_positive, 4),
            "probability_percentage": round(prob_positive * 100, 2),
            "decision_threshold": self.threshold,
            "contributing_factors": contributing_factors,
            "disclaimer": DISCLAIMER_TEXT,
        }

    def _compute_contributing_factors(
        self, input_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Identify top factors influencing the prediction based on feature presence and importance.
        """
        factors = []
        for feat in FEATURE_ORDER:
            val = input_data.get(feat)
            importance = self.feature_importances.get(feat, 0.05)
            readable_name = HUMAN_READABLE_SYMPTOMS.get(feat, feat)

            if feat == "AGE":
                age_val = int(val) if val is not None else 60
                if age_val >= 65:
                    impact = "high" if importance > 0.08 else "moderate"
                    desc = f"Age {age_val} is in an elevated risk demographic in survey data."
                elif age_val >= 50:
                    impact = "moderate"
                    desc = f"Age {age_val} aligns with middle-to-senior age category."
                else:
                    impact = "low"
                    desc = f"Age {age_val} is in a younger demographic cohort."

                factors.append({
                    "feature": feat,
                    "label": readable_name,
                    "value": str(age_val),
                    "importance_score": importance,
                    "impact": impact,
                    "description": desc,
                })

            elif feat == "GENDER":
                g_str = str(val).upper() if val is not None else "M"
                factors.append({
                    "feature": feat,
                    "label": readable_name,
                    "value": "Male" if g_str in ["M", "1"] else "Female",
                    "importance_score": importance,
                    "impact": "low",
                    "description": f"Gender demographic ({'Male' if g_str in ['M', '1'] else 'Female'}) in survey cohort.",
                })

            else:
                # Symptom feature (1 = No/Low, 2 = Yes/High)
                symptom_val = int(val) if val is not None else 1
                is_present = symptom_val == 2

                if is_present:
                    if importance >= 0.09:
                        impact = "high"
                    elif importance >= 0.06:
                        impact = "moderate"
                    else:
                        impact = "low"
                    desc = f"Positive report of {readable_name.lower()}."
                else:
                    impact = "low"
                    desc = f"Absence of reported {readable_name.lower()} (protective)."

                factors.append({
                    "feature": feat,
                    "label": readable_name,
                    "value": "Present" if is_present else "Absent",
                    "importance_score": importance,
                    "impact": impact,
                    "description": desc,
                })

        # Rank factors: high impact first, then descending importance
        impact_order = {"high": 3, "moderate": 2, "low": 1}
        factors.sort(
            key=lambda x: (impact_order.get(x["impact"], 0), x["importance_score"]),
            reverse=True,
        )
        return factors
