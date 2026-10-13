"""
Prediction Service for FastAPI Backend.
Executes inference on validated request schemas using the cached CatGBM pipeline.
"""

from __future__ import annotations

import pandas as pd
from backend.app.schemas import ContributingFactor, PredictionInput, PredictionOutput
from backend.app.services.model_loader import get_metadata, get_pipeline
from ml.predict import DISCLAIMER_TEXT, HUMAN_READABLE_SYMPTOMS
from ml.preprocessing import FEATURE_ORDER


def predict_from_input(payload: PredictionInput) -> PredictionOutput:
    """
    Run prediction on incoming Pydantic input.
    """
    pipeline = get_pipeline()
    metadata = get_metadata()

    threshold = float(metadata.get("optimal_decision_threshold", 0.5))
    feature_importances = metadata.get("feature_importances", {})

    # Convert payload into DataFrame with uppercase column names
    data_dict = {
        k.upper(): v for k, v in payload.model_dump().items()
    }
    df = pd.DataFrame([data_dict])

    probs = pipeline.predict_proba(df)
    prob_pos = float(probs[0, 1])
    predicted_class = "YES" if prob_pos >= threshold else "NO"

    # Compute contributing factors
    factors: list[ContributingFactor] = []
    impact_rank = {"high": 3, "moderate": 2, "low": 1}

    for feat in FEATURE_ORDER:
        val = data_dict.get(feat)
        importance = feature_importances.get(feat, 0.05)
        readable_name = HUMAN_READABLE_SYMPTOMS.get(feat, feat)

        if feat == "AGE":
            age_val = int(val)
            if age_val >= 65:
                impact = "high" if importance > 0.08 else "moderate"
                desc = f"Age {age_val} is in an elevated risk age cohort."
            elif age_val >= 50:
                impact = "moderate"
                desc = f"Age {age_val} represents middle-aged demographic."
            else:
                impact = "low"
                desc = f"Age {age_val} is in younger, lower baseline risk demographic."

            factors.append(ContributingFactor(
                feature=feat,
                label=readable_name,
                value=str(age_val),
                importance_score=round(importance, 4),
                impact=impact,
                description=desc,
            ))

        elif feat == "GENDER":
            g_str = "Male" if str(val).upper() in ["M", "1"] else "Female"
            factors.append(ContributingFactor(
                feature=feat,
                label=readable_name,
                value=g_str,
                importance_score=round(importance, 4),
                impact="low",
                description=f"Participant identified as {g_str}.",
            ))

        else:
            is_present = int(val) == 2
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

            factors.append(ContributingFactor(
                feature=feat,
                label=readable_name,
                value="Present" if is_present else "Absent",
                importance_score=round(importance, 4),
                impact=impact,
                description=desc,
            ))

    # Sort factors by impact then importance
    factors.sort(
        key=lambda x: (impact_rank.get(x.impact, 0), x.importance_score),
        reverse=True,
    )

    return PredictionOutput(
        model_name=metadata.get("model_name", "CatGBM Ensemble (CatBoost + LightGBM + XGBoost)"),
        model_version=metadata.get("version", "1.0.0"),
        predicted_class=predicted_class,
        probability_score=round(prob_pos, 4),
        probability_percentage=round(prob_pos * 100, 2),
        decision_threshold=threshold,
        contributing_factors=factors,
        disclaimer=DISCLAIMER_TEXT,
    )
