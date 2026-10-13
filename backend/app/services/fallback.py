"""
Deterministic Feature-Attribution Fallback Engine.
Generates structured, plain-language educational explanations when LLM APIs
are unavailable, unconfigured, or fail guardrail checks.
"""

from __future__ import annotations

from backend.app.schemas import PredictionOutput


def generate_deterministic_explanation(prediction: PredictionOutput) -> str:
    """
    Generate an educational, plain-language synthesis of the model's prediction
    and top contributing factors.
    """
    pred_class = prediction.predicted_class
    prob_pct = prediction.probability_percentage
    threshold_pct = round(prediction.decision_threshold * 100, 1)

    high_factors = [f for f in prediction.contributing_factors if f.impact == "high"]
    mod_factors = [f for f in prediction.contributing_factors if f.impact == "moderate"]
    protective_factors = [
        f for f in prediction.contributing_factors if f.value == "Absent" and f.impact == "low"
    ]

    lines: list[str] = []

    # Section 1: Overview
    if pred_class == "YES":
        lines.append(
            f"### Survey Analysis: Elevated Risk Pattern Identified\n"
            f"The **CatGBM Ensemble** calculated a **{prob_pct}% probability score**, which exceeds "
            f"the calibrated validation decision threshold of **{threshold_pct}%**. Based on historical "
            f"survey patterns, this questionnaire profile aligns more closely with participants diagnosed with lung cancer."
        )
    else:
        lines.append(
            f"### Survey Analysis: Low / Baseline Risk Pattern Identified\n"
            f"The **CatGBM Ensemble** calculated a **{prob_pct}% probability score**, which is below "
            f"the calibrated validation decision threshold of **{threshold_pct}%**. Based on historical "
            f"survey patterns, this profile indicates a low likelihood within the surveyed cohort."
        )

    # Section 2: Key Risk Drivers
    if high_factors or mod_factors:
        lines.append("\n#### Primary Contributing Factors:")
        for factor in high_factors:
            lines.append(
                f"- **{factor.label}** ({factor.value}): Elevated risk factor (Feature Importance: {round(factor.importance_score * 100, 1)}%). {factor.description}"
            )
        for factor in mod_factors[:3]:
            lines.append(
                f"- **{factor.label}** ({factor.value}): Moderate contributing indicator. {factor.description}"
            )

    # Section 3: Protective Factors
    if protective_factors:
        protective_names = [f.label for f in protective_factors[:4]]
        lines.append(
            f"\n#### Notable Protective Indicators:\n"
            f"The reported absence of key symptoms ({', '.join(protective_names)}) "
            f"positively reduced the predicted risk probability score."
        )

    # Section 4: Educational Perspective
    lines.append(
        "\n#### Educational Note on Survey-Based Modeling:\n"
        "This assessment reflects statistical correlations within our benchmark cohort dataset. "
        "Symptoms such as wheezing, persistent coughing, and chronic fatigue can originate from diverse "
        "benign or treatable conditions (e.g., asthma, allergies, acid reflux, or seasonal respiratory infections)."
    )

    return "\n".join(lines)
