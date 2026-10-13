"""
Unit Tests for LLM Explanation Service, Guardrails, and Fallback Engine.
"""

from __future__ import annotations

import pytest
from backend.app.schemas import ContributingFactor, PredictionOutput
from backend.app.services.explanation import (
    get_explanation_for_prediction,
    validate_explanation_guardrails,
)
from backend.app.services.fallback import generate_deterministic_explanation


@pytest.fixture
def dummy_prediction_yes() -> PredictionOutput:
    factors = [
        ContributingFactor(
            feature="ALLERGY",
            label="Respiratory Allergies",
            value="Present",
            importance_score=0.12,
            impact="high",
            description="Positive report of respiratory allergies.",
        ),
        ContributingFactor(
            feature="WHEEZING",
            label="Wheezing / Stridor",
            value="Present",
            importance_score=0.10,
            impact="high",
            description="Positive report of wheezing.",
        ),
        ContributingFactor(
            feature="CHEST_PAIN",
            label="Chest Pain",
            value="Absent",
            importance_score=0.04,
            impact="low",
            description="Absence of reported chest pain (protective).",
        ),
    ]
    return PredictionOutput(
        model_name="CatGBM Ensemble",
        model_version="1.0.0",
        predicted_class="YES",
        probability_score=0.825,
        probability_percentage=82.5,
        decision_threshold=0.55,
        contributing_factors=factors,
        disclaimer="EDUCATIONAL DISCLAIMER ONLY",
    )


@pytest.fixture
def dummy_prediction_no() -> PredictionOutput:
    factors = [
        ContributingFactor(
            feature="SMOKING",
            label="Smoking History",
            value="Absent",
            importance_score=0.07,
            impact="low",
            description="Absence of reported smoking (protective).",
        )
    ]
    return PredictionOutput(
        model_name="CatGBM Ensemble",
        model_version="1.0.0",
        predicted_class="NO",
        probability_score=0.15,
        probability_percentage=15.0,
        decision_threshold=0.55,
        contributing_factors=factors,
        disclaimer="EDUCATIONAL DISCLAIMER ONLY",
    )


def test_guardrails_valid_text():
    """Verify standard educational response passes guardrail check."""
    valid_text = (
        "Based on the survey data, the model identified elevated statistical correlation "
        "with respiratory symptoms. This is an educational tool and not a clinical diagnosis. "
        "Please speak with a physician for personal medical evaluation."
    )
    assert validate_explanation_guardrails(valid_text, "YES") is True


def test_guardrails_reject_definitive_diagnosis():
    """Verify guardrails reject texts claiming definitive medical certainty."""
    bad_text = "According to our model, you have cancer and the diagnosis is confirmed."
    assert validate_explanation_guardrails(bad_text, "YES") is False


def test_guardrails_reject_contradiction():
    """Verify guardrails reject texts claiming zero risk when model predicted YES."""
    contradictory_text = (
        "Good news, you are completely cancer-free and have zero risk according to this scan."
    )
    assert validate_explanation_guardrails(contradictory_text, "YES") is False


def test_deterministic_fallback_generation(dummy_prediction_yes, dummy_prediction_no):
    """Verify deterministic fallback produces formatted text with factors and disclaimer."""
    text_yes = generate_deterministic_explanation(dummy_prediction_yes)
    assert "Elevated Risk Pattern" in text_yes
    assert "82.5%" in text_yes
    assert "Respiratory Allergies" in text_yes
    assert "Protective Indicators" in text_yes

    text_no = generate_deterministic_explanation(dummy_prediction_no)
    assert "Low / Baseline Risk Pattern" in text_no
    assert "15.0%" in text_no


def test_get_explanation_fallback_activation(dummy_prediction_yes):
    """Verify fallback is used when no LLM API key is configured."""
    res = get_explanation_for_prediction(dummy_prediction_yes)
    assert res.source == "deterministic_fallback"
    assert len(res.explanation) > 100
    assert "EDUCATIONAL DISCLAIMER" in res.disclaimer
