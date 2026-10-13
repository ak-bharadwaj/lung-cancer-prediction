"""
LLM Explanation Service with Guardrails and Deterministic Fallback.
Connects to OpenAI or Gemini when configured, validates outputs against safety
and consistency guardrails, and falls back to deterministic attribution.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from backend.app.config import settings
from backend.app.schemas import ExplanationResponse, PredictionOutput
from backend.app.services.fallback import generate_deterministic_explanation
from ml.predict import DISCLAIMER_TEXT

logger = logging.getLogger(__name__)


def validate_explanation_guardrails(explanation_text: str, predicted_class: str) -> bool:
    """
    Ensure the LLM output conforms to educational and medical safety standards:
    1. Does NOT contradict the model's predicted class.
    2. Does NOT provide clinical diagnosis or claim 100% certainty.
    3. Mentions educational context or consulting healthcare professionals.
    """
    text_lower = explanation_text.lower()

    # Rule 1: No clinical diagnosis claim
    forbidden_claims = [
        "you have cancer",
        "you definitely have",
        "diagnosis is confirmed",
        "clinical diagnosis: yes",
        "clinical diagnosis: no",
        "we diagnose you",
    ]
    for claim in forbidden_claims:
        if claim in text_lower:
            logger.warning(f"Guardrail violation: forbidden diagnostic claim '{claim}'")
            return False

    # Rule 2: No contradiction of prediction
    # Rule 2: No contradiction of prediction
    if predicted_class == "YES" and ("you are completely cancer-free" in text_lower or "zero risk" in text_lower):
        logger.warning("Guardrail violation: LLM contradicts YES prediction.")
        return False
    if predicted_class == "NO" and ("you have high probability of cancer" in text_lower or "you definitely have a malignancy" in text_lower):
        logger.warning("Guardrail violation: LLM contradicts NO prediction.")
        return False

    return True


def query_llm_provider(prediction: PredictionOutput) -> str | None:
    """
    Query the configured LLM provider (OpenAI or Gemini) via direct HTTP request.
    Returns None if unconfigured or on any network/format error.
    """
    provider = settings.llm_provider.lower()
    api_key = settings.llm_api_key.strip()

    if provider == "none" or not api_key:
        return None

    top_factors = [
        f"{f.label}: {f.value} ({f.impact} impact)"
        for f in prediction.contributing_factors[:5]
    ]

    prompt = (
        f"You are an educational AI assistant analyzing survey data for an educational Lung Cancer demonstration.\n"
        f"Model: {prediction.model_name}\n"
        f"Predicted Category: {prediction.predicted_class} (Probability: {prediction.probability_percentage}%, Decision Threshold: {prediction.decision_threshold * 100}%)\n"
        f"Top Survey Factors: {', '.join(top_factors)}\n\n"
        f"Task: Provide an educational summary explaining why the statistical model assigned this probability score, "
        f"how the survey factors influenced the result, and why survey correlations do not equate to a medical diagnosis.\n"
        f"Guidelines:\n"
        f"- Do NOT diagnose or claim medical certainty.\n"
        f"- Always emphasize consulting a licensed physician.\n"
        f"- Format in clean Markdown."
    )

    try:
        if provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            body = json.dumps({
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a helpful educational AI explainer."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 400,
            }).encode("utf-8")

            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=5) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["choices"][0]["message"]["content"]

        elif provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.llm_model}:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            body = json.dumps({
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.3, "maxOutputTokens": 400},
            }).encode("utf-8")

            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=5) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as exc:
        logger.info(f"LLM request bypassed or timed out: {exc}")
        return None

    return None


def get_explanation_for_prediction(prediction: PredictionOutput) -> ExplanationResponse:
    """
    Produce an explanation for the given prediction.
    Attempts LLM generation with guardrails, falling back to deterministic explanation if needed.
    """
    llm_output = query_llm_provider(prediction)

    if llm_output and validate_explanation_guardrails(llm_output, prediction.predicted_class):
        return ExplanationResponse(
            prediction=prediction,
            explanation=llm_output,
            source="llm",
            disclaimer=DISCLAIMER_TEXT,
        )

    # Deterministic fallback
    fallback_text = generate_deterministic_explanation(prediction)
    return ExplanationResponse(
        prediction=prediction,
        explanation=fallback_text,
        source="deterministic_fallback",
        disclaimer=DISCLAIMER_TEXT,
    )
