"""
Model Loader Service.
Caches and provides access to the serialized CatGBM ensemble pipeline and metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from backend.app.config import settings

_PIPELINE: Any | None = None
_METADATA: dict[str, Any] | None = None


def load_model_artifacts() -> tuple[Any, dict[str, Any]]:
    """Load model pipeline and metadata from disk into memory."""
    global _PIPELINE, _METADATA

    model_path = Path(settings.model_path)
    metadata_path = Path(settings.metadata_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at {model_path}. Please run ml/train.py."
        )

    _PIPELINE = joblib.load(model_path)

    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            _METADATA = json.load(f)
    else:
        _METADATA = {
            "model_name": "CatGBM Ensemble",
            "version": "1.0.0",
            "optimal_decision_threshold": 0.5,
            "feature_importances": {},
        }

    return _PIPELINE, _METADATA


def get_pipeline() -> Any:
    """Retrieve the cached pipeline instance."""
    if _PIPELINE is None:
        load_model_artifacts()
    return _PIPELINE


def get_metadata() -> dict[str, Any]:
    """Retrieve the cached metadata dictionary."""
    if _METADATA is None:
        load_model_artifacts()
    return _METADATA if _METADATA is not None else {}
