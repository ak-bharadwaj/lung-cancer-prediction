"""
Application Configuration and Environment Settings.
Loads environment variables for model paths, LLM credentials, and API metadata.
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Server and system settings."""

    # LLM Settings
    llm_provider: Literal["none", "openai", "gemini"] = "none"
    llm_api_key: str = ""
    llm_model: str = "gemini-1.5-flash"

    # Model and Report Artifact Paths
    model_path: str = str(BASE_DIR / "models" / "pipeline.joblib")
    metadata_path: str = str(BASE_DIR / "models" / "metadata.json")
    reports_dir: str = str(BASE_DIR / "reports")

    # Server Settings
    api_title: str = "CatGBM Educational Lung Cancer Prediction API"
    api_version: str = "1.0.0"
    cors_origins: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
