"""
FastAPI Application Entry Point.
Sets up lifespan model initialization, CORS middleware, API route mounting,
and optional frontend static serving.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from backend.app.api.routes import router
from backend.app.config import settings
from backend.app.services.model_loader import load_model_artifacts
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("catgbm_backend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model artifacts at server startup."""
    try:
        load_model_artifacts()
        logger.info("CatGBM Ensemble model artifacts loaded successfully.")
    except Exception as exc:
        logger.error(f"Error loading model artifacts at startup: {exc}")
    yield


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=(
        "Educational API providing machine learning predictions using the "
        "CatGBM Ensemble (CatBoost, LightGBM, XGBoost) and AI explainability."
    ),
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes under both /api and root /
app.include_router(router, prefix="/api", tags=["CatGBM API"])
app.include_router(router, tags=["Root API"])

# Serve frontend build if dist directory exists
frontend_dist = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists() and frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
