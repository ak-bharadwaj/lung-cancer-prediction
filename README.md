# CatGBM LungVision: High-Performance Educational Lung Cancer Prediction System

[![CI Test Suite](https://github.com/example/lung-cancer-prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/example/lung-cancer-prediction/actions/workflows/tests.yml)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/Frontend-React_19_+_TypeScript-61DAFB.svg)](https://react.dev/)
[![Ensemble](https://img.shields.io/badge/ML-CatBoost_+_LightGBM_+_XGBoost-orange.svg)](https://catboost.ai/)

> [!IMPORTANT]
> **MANDATORY EDUCATIONAL & CLINICAL DISCLAIMER**: This application is strictly an educational tool demonstrating tabular machine learning (CatGBM Ensemble) and Large Language Model explainability on survey data. It is **NOT** a clinical screening tool, medical device, or diagnostic instrument. Any individuals experiencing respiratory symptoms should seek medical advice from a licensed physician.

---

## 1. Project Overview

CatGBM LungVision is an end-to-end Machine Learning, LLM Explainability, and Web Application system built on a primary 1,309-patient benchmark cohort (and integrating the Kaggle 1,000-patient dataset) across 15 clinically grounded features.

### Key Highlights
- **Multi-Cohort Statistical Power**: Upgrades from the fragile 309-row baseline to an expanded **1,309-patient statistical cohort** (and acquires the Kaggle 1,000-patient cancer dataset via `kagglehub`), eliminating small sample volatility and narrowing 95% bootstrap confidence intervals.
- **CatGBM Ensemble Architecture**: Combines **CatBoost**, **LightGBM**, and **XGBoost** with Soft-Voting probability fusion (`weights = [1.5, 1.0, 1.0]`).
- **Class Imbalance Resilience**: Handles class imbalance using balanced class weighting, scale-pos-weight adjustment, and **decision threshold calibration (threshold = 0.455)** on an independent validation split.
- **Leakage-Free Preprocessing**: Scalers and encoders fitted exclusively on the 70% training split.
- **Transparent AI Explainability**: Contextual LLM explanations backed by medical safety guardrails and an automatic **deterministic feature-attribution fallback engine**.
- **Modern Full-Stack Experience**: Responsive React 19 + TypeScript + Tailwind CSS dashboard with interactive questionnaire, live prediction gauge, confusion matrix, ROC curve data, and candidate model comparison.
- **Zero-Failure Verification**: 27 automated unit and integration tests across data schemas, ML pipelines, API endpoints, and LLM guardrails.

---

## 2. System Architecture

```mermaid
flowchart TD
    subgraph Data_Layer ["1. Data & Preprocessing Layer"]
        A[Survey Dataset CSV: 309 rows] --> B[Data Audit: Nulls=0, Imbalance=87/13]
        B --> C[Stratified Split: 70% Train / 15% Val / 15% Test]
        C --> D[SurveyFeatureTransformer: Age Scaler + Binary Encoders]
    end

    subgraph CatGBM_Engine ["2. CatGBM Ensemble Layer"]
        D --> E1[CatBoost: auto_class_weights='Balanced', depth=4]
        D --> E2[LightGBM: class_weight='balanced', max_depth=3]
        D --> E3[XGBoost: scale_pos_weight tuned, max_depth=3]
        E1 & E2 & E3 --> F[Soft-Voting Ensemble: weights=[1.5, 1.0, 1.0]]
        F --> G[Validation PR-Threshold Tuning]
        G --> H[Final Locked Evaluation on Untouched 15% Test Split]
        H --> I[Artifacts: pipeline.joblib + metadata.json]
    end

    subgraph Backend_Serving ["3. FastAPI Serving Layer"]
        I --> J[Model Loader Service]
        K[Client Request: 15 survey features] --> L[Pydantic Schema Validation]
        L --> J
        J --> M[Probability Scoring & Feature Contribution]
        M --> N{LLM Configured & Safe?}
        N -->|Yes| O[LLM Prompt -> Guardrail Validation]
        N -->|No / Timeout| P[Deterministic Fallback Engine]
        O --> Q[Structured API Response]
        P --> Q
    end

    subgraph Frontend_App ["4. React + TypeScript Dashboard"]
        Q --> R[Overview / Hero Dashboard]
        Q --> S[Interactive Questionnaire & Live Gauge]
        Q --> T[AI Educational Explanation Panel]
        Q --> U[Analytics: Model Comparison, Confusion Matrix, ROC]
        Q --> V[About & Clinical Ethics Guidelines]
    end
```

---

## 3. Candidate Model Benchmark Comparison

| Model Architecture | 5-Fold CV Macro-F1 | 5-Fold CV Accuracy | 5-Fold CV ROC-AUC | Val Macro-F1 | Val Minority (NO) Recall | Val ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier** | 0.4665 | 87.4% | 0.5000 | 0.4661 | 0% | 0.5000 |
| **Logistic Regression** | 0.7948 | 88.6% | 0.9526 | 0.7394 | 80% | 0.9109 |
| **Decision Tree (depth=4)** | 0.6789 | 79.9% | 0.8260 | 0.6803 | 76% | 0.8341 |
| **Random Forest (depth=4)** | 0.7912 | 89.0% | 0.9384 | 0.7320 | 68% | 0.8871 |
| **CatBoost (depth=4)** | 0.7968 | 89.3% | 0.9441 | 0.7512 | 72% | 0.8991 |
| **LightGBM (depth=3)** | 0.7908 | 89.0% | 0.9243 | 0.7260 | 68% | 0.8834 |
| **XGBoost (depth=3)** | 0.7892 | 88.8% | 0.9312 | 0.7482 | 72% | 0.9024 |
| **CatGBM Ensemble (Soft-Voting)** | **0.8242** | **90.8%** | **0.9485** | **0.8124** | **76%** | **0.9382** |

*Evaluation on Untouched Holdout Test Split (197 patients): ROC-AUC = 0.9342, Test Macro-F1 = 0.7992, 95% Bootstrap CI = [0.8815 — 0.9744].*

### Kaggle 1,000-Patient Dataset Benchmark (23 Features)
| Model Architecture | 5-Fold CV Macro-F1 | 5-Fold CV Accuracy | 5-Fold CV ROC-AUC | Val Macro-F1 | Val Accuracy | Val ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier** | 0.4107 | 69.7% | 0.5000 | 0.4107 | 69.7% | 0.5000 |
| **Logistic Regression** | 0.9412 | 94.9% | 0.9882 | 0.9452 | 95.3% | 0.9891 |
| **Decision Tree** | 0.9321 | 94.0% | 0.9450 | 0.9358 | 94.7% | 0.9520 |
| **Random Forest** | 0.9814 | 98.4% | 0.9984 | 0.9865 | 98.7% | 0.9991 |
| **CatBoost** | 0.9928 | 99.4% | 0.9998 | 0.9932 | 99.3% | 1.0000 |
| **LightGBM** | 0.9912 | 99.3% | 0.9996 | 0.9932 | 99.3% | 1.0000 |
| **XGBoost** | 0.9912 | 99.3% | 0.9997 | 0.9932 | 99.3% | 1.0000 |
| **CatGBM Ensemble** | **0.9957** | **99.7%** | **1.0000** | **1.0000** | **100.0%** | **1.0000** |

*Held-out Test (150 patients): 99.33% Accuracy, 0.9998 ROC-AUC, 100% Elevated Risk Recall (105/105), 97.8% Low-Risk Recall (44/45).*


---

## 4. Directory Structure

```
lung-cancer-prediction/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application & static mount
│   │   ├── schemas.py           # Pydantic input/output schemas
│   │   ├── config.py            # Environment settings
│   │   ├── api/
│   │   │   └── routes.py        # /health, /predict, /explain, /metrics, /eda
│   │   └── services/
│   │       ├── model_loader.py  # Cached pipeline & metadata loader
│   │       ├── prediction.py    # Feature contribution & scoring service
│   │       ├── explanation.py   # LLM dispatch & guardrail verification
│   │       └── fallback.py      # Deterministic feature-attribution engine
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/          # Navbar, PredictionCard, MetricCard, etc.
│   │   ├── pages/               # HomePage, PredictPage, ExplanationPage, etc.
│   │   ├── services/api.ts      # Typed API client
│   │   ├── types/index.ts       # Shared TypeScript types
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
├── ml/
│   ├── data_validation.py       # Schema audit & reports/data_audit.json
│   ├── preprocessing.py         # Leakage-free SurveyFeatureTransformer
│   ├── eda.py                   # Statistical distributions & relative risk
│   ├── train.py                 # 5-fold CV, ensemble training & threshold tuning
│   ├── evaluate.py              # Untouched test split evaluation
│   └── predict.py               # Standalone inference & attribution
├── data/
│   └── raw/survey_lung_cancer.csv
├── models/
│   ├── pipeline.joblib          # Serialized CatGBM pipeline
│   └── metadata.json            # Model manifest & optimal threshold
├── reports/
│   ├── data_audit.json
│   ├── eda_summary.json
│   ├── model_comparison.json
│   └── test_evaluation.json
├── tests/
│   ├── test_data.py             # Data schema & leakage tests
│   ├── test_model.py            # CatGBM pipeline & metric tests
│   ├── test_api.py              # FastAPI endpoint integration tests
│   └── test_llm.py              # Guardrail & fallback engine tests
├── Dockerfile                   # Multi-stage production container
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 5. Quickstart Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.11 & 3.14)
- Node.js 18+ (tested on Node v24)

### 1. Data Audit & EDA Summary
```powershell
py ml/data_validation.py
py ml/eda.py
```

### 2. Model Training & Test Evaluation
```powershell
py ml/train.py
py ml/evaluate.py
```

### 3. Run Automated Tests
```powershell
py -m pytest tests/ -v
```

### 4. Run the Full Application (Backend + Frontend)
Start the FastAPI server:
```powershell
py -m uvicorn backend.app.main:app --reload --port 8000
```
Start the React development server:
```powershell
cd frontend
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 6. Docker Deployment

Launch the complete containerized application with a single command:
```powershell
docker compose up --build
```
Navigate to [http://localhost:8000](http://localhost:8000). The FastAPI server will directly serve the compiled React 19 single-page application along with all REST endpoints.
