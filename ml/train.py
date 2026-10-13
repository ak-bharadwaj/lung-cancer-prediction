"""
Training and Model Selection Module for CatGBM Ensemble.
Trains baseline models, individual gradient boosters, and the CatGBM Ensemble.
Implements 5-fold stratified cross-validation, PR-threshold tuning on the
validation split, and saves pipeline artifacts and comparison reports.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Any

import joblib
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from ml.preprocessing import (
    FEATURE_ORDER,
    SurveyFeatureTransformer,
    load_and_preprocess_raw,
)


def train_and_evaluate_cv(
    model: Any,
    X_train_trans: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    threshold: float | None = None,
) -> dict[str, float]:
    """Run Stratified 5-Fold Cross Validation on the training split."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    f1_scores = []
    auc_scores = []
    rec_minority = []
    acc_scores = []

    for tr_idx, val_idx in skf.split(X_train_trans, y_train):
        X_tr, X_va = X_train_trans[tr_idx], X_train_trans[val_idx]
        y_tr, y_va = y_train[tr_idx], y_train[val_idx]

        model.fit(X_tr, y_tr)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_va)[:, 1]
            try:
                auc_scores.append(roc_auc_score(y_va, probs))
            except ValueError:
                pass
            if threshold is not None:
                preds = (probs >= threshold).astype(int)
            else:
                preds = model.predict(X_va)
        else:
            preds = model.predict(X_va)

        f1_scores.append(f1_score(y_va, preds, average="macro", zero_division=0))
        acc_scores.append(accuracy_score(y_va, preds))
        rec_minority.append(recall_score(y_va, preds, pos_label=0, zero_division=0))

    return {
        "cv_macro_f1_mean": round(float(np.mean(f1_scores)), 4),
        "cv_macro_f1_std": round(float(np.std(f1_scores)), 4),
        "cv_accuracy_mean": round(float(np.mean(acc_scores)), 4),
        "cv_minority_recall_mean": round(float(np.mean(rec_minority)), 4),
        "cv_roc_auc_mean": round(float(np.mean(auc_scores)), 4) if auc_scores else 0.5,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold_range: tuple[float, float] = (0.25, 0.75),
    steps: int = 101,
) -> tuple[float, float]:
    """
    Tune classification threshold on validation set to maximize Macro-F1
    while maintaining robust minority class detection.
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], steps)
    best_th = 0.5
    best_score = -1.0

    for th in thresholds:
        preds = (probs >= th).astype(int)
        macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        # Small tiebreaker favoring threshold closest to 0.5
        score = macro_f1 - 0.05 * abs(th - 0.5)
        if score > best_score:
            best_score = score
            best_th = float(th)

    best_macro_f1 = f1_score(
        y_true, (probs >= best_th).astype(int), average="macro", zero_division=0
    )
    return round(best_th, 4), round(best_macro_f1, 4)


def compute_ensemble_feature_importances(
    catboost_model: CatBoostClassifier,
    lightgbm_model: LGBMClassifier,
    xgboost_model: XGBClassifier,
    weights: list[float],
) -> dict[str, float]:
    """Aggregate normalized feature importances across the 3 boosters according to ensemble weights."""
    cb_imp = np.array(catboost_model.get_feature_importance())
    cb_imp = cb_imp / (cb_imp.sum() + 1e-9)

    lgb_imp = np.array(lightgbm_model.feature_importances_, dtype=float)
    lgb_imp = lgb_imp / (lgb_imp.sum() + 1e-9)

    xgb_imp = np.array(xgboost_model.feature_importances_, dtype=float)
    xgb_imp = xgb_imp / (xgb_imp.sum() + 1e-9)

    total_w = sum(weights)
    ens_imp = (weights[0] * cb_imp + weights[1] * lgb_imp + weights[2] * xgb_imp) / total_w

    importance_dict = {
        FEATURE_ORDER[i]: round(float(ens_imp[i]), 4) for i in range(len(FEATURE_ORDER))
    }
    # Sort descending
    return dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))


def train_models(
    csv_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Execute end-to-end model training, CV benchmarking, threshold optimization,
    and save model pipeline and comparison reports.
    """
    base_dir = Path(__file__).resolve().parent.parent
    if csv_path is None:
        primary_cohort = base_dir / "data" / "lung_cancer_cohort_1309.csv"
        if primary_cohort.exists():
            csv_path = primary_cohort
        else:
            csv_path = base_dir / "data" / "raw" / "survey_lung_cancer.csv"
    else:
        csv_path = Path(csv_path)

    models_dir = base_dir / "models" if output_dir is None else Path(output_dir)
    reports_dir = base_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    X_raw, y_raw = load_and_preprocess_raw(str(csv_path))

    # 2. Stratified 70% Train, 15% Val, 15% Test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_raw, y_raw, test_size=0.15, random_state=random_state, stratify=y_raw
    )
    val_ratio = 0.15 / (0.70 + 0.15)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_train_val,
    )

    # Convert y series to numpy arrays
    y_tr_np = y_train.values
    y_va_np = y_val.values

    # 3. Fit Preprocessing Transformer solely on Training data
    transformer = SurveyFeatureTransformer(scale_age=True)
    X_train_trans = transformer.fit_transform(X_train)
    X_val_trans = transformer.transform(X_val)

    # 4. Instantiate Candidate Models
    models_to_benchmark = {
        "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=4, class_weight="balanced", random_state=random_state
        ),
        "CatBoost": CatBoostClassifier(
            iterations=120,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=3.0,
            auto_class_weights="Balanced",
            random_seed=random_state,
            verbose=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.03,
            class_weight="balanced",
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=random_state,
            verbose=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.03,
            scale_pos_weight=1.0,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            eval_metric="logloss",
        ),
    }

    # Benchmark results table
    comparison_results: list[dict[str, Any]] = []

    for name, model in models_to_benchmark.items():
        # Cross validation metrics
        cv_metrics = train_and_evaluate_cv(
            model, X_train_trans, y_tr_np, n_splits=5, random_state=random_state
        )

        # Fit on full training split
        model.fit(X_train_trans, y_tr_np)

        # Val evaluation
        val_preds = model.predict(X_val_trans)
        val_f1 = f1_score(y_va_np, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(y_va_np, val_preds)
        val_rec_no = recall_score(y_va_np, val_preds, pos_label=0, zero_division=0)

        val_auc = 0.5
        if hasattr(model, "predict_proba"):
            try:
                val_auc = roc_auc_score(y_va_np, model.predict_proba(X_val_trans)[:, 1])
            except ValueError:
                pass

        comparison_results.append({
            "model_name": name,
            "cv_macro_f1": cv_metrics["cv_macro_f1_mean"],
            "cv_macro_f1_std": cv_metrics["cv_macro_f1_std"],
            "cv_accuracy": cv_metrics["cv_accuracy_mean"],
            "cv_roc_auc": cv_metrics["cv_roc_auc_mean"],
            "val_macro_f1": round(val_f1, 4),
            "val_accuracy": round(val_acc, 4),
            "val_minority_recall": round(val_rec_no, 4),
            "val_roc_auc": round(val_auc, 4),
        })

    # 5. Build and Benchmark CatGBM Ensemble
    cb_ens = CatBoostClassifier(
        iterations=120,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=3.0,
        auto_class_weights="Balanced",
        random_seed=random_state,
        verbose=0,
    )
    lgb_ens = LGBMClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        class_weight="balanced",
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=random_state,
        verbose=-1,
    )
    xgb_ens = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        scale_pos_weight=1.0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        eval_metric="logloss",
    )

    ensemble_weights = [1.5, 1.0, 1.0]
    catgbm_voting = VotingClassifier(
        estimators=[
            ("catboost", cb_ens),
            ("lightgbm", lgb_ens),
            ("xgboost", xgb_ens),
        ],
        voting="soft",
        weights=ensemble_weights,
    )

    # Train CatGBM on training set to optimize decision threshold
    catgbm_voting.fit(X_train_trans, y_tr_np)

    # Validation probabilities & threshold search
    val_probs = catgbm_voting.predict_proba(X_val_trans)[:, 1]
    optimal_th, val_f1_th = find_optimal_threshold(y_va_np, val_probs)
    val_preds_th = (val_probs >= optimal_th).astype(int)
    val_acc_th = accuracy_score(y_va_np, val_preds_th)
    val_rec_no_th = recall_score(y_va_np, val_preds_th, pos_label=0, zero_division=0)
    val_auc_th = roc_auc_score(y_va_np, val_probs)

    # CatGBM CV evaluation using optimal threshold
    catgbm_cv = train_and_evaluate_cv(
        catgbm_voting, X_train_trans, y_tr_np, n_splits=5, random_state=random_state, threshold=optimal_th
    )

    comparison_results.append({
        "model_name": "CatGBM Ensemble (Voting)",
        "cv_macro_f1": catgbm_cv["cv_macro_f1_mean"],
        "cv_macro_f1_std": catgbm_cv["cv_macro_f1_std"],
        "cv_accuracy": catgbm_cv["cv_accuracy_mean"],
        "cv_roc_auc": catgbm_cv["cv_roc_auc_mean"],
        "val_macro_f1": val_f1_th,
        "val_accuracy": round(val_acc_th, 4),
        "val_minority_recall": round(val_rec_no_th, 4),
        "val_roc_auc": round(val_auc_th, 4),
        "optimal_threshold": optimal_th,
    })

    # Save model comparison report
    with open(reports_dir / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    # 6. Extract ensemble feature importances
    cb_fitted = catgbm_voting.named_estimators_["catboost"]
    lgb_fitted = catgbm_voting.named_estimators_["lightgbm"]
    xgb_fitted = catgbm_voting.named_estimators_["xgboost"]

    feature_importances = compute_ensemble_feature_importances(
        cb_fitted, lgb_fitted, xgb_fitted, ensemble_weights
    )

    # 7. Construct Full End-to-End Pipeline
    full_pipeline = Pipeline([
        ("transformer", transformer),
        ("ensemble", catgbm_voting),
    ])

    # Save fitted pipeline artifact
    pipeline_path = models_dir / "pipeline.joblib"
    joblib.dump(full_pipeline, pipeline_path)

    # 8. Save Metadata Manifest
    metadata = {
        "model_name": "CatGBM Ensemble",
        "version": "1.0.0",
        "dataset_name": "Lung Cancer Cohort (1309 Records)" if "1309" in str(csv_path) else "Survey Lung Cancer",
        "description": "High-Performance Ensemble of CatBoost, LightGBM, and XGBoost with validation-tuned thresholding and cross-validation.",
        "estimators": ["CatBoost", "LightGBM", "XGBoost"],
        "ensemble_weights": {
            "catboost": ensemble_weights[0],
            "lightgbm": ensemble_weights[1],
            "xgboost": ensemble_weights[2],
        },
        "features": FEATURE_ORDER,
        "optimal_decision_threshold": optimal_th,
        "feature_importances": feature_importances,
        "split_summary": {
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "total_size": len(X_raw),
        },
        "training_cv_performance": catgbm_cv,
        "validation_performance": {
            "macro_f1": val_f1_th,
            "accuracy": round(val_acc_th, 4),
            "minority_recall": round(val_rec_no_th, 4),
            "roc_auc": round(val_auc_th, 4),
            "threshold": optimal_th,
        },
        "artifact_path": str(pipeline_path),
    }

    metadata_path = models_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "pipeline_path": str(pipeline_path),
        "metadata_path": str(metadata_path),
        "optimal_threshold": optimal_th,
        "comparison_results": comparison_results,
    }


if __name__ == "__main__":
    res = train_models()
    print("Model training and CatGBM ensemble generation complete.")
    print(f"Optimal Decision Threshold: {res['optimal_threshold']}")
    print(f"Pipeline saved to: {res['pipeline_path']}")
