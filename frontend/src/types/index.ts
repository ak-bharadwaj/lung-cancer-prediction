export type Gender = 'M' | 'F';
export type SurveyScale = 1 | 2;

export interface PredictionInput {
  gender: Gender;
  age: number;
  smoking: SurveyScale;
  yellow_fingers: SurveyScale;
  anxiety: SurveyScale;
  peer_pressure: SurveyScale;
  chronic_disease: SurveyScale;
  fatigue: SurveyScale;
  allergy: SurveyScale;
  wheezing: SurveyScale;
  alcohol_consuming: SurveyScale;
  coughing: SurveyScale;
  shortness_of_breath: SurveyScale;
  swallowing_difficulty: SurveyScale;
  chest_pain: SurveyScale;
}

export interface ContributingFactor {
  feature: string;
  label: string;
  value: string;
  importance_score: number;
  impact: 'high' | 'moderate' | 'low';
  description: string;
}

export interface PredictionOutput {
  model_name: string;
  model_version: string;
  predicted_class: 'YES' | 'NO';
  probability_score: number;
  probability_percentage: number;
  decision_threshold: number;
  contributing_factors: ContributingFactor[];
  disclaimer: string;
}

export interface ExplanationResponse {
  prediction: PredictionOutput;
  explanation: string;
  source: 'llm' | 'deterministic_fallback';
  disclaimer: string;
}

export interface ModelInfo {
  model_name: string;
  version: string;
  description: string;
  estimators: string[];
  ensemble_weights: Record<string, number>;
  optimal_decision_threshold: number;
  features: string[];
  feature_importances: Record<string, number>;
  split_summary: {
    train_size: number;
    val_size: number;
    test_size: number;
    total_size: number;
  };
}

export interface ModelComparisonItem {
  model_name: string;
  cv_macro_f1: number;
  cv_macro_f1_std: number;
  cv_accuracy: number;
  cv_roc_auc: number;
  val_macro_f1: number;
  val_accuracy: number;
  val_minority_recall: number;
  val_roc_auc: number;
  optimal_threshold?: number;
}

export interface TestEvaluation {
  model_name: string;
  eval_dataset: string;
  test_sample_count: number;
  test_positive_count: number;
  test_negative_count: number;
  applied_threshold: number;
  metrics: {
    accuracy: number;
    macro_f1: number;
    weighted_f1: number;
    macro_precision: number;
    macro_recall: number;
    roc_auc: number;
    pr_auc: number;
    minority_class_no: {
      recall: number;
      precision: number;
      f1_score: number;
      true_count: number;
    };
    majority_class_yes: {
      recall: number;
      precision: number;
      f1_score: number;
      true_count: number;
    };
  };
  default_threshold_comparison: {
    threshold_0_5_macro_f1: number;
    threshold_0_5_minority_recall: number;
  };
  confusion_matrix: {
    true_negatives: number;
    false_positives: number;
    false_negatives: number;
    true_positives: number;
    raw_matrix: number[][];
  };
  roc_curve: Array<{ fpr: number; tpr: number }>;
  pr_curve: Array<{ recall: number; precision: number }>;
  bootstrap_confidence_intervals_95?: Record<string, { mean: number; ci_lower: number; ci_upper: number }>;
  out_of_fold_evaluation?: {
    total_cohort_size: number;
    minority_count_total: number;
    majority_count_total: number;
    metrics: {
      accuracy: number;
      macro_f1: number;
      roc_auc: number;
      pr_auc: number;
      minority_recall: number;
      majority_recall: number;
    };
    confusion_matrix: {
      true_negatives: number;
      false_positives: number;
      false_negatives: number;
      true_positives: number;
    };
  };
  out_of_fold_evaluation_309_samples?: {
    total_cohort_size: number;
    minority_count_total: number;
    majority_count_total: number;
    metrics: {
      accuracy: number;
      macro_f1: number;
      roc_auc: number;
      pr_auc: number;
      minority_recall: number;
      majority_recall: number;
    };
    confusion_matrix: {
      true_negatives: number;
      false_positives: number;
      false_negatives: number;
      true_positives: number;
    };
  };
  synthetic_cohort_stress_test_1000_samples?: {
    sample_size: number;
    synthetic_positives: number;
    synthetic_negatives: number;
    metrics: {
      accuracy: number;
      macro_f1: number;
      roc_auc: number;
      minority_recall: number;
      majority_recall: number;
    };
    confusion_matrix: {
      true_negatives: number;
      false_positives: number;
      false_negatives: number;
      true_positives: number;
    };
  };
  sample_size_analysis?: {
    rationale: string;
    holdout_samples: number;
    oof_samples: number;
    synthetic_stress_samples: number;
  };
}

export interface EdaSummary {
  dataset_name: string;
  total_records: number;
  target_distribution: {
    positive: { count: number; percentage: number };
    negative: { count: number; percentage: number };
    imbalance_ratio: number;
  };
  age_statistics: {
    overall: { mean: number; median: number; min: number; max: number; std: number };
    cancer_yes: { mean: number; median: number; min: number; max: number; std: number };
    cancer_no: { mean: number; median: number; min: number; max: number; std: number };
  };
  gender_breakdown: Record<string, { total: number; cancer_yes: number; cancer_no: number; cancer_prevalence_pct: number }>;
  symptom_analysis: Record<string, {
    prevalence_in_cancer_positive_pct: number;
    prevalence_in_cancer_negative_pct: number;
    relative_risk: number;
    risk_elevation: string;
  }>;
  correlation_with_target: Record<string, number>;
  top_correlated_feature_pairs: Array<{ feature_a: string; feature_b: string; correlation: number }>;
}

export interface CancerPatientEvaluation {
  model_name: string;
  dataset_name: string;
  total_samples: number;
  test_sample_count: number;
  test_elevated_risk_count: number;
  test_low_risk_count: number;
  applied_threshold: number;
  metrics: {
    accuracy: number;
    macro_f1: number;
    weighted_f1: number;
    macro_precision: number;
    macro_recall: number;
    roc_auc: number;
    pr_auc: number;
    minority_class_low_risk: {
      recall: number;
      precision: number;
      f1_score: number;
      true_count: number;
    };
    majority_class_elevated_risk: {
      recall: number;
      precision: number;
      f1_score: number;
      true_count: number;
    };
  };
  confusion_matrix: {
    true_negatives: number;
    false_positives: number;
    false_negatives: number;
    true_positives: number;
    raw_matrix: number[][];
  };
  feature_importances: Record<string, number>;
  deduplication_audit: {
    total_records: number;
    unique_patient_profiles: number;
    repetition_factor: number;
    note: string;
  };
}

