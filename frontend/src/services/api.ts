import {
  EdaSummary,
  ExplanationResponse,
  ModelComparisonItem,
  ModelInfo,
  PredictionInput,
  PredictionOutput,
  TestEvaluation,
} from '../types';

const API_BASE = '/api';

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const errorText = await res.text();
    let msg = `HTTP ${res.status}`;
    try {
      const parsed = JSON.parse(errorText);
      msg = parsed.detail || msg;
    } catch {
      msg = errorText || msg;
    }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  async getHealth(): Promise<{ status: string; model_loaded: boolean; version: string }> {
    const res = await fetch(`${API_BASE}/health`);
    return handleResponse(res);
  },

  async getModelInfo(): Promise<ModelInfo> {
    const res = await fetch(`${API_BASE}/model-info`);
    return handleResponse<ModelInfo>(res);
  },

  async getMetrics(): Promise<{
    model_comparison: ModelComparisonItem[];
    test_evaluation: TestEvaluation;
    cancer_patient_comparison?: ModelComparisonItem[];
    cancer_patient_test_evaluation?: import('../types').CancerPatientEvaluation;
  }> {
    const res = await fetch(`${API_BASE}/metrics`);
    return handleResponse(res);
  },

  async getEda(): Promise<{
    eda_summary: EdaSummary;
    data_audit: any;
  }> {
    const res = await fetch(`${API_BASE}/eda`);
    return handleResponse(res);
  },

  async predict(input: PredictionInput): Promise<PredictionOutput> {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
    });
    return handleResponse<PredictionOutput>(res);
  },

  async explain(input: PredictionInput): Promise<ExplanationResponse> {
    const res = await fetch(`${API_BASE}/explain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
    });
    return handleResponse<ExplanationResponse>(res);
  },
};
