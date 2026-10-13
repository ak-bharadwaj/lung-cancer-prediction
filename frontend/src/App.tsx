import React, { useEffect, useState } from 'react';
import { Navbar, NavTab } from './components/Navbar';
import { AboutPage } from './pages/AboutPage';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { PredictPage } from './pages/PredictPage';
import { api } from './services/api';
import {
  EdaSummary,
  ExplanationResponse,
  ModelComparisonItem,
  ModelInfo,
  PredictionInput,
  PredictionOutput,
  TestEvaluation,
} from './types';

const INITIAL_FORM: PredictionInput = {
  gender: 'M',
  age: 65,
  smoking: 2,
  yellow_fingers: 2,
  anxiety: 2,
  peer_pressure: 1,
  chronic_disease: 2,
  fatigue: 2,
  allergy: 1,
  wheezing: 2,
  alcohol_consuming: 2,
  coughing: 2,
  shortness_of_breath: 2,
  swallowing_difficulty: 2,
  chest_pain: 2,
};

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<NavTab>('predict');
  const [systemStatus, setSystemStatus] = useState<{ status: string; model_loaded: boolean }>({
    status: 'checking',
    model_loaded: false,
  });

  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [metrics, setMetrics] = useState<{
    model_comparison: ModelComparisonItem[];
    test_evaluation: TestEvaluation;
    cancer_patient_comparison?: ModelComparisonItem[];
    cancer_patient_test_evaluation?: import('./types').CancerPatientEvaluation;
  } | null>(null);
  const [edaData, setEdaData] = useState<{
    eda_summary: EdaSummary;
    data_audit: any;
  } | null>(null);

  const [formState, setFormState] = useState<PredictionInput>(INITIAL_FORM);
  const [prediction, setPrediction] = useState<PredictionOutput | null>(null);
  const [explanationData, setExplanationData] = useState<ExplanationResponse | null>(null);
  const [predictLoading, setPredictLoading] = useState<boolean>(false);
  const [explainLoading, setExplainLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Load initial backend resources on startup
  useEffect(() => {
    async function loadData() {
      try {
        const health = await api.getHealth();
        setSystemStatus(health);

        const [info, met, eda] = await Promise.allSettled([
          api.getModelInfo(),
          api.getMetrics(),
          api.getEda(),
        ]);

        if (info.status === 'fulfilled') setModelInfo(info.value);
        if (met.status === 'fulfilled') setMetrics(met.value);
        if (eda.status === 'fulfilled') setEdaData(eda.value);
      } catch (err: any) {
        console.warn('Initial data load warning:', err.message);
      }
    }
    loadData();
  }, []);

  const handlePredict = async (input: PredictionInput) => {
    setPredictLoading(true);
    setExplainLoading(true);
    setErrorMsg(null);
    try {
      // 1. Obtain instant CatGBM prediction (< 25ms inference)
      const predResult = await api.predict(input);
      setPrediction(predResult);
      setPredictLoading(false);

      // 2. Obtain AI or deterministic plain-language explanation
      try {
        const explResult = await api.explain(input);
        setExplanationData(explResult);
      } catch (explErr: any) {
        console.warn('AI explanation warning:', explErr);
      } finally {
        setExplainLoading(false);
      }
    } catch (err: any) {
      console.error('Prediction inference error:', err);
      setErrorMsg(err.message || 'Failed to obtain prediction from backend service. Please check connection.');
      setPredictLoading(false);
      setExplainLoading(false);
    }
  };

  const handleRefreshExplanation = async () => {
    setExplainLoading(true);
    try {
      const res = await api.explain(formState);
      setExplanationData(res);
      setPrediction(res.prediction);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to refresh explanation.');
    } finally {
      setExplainLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-teal-500 selection:text-slate-950">
      {/* Navigation Bar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        systemStatus={systemStatus}
      />

      {/* Global Error Banner */}
      {errorMsg && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-4">
          <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-500/50 text-rose-300 text-xs flex justify-between items-center">
            <span>{errorMsg}</span>
            <button
              onClick={() => setErrorMsg(null)}
              className="text-xs uppercase font-bold text-rose-400 hover:text-white"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {/* Main Content Pages */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'predict' && (
          <PredictPage
            onPredict={handlePredict}
            prediction={prediction}
            loading={predictLoading}
            explanationData={explanationData}
            explainLoading={explainLoading}
            onRefreshExplanation={handleRefreshExplanation}
            formState={formState}
            setFormState={setFormState}
          />
        )}

        {activeTab === 'analytics' && (
          <AnalyticsPage
            metrics={metrics}
            edaData={edaData}
            modelInfo={modelInfo}
          />
        )}

        {activeTab === 'about' && <AboutPage />}
      </main>

      {/* Persistent Clean Footer */}
      <footer className="border-t border-slate-800 bg-slate-900/40 text-slate-400 py-6 text-center text-xs">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <p>
            CatGBM LungVision &copy; 2026 &mdash; Educational Tabular ML & Explainability Demonstration
          </p>
          <p className="text-[11px] text-slate-500">
            CatBoost 1.2.10 &bull; LightGBM 4.7.0 &bull; XGBoost 3.4.1 &bull; FastAPI Backend
          </p>
        </div>
      </footer>
    </div>
  );
};

export default App;
