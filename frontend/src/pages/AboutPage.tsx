import React from 'react';
import {
  AlertTriangle,
  Award,
  BookOpen,
  Code,
  Cpu,
  Database,
  Layers,
  Shield,
  Stethoscope,
} from 'lucide-react';
import { DisclaimerBanner } from '../components/DisclaimerBanner';

export const AboutPage: React.FC = () => {
  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <DisclaimerBanner />

      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 sm:p-8">
        <h1 className="text-2xl sm:text-3xl font-black text-white">
          About CatGBM LungVision
        </h1>
        <p className="text-sm text-slate-400 mt-2 leading-relaxed">
          An educational machine learning and AI explainability project demonstrating robust tabular classification and plain-language reasoning.
        </p>
      </div>

      {/* 3 Clean Information Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Card 1: Machine Learning Engine */}
        <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 space-y-3">
          <div className="flex items-center gap-2.5 text-teal-400">
            <Layers className="w-5 h-5" />
            <h2 className="text-base font-bold text-white">CatGBM Ensemble Architecture</h2>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Combines three premier gradient boosting engines into a soft-voting ensemble with validation threshold calibration:
          </p>
          <ul className="text-xs text-slate-400 space-y-1.5 list-disc list-inside">
            <li><strong className="text-slate-200">CatBoost:</strong> Symmetric oblivious decision trees with categorical target encoding.</li>
            <li><strong className="text-slate-200">LightGBM:</strong> Leaf-wise tree growth with gradient-based one-side sampling.</li>
            <li><strong className="text-slate-200">XGBoost:</strong> Histogram-regularized gradient boosted trees with L1/L2 penalties.</li>
          </ul>
        </div>

        {/* Card 2: Datasets Evaluated */}
        <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 space-y-3">
          <div className="flex items-center gap-2.5 text-teal-400">
            <Database className="w-5 h-5" />
            <h2 className="text-base font-bold text-white">Evaluated Datasets</h2>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            Trained and tested across two comprehensive benchmarks:
          </p>
          <ul className="text-xs text-slate-400 space-y-1.5 list-disc list-inside">
            <li><strong className="text-slate-200">Primary 1,309 Survey Cohort:</strong> 15 survey indicators (Smoking, Yellow Fingers, Coughing, Fatigue, etc.) with 90.4% test accuracy and 0.934 ROC-AUC.</li>
            <li><strong className="text-slate-200">Kaggle 1,000 Clinical Dataset:</strong> 23 environmental & clinical predictors (Air Pollution, Genetic Risk, Coughing of Blood) with 99.3% test accuracy.</li>
          </ul>
        </div>
      </div>

      {/* Card 3: Technology Stack */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 space-y-3">
        <div className="flex items-center gap-2.5 text-teal-400">
          <Code className="w-5 h-5" />
          <h2 className="text-base font-bold text-white">Full Stack Technology Stack</h2>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
            <span className="font-bold text-slate-200 block">ML Core</span>
            <span className="text-slate-400">CatBoost, LightGBM, XGBoost, Scikit-learn</span>
          </div>
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
            <span className="font-bold text-slate-200 block">Backend</span>
            <span className="text-slate-400">FastAPI, Pydantic, Uvicorn, Python 3.14</span>
          </div>
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
            <span className="font-bold text-slate-200 block">Frontend</span>
            <span className="text-slate-400">React 19, TypeScript, Vite, Tailwind CSS</span>
          </div>
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
            <span className="font-bold text-slate-200 block">Testing & CI</span>
            <span className="text-slate-400">Pytest, Ruff, Docker, GitHub Actions</span>
          </div>
        </div>
      </div>

      {/* Card 4: Ethical & Educational Notice */}
      <div className="bg-slate-900 border border-amber-500/30 rounded-3xl p-6 space-y-3 bg-amber-950/10">
        <div className="flex items-center gap-2.5 text-amber-400">
          <Shield className="w-5 h-5" />
          <h2 className="text-base font-bold text-amber-300">Important Educational Notice</h2>
        </div>
        <p className="text-xs text-slate-300 leading-relaxed">
          This system is strictly an educational engineering demonstration. It is designed to illustrate tabular machine learning pipelines, class-imbalance mitigation, and LLM explainability guardrails. It must <strong>not</strong> be used as a clinical diagnostic tool, medical device, or replacement for qualified healthcare consultation.
        </p>
      </div>
    </div>
  );
};
