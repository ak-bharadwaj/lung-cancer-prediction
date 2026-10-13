import React, { useState } from 'react';
import {
  AlertCircle,
  AlertTriangle,
  Bot,
  Check,
  CheckCircle2,
  Copy,
  Cpu,
  HeartPulse,
  Info,
  Lightbulb,
  RefreshCw,
  RotateCcw,
  Shield,
  Sparkles,
  Stethoscope,
  User,
  Wind,
} from 'lucide-react';
import { DisclaimerBanner } from '../components/DisclaimerBanner';
import { PredictionCard } from '../components/PredictionCard';
import {
  ExplanationResponse,
  Gender,
  PredictionInput,
  PredictionOutput,
  SurveyScale,
} from '../types';

interface Props {
  onPredict: (input: PredictionInput) => Promise<void>;
  prediction: PredictionOutput | null;
  loading: boolean;
  explanationData: ExplanationResponse | null;
  explainLoading: boolean;
  onRefreshExplanation: () => void;
  formState: PredictionInput;
  setFormState: React.Dispatch<React.SetStateAction<PredictionInput>>;
}

const PRESETS: Record<string, { label: string; tag: string; description: string; data: PredictionInput }> = {
  highRisk: {
    label: 'Senior Heavy Smoker',
    tag: 'High Risk Preset',
    description: 'Senior (69 yrs), active smoker, nicotine staining, persistent cough, dyspnea, chest pain',
    data: {
      gender: 'M',
      age: 69,
      smoking: 2,
      yellow_fingers: 2,
      anxiety: 2,
      peer_pressure: 1,
      chronic_disease: 2,
      fatigue: 2,
      allergy: 2,
      wheezing: 2,
      alcohol_consuming: 2,
      coughing: 2,
      shortness_of_breath: 2,
      swallowing_difficulty: 2,
      chest_pain: 2,
    },
  },
  lowRisk: {
    label: 'Young Non-Smoker',
    tag: 'Low Risk Preset',
    description: 'Young adult (32 yrs), non-smoker, clear lungs, no respiratory or systemic symptoms',
    data: {
      gender: 'F',
      age: 32,
      smoking: 1,
      yellow_fingers: 1,
      anxiety: 1,
      peer_pressure: 1,
      chronic_disease: 1,
      fatigue: 1,
      allergy: 1,
      wheezing: 1,
      alcohol_consuming: 1,
      coughing: 1,
      shortness_of_breath: 1,
      swallowing_difficulty: 1,
      chest_pain: 1,
    },
  },
  borderline: {
    label: 'Middle-Aged Borderline',
    tag: 'Moderate Risk Preset',
    description: 'Middle age (54 yrs), non-smoker, exertional dyspnea and fatigue without chronic cough',
    data: {
      gender: 'M',
      age: 54,
      smoking: 1,
      yellow_fingers: 1,
      anxiety: 2,
      peer_pressure: 1,
      chronic_disease: 1,
      fatigue: 2,
      allergy: 1,
      wheezing: 1,
      alcohol_consuming: 2,
      coughing: 1,
      shortness_of_breath: 2,
      swallowing_difficulty: 1,
      chest_pain: 1,
    },
  },
};

export const PredictPage: React.FC<Props> = ({
  onPredict,
  prediction,
  loading,
  explanationData,
  explainLoading,
  onRefreshExplanation,
  formState,
  setFormState,
}) => {
  const [activePreset, setActivePreset] = useState<string | null>(null);
  const [copied, setCopied] = useState<boolean>(false);

  const applyPreset = (key: string) => {
    setActivePreset(key);
    setFormState(PRESETS[key].data);
  };

  const handleScaleToggle = (field: keyof PredictionInput, val: SurveyScale) => {
    setActivePreset(null);
    setFormState((prev) => ({ ...prev, [field]: val }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onPredict(formState);
  };

  const handleCopyExplanation = () => {
    if (explanationData?.explanation) {
      navigator.clipboard.writeText(explanationData.explanation);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const getAgeCohort = (age: number) => {
    if (age < 35) return 'Young Adult (18-34)';
    if (age < 60) return 'Middle-Aged (35-59)';
    return 'Senior Adult (60+)';
  };

  const renderRadioGroup = (
    field: keyof PredictionInput,
    label: string,
    clinicalSubtext: string,
    negativeLabel = 'No / Absent',
    positiveLabel = 'Yes / Present'
  ) => {
    const val = formState[field] as SurveyScale;
    return (
      <div className="p-3.5 rounded-2xl bg-slate-900 border border-slate-800 hover:border-slate-700/80 transition-all flex flex-col justify-between">
        <div>
          <span className="text-xs font-bold text-slate-200 block">{label}</span>
          <span className="text-[11px] text-slate-400 block mt-0.5 leading-snug">
            {clinicalSubtext}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-2 mt-3">
          <button
            type="button"
            onClick={() => handleScaleToggle(field, 1)}
            className={`py-1.5 px-2.5 rounded-xl text-xs font-semibold transition-all ${
              val === 1
                ? 'bg-slate-700 text-white font-bold border border-slate-600 shadow-sm'
                : 'bg-slate-950/70 text-slate-400 border border-slate-800/80 hover:bg-slate-800/80'
            }`}
          >
            {negativeLabel}
          </button>
          <button
            type="button"
            onClick={() => handleScaleToggle(field, 2)}
            className={`py-1.5 px-2.5 rounded-xl text-xs font-semibold transition-all ${
              val === 2
                ? 'bg-teal-500 text-slate-950 font-bold shadow-md shadow-teal-500/20'
                : 'bg-slate-950/70 text-slate-400 border border-slate-800/80 hover:bg-slate-800/80'
            }`}
          >
            {positiveLabel}
          </button>
        </div>
      </div>
    );
  };

  const isFallback = explanationData?.source === 'deterministic_fallback';

  return (
    <div className="space-y-6">
      {/* Educational Notice Banner */}
      <DisclaimerBanner />

      {/* Preset Profiles Bar */}
      <div className="bg-slate-900 border border-slate-800 rounded-3xl p-5 shadow-lg">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-teal-400" />
            <h2 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
              Quick Test Patient Profiles
            </h2>
          </div>
          <span className="text-[11px] text-slate-400">
            Select a verified clinical cohort preset to populate survey inputs
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {Object.entries(PRESETS).map(([key, item]) => {
            const isSelected = activePreset === key;
            return (
              <button
                key={key}
                type="button"
                onClick={() => applyPreset(key)}
                className={`text-left p-3.5 rounded-2xl border transition-all ${
                  isSelected
                    ? 'bg-teal-500/10 border-teal-500/50 text-teal-300 shadow-md shadow-teal-950/30'
                    : 'bg-slate-950/60 border-slate-800 text-slate-300 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="font-bold text-xs text-white">{item.label}</span>
                  <span className="text-[9px] uppercase font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                    {item.tag}
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 mt-1 line-clamp-1">
                  {item.description}
                </p>
              </button>
            );
          })}
        </div>
      </div>

      {/* Two Column Layout: Questionnaire & Live Prediction + Explanation */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Questionnaire Form Column (7 cols) */}
        <div className="lg:col-span-7 space-y-5">
          <form onSubmit={handleSubmit} className="space-y-5">
            {/* 1. Demographics */}
            <div className="bg-slate-900/60 border border-slate-800/80 rounded-3xl p-5 space-y-3">
              <div className="flex items-center gap-2 border-b border-slate-800/80 pb-2.5">
                <User className="w-4 h-4 text-teal-400" />
                <h3 className="text-xs uppercase font-extrabold text-teal-400 tracking-wider">
                  1. Demographic Characteristics
                </h3>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                <div className="p-3.5 rounded-2xl bg-slate-900 border border-slate-800">
                  <label className="text-xs font-bold text-slate-200 block mb-2">
                    Biological Sex
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      onClick={() => {
                        setActivePreset(null);
                        setFormState((prev) => ({ ...prev, gender: 'M' }));
                      }}
                      className={`py-2 px-3 rounded-xl text-xs font-semibold transition-all ${
                        formState.gender === 'M'
                          ? 'bg-teal-500 text-slate-950 font-bold shadow-md shadow-teal-500/20'
                          : 'bg-slate-950/70 text-slate-400 border border-slate-800'
                      }`}
                    >
                      Male (M)
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setActivePreset(null);
                        setFormState((prev) => ({ ...prev, gender: 'F' }));
                      }}
                      className={`py-2 px-3 rounded-xl text-xs font-semibold transition-all ${
                        formState.gender === 'F'
                          ? 'bg-teal-500 text-slate-950 font-bold shadow-md shadow-teal-500/20'
                          : 'bg-slate-950/70 text-slate-400 border border-slate-800'
                      }`}
                    >
                      Female (F)
                    </button>
                  </div>
                </div>

                <div className="p-3.5 rounded-2xl bg-slate-900 border border-slate-800">
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-xs font-bold text-slate-200">
                      Patient Age:{' '}
                      <span className="text-teal-400 font-mono font-bold text-sm">
                        {formState.age} yrs
                      </span>
                    </label>
                    <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">
                      {getAgeCohort(formState.age)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="18"
                    max="90"
                    value={formState.age}
                    onChange={(e) => {
                      setActivePreset(null);
                      setFormState((prev) => ({ ...prev, age: parseInt(e.target.value) || 50 }));
                    }}
                    className="w-full accent-teal-400 bg-slate-800 rounded-lg h-2 cursor-pointer mt-1"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 mt-2 font-mono">
                    <span>18 yrs</span>
                    <span>50 yrs</span>
                    <span>90 yrs</span>
                  </div>
                </div>
              </div>
            </div>

            {/* 2. Habits & Behavioral Exposures */}
            <div className="bg-slate-900/60 border border-slate-800/80 rounded-3xl p-5 space-y-3">
              <div className="flex items-center gap-2 border-b border-slate-800/80 pb-2.5">
                <Wind className="w-4 h-4 text-teal-400" />
                <h3 className="text-xs uppercase font-extrabold text-teal-400 tracking-wider">
                  2. Lifestyle & Environmental Exposure
                </h3>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
                {renderRadioGroup(
                  'smoking',
                  'Active Tobacco Smoking',
                  'Active or regular daily cigarette tobacco smoking'
                )}
                {renderRadioGroup(
                  'alcohol_consuming',
                  'Alcohol Intake',
                  'Frequent consumption of alcoholic beverages'
                )}
                {renderRadioGroup(
                  'peer_pressure',
                  'Secondhand Smoke (ETS)',
                  'Passive environmental tobacco smoke exposure'
                )}
              </div>
            </div>

            {/* 3. Systemic & Physical Indicators */}
            <div className="bg-slate-900/60 border border-slate-800/80 rounded-3xl p-5 space-y-3">
              <div className="flex items-center gap-2 border-b border-slate-800/80 pb-2.5">
                <HeartPulse className="w-4 h-4 text-teal-400" />
                <h3 className="text-xs uppercase font-extrabold text-teal-400 tracking-wider">
                  3. Systemic & Physical Indicators
                </h3>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
                {renderRadioGroup(
                  'yellow_fingers',
                  'Yellow Nicotine Staining',
                  'Visible digital pigment staining from tobacco tar'
                )}
                {renderRadioGroup(
                  'anxiety',
                  'Anxiety & Agitation',
                  'Frequent psychosomatic nervousness or tension'
                )}
                {renderRadioGroup(
                  'fatigue',
                  'Chronic Daily Fatigue',
                  'Persistent lethargy or abnormal physical exhaustion'
                )}
              </div>
            </div>

            {/* 4. Respiratory & Clinical Complaints */}
            <div className="bg-slate-900/60 border border-slate-800/80 rounded-3xl p-5 space-y-3">
              <div className="flex items-center gap-2 border-b border-slate-800/80 pb-2.5">
                <Stethoscope className="w-4 h-4 text-teal-400" />
                <h3 className="text-xs uppercase font-extrabold text-teal-400 tracking-wider">
                  4. Respiratory Symptoms & Clinical Complaints
                </h3>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 pt-1">
                {renderRadioGroup(
                  'coughing',
                  'Persistent Chronic Cough',
                  'Unresolved cough lasting longer than 3 weeks'
                )}
                {renderRadioGroup(
                  'shortness_of_breath',
                  'Dyspnea (Shortness of Breath)',
                  'Breathlessness occurring upon mild physical exertion'
                )}
                {renderRadioGroup(
                  'wheezing',
                  'Audible Wheezing',
                  'Whistling or musical breathing sounds in chest'
                )}
                {renderRadioGroup(
                  'chest_pain',
                  'Thoracic / Chest Pain',
                  'Pleuritic chest discomfort or persistent dull ache'
                )}
                {renderRadioGroup(
                  'swallowing_difficulty',
                  'Dysphagia (Difficulty Swallowing)',
                  'Impaired or painful passage of food or liquids'
                )}
                {renderRadioGroup(
                  'chronic_disease',
                  'Chronic Lung Condition (COPD)',
                  'Prior clinical history of bronchitis or emphysema'
                )}
                {renderRadioGroup(
                  'allergy',
                  'Respiratory Allergies',
                  'Chronic environmental allergic rhinitis or asthma'
                )}
              </div>
            </div>

            {/* Action Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center gap-2.5 py-4 px-6 rounded-2xl bg-teal-500 hover:bg-teal-400 text-slate-950 font-black text-sm sm:text-base transition-all shadow-xl shadow-teal-500/20 disabled:opacity-50"
            >
              {loading ? (
                <div className="flex items-center gap-2">
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Computing CatGBM Ensemble Risk & AI Explanation...</span>
                </div>
              ) : (
                <>
                  <Sparkles className="w-5 h-5" />
                  <span>Execute CatGBM Prediction & AI Explanation</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Results & AI Explanation Column (5 cols) */}
        <div className="lg:col-span-5 space-y-5">
          {prediction ? (
            <>
              {/* Prediction Card with Gauge */}
              <PredictionCard prediction={prediction} />

              {/* Integrated AI Explanation Panel */}
              <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 space-y-4 shadow-xl">
                <div className="flex items-center justify-between gap-2 border-b border-slate-800 pb-3">
                  <div className="flex items-center gap-2">
                    <Lightbulb className="w-5 h-5 text-teal-400" />
                    <h3 className="text-sm font-bold text-white">AI Educational Explanation</h3>
                  </div>

                  <div className="flex items-center gap-2">
                    <span
                      className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-bold border ${
                        isFallback
                          ? 'bg-amber-500/10 text-amber-300 border-amber-500/30'
                          : 'bg-teal-500/10 text-teal-300 border-teal-500/30'
                      }`}
                    >
                      {isFallback ? <Cpu className="w-3 h-3" /> : <Bot className="w-3 h-3" />}
                      <span>{isFallback ? 'Deterministic Attribution' : 'LLM Generated'}</span>
                    </span>

                    <button
                      type="button"
                      onClick={handleCopyExplanation}
                      title="Copy explanation"
                      className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-all"
                    >
                      {copied ? <Check className="w-3.5 h-3.5 text-teal-400" /> : <Copy className="w-3.5 h-3.5" />}
                    </button>

                    <button
                      type="button"
                      onClick={onRefreshExplanation}
                      disabled={explainLoading}
                      title="Regenerate explanation"
                      className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-all"
                    >
                      <RefreshCw className={`w-3.5 h-3.5 ${explainLoading ? 'animate-spin' : ''}`} />
                    </button>
                  </div>
                </div>

                {explainLoading ? (
                  <div className="py-8 text-center text-xs text-slate-400 animate-pulse flex flex-col items-center justify-center gap-2">
                    <RefreshCw className="w-5 h-5 animate-spin text-teal-400" />
                    <span>Synthesizing plain-language clinical explanation...</span>
                  </div>
                ) : explanationData ? (
                  <div className="text-xs text-slate-300 leading-relaxed space-y-3 font-sans">
                    <div className="bg-slate-950/70 p-4 rounded-2xl border border-slate-800/80 whitespace-pre-line leading-relaxed text-slate-200">
                      {explanationData.explanation}
                    </div>
                    <div className="flex items-start gap-2 bg-amber-500/5 p-3 rounded-xl border border-amber-500/20 text-[11px] text-amber-300/90 italic">
                      <Info className="w-4 h-4 flex-shrink-0 mt-0.5 text-amber-400" />
                      <span>{explanationData.disclaimer}</span>
                    </div>
                  </div>
                ) : (
                  <div className="text-xs text-slate-400 py-2">
                    Click regenerate to synthesize plain-language explanation.
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="bg-slate-900 border border-slate-800 rounded-3xl p-8 text-center flex flex-col items-center justify-center min-h-[420px] shadow-lg">
              <div className="w-16 h-16 rounded-2xl bg-teal-500/10 border border-teal-500/20 flex items-center justify-center text-teal-400 mb-4">
                <Stethoscope className="w-8 h-8" />
              </div>
              <h3 className="text-lg font-bold text-white">Live Prediction Ready</h3>
              <p className="text-xs text-slate-400 max-w-sm mt-2 leading-relaxed">
                Choose a clinical profile preset above or complete the 15 questions, then click{' '}
                <strong className="text-teal-300">Execute CatGBM Prediction</strong> to view instant ensemble risk probabilities and AI explanations.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
