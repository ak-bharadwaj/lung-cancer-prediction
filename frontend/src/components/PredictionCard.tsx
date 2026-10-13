import React from 'react';
import {
  AlertCircle,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  CheckCircle2,
  Clock,
  HelpCircle,
  Info,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';
import { PredictionOutput } from '../types';

interface Props {
  prediction: PredictionOutput;
}

const FEATURE_NAMES_MAP: Record<string, { title: string; clinicalTerm: string }> = {
  SMOKING: { title: 'Active Tobacco Smoking', clinicalTerm: 'Tobacco Exposure History' },
  YELLOW_FINGERS: { title: 'Yellow Nicotine Staining', clinicalTerm: 'Digital Nicotine Pigmentation' },
  ANXIETY: { title: 'Reported Anxiety / Tension', clinicalTerm: 'Autonomic / Psychosomatic Agitation' },
  PEER_PRESSURE: { title: 'Secondhand Smoke Exposure', clinicalTerm: 'Passive Environmental Tobacco Exposure' },
  CHRONIC_DISEASE: { title: 'Chronic Lung Disease', clinicalTerm: 'COPD / Chronic Respiratory Condition' },
  FATIGUE: { title: 'Chronic Daily Fatigue', clinicalTerm: 'Persistent Asthenia & Lethargy' },
  ALLERGY: { title: 'Respiratory Allergies', clinicalTerm: 'Bronchial Allergic Sensitivity' },
  WHEEZING: { title: 'Airway Wheezing Sounds', clinicalTerm: 'Audible Rhonchi / Stridor' },
  ALCOHOL_CONSUMING: { title: 'Alcohol Consumption', clinicalTerm: 'Regular Ethanol Intake' },
  COUGHING: { title: 'Persistent Chronic Cough', clinicalTerm: 'Chronic Bronchial Irritation' },
  SHORTNESS_OF_BREATH: { title: 'Shortness of Breath', clinicalTerm: 'Exertional Dyspnea' },
  SWALLOWING_DIFFICULTY: { title: 'Difficulty Swallowing', clinicalTerm: 'Oropharyngeal Dysphagia' },
  CHEST_PAIN: { title: 'Thoracic / Chest Pain', clinicalTerm: 'Pleuritic Thoracic Discomfort' },
  AGE: { title: 'Participant Age', clinicalTerm: 'Biological Chronological Age' },
  GENDER: { title: 'Biological Sex', clinicalTerm: 'Biological Gender Marker' },
};

export const PredictionCard: React.FC<Props> = ({ prediction }) => {
  const isHighRisk = prediction.predicted_class === 'YES';
  const scorePct = prediction.probability_percentage;
  const thPct = Math.round(prediction.decision_threshold * 1000) / 10; // e.g. 45.5%
  const deltaFromCutoff = Math.round((scorePct - thPct) * 10) / 10;
  const isAboveThreshold = scorePct >= thPct;

  // Semi-circular gauge parameters
  // Semi-circle arc: r = 85, center at (140, 115)
  // Arc length = pi * r = 3.14159 * 85 = 267.035
  const arcRadius = 85;
  const arcLength = Math.PI * arcRadius;
  const clampedScore = Math.max(0, Math.min(100, scorePct));
  const strokeOffset = arcLength * (1 - clampedScore / 100);

  // Threshold tick coordinates
  const thFraction = thPct / 100;
  const thAngle = Math.PI * (1 - thFraction); // 0 at right (100%), PI at left (0%)
  const cx = 140;
  const cy = 115;
  const tickInnerR = 72;
  const tickOuterR = 98;
  const tickX1 = cx + tickInnerR * Math.cos(thAngle);
  const tickY1 = cy - tickInnerR * Math.sin(thAngle);
  const tickX2 = cx + tickOuterR * Math.cos(thAngle);
  const tickY2 = cy - tickOuterR * Math.sin(thAngle);

  // Score needle coordinates
  const scoreAngle = Math.PI * (1 - clampedScore / 100);
  const needleLen = 65;
  const needleX = cx + needleLen * Math.cos(scoreAngle);
  const needleY = cy - needleLen * Math.sin(scoreAngle);

  return (
    <div
      className={`rounded-3xl border p-6 transition-all shadow-2xl relative overflow-hidden backdrop-blur-xl ${
        isHighRisk
          ? 'bg-gradient-to-br from-rose-950/40 via-slate-900/90 to-slate-950 border-rose-500/35 shadow-rose-950/20'
          : 'bg-gradient-to-br from-emerald-950/40 via-slate-900/90 to-slate-950 border-emerald-500/35 shadow-emerald-950/20'
      }`}
    >
      {/* Background glow orb */}
      <div
        className={`absolute -top-20 -right-20 w-64 h-64 rounded-full blur-3xl opacity-25 pointer-events-none ${
          isHighRisk ? 'bg-rose-500' : 'bg-emerald-500'
        }`}
      />

      {/* Header Status & Provenance */}
      <div className="flex items-start justify-between relative z-10">
        <div className="flex items-center gap-3">
          <div
            className={`w-12 h-12 rounded-2xl flex items-center justify-center border shadow-lg ${
              isHighRisk
                ? 'bg-rose-500/20 text-rose-400 border-rose-500/40'
                : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40'
            }`}
          >
            {isHighRisk ? <AlertCircle className="w-6 h-6" /> : <ShieldCheck className="w-6 h-6" />}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] uppercase font-extrabold tracking-wider text-slate-400">
                CatGBM Risk Assessment
              </span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                v{prediction.model_version}
              </span>
            </div>
            <h3
              className={`text-xl sm:text-2xl font-black tracking-tight mt-0.5 ${
                isHighRisk ? 'text-rose-400' : 'text-emerald-400'
              }`}
            >
              {isHighRisk ? 'Elevated Survey Risk Indicated' : 'Low / Baseline Risk Indicated'}
            </h3>
          </div>
        </div>

        {/* Confidence / Delta Pill */}
        <div className="text-right hidden sm:block">
          <span
            className={`inline-flex items-center gap-1 text-[11px] font-bold px-2.5 py-1 rounded-full border ${
              isAboveThreshold
                ? 'bg-rose-500/10 text-rose-300 border-rose-500/30'
                : 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30'
            }`}
          >
            {isAboveThreshold ? (
              <ArrowUpRight className="w-3.5 h-3.5 text-rose-400" />
            ) : (
              <ArrowDownRight className="w-3.5 h-3.5 text-emerald-400" />
            )}
            <span>
              {deltaFromCutoff >= 0 ? `+${deltaFromCutoff}%` : `${deltaFromCutoff}%`} vs Cutoff
            </span>
          </span>
        </div>
      </div>

      {/* Radial Semi-Circular Risk Gauge */}
      <div className="mt-6 bg-slate-950/80 p-5 rounded-2xl border border-slate-800/90 relative z-10 shadow-inner">
        <div className="flex flex-col items-center justify-center">
          <div className="relative w-[280px] h-[145px] overflow-hidden flex justify-center items-end">
            <svg
              viewBox="0 0 280 145"
              className="w-[280px] h-[145px] overflow-visible"
            >
              <defs>
                {/* Arc gradient: Emerald -> Amber -> Crimson */}
                <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#10b981" />
                  <stop offset="45%" stopColor="#f59e0b" />
                  <stop offset="100%" stopColor="#f43f5e" />
                </linearGradient>
                <filter id="gaugeGlow" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="4" result="blur" />
                  <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
              </defs>

              {/* Background inactive track */}
              <path
                d="M 55,115 A 85 85 0 0 1 225,115"
                fill="none"
                stroke="#1e293b"
                strokeWidth="16"
                strokeLinecap="round"
              />

              {/* Active illuminated value track */}
              <path
                d="M 55,115 A 85 85 0 0 1 225,115"
                fill="none"
                stroke="url(#gaugeGradient)"
                strokeWidth="16"
                strokeDasharray={`${arcLength}`}
                strokeDashoffset={`${strokeOffset}`}
                strokeLinecap="round"
                filter="url(#gaugeGlow)"
                className="transition-all duration-1000 ease-out"
              />

              {/* Decision threshold cutoff tick notch */}
              <line
                x1={tickX1}
                y1={tickY1}
                x2={tickX2}
                y2={tickY2}
                stroke="#ffffff"
                strokeWidth="2.5"
                strokeLinecap="round"
                className="drop-shadow-md"
              />

              {/* Score indicator needle */}
              <line
                x1={cx}
                y1={cy}
                x2={needleX}
                y2={needleY}
                stroke={isHighRisk ? '#fb7185' : '#34d399'}
                strokeWidth="3"
                strokeLinecap="round"
                className="transition-all duration-700 ease-out"
              />

              {/* Center needle pivot pin */}
              <circle cx={cx} cy={cy} r="6" fill="#0f172a" stroke="#94a3b8" strokeWidth="2.5" />
            </svg>

            {/* In-gauge live numeric score readout */}
            <div className="absolute bottom-1 text-center flex flex-col items-center">
              <span
                className={`text-4xl font-black font-mono tracking-tight leading-none ${
                  isHighRisk ? 'text-rose-400' : 'text-emerald-400'
                }`}
              >
                {scorePct}%
              </span>
              <span className="text-[10px] uppercase font-bold text-slate-400 tracking-wider mt-1">
                {isHighRisk ? 'Elevated Likelihood' : 'Minimal Likelihood'}
              </span>
            </div>
          </div>

          {/* Sub-gauge metrics summary bar */}
          <div className="w-full grid grid-cols-3 gap-2 pt-3 border-t border-slate-800/80 mt-2 text-center text-xs">
            <div className="bg-slate-900/60 p-2 rounded-xl border border-slate-800/50">
              <span className="text-[10px] text-slate-500 uppercase font-mono block">Baseline</span>
              <span className="font-mono font-bold text-emerald-400">0%</span>
            </div>
            <div className="bg-slate-900/60 p-2 rounded-xl border border-slate-800/50">
              <span className="text-[10px] text-slate-500 uppercase font-mono block">Cutoff Notch</span>
              <span className="font-mono font-bold text-amber-400">{thPct}%</span>
            </div>
            <div className="bg-slate-900/60 p-2 rounded-xl border border-slate-800/50">
              <span className="text-[10px] text-slate-500 uppercase font-mono block">Max Risk</span>
              <span className="font-mono font-bold text-rose-400">100%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Top Influential Factors with Clinical Terminology */}
      <div className="mt-6 relative z-10">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5 text-teal-400" />
            <span>Top Influential Survey Indicators</span>
          </h4>
          <span className="text-[10px] text-slate-500 font-mono">Ensemble Split Weight</span>
        </div>

        <div className="space-y-2.5">
          {prediction.contributing_factors.slice(0, 3).map((factor, idx) => {
            const mapped = FEATURE_NAMES_MAP[factor.feature];
            const title = mapped?.title || factor.label || factor.feature;
            const clinicalTerm = mapped?.clinicalTerm;
            const isHighImpact = factor.impact === 'high';
            const isModerate = factor.impact === 'moderate';

            return (
              <div
                key={idx}
                className="flex items-center justify-between p-3.5 rounded-2xl bg-slate-900/80 border border-slate-800/90 text-xs hover:border-slate-700 transition-all shadow-sm"
              >
                <div className="space-y-1 pr-3">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-slate-200 text-xs">{title}</span>
                    <span className="text-[11px] font-mono text-teal-400/90 font-medium">
                      &bull;{' '}
                      {factor.value === 'Present' || factor.value === '2'
                        ? 'Reported Active'
                        : factor.value === 'Absent' || factor.value === '1'
                        ? 'Not Reported'
                        : factor.value}
                    </span>
                  </div>
                  {clinicalTerm && (
                    <span className="text-[10px] font-mono text-slate-400 block">
                      Clinical Term: <span className="text-slate-300">{clinicalTerm}</span>
                    </span>
                  )}
                  <p className="text-[11px] text-slate-400 leading-snug">{factor.description}</p>
                </div>

                <span
                  className={`text-[10px] uppercase font-mono font-extrabold px-2.5 py-1 rounded-lg border flex-shrink-0 ${
                    isHighImpact
                      ? 'bg-rose-500/15 text-rose-300 border-rose-500/30'
                      : isModerate
                      ? 'bg-amber-500/15 text-amber-300 border-amber-500/30'
                      : 'bg-slate-800 text-slate-300 border-slate-700'
                  }`}
                >
                  {factor.impact}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
