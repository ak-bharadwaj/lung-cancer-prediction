import React from 'react';
import { AlertTriangle, ShieldAlert } from 'lucide-react';

interface Props {
  compact?: boolean;
}

export const DisclaimerBanner: React.FC<Props> = ({ compact = false }) => {
  if (compact) {
    return (
      <div className="bg-amber-950/30 border border-amber-500/30 text-amber-200 text-xs px-3.5 py-2 rounded-xl flex items-center gap-2.5 backdrop-blur-sm">
        <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0" />
        <span className="leading-snug">
          <strong className="text-amber-300">Educational Research Demo:</strong> Strictly for machine learning & explainability demonstration. <span className="underline decoration-amber-500/50">Not a clinical diagnostic device.</span>
        </span>
      </div>
    );
  }

  return (
    <div className="relative overflow-hidden bg-gradient-to-r from-amber-950/40 via-slate-900 to-slate-900 border border-amber-500/40 p-4 sm:p-5 rounded-2xl shadow-lg shadow-amber-950/10 mb-6 backdrop-blur-sm">
      <div className="flex items-start gap-3.5">
        <div className="w-9 h-9 rounded-xl bg-amber-500/20 text-amber-400 flex items-center justify-center flex-shrink-0 border border-amber-500/30 mt-0.5">
          <ShieldAlert className="w-5 h-5" />
        </div>
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <h4 className="text-xs font-bold text-amber-300 uppercase tracking-wider">
              Educational & Clinical Non-Diagnostic Notice
            </h4>
            <span className="text-[10px] uppercase font-bold bg-amber-500/10 text-amber-400 px-2 py-0.5 rounded border border-amber-500/20">
              Portfolio Demonstration
            </span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            This application is an educational simulation developed to illustrate tabular machine learning (CatGBM Ensemble) and Large Language Model explainability on survey questionnaire data. It is <strong className="text-amber-200 font-semibold">NOT a medical device, screening program, or medical diagnostic instrument</strong>. Model outputs reflect statistical survey correlations. Anyone experiencing persistent respiratory symptoms must consult a licensed healthcare practitioner.
          </p>
        </div>
      </div>
    </div>
  );
};
