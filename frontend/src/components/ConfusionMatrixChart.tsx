import React from 'react';

interface Props {
  confusionMatrix: {
    true_negatives: number;
    false_positives: number;
    false_negatives: number;
    true_positives: number;
  };
  title?: string;
  subtitle?: string;
}

export const ConfusionMatrixChart: React.FC<Props> = ({
  confusionMatrix,
  title = 'Confusion Matrix (Untouched Test Split)',
  subtitle,
}) => {
  const { true_negatives, false_positives, false_negatives, true_positives } = confusionMatrix;
  const total = true_negatives + false_positives + false_negatives + true_positives;

  const pct = (n: number) => (total > 0 ? ((n / total) * 100).toFixed(1) : '0.0');

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h4 className="text-sm font-bold text-white uppercase tracking-wider">{title}</h4>
          <p className="text-xs text-slate-400">
            {subtitle || `Total Held-Out Evaluated Patients: ${total}`}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 max-w-md mx-auto text-center">
        {/* True Negatives */}
        <div className="bg-emerald-950/40 border border-emerald-500/40 p-4 rounded-xl shadow-sm">
          <span className="text-[11px] font-bold uppercase text-emerald-400 block tracking-wider">
            True Negatives (TN)
          </span>
          <span className="text-2xl font-black text-white mt-1 block font-mono">
            {true_negatives}
          </span>
          <span className="text-[11px] text-slate-400 font-mono">{pct(true_negatives)}% of test set</span>
          <p className="text-[10px] text-emerald-300 font-medium mt-1">
            Correctly Identified Low-Risk (NO)
          </p>
        </div>

        {/* False Positives */}
        <div className="bg-amber-950/30 border border-amber-500/40 p-4 rounded-xl shadow-sm">
          <span className="text-[11px] font-bold uppercase text-amber-400 block tracking-wider">
            False Positives (FP)
          </span>
          <span className="text-2xl font-black text-white mt-1 block font-mono">
            {false_positives}
          </span>
          <span className="text-[11px] text-slate-400 font-mono">{pct(false_positives)}% of test set</span>
          <p className="text-[10px] text-amber-300 font-medium mt-1">
            Over-Flagged (Predicted YES, Actual NO)
          </p>
        </div>

        {/* False Negatives */}
        <div className="bg-rose-950/40 border border-rose-500/40 p-4 rounded-xl shadow-sm">
          <span className="text-[11px] font-bold uppercase text-rose-400 block tracking-wider">
            False Negatives (FN)
          </span>
          <span className="text-2xl font-black text-white mt-1 block font-mono">
            {false_negatives}
          </span>
          <span className="text-[11px] text-slate-400 font-mono">{pct(false_negatives)}% of test set</span>
          <p className="text-[10px] text-rose-300 font-medium mt-1">
            Under-Flagged (Predicted NO, Actual YES)
          </p>
        </div>

        {/* True Positives */}
        <div className="bg-teal-950/40 border border-teal-500/40 p-4 rounded-xl shadow-sm">
          <span className="text-[11px] font-bold uppercase text-teal-400 block tracking-wider">
            True Positives (TP)
          </span>
          <span className="text-2xl font-black text-white mt-1 block font-mono">
            {true_positives}
          </span>
          <span className="text-[11px] text-slate-400 font-mono">{pct(true_positives)}% of test set</span>
          <p className="text-[10px] text-teal-300 font-medium mt-1">
            Correctly Identified Elevated Risk (YES)
          </p>
        </div>
      </div>
    </div>
  );
};
