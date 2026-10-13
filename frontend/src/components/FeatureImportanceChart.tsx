import React from 'react';

interface Props {
  featureImportances: Record<string, number>;
}

const HUMAN_LABELS: Record<string, string> = {
  SMOKING: 'Smoking History',
  YELLOW_FINGERS: 'Yellow Finger Staining',
  ANXIETY: 'Anxiety Symptoms',
  PEER_PRESSURE: 'Peer Smoke Exposure',
  CHRONIC_DISEASE: 'Chronic Lung Condition',
  FATIGUE: 'Fatigue / Lethargy',
  ALLERGY: 'Respiratory Allergies',
  WHEEZING: 'Wheezing Sound',
  ALCOHOL_CONSUMING: 'Alcohol Consumption',
  COUGHING: 'Persistent Cough',
  SHORTNESS_OF_BREATH: 'Shortness of Breath',
  SWALLOWING_DIFFICULTY: 'Difficulty Swallowing',
  CHEST_PAIN: 'Chest Pain',
  AGE: 'Participant Age',
  GENDER: 'Gender',
};

export const FeatureImportanceChart: React.FC<Props> = ({ featureImportances }) => {
  const items = Object.entries(featureImportances).sort((a, b) => b[1] - a[1]);
  const maxVal = items.length > 0 ? items[0][1] : 1.0;

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
      <h4 className="text-sm font-bold text-white uppercase tracking-wider mb-1">
        Ensemble Feature Importance Ranking
      </h4>
      <p className="text-xs text-slate-400 mb-5">
        Normalized weight contributions aggregated across CatBoost, LightGBM, and XGBoost trees.
      </p>

      <div className="space-y-2.5">
        {items.map(([feat, score]) => {
          const pct = (score * 100).toFixed(1);
          const barWidth = Math.max(4, Math.round((score / maxVal) * 100));
          const label = HUMAN_LABELS[feat] || feat;

          return (
            <div key={feat} className="text-xs">
              <div className="flex justify-between items-center mb-1">
                <span className="font-semibold text-slate-300">{label}</span>
                <span className="text-slate-400 font-mono text-[11px]">{pct}%</span>
              </div>
              <div className="w-full h-2.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-teal-500 to-emerald-400 rounded-full transition-all duration-500"
                  style={{ width: `${barWidth}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
