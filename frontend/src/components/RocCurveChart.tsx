import React from 'react';
import { TrendingUp } from 'lucide-react';

interface Props {
  rocCurve: Array<{ fpr: number; tpr: number }>;
  rocAuc: number;
  title?: string;
}

export const RocCurveChart: React.FC<Props> = ({
  rocCurve,
  rocAuc,
  title = 'Receiver Operating Characteristic (ROC Curve)',
}) => {
  // SVG Dimensions
  const width = 360;
  const height = 240;
  const padding = 38;

  const innerWidth = width - padding * 2;
  const innerHeight = height - padding * 2;

  // Scale coordinates: (0,0) is bottom-left, (1,1) is top-right
  const scaleX = (fpr: number) => padding + fpr * innerWidth;
  const scaleY = (tpr: number) => height - padding - tpr * innerHeight;

  // Build path data
  const points = rocCurve.map((pt) => `${scaleX(pt.fpr)},${scaleY(pt.tpr)}`);
  const pathD = points.length > 0 ? `M ${points.join(' L ')}` : '';

  // Area path closing to bottom right
  const areaD =
    points.length > 0
      ? `M ${scaleX(0)},${scaleY(0)} L ${points.join(' L ')} L ${scaleX(1)},${scaleY(0)} Z`
      : '';

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-teal-400" />
          <h4 className="text-sm font-bold text-white uppercase tracking-wider">{title}</h4>
        </div>
        <span className="text-xs font-mono font-bold px-2 py-0.5 rounded-full bg-teal-500/20 text-teal-300 border border-teal-500/30">
          AUC: {rocAuc.toFixed(4)}
        </span>
      </div>
      <p className="text-[11px] text-slate-400 mb-3">
        Discriminative trade-off between True Positive Rate (Sensitivity) and False Positive Rate (1 &minus; Specificity).
      </p>

      <div className="relative flex justify-center">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full max-w-sm overflow-visible text-slate-400"
        >
          <defs>
            <linearGradient id="rocAreaGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#14b8a6" stopOpacity="0.35" />
              <stop offset="100%" stopColor="#14b8a6" stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((v) => (
            <g key={v}>
              {/* Horizontal line */}
              <line
                x1={scaleX(0)}
                y1={scaleY(v)}
                x2={scaleX(1)}
                y2={scaleY(v)}
                stroke="#1e293b"
                strokeDasharray="2,2"
              />
              <text
                x={scaleX(0) - 6}
                y={scaleY(v) + 3}
                textAnchor="end"
                className="text-[9px] fill-slate-500 font-mono"
              >
                {v.toFixed(2)}
              </text>

              {/* Vertical line */}
              <line
                x1={scaleX(v)}
                y1={scaleY(0)}
                x2={scaleX(v)}
                y2={scaleY(1)}
                stroke="#1e293b"
                strokeDasharray="2,2"
              />
              <text
                x={scaleX(v)}
                y={scaleY(0) + 14}
                textAnchor="middle"
                className="text-[9px] fill-slate-500 font-mono"
              >
                {v.toFixed(2)}
              </text>
            </g>
          ))}

          {/* Diagonal random chance baseline (y = x) */}
          <line
            x1={scaleX(0)}
            y1={scaleY(0)}
            x2={scaleX(1)}
            y2={scaleY(1)}
            stroke="#475569"
            strokeDasharray="4,4"
            strokeWidth="1.2"
          />

          {/* Filled Area */}
          {areaD && <path d={areaD} fill="url(#rocAreaGrad)" />}

          {/* ROC Curve Path */}
          {pathD && (
            <path
              d={pathD}
              fill="none"
              stroke="#2dd4bf"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}

          {/* Axis Labels */}
          <text
            x={width / 2}
            y={height - 2}
            textAnchor="middle"
            className="text-[10px] fill-slate-400 font-medium"
          >
            False Positive Rate (1 &minus; Specificity)
          </text>
          <text
            x={-height / 2 + padding / 2}
            y={12}
            transform="rotate(-90)"
            textAnchor="middle"
            className="text-[10px] fill-slate-400 font-medium"
          >
            True Positive Rate (Sensitivity)
          </text>
        </svg>
      </div>

      <div className="flex justify-between items-center text-[10px] text-slate-400 mt-2 px-1 border-t border-slate-800/80 pt-2">
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-0.5 bg-teal-400 inline-block rounded-full" />
          <span>CatGBM Ensemble (AUC = {rocAuc.toFixed(3)})</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-0.5 bg-slate-500 inline-block stroke-dasharray" />
          <span>Random Chance (AUC = 0.500)</span>
        </span>
      </div>
    </div>
  );
};
