import React from 'react';
import { LucideIcon } from 'lucide-react';

interface Props {
  title: string;
  value: string | number;
  subtitle: string;
  icon: LucideIcon;
  badge?: string;
  highlight?: boolean;
}

export const MetricCard: React.FC<Props> = ({
  title,
  value,
  subtitle,
  icon: Icon,
  badge,
  highlight = false,
}) => {
  return (
    <div
      className={`p-5 rounded-2xl border transition-all ${
        highlight
          ? 'bg-gradient-to-br from-teal-950/40 via-slate-900 to-slate-900 border-teal-500/40 shadow-lg shadow-teal-950/20'
          : 'bg-slate-900/60 border-slate-800 hover:border-slate-700'
      }`}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          {title}
        </span>
        <div
          className={`w-9 h-9 rounded-xl flex items-center justify-center ${
            highlight ? 'bg-teal-500/20 text-teal-300' : 'bg-slate-800 text-slate-400'
          }`}
        >
          <Icon className="w-5 h-5" />
        </div>
      </div>

      <div className="mt-4 flex items-baseline gap-2">
        <span className="text-3xl font-black tracking-tight text-white">{value}</span>
        {badge && (
          <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-teal-500/10 text-teal-400 border border-teal-500/20">
            {badge}
          </span>
        )}
      </div>

      <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">{subtitle}</p>
    </div>
  );
};
