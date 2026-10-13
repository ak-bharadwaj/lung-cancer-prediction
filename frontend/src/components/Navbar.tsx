import React from 'react';
import { Activity, BarChart3, HelpCircle, Stethoscope } from 'lucide-react';

export type NavTab = 'predict' | 'analytics' | 'about';

interface Props {
  activeTab: NavTab;
  setActiveTab: (tab: NavTab) => void;
  systemStatus?: { status: string; model_loaded: boolean };
}

export const Navbar: React.FC<Props> = ({ activeTab, setActiveTab, systemStatus }) => {
  const tabs = [
    { id: 'predict', label: 'Predict & Explain', icon: Stethoscope },
    { id: 'analytics', label: 'Model Analytics', icon: BarChart3 },
    { id: 'about', label: 'About & Ethics', icon: HelpCircle },
  ];

  return (
    <header className="sticky top-0 z-50 bg-slate-900 border-b border-slate-800 text-white shadow-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Project Title */}
          <div
            className="flex items-center gap-3 cursor-pointer"
            onClick={() => setActiveTab('predict')}
          >
            <div className="w-10 h-10 rounded-xl bg-teal-500/20 border border-teal-400/40 flex items-center justify-center text-teal-300">
              <Activity className="w-6 h-6 text-teal-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-lg tracking-tight bg-gradient-to-r from-teal-300 via-emerald-200 to-cyan-300 bg-clip-text text-transparent">
                  CatGBM LungVision
                </span>
                <span className="text-[10px] uppercase font-semibold bg-teal-900/60 text-teal-300 px-2 py-0.5 rounded border border-teal-700/50">
                  Ensemble ML
                </span>
              </div>
              <p className="text-[11px] text-slate-400 hidden sm:block">
                CatBoost + LightGBM + XGBoost AI Prediction Engine
              </p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="flex items-center gap-1 sm:gap-2">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as NavTab)}
                  className={`flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-xs sm:text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-teal-500 text-slate-950 font-bold shadow-sm shadow-teal-500/20'
                      : 'text-slate-300 hover:text-white hover:bg-slate-800'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Model Health Badge */}
          <div className="hidden sm:flex items-center gap-2 text-xs">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                systemStatus?.model_loaded ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'
              }`}
            />
            <span className="text-slate-300 text-[11px]">
              {systemStatus?.model_loaded ? 'CatGBM Ready' : 'Connecting...'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};
