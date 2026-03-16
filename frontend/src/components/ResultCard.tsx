import React from 'react';

interface ResultCardProps {
  label: string;
  value: string;
  subValue: string;
  valueColor?: string; // e.g. 'text-accent', 'text-accent2'
}

export default function ResultCard({ label, value, subValue, valueColor = 'text-text' }: ResultCardProps) {
  return (
    <div className="bg-surface border border-border rounded-2xl p-6 text-center shadow-lg hover:border-accent/40 transition-colors">
      <div className="text-muted uppercase text-xs font-bold tracking-widest mb-3">
        {label}
      </div>
      <div className={`font-display text-5xl font-bold mb-3 ${valueColor}`}>
        {value}
      </div>
      <div className="text-muted text-sm px-2">
        {subValue}
      </div>
    </div>
  );
}
