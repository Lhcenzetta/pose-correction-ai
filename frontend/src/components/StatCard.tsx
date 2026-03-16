import React from 'react';

interface StatCardProps {
  label: string;
  value: string;
  subValue: string;
  valueColor?: string; // e.g. 'text-accent', 'text-accent3', 'text-yellow-400'
}

export default function StatCard({ label, value, subValue, valueColor = 'text-text' }: StatCardProps) {
  return (
    <div className="bg-surface border border-border rounded-2xl p-5 hover:border-accent/30 transition-colors">
      <div className="text-muted uppercase text-xs font-bold tracking-wider mb-2">
        {label}
      </div>
      <div className={`font-display text-4xl font-bold mb-1 ${valueColor}`}>
        {value}
      </div>
      <div className="text-muted text-sm">
        {subValue}
      </div>
    </div>
  );
}
