import React from 'react';

interface ProgressBarProps {
  label: string;
  percentage: number;
  colorClass: string; // e.g., 'bg-accent', 'bg-accent2'
}

export default function ProgressBar({ label, percentage, colorClass }: ProgressBarProps) {
  return (
    <div className="mb-4">
      <div className="flex justify-between items-center mb-2">
        <span className="text-muted text-sm font-medium">{label}</span>
      </div>
      <div className="h-3 bg-surface2 rounded-full overflow-hidden border border-border">
        <div 
          className={`h-full rounded-full transition-all duration-1000 ${colorClass}`} 
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
