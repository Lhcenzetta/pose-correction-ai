import React from 'react';

interface DurationPillProps {
  minutes: number;
  selected?: boolean;
  onClick?: () => void;
}

export default function DurationPill({ minutes, selected, onClick }: DurationPillProps) {
  return (
    <div 
      onClick={onClick}
      className={`cursor-pointer transition-all duration-300 border rounded-xl py-6 flex flex-col items-center justify-center
        ${selected ? 'bg-accent/10 border-accent shadow-[0_0_15px_rgba(79,255,176,0.15)] text-accent' : 'bg-surface border-border text-text hover:border-accent/50'}
      `}
    >
      <div className="font-display text-4xl font-bold mb-1">
        {minutes}
      </div>
      <div className={`text-sm font-medium ${selected ? 'text-accent/80' : 'text-muted'}`}>
        min
      </div>
    </div>
  );
}
