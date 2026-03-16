import React from 'react';

interface ExerciseCardProps {
  name: string;
  description: string;
  emoji: string;
  status: 'Available' | 'Soon';
  selected?: boolean;
  onClick?: () => void;
}

export default function ExerciseCard({ name, description, emoji, status, selected, onClick }: ExerciseCardProps) {
  const isLocked = status === 'Soon';
  
  return (
    <div 
      onClick={!isLocked ? onClick : undefined}
      className={`relative bg-surface border-2 rounded-2xl p-7 text-center transition-all duration-300
        ${isLocked ? 'opacity-40 cursor-not-allowed border-border' : 'cursor-pointer hover:-translate-y-1 hover:border-accent/50'}
        ${selected ? 'border-accent bg-accent/5 shadow-[0_0_20px_rgba(79,255,176,0.1)]' : 'border-border'}
      `}
    >
      <div className={`absolute top-4 right-4 text-xs font-bold px-3 py-1 rounded-full ${isLocked ? 'bg-surface2 text-muted border border-border' : 'bg-accent/20 text-accent border border-accent/30'}`}>
        {status}
      </div>
      <div className="text-6xl mb-4 mt-2">
        {emoji}
      </div>
      <h3 className="font-display font-bold text-xl text-text mb-2">
        {name}
      </h3>
      <p className="text-muted text-sm line-clamp-2">
        {description}
      </p>
    </div>
  );
}
