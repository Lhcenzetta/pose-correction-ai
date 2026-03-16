import React from 'react';

interface SessionItemProps {
  icon: string;
  name: string;
  meta: string;
  score: number;
}

export default function SessionItem({ icon, name, meta, score }: SessionItemProps) {
  let scoreColor = 'text-accent2'; // default low (red)
  if (score >= 80) scoreColor = 'text-accent'; // high (green)
  else if (score >= 60) scoreColor = 'text-yellow-400'; // mid (yellow)

  return (
    <div className="bg-surface border border-border rounded-xl p-4 flex items-center justify-between hover:border-accent/30 transition-colors mb-3">
      <div className="flex items-center gap-4">
        <div className="w-12 h-12 rounded-full bg-accent/20 text-xl flex items-center justify-center border border-accent/20">
          {icon}
        </div>
        <div>
          <h4 className="font-bold text-text text-lg">{name}</h4>
          <p className="text-muted text-sm">{meta}</p>
        </div>
      </div>
      <div className={`font-display text-2xl font-bold ${scoreColor}`}>
        {score}%
      </div>
    </div>
  );
}
