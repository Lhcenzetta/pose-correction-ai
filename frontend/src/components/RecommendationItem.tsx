import React from 'react';

interface RecommendationItemProps {
  text: React.ReactNode;
}

export default function RecommendationItem({ text }: RecommendationItemProps) {
  return (
    <div className="flex items-start gap-4 p-4 rounded-xl bg-surface2/50 border border-border/50 hover:bg-surface2 transition-colors mb-3">
      <div className="text-accent text-xl mt-0.5">💡</div>
      <div className="text-text text-sm leading-relaxed">
        {text}
      </div>
    </div>
  );
}
