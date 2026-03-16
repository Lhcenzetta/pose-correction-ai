import React, { ButtonHTMLAttributes } from 'react';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger';
  fullWidth?: boolean;
}

export default function Button({ children, variant = 'primary', fullWidth = false, className = '', ...props }: ButtonProps) {
  const baseStyle = "font-bold rounded-full transition-all duration-300 flex justify-center items-center py-3 px-6 h-12";
  const widthStyle = fullWidth ? "w-full" : "w-auto";
  
  const variants = {
    primary: "bg-accent text-[#000] hover:bg-[#3be09b] hover:shadow-[0_0_20px_rgba(79,255,176,0.4)]",
    secondary: "bg-surface2 text-text border border-border hover:border-accent hover:text-accent",
    danger: "bg-accent2 text-white hover:bg-opacity-90 hover:shadow-[0_0_20px_rgba(255,79,123,0.4)]"
  };

  return (
    <button 
      className={`${baseStyle} ${widthStyle} ${variants[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
