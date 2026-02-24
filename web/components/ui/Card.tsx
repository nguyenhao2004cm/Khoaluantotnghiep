import React from "react";

interface CardProps {
  children: React.ReactNode;
  className?: string;
}

export const Card: React.FC<CardProps> = ({ children, className = "" }) => (
  <div
    className={`bg-[#1a1a1a] rounded-2xl shadow-lg border border-[#2a2a2a] p-6 hover:border-[#333] transition-all duration-300 ${className}`}
  >
    {children}
  </div>
);
