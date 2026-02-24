import React from "react";

interface PrimaryButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  type?: "button" | "submit";
  className?: string;
}

export const PrimaryButton: React.FC<PrimaryButtonProps> = ({
  children,
  onClick,
  disabled = false,
  type = "button",
  className = "",
}) => (
  <button
    type={type}
    onClick={onClick}
    disabled={disabled}
    className={`px-6 py-3 rounded-xl bg-sky-400 text-black text-sm font-semibold hover:bg-sky-300 transition-all shadow-lg shadow-sky-400/40 disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none ${className}`}
  >
    {children}
  </button>
);
