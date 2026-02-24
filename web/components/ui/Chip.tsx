import React from "react";

interface ChipProps {
  label: string;
  active?: boolean;
  onClick?: () => void;
}

export const Chip: React.FC<ChipProps> = ({ label, active = false, onClick }) => (
  <button
    onClick={onClick}
    type="button"
    className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${
      active
        ? "bg-sky-400 text-black shadow-lg shadow-sky-400/40"
        : "bg-[#252525] text-gray-300 hover:bg-[#333]"
    }`}
  >
    {label}
  </button>
);
