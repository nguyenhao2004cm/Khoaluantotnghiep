import React, { useState, useEffect, useRef } from "react";
import { API_BASE } from "../apiConfig";

export interface CompanySuggestion {
  ma: string;
  ten: string;
}

interface StockSelectorProps {
  onSelect: (item: CompanySuggestion) => void;
  placeholder?: string;
  className?: string;
}

export const StockSelector: React.FC<StockSelectorProps> = ({
  onSelect,
  placeholder = "Nhập mã cổ phiếu...",
  className = "",
}) => {
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState<CompanySuggestion[]>([]);
  const [loading, setLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (query.length > 0) {
      setLoading(true);
      fetch(`${API_BASE}/api/search-company?query=${encodeURIComponent(query)}`)
        .then((res) => res.json())
        .then((data) => {
          setSuggestions(Array.isArray(data) ? data : []);
          setIsOpen(true);
        })
        .catch(() => setSuggestions([]))
        .finally(() => setLoading(false));
    } else {
      setSuggestions([]);
      setIsOpen(false);
    }
  }, [query]);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSelect = (item: CompanySuggestion) => {
    onSelect(item);
    setQuery("");
    setSuggestions([]);
    setIsOpen(false);
  };

  return (
    <div ref={wrapperRef} className={`relative ${className}`}>
      <input
        type="text"
        placeholder={placeholder}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => suggestions.length > 0 && setIsOpen(true)}
        className="w-full px-4 py-3 rounded-xl border border-[#2a2a2a] bg-[#1a1a1a] text-white placeholder-gray-500 focus:ring-2 focus:ring-sky-500 focus:border-sky-500 outline-none text-sm transition-all"
      />

      {isOpen && suggestions.length > 0 && (
        <div className="absolute w-full mt-2 bg-[#1a1a1a] border border-[#2a2a2a] rounded-xl shadow-xl z-50 overflow-hidden">
          {suggestions.map((item, index) => (
            <div
              key={`${item.ma}-${index}`}
              onClick={() => handleSelect(item)}
              className="px-4 py-3 hover:bg-[#252525] cursor-pointer transition-all flex items-center gap-2 border-b border-[#2a2a2a] last:border-b-0"
            >
              <span className="font-semibold text-sky-400 min-w-[3rem]">
                {item.ma}
              </span>
              <span className="text-gray-300 text-sm truncate">{item.ten}</span>
            </div>
          ))}
        </div>
      )}

      {loading && query.length > 0 && (
        <div className="absolute right-4 top-1/2 -translate-y-1/2">
          <div className="w-4 h-4 border-2 border-sky-500 border-t-transparent rounded-full animate-spin" />
        </div>
      )}
    </div>
  );
};
