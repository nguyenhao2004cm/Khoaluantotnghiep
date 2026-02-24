import React from "react";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";
import { Calendar } from "lucide-react";

interface StartDatePickerProps {
  value: Date;
  onChange: (date: Date) => void;
  label?: string;
}

export const StartDatePicker: React.FC<StartDatePickerProps> = ({
  value,
  onChange,
  label = "Ngày bắt đầu",
}) => (
  <div>
    <label className="block text-sm text-gray-400 mb-2">{label}</label>
    <div className="relative">
      <DatePicker
        selected={value}
        onChange={(date) => date && onChange(date)}
        dateFormat="dd/MM/yyyy"
        maxDate={new Date()}
        className="w-full pl-11 pr-4 py-3 rounded-xl border border-[#2a2a2a] bg-[#1a1a1a] text-white focus:ring-2 focus:ring-sky-500 focus:border-sky-500 outline-none text-sm"
      />
      <Calendar
        size={18}
        className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none"
      />
    </div>
    <p className="mt-1.5 text-xs text-gray-500">
      Ngày kết thúc: {new Date().toLocaleDateString("vi-VN")}
    </p>
  </div>
);
