import React from "react";
import PortfolioOptimizer from "./components/PortfolioOptimizer";

const App: React.FC = () => {

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Top Navigation - Dark glass */}
      <header className="sticky top-0 z-50 backdrop-blur-md bg-black/90 border-b border-[#2a2a2a] shadow-lg">
        <div className="max-w-7xl mx-auto flex items-center justify-between px-6 py-4">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <img
              src="/LOGO.png"
              alt="Logo"
              className="h-10 w-10 object-contain"
            />
            <div>
              <span className="text-lg font-semibold tracking-tight text-white">
                Nền tảng tối ưu hóa danh mục đầu tư
              </span>
              <span className="hidden sm:inline ml-2 text-[10px] font-medium text-sky-400 uppercase tracking-wider">
                Fintech
              </span>
            </div>
          </div>

        </div>
      </header>

      {/* Main Content - Full Width */}
      <main className="max-w-7xl mx-auto px-6 py-10">
        <PortfolioOptimizer />
      </main>

      <style>{`
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .animate-fade-in { animation: fadeIn 0.4s ease-out forwards; }
        .animate-slide-up { animation: slideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
        .react-datepicker { border-radius: 16px !important; border: 1px solid #2a2a2a !important; box-shadow: 0 10px 25px rgba(0,0,0,0.5); background: #1a1a1a !important; }
      `}</style>
    </div>
  );
};

export default App;
